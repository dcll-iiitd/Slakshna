use crate::chain::{LatticeBlock, Blockchain, BoxError};
use crate::config::Config;
use crate::state::{State, StateSnapshot};
use crate::network::Network;

use async_trait::async_trait;
use axum::extract::ws::{Message, WebSocket};
use futures::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{RwLock, mpsc};
use tracing::info;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum P2PMessage {
    Hello { node_id: String, node_type: String },
    Welcome { node_id: String, height: u64, peers: Vec<String> },
    GetState,
    StateSnapshot(StateSnapshot),
    NewLatticeBlock(LatticeBlock),
    GetBlock { height: u64 },
    Ping,
    Pong,
}

#[derive(Clone)]
pub struct ConnectedPeer {
    pub node_id: String,
    pub node_type: String,
    pub tx: mpsc::Sender<P2PMessage>,
}

pub struct StarNetwork {
    config: Config,
    blockchain: Arc<RwLock<Blockchain>>,
    state: Arc<RwLock<State>>,
    peers: Arc<RwLock<HashMap<String, ConnectedPeer>>>,
    browsers: Arc<RwLock<HashMap<String, mpsc::Sender<P2PMessage>>>>,
}

impl StarNetwork {
    pub fn new(
        config: Config,
        blockchain: Arc<RwLock<Blockchain>>,
        state: Arc<RwLock<State>>,
    ) -> Self {
        StarNetwork {
            config,
            blockchain,
            state,
            peers: Arc::new(RwLock::new(HashMap::new())),
            browsers: Arc::new(RwLock::new(HashMap::new())),
        }
    }


    pub async fn handle_peer_connection(
        network_lock: Arc<RwLock<Self>>,
        ws: WebSocket,
        peer_id: String,
    ) {
        let (mut sender, mut receiver) = ws.split();
        let (tx, mut rx) = mpsc::channel::<P2PMessage>(100);

        // Send welcome message safely avoiding deadlocks
        let height;
        let peers_list;
        let node_id;
        
        {
            let net = network_lock.read().await;
            let state_guard = net.state.read().await;
            height = state_guard.get_height().unwrap_or(0);
            
            let peers_guard = net.peers.read().await;
            peers_list = peers_guard.keys().cloned().collect::<Vec<String>>();
            
            node_id = net.config.node.id.clone();
        }

        let welcome = P2PMessage::Welcome {
            node_id,
            height,
            peers: peers_list,
        };

        if let Ok(msg) = serde_json::to_string(&welcome) {
            let _ = sender.send(Message::Text(msg)).await;
        }

        // Spawn sender task
        let sender_task = tokio::spawn(async move {
            while let Some(msg) = rx.recv().await {
                if let Ok(text) = serde_json::to_string(&msg) {
                    if sender.send(Message::Text(text)).await.is_err() {
                        break;
                    }
                }
            }
        });

        let network_lock_clone = network_lock.clone();
        let peer_id_clone = peer_id.clone();
        let tx_clone = tx.clone();

        while let Some(Ok(msg)) = receiver.next().await {
            if let Message::Text(text) = msg {
                if let Ok(p2p_msg) = serde_json::from_str::<P2PMessage>(&text) {
                    match p2p_msg {
                        P2PMessage::Hello { node_id, node_type } => {
                            tracing::info!("🔗 Peer connected: {} ({})", node_id, node_type);
                            let peer = ConnectedPeer {
                                node_id: node_id.clone(),
                                node_type,
                                tx: tx_clone.clone(),
                            };
                            let net = network_lock_clone.read().await;
                            net.peers.write().await.insert(node_id, peer);
                        }
                        // ==========================================
                        // THE MISSING LOGIC: Handle new dual-layer messages!
                        // ==========================================
                        P2PMessage::NewLatticeBlock(block) => {
                            tracing::info!("📡 Received Lattice Block from P2P");
                            let net = network_lock_clone.read().await;
                            let mut bc = net.blockchain.write().await;
                            bc.add_lattice_block(block);
                        }
                        P2PMessage::Ping => {
                            let _ = tx_clone.send(P2PMessage::Pong).await;
                        }
                        _ => {}
                    }
                }
            }
        }

        // Clean up
        let net = network_lock.read().await;
        net.peers.write().await.remove(&peer_id_clone);
        sender_task.abort();
        tracing::info!("🔌 Peer disconnected: {}", peer_id_clone);
    }
}

#[async_trait]
impl Network for StarNetwork {
 
    async fn start(&mut self) -> Result<(), BoxError> {
        let is_master = self.config.node.node_type == "master";
        
        if is_master {
            info!("Starting P2P server for master node...");
        } else {
            let master_url = if let Some(ref star) = self.config.network.star {
                star.master_url.clone()
            } else {
                "http://localhost:8000".to_string()
            };
            let my_node_id = self.config.node.id.clone();
            let my_node_type = self.config.node.node_type.clone();

            let peers_clone = self.peers.clone();
            let bc_clone = self.blockchain.clone();
            let _state_clone = self.state.clone();

            if !master_url.is_empty() {
                info!("Dialing master node at: {}/p2p", master_url);
                
                tokio::spawn(async move {
                    let url = format!("{}/p2p", master_url).replace("http://", "ws://");
                    match tokio_tungstenite::connect_async(&url).await {
                        Ok((ws_stream, _)) => {
                            info!("✅ Successfully connected to Master Node!");
                            
                            let (mut write, mut read) = ws_stream.split();
                            
                            // 1. Send Hello Message to Master
                            let hello = P2PMessage::Hello { 
                                node_id: my_node_id.clone(), 
                                node_type: my_node_type 
                            };
                            if let Ok(msg) = serde_json::to_string(&hello) {
                                let _ = write.send(tokio_tungstenite::tungstenite::Message::Text(msg)).await;
                            }
                            
                            // 2. Map OUTBOUND messages so Node 2 can broadcast FL Work TO the master
                            let (tx, mut rx) = mpsc::channel::<P2PMessage>(100);
                            peers_clone.write().await.insert(
                                "master".to_string(),
                                ConnectedPeer {
                                    node_id: "master".to_string(),
                                    node_type: "master".to_string(),
                                    tx: tx.clone(),
                                }
                            );
                            
                            let outbound_task = tokio::spawn(async move {
                                while let Some(msg) = rx.recv().await {
                                    if let Ok(text) = serde_json::to_string(&msg) {
                                        if write.send(tokio_tungstenite::tungstenite::Message::Text(text)).await.is_err() {
                                            break;
                                        }
                                    }
                                }
                            });

                            // 3. Listen for INCOMING blocks from the master
                            while let Some(Ok(msg)) = read.next().await {
                                if let tokio_tungstenite::tungstenite::Message::Text(text) = msg {
                                    if let Ok(p2p_msg) = serde_json::from_str::<P2PMessage>(&text) {
                                        match p2p_msg {
                                            P2PMessage::NewLatticeBlock(block) => {
                                                tracing::info!("📘 Received Lattice Block from Master");
                                                let mut bc = bc_clone.write().await;
                                                bc.add_lattice_block(block);
                                            }
                                            _ => {}
                                        }
                                    }
                                }
                            }
                            
                            outbound_task.abort();
                            peers_clone.write().await.remove("master");
                            tracing::error!("🔌 Disconnected from Master Node");
                        }
                        Err(e) => {
                            tracing::error!("❌ Failed to connect to Master Node: {}", e);
                        }
                    }
                });
            }
        }
        
        Ok(())
    }
    
    // ... keep rest of the trait implementation ...

    
    async fn broadcast_lattice_block(&self, block: &LatticeBlock) -> Result<(), BoxError> {
        let msg = P2PMessage::NewLatticeBlock(block.clone());
        let peers = self.peers.read().await;
        for (_, peer) in peers.iter() {
            let _ = peer.tx.send(msg.clone()).await;
        }
        Ok(())
    }

    async fn get_active_peer_ids(&self) -> Vec<String> {
        let peers = self.peers.read().await;
        peers.keys().cloned().collect()
    }

    fn peer_count(&self) -> usize {
        self.peers.try_read().map(|p| p.len()).unwrap_or(0)
    }

    fn browser_count(&self) -> usize {
        self.browsers.try_read().map(|b| b.len()).unwrap_or(0)
    }
}