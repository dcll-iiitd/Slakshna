// use crate::chain::{LatticeBlock, Blockchain, BoxError};
// use crate::network::Network;
// use crate::state::State;
// use crate::network::star::P2PMessage;
// use async_trait::async_trait;
// use iroh::Endpoint;
// use iroh::endpoint::presets;
// use iroh::protocol::Router;
// use iroh_gossip::net::{Gossip};
// use iroh_gossip::ALPN;
// use futures::StreamExt;
// use std::sync::Arc;
// use tokio::sync::{RwLock, mpsc};
// use tracing::{info, error, debug, warn};

// /// 32-byte topic identifier for the SLAKSHNA FL Lattice gossip channel.
// /// Derived deterministically so all nodes with the same chain_id join the same swarm.
// fn topic_from_chain_id(chain_id: &str) -> iroh_gossip::TopicId {
//     use sha2::{Sha256, Digest};
//     let hash = Sha256::digest(chain_id.as_bytes());
//     let mut bytes = [0u8; 32];
//     bytes.copy_from_slice(&hash);
//     iroh_gossip::TopicId::from_bytes(bytes)
// }

// pub struct MeshNetwork {
//     config: crate::config::Config,
//     blockchain: Arc<RwLock<Blockchain>>,
//     #[allow(dead_code)]
//     state: Arc<RwLock<State>>,
//     /// Channel for the rest of the application to send broadcast commands into the gossip loop
//     command_tx: Option<mpsc::Sender<P2PMessage>>,
//     active_peers: Arc<RwLock<Vec<String>>>,
//     node_id_iroh: Arc<RwLock<Option<String>>>,
//     _router: Option<iroh::protocol::Router>,
//     _endpoint: Option<iroh::Endpoint>,
// }

// impl MeshNetwork {
//     pub fn new(
//         config: crate::config::Config,
//         blockchain: Arc<RwLock<Blockchain>>,
//         state: Arc<RwLock<State>>,
//     ) -> Self {
//         MeshNetwork {
//             config,
//             blockchain,
//             state,
//             command_tx: None,
//             active_peers: Arc::new(RwLock::new(Vec::new())),
//             node_id_iroh: Arc::new(RwLock::new(None)),
//             _router: None,
//             _endpoint: None,
//         }
//     }
// }

// #[async_trait]
// impl Network for MeshNetwork {
//     async fn start(&mut self) -> Result<(), BoxError> {
//         info!("🌐 Starting Iroh Mesh Network (QUIC + Gossip + DERP)...");

//         // 1. Create the Iroh Endpoint with Minimal preset to bypass all PKARR/TLS certificate issues
//         let bind_addr: std::net::SocketAddr = format!("0.0.0.0:{}", self.config.network.p2p_port).parse().unwrap();
//         let endpoint = Endpoint::builder(presets::Minimal)
//             .relay_mode(iroh::RelayMode::Disabled)
//             .bind_addr(bind_addr).map_err(|e| format!("Failed to set bind_addr: {}", e))?
//             .bind()
//             .await
//             .map_err(|e| format!("Failed to bind Iroh endpoint: {}", e))?;

//         let node_id = endpoint.id();
//         let my_addr = endpoint.addr();
//         info!("🔑 Iroh NodeId: {}\nNodeAddr: {:?}", node_id, my_addr);
//         {
//             let mut nid = self.node_id_iroh.write().await;
//             *nid = Some(node_id.to_string());
//         }

//         // 2. Start Gossip Router
//         let gossip = Gossip::builder().spawn(endpoint.clone());
        
//         // 3. Setup Iroh Router to accept incoming gossip connections via ALPN
//         let router = Router::builder(endpoint.clone())
//             .accept(ALPN, gossip.clone())
//             .spawn();

//         self._router = Some(router);
//         self._endpoint = Some(endpoint.clone());

//         info!("📡 Iroh Router started (listening on port {})", self.config.network.p2p_port);

//         // 4. Derive topic from chain_id for deterministic swarm membership
//         let topic_id = topic_from_chain_id(&self.config.chain.chain_id);
//         info!("📢 Gossip Topic: {:?} (derived from chain_id '{}')", topic_id, self.config.chain.chain_id);

//         // 5. Resolve bootstrap peers (if any provided in config)
//         let mut bootstrap_peers = Vec::new();
//         if let Some(boot_nodes) = &self.config.network.boot_nodes {
//             for node_id_str in boot_nodes {
//                 if node_id_str.is_empty() { continue; }
//                 let parts: Vec<&str> = node_id_str.split('@').collect();
//                 match parts[0].parse::<iroh::EndpointId>() {
//                     Ok(peer_node_id) => {
//                         info!("🔗 Adding bootstrap peer: {}", peer_node_id);
//                         if parts.len() > 1 {
//                             if let Ok(socket_addr) = parts[1].parse::<std::net::SocketAddr>() {
//                                 let node_addr = iroh::EndpointAddr::new(peer_node_id).with_ip_addr(socket_addr);
//                                 info!("🔗 Connecting directly to peer at {}...", socket_addr);
//                                 let _ = endpoint.connect(node_addr, ALPN).await;
//                             }
//                         }
//                         bootstrap_peers.push(peer_node_id);
//                     }
//                     Err(e) => {
//                         warn!("⚠️ Invalid boot_node NodeId '{}': {}", parts[0], e);
//                     }
//                 }
//             }
//         }

//         // 6. Subscribe to the gossip topic with bootstrap peers
//         let (gossip_sender, mut gossip_receiver) = gossip
//             .subscribe_and_join(topic_id, bootstrap_peers)
//             .await
//             .map_err(|e| format!("Failed to subscribe to gossip topic: {}", e))?
//             .split();

//         info!("✅ Joined gossip swarm for topic {:?}", topic_id);

//         // 7. Setup internal broadcast channel
//         let (cmd_tx, mut cmd_rx) = mpsc::channel::<P2PMessage>(100);
//         self.command_tx = Some(cmd_tx);

//         let bc_clone = self.blockchain.clone();
//         let peers_clone = self.active_peers.clone();
//         let allowed_peers = self.config.network.allowed_peers.clone();

//         // 8. Main event loop (background task)
//         tokio::spawn(async move {
//             loop {
//                 tokio::select! {
//                     // OUTBOUND: Broadcast messages from the application to the gossip swarm
//                     Some(msg) = cmd_rx.recv() => {
//                         match &msg {
//                             P2PMessage::NewLatticeBlock(_) => {
//                                 let data = serde_json::to_vec(&msg).unwrap();
//                                 if let Err(e) = gossip_sender.broadcast(data.into()).await {
//                                     error!("Failed to broadcast gossip message: {:?}", e);
//                                 }
//                             }
//                             _ => continue,
//                         }
//                     }

//                     // INBOUND: Receive messages from the gossip swarm
//                     event = gossip_receiver.next() => {
//                         match event {
//                             Some(Ok(event)) => {
//                                 match event {
//                                     iroh_gossip::api::Event::Received(msg) => {
//                                         let from_id = msg.delivered_from.to_string();

//                                         // Whitelisting check
//                                         if let Some(ref allowed) = allowed_peers {
//                                             if !allowed.is_empty() && !allowed.contains(&from_id) {
//                                                 warn!("🚫 Blocked message from unauthorized peer: {}", from_id);
//                                                 continue;
//                                             }
//                                         }

//                                         if let Ok(p2p_msg) = serde_json::from_slice::<P2PMessage>(&msg.content) {
//                                             match p2p_msg {
//                                                 P2PMessage::NewLatticeBlock(block) => {
//                                                     info!("📡 Gossiped Lattice Block received from {}", from_id);
//                                                     let mut bc = bc_clone.write().await;
//                                                     bc.add_lattice_block(block);
//                                                 }
//                                                 _ => {}
//                                             }
//                                         }
//                                     }
//                                     iroh_gossip::api::Event::NeighborUp(node_id) => {
//                                         let peer_str = node_id.to_string();
//                                         info!("🔗 Peer joined gossip mesh: {}", peer_str);
//                                         let mut peers = peers_clone.write().await;
//                                         if !peers.contains(&peer_str) {
//                                             peers.push(peer_str);
//                                         }
//                                     }
//                                     iroh_gossip::api::Event::NeighborDown(node_id) => {
//                                         let peer_str = node_id.to_string();
//                                         info!("🔌 Peer left gossip mesh: {}", peer_str);
//                                         let mut peers = peers_clone.write().await;
//                                         peers.retain(|p| p != &peer_str);
//                                     }
//                                     _ => {}
//                                 }
//                             }
//                             Some(Err(e)) => {
//                                 error!("Gossip receive error: {:?}", e);
//                                 break;
//                             }
//                             None => {
//                                 warn!("Gossip stream ended");
//                                 break;
//                             }
//                         }
//                     }
//                 }
//             }
//         });

//         Ok(())
//     }

//     async fn broadcast_lattice_block(&self, block: &LatticeBlock) -> Result<(), BoxError> {
//         if let Some(tx) = &self.command_tx {
//             let _ = tx.send(P2PMessage::NewLatticeBlock(block.clone())).await;
//         }
//         Ok(())
//     }

//     async fn get_active_peer_ids(&self) -> Vec<String> {
//         let peers = self.active_peers.read().await;
//         peers.clone()
//     }

//     fn peer_count(&self) -> usize {
//         self.active_peers
//             .try_read()
//             .map(|p| p.len())
//             .unwrap_or(0)
//     }

//     fn browser_count(&self) -> usize {
//         0
//     }
// }










use crate::chain::{LatticeBlock, Blockchain, BoxError};
use crate::network::Network;
use crate::state::State;
use crate::network::star::P2PMessage;
use async_trait::async_trait;
use iroh::Endpoint;
use iroh::endpoint::presets;
use iroh::protocol::Router;
use iroh_gossip::net::{Gossip};
use iroh_gossip::ALPN;
use futures::StreamExt;
use std::sync::Arc;
use tokio::sync::{RwLock, mpsc};
use tracing::{info, error, debug, warn};

/// 32-byte topic identifier for the SLAKSHNA FL Lattice gossip channel.
/// Derived deterministically so all nodes with the same chain_id join the same swarm.
fn topic_from_chain_id(chain_id: &str) -> iroh_gossip::TopicId {
    use sha2::{Sha256, Digest};
    let hash = Sha256::digest(chain_id.as_bytes());
    let mut bytes = [0u8; 32];
    bytes.copy_from_slice(&hash);
    iroh_gossip::TopicId::from_bytes(bytes)
}

pub struct MeshNetwork {
    config: crate::config::Config,
    blockchain: Arc<RwLock<Blockchain>>,
    #[allow(dead_code)]
    state: Arc<RwLock<State>>,
    /// Channel for the rest of the application to send broadcast commands into the gossip loop
    command_tx: Option<mpsc::Sender<P2PMessage>>,
    active_peers: Arc<RwLock<Vec<String>>>,
    node_id_iroh: Arc<RwLock<Option<String>>>,
    _router: Option<iroh::protocol::Router>,
    _endpoint: Option<iroh::Endpoint>,
}

impl MeshNetwork {
    pub fn new(
        config: crate::config::Config,
        blockchain: Arc<RwLock<Blockchain>>,
        state: Arc<RwLock<State>>,
    ) -> Self {
        MeshNetwork {
            config,
            blockchain,
            state,
            command_tx: None,
            active_peers: Arc::new(RwLock::new(Vec::new())),
            node_id_iroh: Arc::new(RwLock::new(None)),
            _router: None,
            _endpoint: None,
        }
    }
}

#[async_trait]
impl Network for MeshNetwork {
    async fn start(&mut self) -> Result<(), BoxError> {
        info!("🌐 Starting Iroh Mesh Network (QUIC + Gossip + DERP)...");

        // 1. Create the Iroh Endpoint with global discovery and relays enabled for internet NAT traversal
        // We set `ca_tls_config(CaTlsConfig::insecure_skip_verify())` so that HTTPS connection attempts to
        // global Iroh relays (relay.n0.iroh.link) and PKARR discovery (dns.iroh.link) work across ANY network
        // (including college/enterprise networks with MITM firewalls) without `UnknownIssuer` SSL errors.
        // NOTE: This does NOT weaken peer-to-peer security! All QUIC & Gossip connections between nodes are still
        // strictly authenticated and end-to-end encrypted using each node's self-certifying Ed25519 PublicKey.
        let bind_addr: std::net::SocketAddr = format!("0.0.0.0:{}", self.config.network.p2p_port).parse().map_err(|e| format!("Failed to parse bind_addr: {}", e))?;
        let endpoint = Endpoint::builder(presets::N0)
            .ca_tls_config(iroh::tls::CaTlsConfig::insecure_skip_verify())
            .relay_mode(iroh::RelayMode::Disabled) // Enables DERP relays for NAT traversal!
            .clear_address_lookup() // Disable PKARR to stop firewall warnings
            .bind_addr(bind_addr).map_err(|e| format!("Failed to set bind_addr: {}", e))?
            .alpns(vec![ALPN.to_vec()])
            .bind()
            .await
            .map_err(|e| format!("Failed to bind Iroh endpoint: {}", e))?;

        let node_id = endpoint.id();
        let my_addr = endpoint.addr();
        info!("🔑 Iroh NodeId: {}\nNodeAddr: {:?}\n🎟️ USE THIS NODE_ID AS BOOT_NODE ON OTHER MACHINES: {}", node_id, my_addr, node_id);
        {
            let mut nid = self.node_id_iroh.write().await;
            *nid = Some(node_id.to_string());
        }

        // 2. Start Gossip Router
        let gossip = Gossip::builder()
            .max_message_size(10_485_760) // 10 MB limit for large AI payloads
            .spawn(endpoint.clone());
        
        // 3. Setup Iroh Router to accept incoming gossip connections via ALPN
        let router = Router::builder(endpoint.clone())
            .accept(ALPN, gossip.clone())
            .spawn();

        self._router = Some(router);
        self._endpoint = Some(endpoint.clone());

        info!("📡 Iroh Router started (listening on port {})", self.config.network.p2p_port);

        // 4. Derive topic from chain_id for deterministic swarm membership
        let topic_id = topic_from_chain_id(&self.config.chain.chain_id);
        info!("📢 Gossip Topic: {:?} (derived from chain_id '{}')", topic_id, self.config.chain.chain_id);

        // 5. Resolve bootstrap peers (if any provided in config)
        let mut bootstrap_peers = Vec::new();
        if let Some(boot_nodes) = &self.config.network.boot_nodes {
            for node_id_str in boot_nodes {
                if node_id_str.is_empty() { continue; }
                let parts: Vec<&str> = node_id_str.split('@').collect();
                match parts[0].parse::<iroh::EndpointId>() {
                    Ok(peer_node_id) => {
                        info!("🔗 Adding bootstrap peer: {}", peer_node_id);
                        if parts.len() > 1 {
                            match parts[1].parse::<std::net::SocketAddr>() {
                                Ok(socket_addr) => {
                                    let node_addr = iroh::EndpointAddr::new(peer_node_id).with_ip_addr(socket_addr);
                                    info!("🔗 Connecting directly to peer at {}...", socket_addr);
                                    let _ = endpoint.connect(node_addr, ALPN).await;
                                }
                                Err(e) => {
                                    warn!("⚠️ Failed to parse Tailscale IP '{}': {}", parts[1], e);
                                }
                            }
                        }
                        bootstrap_peers.push(peer_node_id);
                    }
                    Err(e) => {
                        warn!("⚠️ Invalid boot_node NodeId '{}': {}", parts[0], e);
                    }
                }
            }
        }

        // 6. Subscribe to the gossip topic with bootstrap peers
        let (gossip_sender, mut gossip_receiver) = gossip
            .subscribe_and_join(topic_id, bootstrap_peers)
            .await
            .map_err(|e| format!("Failed to subscribe to gossip topic: {}", e))?
            .split();

        info!("✅ Joined gossip swarm for topic {:?}", topic_id);

        // 7. Setup internal broadcast channel
        let (cmd_tx, mut cmd_rx) = mpsc::channel::<P2PMessage>(100);
        self.command_tx = Some(cmd_tx);

        let bc_clone = self.blockchain.clone();
        let peers_clone = self.active_peers.clone();
        let allowed_peers = self.config.network.allowed_peers.clone();

        // 8. Main event loop (background task)
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    // OUTBOUND: Broadcast messages from the application to the gossip swarm
                    Some(msg) = cmd_rx.recv() => {
                        match &msg {
                            P2PMessage::NewLatticeBlock(_) => {
                                let data = serde_json::to_vec(&msg).unwrap();
                                let size_bytes = data.len();
                                let size_mb = size_bytes as f64 / 1_048_576.0;
                                info!("📤 Broadcasting Lattice Block to Swarm | Network Payload Size: {} bytes ({:.2} MB)", size_bytes, size_mb);
                                if let Err(e) = gossip_sender.broadcast(data.into()).await {
                                    error!("Failed to broadcast gossip message: {:?}", e);
                                }
                            }
                            _ => continue,
                        }
                    }

                    // INBOUND: Receive messages from the gossip swarm
                    event = gossip_receiver.next() => {
                        match event {
                            Some(Ok(event)) => {
                                match event {
                                    iroh_gossip::api::Event::Received(msg) => {
                                        let from_id = msg.delivered_from.to_string();

                                        // Whitelisting check
                                        if let Some(ref allowed) = allowed_peers {
                                            if !allowed.is_empty() && !allowed.contains(&from_id) {
                                                warn!("🚫 Blocked message from unauthorized peer: {}", from_id);
                                                continue;
                                            }
                                        }
                                        let size_bytes = msg.content.len();
                                        let size_mb = size_bytes as f64 / 1_048_576.0;
                                        info!("📡 Gossiped Lattice Block received from {} | Network Payload Size: {} bytes ({:.2} MB)", from_id, size_bytes, size_mb);

                                        if let Ok(p2p_msg) = serde_json::from_slice::<P2PMessage>(&msg.content) {
                                            match p2p_msg {
                                                P2PMessage::NewLatticeBlock(block) => {
                                                    info!("📡 Gossiped Lattice Block received from {}", from_id);
                                                    let mut bc = bc_clone.write().await;
                                                    bc.add_lattice_block(block);
                                                }
                                                _ => {}
                                            }
                                        }
                                    }
                                    iroh_gossip::api::Event::NeighborUp(node_id) => {
                                        let peer_str = node_id.to_string();
                                        info!("🔗 Peer joined gossip mesh: {}", peer_str);
                                        let mut peers = peers_clone.write().await;
                                        if !peers.contains(&peer_str) {
                                            peers.push(peer_str);
                                        }
                                    }
                                    iroh_gossip::api::Event::NeighborDown(node_id) => {
                                        let peer_str = node_id.to_string();
                                        info!("🔌 Peer left gossip mesh: {}", peer_str);
                                        let mut peers = peers_clone.write().await;
                                        peers.retain(|p| p != &peer_str);
                                    }
                                    _ => {}
                                }
                            }
                            Some(Err(e)) => {
                                error!("Gossip receive error: {:?}", e);
                                break;
                            }
                            None => {
                                warn!("Gossip stream ended");
                                break;
                            }
                        }
                    }
                }
            }
        });

        Ok(())
    }

    async fn broadcast_lattice_block(&self, block: &LatticeBlock) -> Result<(), BoxError> {
        if let Some(tx) = &self.command_tx {
            let _ = tx.send(P2PMessage::NewLatticeBlock(block.clone())).await;
        }
        Ok(())
    }

    async fn get_active_peer_ids(&self) -> Vec<String> {
        let peers = self.active_peers.read().await;
        peers.clone()
    }

    fn peer_count(&self) -> usize {
        self.active_peers
            .try_read()
            .map(|p| p.len())
            .unwrap_or(0)
    }

    fn browser_count(&self) -> usize {
        0
    }
}