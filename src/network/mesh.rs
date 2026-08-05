use crate::history::{UpdateRecord, UpdateHistory, BoxError};
use crate::network::{Network, P2PMessage};
use crate::state::State;
use async_trait::async_trait;
use futures::StreamExt;
use iroh::address_lookup::MemoryLookup;
use iroh::endpoint::presets;
use iroh::protocol::Router;
use iroh::{Endpoint, EndpointAddr, EndpointId, RelayMode, SecretKey};
use iroh_gossip::net::Gossip;
use iroh_gossip::ALPN;
use iroh_mainline_address_lookup::DhtAddressLookup;
use iroh_mdns_address_lookup::{DiscoveryEvent, MdnsAddressLookup};
use sha2::{Digest, Sha256};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{RwLock, mpsc};
use tracing::{info, error, debug, warn};

/// How often every node re-offers its full set of known peers to the gossip
/// membership layer. This is what heals a split federation without anyone
/// playing coordinator: each node keeps reaching for everyone it has ever met.
const REJOIN_INTERVAL: Duration = Duration::from_secs(60);

/// 32-byte topic identifier for the SLAKSHNA federated learning gossip channel.
/// Derived deterministically so all nodes in the same federation join the same swarm.
fn topic_from_federation_id(federation_id: &str) -> iroh_gossip::TopicId {
    let hash = Sha256::digest(federation_id.as_bytes());
    let mut bytes = [0u8; 32];
    bytes.copy_from_slice(&hash);
    iroh_gossip::TopicId::from_bytes(bytes)
}

/// DNS-SD service label for this federation's mDNS advertisements.
///
/// Scoping the label by federation means a node only ever discovers members of
/// its own federation on the LAN. Kept short and lowercase-alphanumeric: DNS-SD
/// labels are capped at 15 characters.
fn mdns_service_name(federation_id: &str) -> String {
    let hash = Sha256::digest(federation_id.as_bytes());
    format!("sl{}", hex::encode(&hash[..5]))
}

pub struct MeshNetwork {
    config: crate::config::Config,
    history: Arc<RwLock<UpdateHistory>>,
    state: Arc<RwLock<State>>,
    dynamic_allowed_peers: Arc<RwLock<Option<Vec<String>>>>,
    dynamic_blocked_peers: Arc<RwLock<Option<Vec<String>>>>,
    /// Channel for the rest of the application to send broadcast commands into the gossip loop
    command_tx: Option<mpsc::Sender<P2PMessage>>,
    active_peers: Arc<RwLock<Vec<String>>>,
    endpoint_id: Arc<RwLock<Option<String>>>,
    _router: Option<iroh::protocol::Router>,
    _endpoint: Option<iroh::Endpoint>,
}

impl MeshNetwork {
    pub fn new(
        config: crate::config::Config,
        history: Arc<RwLock<UpdateHistory>>,
        state: Arc<RwLock<State>>,
    ) -> Self {
        MeshNetwork {
            dynamic_allowed_peers: Arc::new(RwLock::new(config.network.allowed_peers.clone())),
            dynamic_blocked_peers: Arc::new(RwLock::new(config.network.blocked_peers.clone())),
            config,
            history,
            state,
            command_tx: None,
            active_peers: Arc::new(RwLock::new(Vec::new())),
            endpoint_id: Arc::new(RwLock::new(None)),
            _router: None,
            _endpoint: None,
        }
    }

    pub async fn update_permissions(&self, allowed: Option<Vec<String>>, blocked: Option<Vec<String>>) {
        let mut wl = self.dynamic_allowed_peers.write().await;
        *wl = allowed;
        let mut bl = self.dynamic_blocked_peers.write().await;
        *bl = blocked;
    }

    /// This node's Iroh EndpointId, once the endpoint has been bound.
    pub async fn endpoint_id(&self) -> Option<String> {
        self.endpoint_id.read().await.clone()
    }

    /// Iroh transport key, derived from the federation keypair persisted in
    /// RocksDB so the EndpointId survives restarts.
    async fn transport_secret_key(&self) -> Result<SecretKey, BoxError> {
        let state = self.state.read().await;
        let keypair = state
            .get_keypair()
            .ok_or_else(|| BoxError::from("node identity not loaded before starting the network"))?;
        Ok(SecretKey::from_bytes(&keypair.transport_seed()))
    }

    /// Seed peers to reach for on startup: whatever the config lists, plus every
    /// peer this node has met in a previous run. Entries may carry an explicit
    /// `@host:port`, which is pinned into `direct` for closed networks where no
    /// discovery mechanism is reachable.
    async fn seed_peers(&self, direct: &MemoryLookup) -> Vec<EndpointId> {
        let mut seeds: Vec<EndpointId> = Vec::new();

        let mut push = |id: EndpointId| {
            if !seeds.contains(&id) {
                seeds.push(id);
            }
        };

        for entry in self.config.network.peers.iter().flatten() {
            if entry.is_empty() {
                continue;
            }
            let (id_str, addr_str) = match entry.split_once('@') {
                Some((id, addr)) => (id, Some(addr)),
                None => (entry.as_str(), None),
            };

            let peer_id = match id_str.parse::<EndpointId>() {
                Ok(id) => id,
                Err(e) => {
                    warn!("⚠️ Ignoring unparseable peer '{}': {}", id_str, e);
                    continue;
                }
            };

            let peer_id_str = peer_id.to_string();
            if let Some(ref allowed) = self.config.network.allowed_peers {
                if !allowed.is_empty() && !allowed.contains(&peer_id_str) {
                    warn!("🚫 Seed node not in whitelist: {}", peer_id_str);
                    continue;
                }
            }
            if let Some(ref blocked) = self.config.network.blocked_peers {
                if blocked.contains(&peer_id_str) {
                    warn!("🚫 Seed node is blacklisted: {}", peer_id_str);
                    continue;
                }
            }

            if let Some(addr_str) = addr_str {
                match addr_str.parse::<std::net::SocketAddr>() {
                    Ok(socket_addr) => {
                        info!("📌 Pinned direct address for {}: {}", peer_id.fmt_short(), socket_addr);
                        direct.add_endpoint_info(
                            EndpointAddr::new(peer_id).with_ip_addr(socket_addr),
                        );
                    }
                    Err(e) => warn!("⚠️ Ignoring unparseable address '{}': {}", addr_str, e),
                }
            }

            push(peer_id);
        }

        let remembered = {
            let state = self.state.read().await;
            state.known_peers()
        };
        for id_str in remembered {
            match id_str.parse::<EndpointId>() {
                Ok(id) => {
                    let peer_id_str = id.to_string();
                    if let Some(ref allowed) = self.config.network.allowed_peers {
                        if !allowed.is_empty() && !allowed.contains(&peer_id_str) {
                            continue;
                        }
                    }
                    if let Some(ref blocked) = self.config.network.blocked_peers {
                        if blocked.contains(&peer_id_str) {
                            continue;
                        }
                    }
                    push(id)
                },
                Err(e) => warn!("⚠️ Dropping unparseable cached peer '{}': {}", id_str, e),
            }
        }

        seeds
    }
}

#[async_trait]
impl Network for MeshNetwork {
    async fn start(&mut self) -> Result<(), BoxError> {
        info!("🌐 Starting Iroh mesh (QUIC + gossip, no bootstrap node)...");

        let discovery = self.config.discovery.clone();
        let secret_key = self.transport_secret_key().await?;
        let my_endpoint_id = secret_key.public();

        // 1. Serverless address lookup. mDNS covers the local network, the
        //    BitTorrent mainline DHT covers the internet; both are symmetric —
        //    this node advertises itself on exactly the services it queries.
        //    `MemoryLookup` carries any addresses pinned in the config.
        let direct = MemoryLookup::new();

        let mdns = if discovery.mdns {
            let service_name = mdns_service_name(&self.config.federation.id);
            match MdnsAddressLookup::builder().service_name(service_name.clone()).build(my_endpoint_id) {
                Ok(mdns) => {
                    info!("📻 mDNS discovery active on the local network (service '{}')", service_name);
                    Some(mdns)
                }
                Err(e) => {
                    warn!("⚠️ mDNS discovery unavailable, continuing without it: {}", e);
                    None
                }
            }
        } else {
            None
        };

        // 2. Bind the endpoint. `presets::N0` bundles the public pkarr/DNS
        //    address lookup and the default relays; both are opt-out.
        //    `insecure_skip_verify` keeps HTTPS lookups working behind campus
        //    firewalls that MITM TLS. It does not weaken peer-to-peer security:
        //    every QUIC and gossip connection is still authenticated and
        //    end-to-end encrypted against the peer's Ed25519 public key.
        let bind_addr: std::net::SocketAddr = format!("0.0.0.0:{}", self.config.network.p2p_port)
            .parse()
            .map_err(|e| format!("Failed to parse bind_addr: {}", e))?;

        let mut builder = Endpoint::builder(presets::N0)
            .secret_key(secret_key)
            .ca_tls_config(iroh::tls::CaTlsConfig::insecure_skip_verify())
            .bind_addr(bind_addr)
            .map_err(|e| format!("Failed to set bind_addr: {}", e))?
            .alpns(vec![ALPN.to_vec()]);

        if !discovery.dns {
            // Drop the n0 pkarr publisher/resolver the preset installed.
            builder = builder.clear_address_lookup();
        }
        builder = builder.address_lookup(direct.clone());
        if let Some(mdns) = mdns.clone() {
            builder = builder.address_lookup(mdns);
        }
        if discovery.dht {
            builder = builder.address_lookup(DhtAddressLookup::builder());
            info!("🕸️ Mainline DHT address lookup active (serverless global resolution)");
        }
        if !discovery.relay {
            builder = builder.relay_mode(RelayMode::Disabled);
        }

        let endpoint = builder
            .bind()
            .await
            .map_err(|e| format!("Failed to bind Iroh endpoint: {}", e))?;

        let endpoint_id = endpoint.id();
        info!("🔑 Iroh EndpointId: {}", endpoint_id);
        info!("📍 Endpoint address: {:?}", endpoint.addr());
        info!("🤝 Any peer can join this federation with: peers = [\"{}\"]", endpoint_id);
        {
            let mut eid = self.endpoint_id.write().await;
            *eid = Some(endpoint_id.to_string());
        }

        // 3. Gossip + router. Every node accepts inbound gossip, so any member
        //    is a valid entry point into the federation.
        let gossip = Gossip::builder()
            .max_message_size(10_485_760) // 10 MB limit for large AI payloads
            .spawn(endpoint.clone());

        let router = Router::builder(endpoint.clone())
            .accept(ALPN, gossip.clone())
            .spawn();

        self._router = Some(router);
        self._endpoint = Some(endpoint.clone());

        info!("📡 Iroh router listening on port {}", self.config.network.p2p_port);

        // 4. Topic is derived from the federation id, so membership is decided
        //    by shared configuration rather than by dialling a specific node.
        let topic_id = topic_from_federation_id(&self.config.federation.id);
        info!(
            "📢 Gossip topic {:?} (derived from federation '{}')",
            topic_id,
            self.config.federation.id
        );

        let seeds = self.seed_peers(&direct).await;
        if seeds.is_empty() {
            info!("🌱 No peers known yet — starting alone and waiting to be discovered");
        } else {
            info!("🌱 Reaching for {} known peer(s)", seeds.len());
        }

        // 5. `subscribe` rather than `subscribe_and_join`: a node must come up
        //    even when it is the only member so far. Peers merge into the swarm
        //    whenever they appear, in any order.
        let (gossip_sender, mut gossip_receiver) = gossip
            .subscribe(topic_id, seeds)
            .await
            .map_err(|e| format!("Failed to subscribe to gossip topic: {}", e))?
            .split();

        info!("✅ Subscribed to the federation gossip topic");

        // 6. Setup internal broadcast channel
        let (cmd_tx, mut cmd_rx) = mpsc::channel::<P2PMessage>(100);
        self.command_tx = Some(cmd_tx);

        let history_clone = self.history.clone();
        let peers_clone = self.active_peers.clone();
        let state_clone = self.state.clone();
        let dyn_allowed = self.dynamic_allowed_peers.clone();
        let dyn_blocked = self.dynamic_blocked_peers.clone();

        // 7. Feed mDNS discoveries straight into the gossip membership layer.
        if let Some(mdns) = mdns {
            let sender = gossip_sender.clone();
            let state = self.state.clone();
            tokio::spawn(async move {
                let mut events = mdns.subscribe().await;
                while let Some(event) = events.next().await {
                    match event {
                        DiscoveryEvent::Discovered { endpoint_info, .. } => {
                            let peer_id = endpoint_info.endpoint_id;
                            if peer_id == my_endpoint_id {
                                continue;
                            }
                            debug!("📻 Discovered federation peer on the local network: {}", peer_id.fmt_short());
                            remember_peer(&state, &peer_id.to_string()).await;
                            if let Err(e) = sender.join_peers(vec![peer_id]).await {
                                warn!("⚠️ Failed to join discovered peer {}: {:?}", peer_id.fmt_short(), e);
                            }
                        }
                        DiscoveryEvent::Expired { endpoint_id } => {
                            debug!("📻 Local-network peer went quiet: {}", endpoint_id.fmt_short());
                        }
                        _ => {}
                    }
                }
                debug!("mDNS discovery stream ended");
            });
        }

        // 8. Keep reaching for everyone we have ever met. Without a bootstrap
        //    node this is what lets a federation reassemble after a restart or
        //    a network partition.
        {
            let sender = gossip_sender.clone();
            let state = self.state.clone();
            tokio::spawn(async move {
                let mut ticker = tokio::time::interval(REJOIN_INTERVAL);
                ticker.tick().await; // the first tick fires immediately
                loop {
                    ticker.tick().await;
                    let known: Vec<EndpointId> = {
                        let state = state.read().await;
                        state
                            .known_peers()
                            .iter()
                            .filter_map(|id| id.parse::<EndpointId>().ok())
                            .filter(|id| *id != my_endpoint_id)
                            .collect()
                    };
                    if known.is_empty() {
                        continue;
                    }
                    debug!("🔁 Re-offering {} remembered peer(s) to the swarm", known.len());
                    if let Err(e) = sender.join_peers(known).await {
                        debug!("Re-join attempt failed: {:?}", e);
                    }
                }
            });
        }

        // 9. Main event loop (background task)
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    // OUTBOUND: Broadcast messages from the application to the gossip swarm
                    Some(msg) = cmd_rx.recv() => {
                        match &msg {
                            P2PMessage::NewUpdate(_) => {
                                let data = match serde_json::to_vec(&msg) {
                                    Ok(data) => data,
                                    Err(e) => {
                                        error!("Failed to serialize gossip message: {}", e);
                                        continue;
                                    }
                                };
                                let size_bytes = data.len();
                                let size_mb = size_bytes as f64 / 1_048_576.0;
                                info!("📤 Broadcasting model update to swarm | Network Payload Size: {} bytes ({:.2} MB)", size_bytes, size_mb);
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

                                        let allowed_peers = dyn_allowed.read().await;
                                        let blocked_peers = dyn_blocked.read().await;

                                        let size_bytes = msg.content.len();
                                        let size_mb = size_bytes as f64 / 1_048_576.0;

                                        if let Ok(p2p_msg) = serde_json::from_slice::<P2PMessage>(&msg.content) {
                                            match p2p_msg {
                                                P2PMessage::NewUpdate(record) => {
                                                    // Application-layer Blacklist Check
                                                    if let Some(ref blocked) = *blocked_peers {
                                                        if blocked.contains(&record.node_id) {
                                                            warn!("🚫 Ignored model update from blacklisted author: {}", record.node_id);
                                                            continue;
                                                        }
                                                    }
                                                    
                                                    // Application-layer Whitelist Check
                                                    if let Some(ref allowed) = *allowed_peers {
                                                        if !allowed.is_empty() && !allowed.contains(&record.node_id) {
                                                            warn!("🚫 Ignored model update from unauthorized author: {}", record.node_id);
                                                            continue;
                                                        }
                                                    }

                                                    info!("📡 Gossiped model update received from author {} | Network Payload Size: {} bytes ({:.2} MB)", record.node_id, size_bytes, size_mb);
                                                    let mut history = history_clone.write().await;
                                                    history.record_update(record);
                                                }
                                                _ => {}
                                            }
                                        }
                                    }
                                    iroh_gossip::api::Event::NeighborUp(endpoint_id) => {
                                        let peer_str = endpoint_id.to_string();

                                        let allowed_peers = dyn_allowed.read().await;
                                        let blocked_peers = dyn_blocked.read().await;

                                        // Whitelisting & Blacklisting for neighbors
                                        if let Some(ref allowed) = *allowed_peers {
                                            if !allowed.is_empty() && !allowed.contains(&peer_str) {
                                                warn!("🚫 Unauthorized peer joined gossip mesh (ignored): {}", peer_str);
                                                continue;
                                            }
                                        }
                                        if let Some(ref blocked) = *blocked_peers {
                                            if blocked.contains(&peer_str) {
                                                warn!("🚫 Blacklisted peer joined gossip mesh (ignored): {}", peer_str);
                                                continue;
                                            }
                                        }

                                        info!("🔗 Peer joined gossip mesh: {}", peer_str);
                                        remember_peer(&state_clone, &peer_str).await;
                                        let mut peers = peers_clone.write().await;
                                        if !peers.contains(&peer_str) {
                                            peers.push(peer_str);
                                        }
                                    }
                                    iroh_gossip::api::Event::NeighborDown(endpoint_id) => {
                                        let peer_str = endpoint_id.to_string();
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

    async fn broadcast_update(&self, record: &UpdateRecord) -> Result<(), BoxError> {
        if let Some(tx) = &self.command_tx {
            let _ = tx.send(P2PMessage::NewUpdate(record.clone())).await;
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

/// Persist a peer so a future run of this node can find the federation again
/// on its own. Failing to write is not fatal — it only costs us the shortcut.
async fn remember_peer(state: &Arc<RwLock<State>>, endpoint_id: &str) {
    let mut state = state.write().await;
    if let Err(e) = state.remember_peer(endpoint_id) {
        debug!("Could not cache peer {}: {}", endpoint_id, e);
    }
}
