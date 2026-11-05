use crate::agents::protocol::{AgentProtocol, AGENT_ALPN};
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
    /// Channel for the rest of the application to send broadcast commands into the gossip loop
    command_tx: Option<mpsc::Sender<P2PMessage>>,
    active_peers: Arc<RwLock<Vec<String>>>,
    endpoint_id: Arc<RwLock<Option<String>>>,
    _router: Option<iroh::protocol::Router>,
    _endpoint: Option<iroh::Endpoint>,
    agent_slot: crate::agents::protocol::AgentSlot,
    topics: Option<Arc<ProgramTopics>>,
    blob_store: Option<iroh_blobs::api::Store>,
}

impl MeshNetwork {
    pub fn new(
        config: crate::config::Config,
        history: Arc<RwLock<UpdateHistory>>,
        state: Arc<RwLock<State>>,
    ) -> Self {
        MeshNetwork {
            config,
            history,
            state,
            command_tx: None,
            active_peers: Arc::new(RwLock::new(Vec::new())),
            endpoint_id: Arc::new(RwLock::new(None)),
            _router: None,
            _endpoint: None,
            agent_slot: Default::default(),
            topics: None,
            blob_store: None,
        }
    }

    /// Per-program gossip topics (None before `start`).
    pub fn topics(&self) -> Option<Arc<ProgramTopics>> {
        self.topics.clone()
    }

    /// The iroh-blobs store served on this endpoint (None before `start`, or with agents disabled).
    pub fn blob_store(&self) -> Option<iroh_blobs::api::Store> {
        self.blob_store.clone()
    }

    /// Broadcast an emergency halt/resume to the federation.
    pub async fn broadcast_halt(&self, msg: crate::agents::safety::HaltMsg) {
        if let Some(tx) = &self.command_tx {
            let _ = tx.send(P2PMessage::Halt(msg)).await;
        }
    }

    /// EndpointIds of current gossip neighbours (for blob providers).
    pub fn neighbour_ids(&self) -> Vec<iroh::EndpointId> {
        self.active_peers
            .try_read()
            .map(|p| p.iter().filter_map(|s| s.parse().ok()).collect())
            .unwrap_or_default()
    }

    /// Shared handle to the live neighbour list.
    pub fn neighbours_handle(&self) -> Arc<RwLock<Vec<String>>> {
        self.active_peers.clone()
    }

    /// Attach the agent service once it exists; the agent ALPN handler picks it up.
    pub fn agent_slot(&self) -> crate::agents::protocol::AgentSlot {
        self.agent_slot.clone()
    }

    /// The bound Iroh endpoint (None before `start`).
    pub fn endpoint(&self) -> Option<iroh::Endpoint> {
        self._endpoint.clone()
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
                Ok(id) => push(id),
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
            .alpns(vec![ALPN.to_vec(), AGENT_ALPN.to_vec(), iroh_blobs::ALPN.to_vec()]);

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

        let topics = Arc::new(ProgramTopics::new(
            gossip.clone(),
            self.history.clone(),
            self.active_peers.clone(),
            self.config.network.allowed_peers.clone(),
        ));
        self.topics = Some(topics);

        // Routing by ALPN: gossip for the training mesh, agent protocol for
        // Steps 4-5 traffic. Both share this one authenticated endpoint.
        let agent_node_id = self.history.read().await.node_identity.to_string();
        let agent_protocol = AgentProtocol::new(
            self.history.clone(),
            agent_node_id,
            self.config.network.allowed_peers.clone(),
        )
        .with_agent_slot(self.agent_slot.clone());
        let mut router_builder = Router::builder(endpoint.clone())
            .accept(ALPN, gossip.clone())
            .accept(AGENT_ALPN, agent_protocol);
        if self.config.agents.enabled {
            // Artifacts (candidate diffs, ...) move by BLAKE3 hash over iroh-blobs.
            let dir = std::path::Path::new(&self.config.node.data_dir).join("blobs");
            let store = iroh_blobs::store::fs::FsStore::load(&dir)
                .await
                .map_err(|e| format!("Failed to open blob store: {}", e))?;
            router_builder = router_builder.accept(iroh_blobs::ALPN, iroh_blobs::BlobsProtocol::new(&store, None));
            self.blob_store = Some((*store).clone());
        }
        let router = router_builder.spawn();

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
        let allowed_peers = self.config.network.allowed_peers.clone();
        let agent_slot = self.agent_slot.clone();

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
                            P2PMessage::Halt(_) => {
                                match serde_json::to_vec(&msg) {
                                    Ok(data) => {
                                        if let Err(e) = gossip_sender.broadcast(data.into()).await {
                                            error!("Failed to broadcast halt: {:?}", e);
                                        }
                                    }
                                    Err(e) => error!("Failed to serialize halt: {}", e),
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
                                        info!("📡 Gossiped model update received from {} | Network Payload Size: {} bytes ({:.2} MB)", from_id, size_bytes, size_mb);

                                        if let Ok(p2p_msg) = serde_json::from_slice::<P2PMessage>(&msg.content) {
                                            match p2p_msg {
                                                P2PMessage::NewUpdate(record) => {
                                                    let mut history = history_clone.write().await;
                                                    let _ = history.record_update(record);
                                                }
                                                P2PMessage::Halt(h) => {
                                                    match agent_slot.get() {
                                                        Some(svc) => match svc.kill.apply(&h) {
                                                            Ok(true) => warn!("🛑 Halt message applied: {:?} seq {} ({})", h.action, h.seq, h.reason),
                                                            Ok(false) => {}
                                                            Err(e) => warn!("Ignored halt message from {}: {}", from_id, e),
                                                        },
                                                        None => debug!("Halt message ignored: no agent service on this node"),
                                                    }
                                                }
                                                _ => {}
                                            }
                                        }
                                    }
                                    iroh_gossip::api::Event::NeighborUp(endpoint_id) => {
                                        let peer_str = endpoint_id.to_string();
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


/// Per-program gossip topics (A2): one topic per research program, so only nodes
/// taking part in a program receive its records. Records still verify and land in
/// the same history as everything else.
pub struct ProgramTopics {
    gossip: Gossip,
    history: Arc<RwLock<UpdateHistory>>,
    active_peers: Arc<RwLock<Vec<String>>>,
    allowed_peers: Option<Vec<String>>,
    senders: tokio::sync::Mutex<std::collections::HashMap<String, iroh_gossip::api::GossipSender>>,
}

pub fn program_topic(program: &str) -> iroh_gossip::TopicId {
    let hash = Sha256::digest([b"slakshna/program-topic/v1\0".as_slice(), program.as_bytes()].concat());
    let mut bytes = [0u8; 32];
    bytes.copy_from_slice(&hash);
    iroh_gossip::TopicId::from_bytes(bytes)
}

impl ProgramTopics {
    fn new(gossip: Gossip, history: Arc<RwLock<UpdateHistory>>, active_peers: Arc<RwLock<Vec<String>>>, allowed_peers: Option<Vec<String>>) -> Self {
        ProgramTopics { gossip, history, active_peers, allowed_peers, senders: Default::default() }
    }

    /// Subscribe to a program's topic (idempotent), bootstrapping from current neighbours.
    pub async fn join(&self, program: &str) -> Result<(), BoxError> {
        let mut senders = self.senders.lock().await;
        if senders.contains_key(program) {
            return Ok(());
        }
        let peers: Vec<EndpointId> = self.active_peers.read().await.iter().filter_map(|s| s.parse().ok()).collect();
        let (sender, mut receiver) = self
            .gossip
            .subscribe(program_topic(program), peers)
            .await
            .map_err(|e| format!("Failed to subscribe to program topic: {}", e))?
            .split();
        senders.insert(program.to_string(), sender);
        let history = self.history.clone();
        let allowed = self.allowed_peers.clone();
        let program = program.to_string();
        tokio::spawn(async move {
            while let Some(ev) = receiver.next().await {
                match ev {
                    Ok(iroh_gossip::api::Event::Received(msg)) => {
                        let from = msg.delivered_from.to_string();
                        if let Some(a) = &allowed {
                            if !a.is_empty() && !a.contains(&from) {
                                continue;
                            }
                        }
                        if let Ok(P2PMessage::NewUpdate(rec)) = serde_json::from_slice::<P2PMessage>(&msg.content) {
                            let _ = history.write().await.record_update(rec);
                        }
                    }
                    Ok(_) => {}
                    Err(e) => {
                        warn!("program topic {} receive error: {:?}", program, e);
                        break;
                    }
                }
            }
        });
        Ok(())
    }

    /// Broadcast a record on the program's topic, joining it first if needed.
    pub async fn broadcast(&self, program: &str, record: &UpdateRecord) -> Result<(), BoxError> {
        self.join(program).await?;
        let data = serde_json::to_vec(&P2PMessage::NewUpdate(record.clone()))?;
        let senders = self.senders.lock().await;
        if let Some(s) = senders.get(program) {
            s.broadcast(data.into()).await.map_err(|e| format!("program broadcast failed: {:?}", e))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agents::graph::make_record;
    use crate::agents::records::*;
    use crate::config::Config;
    use crate::identity::Keypair;
    use iroh::address_lookup::MemoryLookup;

    async fn history(kp: &Keypair) -> Arc<RwLock<UpdateHistory>> {
        let dir = std::env::temp_dir().join(format!("slk-mesh-{}", rand::random::<u64>()));
        let cfg: Config = toml::from_str(&format!(
            "[federation]\nid=\"t\"\nname=\"t\"\n[node]\nid=\"x\"\ndata_dir=\"{}\"\n[network]\nhost=\"127.0.0.1\"\napi_port=0\nws_port=0\np2p_port=0\n[logging]\nlevel=\"error\"\n",
            dir.display()
        ))
        .unwrap();
        let state = Arc::new(RwLock::new(State::new(&cfg.node.data_dir).unwrap()));
        Arc::new(RwLock::new(UpdateHistory::new(cfg, state, kp.node_id()).await.unwrap()))
    }

    struct Peer { topics: ProgramTopics, hist: Arc<RwLock<UpdateHistory>>, kp: Keypair, _router: Router, ep: Endpoint }

    async fn peer(lookup: &MemoryLookup, neighbours: Arc<RwLock<Vec<String>>>) -> Peer {
        let kp = Keypair::generate();
        let hist = history(&kp).await;
        let ep = Endpoint::builder(presets::Minimal).relay_mode(RelayMode::Disabled).alpns(vec![ALPN.to_vec()]).address_lookup(lookup.clone()).bind().await.unwrap();
        let gossip = Gossip::builder().spawn(ep.clone());
        let router = Router::builder(ep.clone()).accept(ALPN, gossip.clone()).spawn();
        lookup.add_endpoint_info(ep.addr());
        Peer { topics: ProgramTopics::new(gossip, hist.clone(), neighbours, None), hist, kp, _router: router, ep }
    }

    fn candidate(kp: &Keypair, program: &str, n: u8) -> crate::history::UpdateRecord {
        let mut r = make_record(&kp.node_id().to_string(), &"0".repeat(64), KIND_CANDIDATE, &Candidate { program: program.into(), diff_hash: format!("{:064x}", n), parents: vec![], agenda_item: None });
        r.sign(kp);
        r
    }

    #[test]
    fn program_topics_are_distinct_per_program_and_from_the_federation_topic() {
        assert_eq!(program_topic("p1"), program_topic("p1"));
        assert_ne!(program_topic("p1"), program_topic("p2"));
        assert_ne!(program_topic("p1"), topic_from_federation_id("p1"));
    }

    /// Records for a program reach exactly the peers that joined its topic, and bad records are refused on receipt.
    #[tokio::test]
    async fn program_records_travel_only_on_the_programs_own_topic() {
        let lookup = MemoryLookup::new();
        let na: Arc<RwLock<Vec<String>>> = Default::default();
        let nb: Arc<RwLock<Vec<String>>> = Default::default();
        let nc: Arc<RwLock<Vec<String>>> = Default::default();
        let (a, b, c) = (peer(&lookup, na.clone()).await, peer(&lookup, nb.clone()).await, peer(&lookup, nc.clone()).await);
        // bootstrap: a knows b; b knows a; c knows a (but will not join the program)
        na.write().await.push(b.ep.id().to_string());
        nb.write().await.push(a.ep.id().to_string());
        nc.write().await.push(a.ep.id().to_string());

        let program = "a".repeat(64);
        a.topics.join(&program).await.unwrap();
        b.topics.join(&program).await.unwrap();
        // wait for the swarm to form, then broadcast until delivered (gossip is best effort while neighbours connect)
        let rec = candidate(&a.kp, &program, 1);
        let mut delivered = false;
        for _ in 0..40 {
            a.topics.broadcast(&program, &rec).await.unwrap();
            tokio::time::sleep(Duration::from_millis(150)).await;
            if b.hist.read().await.peer_updates.get(&a.kp.node_id().to_string()).map_or(false, |l| !l.is_empty()) {
                delivered = true;
                break;
            }
        }
        assert!(delivered, "B joined the program topic and must receive A's record");
        assert_eq!(b.hist.read().await.peer_updates[&a.kp.node_id().to_string()][0].hash, rec.hash);
        assert!(c.hist.read().await.peer_updates.is_empty(), "C never joined the program topic and saw nothing");
        // a forged record on the topic is dropped by the receiver's verification
        let mut forged = candidate(&a.kp, &program, 2);
        forged.signature = "00".repeat(64);
        a.topics.broadcast(&program, &forged).await.unwrap();
        tokio::time::sleep(Duration::from_millis(600)).await;
        assert_eq!(b.hist.read().await.peer_updates[&a.kp.node_id().to_string()].len(), 1);
    }
}
