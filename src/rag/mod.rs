//! Retrieval-augmented generation on top of the mesh.
//!
//! The node itself does no ML: document search and answer writing live in a
//! local Python sidecar (`rag_engine.py`). This module is the node's side of
//! that — the sidecar client and the query pipeline behind `POST /rag/query`.

pub mod adversary;
pub mod directory;
pub mod merkle;
pub mod protocol;
pub mod registry;
pub mod retrieval;
pub mod sidecar;

use crate::config::{ RagConfig, Retrieval };
use crate::history::BoxError;
use directory::{ CentroidDirectory, DirectoryStatus };
use merkle::{ Commitment, SignedRoot };
use protocol::{ RagRequest, RequestBody, ResponseBody };
use registry::{ PeerRootStatus, RootOutcome, RootRegistry };
use serde::{ Deserialize, Serialize };
use sha2::{ Digest, Sha256 };
use sidecar::SidecarClient;
use std::collections::HashSet;
use std::sync::atomic::AtomicU64;
use std::sync::{ Arc, Mutex, MutexGuard };
use std::time::{ Duration, Instant };
use tokio::sync::RwLock;
use tracing::{ info, warn };

/// One retrieved passage. `node_id` records which node's index it came from;
/// the sidecar does not know it, so the node fills it in.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Chunk {
    pub chunk_id: String,
    pub source: String,
    pub text: String,
    /// For peers' chunks with `rescore` on, the score this node computed.
    pub score: f64,
    #[serde(default)]
    pub node_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cluster_id: Option<u32>,
    /// "local", "verified", "verified_new_root" or "failed_proof".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub verification: Option<String>,
    /// The score the peer claimed, when this node recomputed it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub claimed_score: Option<f64>,
    /// E.g. "score_mismatch".
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub flags: Vec<String>,
}

/// Which indexes a query consults.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Scope {
    /// This node's index plus every reachable peer's.
    #[default]
    All,
    /// Only this node's index.
    Local,
    /// Only peers' indexes.
    Peers,
}

#[derive(Debug, Deserialize)]
pub struct QueryRequest {
    pub query: String,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub final_k: Option<usize>,
    #[serde(default)]
    pub scope: Scope,
    /// Overrides `[rag] retrieval` for this query ("two_round", "flat" or "directory").
    #[serde(default)]
    pub retrieval: Option<Retrieval>,
}

/// What a query needs from the mesh to reach other nodes.
pub struct PeerLink {
    pub endpoint: iroh::Endpoint,
    pub targets: Vec<iroh::EndpointId>,
}

/// Outcome of asking one peer for chunks.
#[derive(Debug, Clone, Serialize)]
pub struct PeerResult {
    pub endpoint_id: String,
    /// "ok", "timeout", "error", "no_root" (its answers matched no signed
    /// root), "skipped_stale" (not asked: down lately), "refused" (not
    /// asked: caught equivocating) or "no_directory" (directory mode without
    /// fallback: no fresh directory entry).
    pub status: String,
    /// Chunks kept from this peer.
    pub chunks: usize,
    pub rtt_ms: u64,
    /// The root version the peer answered under.
    pub root_version: Option<u64>,
    /// Clusters (two_round) or cluster entries (flat) the peer offered.
    pub offered_clusters: usize,
    /// Clusters of this peer chosen in round 1; round 2 went only to peers with some.
    pub selected_clusters: Vec<u32>,
    /// Chunks dropped: failed proof, not requested, or over the requested count.
    pub rejected_chunks: usize,
    pub flags: Vec<String>,
    pub round1_ms: Option<u64>,
    pub round2_ms: Option<u64>,
    /// How this peer's clusters were chosen: "directory" or "round1".
    pub selected_via: Option<String>,
    /// JSON bytes sent to and received from this peer during this query, over
    /// every request (round 1, round 2, a search, or an inline `hello`).
    pub bytes_sent: u64,
    pub bytes_received: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Outcome of fetching one peer's signed root (`hello`).
#[derive(Debug, Clone, Serialize)]
pub struct PeerHello {
    pub endpoint_id: String,
    /// "ok", "timeout" or "error".
    pub status: String,
    pub rtt_ms: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub node_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub root: Option<SignedRoot>,
    /// Whether `root` is signed by the key behind `endpoint_id`.
    pub signature_valid: bool,
    /// What the root registry made of `root`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub outcome: Option<RootOutcome>,
    pub bytes_sent: u64,
    pub bytes_received: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl PeerResult {
    fn new(peer: iroh::EndpointId, status: &str) -> Self {
        PeerResult {
            endpoint_id: peer.to_string(),
            status: status.to_string(),
            chunks: 0,
            rtt_ms: 0,
            root_version: None,
            offered_clusters: 0,
            selected_clusters: Vec::new(),
            rejected_chunks: 0,
            flags: Vec::new(),
            round1_ms: None,
            round2_ms: None,
            selected_via: None,
            bytes_sent: 0,
            bytes_received: 0,
            error: None,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct Timings {
    pub local_retrieve: u64,
    /// `select + fetch + rescore`.
    pub peer_retrieve: u64,
    /// Round 1 and cluster selection (two_round only).
    pub select: u64,
    /// Round 2, or the single round of flat mode.
    pub fetch: u64,
    /// Re-embedding peers' chunks.
    pub rescore: u64,
    pub generate: u64,
    pub total: u64,
}

#[derive(Debug, Serialize)]
pub struct QueryResponse {
    pub retrieval: Retrieval,
    pub answer: String,
    pub contexts: Vec<Chunk>,
    pub peers: Vec<PeerResult>,
    pub timings_ms: Timings,
    /// Notes about the whole query (e.g. "select_by_centroid_for_directory").
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub flags: Vec<String>,
}

/// Pool chunks from several nodes into one ranked list of at most `final_k`.
///
/// Every node embeds with the same model, so cosine scores are comparable
/// across nodes and a plain sort is a fair ranking. Chunks with identical text
/// (the same document held by two nodes) are kept once, at their best score.
pub fn merge_chunks(mut chunks: Vec<Chunk>, final_k: usize) -> Vec<Chunk> {
    chunks.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.chunk_id.cmp(&b.chunk_id))
    });

    let mut seen = HashSet::new();
    chunks.retain(|chunk| seen.insert(Sha256::digest(chunk.text.as_bytes())));
    chunks.truncate(final_k);
    chunks
}

pub struct RagService {
    pub config: RagConfig,
    pub sidecar: SidecarClient,
    node_id: String,
    /// This node's Iroh transport key; its public half is our EndpointId, so
    /// peers can check our root's signature against the id they dial.
    secret_key: iroh::SecretKey,
    /// Our latest signed Merkle root, served to peers in `hello`.
    own_root: RwLock<Option<SignedRoot>>,
    /// Peers' signed roots and when each peer last answered.
    registry: Mutex<RootRegistry>,
    /// `hello` answers given, used only by `[rag] adversary = "equivocate"`.
    hellos_served: AtomicU64,
    /// Verified copies of peers' centroids (directory mode).
    directory: Mutex<CentroidDirectory>,
}

impl RagService {
    pub fn new(config: RagConfig, sidecar: SidecarClient, node_id: String, secret_key: iroh::SecretKey) -> Self {
        RagService {
            config,
            sidecar,
            node_id,
            secret_key,
            own_root: RwLock::new(None),
            registry: Mutex::new(RootRegistry::default()),
            hellos_served: AtomicU64::new(0),
            directory: Mutex::new(CentroidDirectory::default()),
        }
    }

    /// The centroid directory. Never hold the guard across an `.await`.
    fn directory(&self) -> MutexGuard<'_, CentroidDirectory> {
        self.directory.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// The directory's view of every peer, for `GET /rag/status`.
    pub fn directory_status(&self) -> DirectoryStatus {
        self.directory().status(self.config.directory, Instant::now())
    }

    /// The root registry. Never hold the guard across an `.await`.
    fn registry(&self) -> MutexGuard<'_, RootRegistry> {
        // A panic while holding the lock leaves the data usable; keep going.
        self.registry.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// The registry's view of every peer, for `GET /rag/status`.
    pub fn registry_status(&self, targets: &[iroh::EndpointId]) -> Vec<PeerRootStatus> {
        self.registry().status(targets, Instant::now(), Duration::from_secs(self.config.peer_stale_secs))
    }

    pub fn node_id(&self) -> &str {
        &self.node_id
    }

    pub async fn own_root(&self) -> Option<SignedRoot> {
        self.own_root.read().await.clone()
    }

    /// Sign `commitment` and make it the root served to peers.
    pub async fn adopt_commitment(&self, commitment: &Commitment) -> Result<SignedRoot, BoxError> {
        let signed = SignedRoot::sign(&self.secret_key, commitment)?;
        let mut own_root = self.own_root.write().await;
        if own_root.as_ref().map(|r| r.version) != Some(signed.version) {
            info!(
                "🔏 Signed RAG root {:.16}… (version {}, {} chunks in {} clusters)",
                signed.root,
                signed.version,
                signed.num_chunks,
                signed.num_clusters
            );
        }
        *own_root = Some(signed.clone());
        Ok(signed)
    }

    /// Fetch the sidecar's current commitment and sign it. Run at startup and
    /// after every ingest.
    pub async fn refresh_own_root(&self) -> Result<SignedRoot, BoxError> {
        let commitment = self.sidecar.commitment().await?;
        self.adopt_commitment(&commitment).await
    }

    /// Our signed root, signing one first if there is none yet.
    pub async fn current_root(&self) -> Result<SignedRoot, BoxError> {
        match self.own_root().await {
            Some(root) => Ok(root),
            None => self.refresh_own_root().await,
        }
    }

    /// Re-sign if the sidecar answered under a root version we have not signed
    /// (an ingest that bypassed the node), so `hello` matches what we serve.
    pub async fn ensure_root_version(&self, version: u64) {
        if self.own_root().await.map(|r| r.version) != Some(version) {
            if let Err(e) = self.refresh_own_root().await {
                warn!("⚠️ Could not re-sign the RAG root: {}", e);
            }
        }
    }

    /// Fetch one peer's signed root, check its signature against the peer's
    /// EndpointId, and record it in the registry. Never fails; problems are
    /// reported in the result.
    pub async fn hello_peer(&self, endpoint: &iroh::Endpoint, peer: iroh::EndpointId) -> PeerHello {
        let started = Instant::now();
        let request = RagRequest::new(RequestBody::Hello);
        let mut bytes = protocol::ByteCount::default();
        let outcome = tokio::time::timeout(
            Duration::from_millis(self.config.select_timeout_ms),
            protocol::request_peer(endpoint, peer, &request, self.config.max_message_bytes, &mut bytes),
        ).await;

        let mut result = PeerHello {
            endpoint_id: peer.to_string(),
            status: "ok".to_string(),
            rtt_ms: started.elapsed().as_millis() as u64,
            node_id: None,
            root: None,
            signature_valid: false,
            outcome: None,
            bytes_sent: bytes.sent,
            bytes_received: bytes.received,
            error: None,
        };
        let now = Instant::now();
        match outcome {
            Ok(Ok(response)) => {
                result.node_id = Some(response.node_id);
                if let ResponseBody::Hello { root } = response.body {
                    result.signature_valid = root.verify(&peer);
                    let outcome = self.registry().record_root(peer, root.clone(), now);
                    match outcome {
                        RootOutcome::Accepted => info!(
                            "📜 Accepted RAG root {:.16}… (version {}) from peer {}",
                            root.root,
                            root.version,
                            peer.fmt_short()
                        ),
                        RootOutcome::Unchanged => {}
                        flagged => warn!("🚩 Peer {} sent a root flagged {:?}", peer.fmt_short(), flagged),
                    }
                    result.outcome = Some(outcome);
                    result.root = Some(root);
                }
            }
            Ok(Err(e)) => {
                result.status = "error".to_string();
                result.error = Some(e.to_string());
                self.registry().record_failure(peer, e.to_string(), now);
            }
            Err(_) => {
                result.status = "timeout".to_string();
                self.registry().record_failure(peer, "hello timed out".to_string(), now);
            }
        }
        result
    }

    /// Fetch every target's signed root at once. Run in the background by
    /// `run_root_refresh`, and by `GET /rag/debug/hello`.
    pub async fn refresh_peer_roots(&self, link: &PeerLink) -> Vec<PeerHello> {
        futures::future::join_all(link.targets.iter().map(|peer| self.hello_peer(&link.endpoint, *peer))).await
    }

    /// Chunks from this node's own index, tagged with this node's id.
    pub async fn retrieve_local(&self, query: &str, top_k: usize) -> Result<Vec<Chunk>, BoxError> {
        let mut chunks = self.sidecar.retrieve(query, top_k).await?.chunks;
        for chunk in &mut chunks {
            chunk.node_id = Some(self.node_id.clone());
            chunk.verification = Some(retrieval::LOCAL.to_string());
        }
        Ok(chunks)
    }

    /// Run the full pipeline for one query: retrieve, merge, generate.
    ///
    /// `link` is the way to other nodes; without it only the local index is used.
    pub async fn query(
        self: &Arc<Self>,
        request: QueryRequest,
        link: Option<PeerLink>,
    ) -> Result<QueryResponse, BoxError> {
        let started = Instant::now();
        let top_k = request.top_k.unwrap_or(self.config.top_k);
        let final_k = request.final_k.unwrap_or(self.config.final_k);
        let mode = request.retrieval.unwrap_or(self.config.retrieval);

        // The local index and every peer are consulted at the same time, so
        // retrieval takes as long as the slowest one, not the sum.
        let local = async {
            let local_started = Instant::now();
            let chunks = if request.scope != Scope::Peers {
                self.retrieve_local(&request.query, top_k).await?
            } else {
                Vec::new()
            };
            Ok::<_, BoxError>((chunks, local_started.elapsed().as_millis() as u64))
        };

        let remote = async {
            match link.as_ref().filter(|_| request.scope != Scope::Local) {
                Some(link) => match mode {
                    Retrieval::TwoRound => self.two_round(&request.query, link, started).await,
                    Retrieval::Flat => self.flat(&request.query, top_k, link, started).await,
                    Retrieval::Directory => self.directory_mode(&request.query, link, started).await,
                },
                None => Ok(retrieval::RemoteRetrieval::default()),
            }
        };

        let (local, remote) = tokio::join!(local, remote);
        let (mut pooled, local_retrieve) = local?;
        let remote = remote?;
        pooled.extend(remote.chunks);
        let peers = remote.peers;

        let contexts = merge_chunks(pooled, final_k);

        // With nothing retrieved the generator would answer from its own
        // memory, which is exactly what RAG is meant to avoid.
        let generate_started = Instant::now();
        let answer = if contexts.is_empty() {
            String::new()
        } else {
            self.sidecar.generate(&request.query, &contexts).await?.answer
        };
        let generate = generate_started.elapsed().as_millis() as u64;

        Ok(QueryResponse {
            flags: remote.flags.clone(),
            retrieval: mode,
            answer,
            contexts,
            peers,
            timings_ms: Timings {
                local_retrieve,
                peer_retrieve: remote.select_ms + remote.fetch_ms + remote.rescore_ms,
                select: remote.select_ms,
                fetch: remote.fetch_ms,
                rescore: remote.rescore_ms,
                generate,
                total: started.elapsed().as_millis() as u64,
            },
        })
    }
}

/// Fetch every peer's signed root 5 s after startup and then every
/// `root_refresh_secs`, so the registry knows who is up before queries arrive.
pub async fn run_root_refresh(
    service: Arc<RagService>,
    network: Arc<RwLock<crate::network::mesh::MeshNetwork>>,
) {
    tokio::time::sleep(Duration::from_secs(5)).await;
    loop {
        let link = network.read().await.rag_link().await;
        if let Some(link) = link {
            let results = service.refresh_peer_roots(&link).await;
            let ok = results.iter().filter(|r| r.status == "ok").count();
            tracing::debug!("Refreshed RAG roots: {} of {} peers answered", ok, results.len());
            // Directory mode: fetch the centroids of peers whose root changed.
            if service.config.directory {
                service.refresh_directories(&link).await;
            }
        }
        tokio::time::sleep(Duration::from_secs(service.config.root_refresh_secs)).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk(id: &str, text: &str, score: f64) -> Chunk {
        Chunk {
            chunk_id: id.to_string(),
            source: "doc".to_string(),
            text: text.to_string(),
            score,
            ..Default::default()
        }
    }

    #[test]
    fn merge_sorts_by_score_and_truncates() {
        let merged = merge_chunks(
            vec![chunk("a", "one", 0.2), chunk("b", "two", 0.9), chunk("c", "three", 0.5)],
            2,
        );
        let ids: Vec<_> = merged.iter().map(|c| c.chunk_id.as_str()).collect();
        assert_eq!(ids, vec!["b", "c"]);
    }

    #[test]
    fn merge_keeps_duplicate_text_once_at_best_score() {
        let merged = merge_chunks(
            vec![chunk("n1#0", "same text", 0.4), chunk("n2#0", "same text", 0.7), chunk("x", "other", 0.1)],
            5,
        );
        let ids: Vec<_> = merged.iter().map(|c| c.chunk_id.as_str()).collect();
        assert_eq!(ids, vec!["n2#0", "x"]);
    }

    #[test]
    fn a_query_can_ask_for_directory_mode() {
        let request: QueryRequest = serde_json::from_str(r#"{"query": "q", "retrieval": "directory"}"#).unwrap();
        assert_eq!(request.retrieval, Some(Retrieval::Directory));
    }

    /// Write `toml` to a temporary file and load it as a node config.
    fn load_config(name: &str, toml: &str) -> Result<crate::config::Config, BoxError> {
        let path = std::env::temp_dir().join(format!("slakshna-test-{}-{}.toml", std::process::id(), name));
        std::fs::write(&path, toml).unwrap();
        let loaded = crate::config::Config::load(path.to_str().unwrap());
        std::fs::remove_file(&path).ok();
        loaded
    }

    #[test]
    fn example_configs_load_and_directory_mode_needs_the_switch() {
        let example = include_str!("../../examples/rag-2node/node1.toml");
        let config = load_config("example", example).unwrap();
        assert!(config.rag.directory);
        assert!(config.rag.directory_fallback);
        assert_eq!(config.rag.retrieval, Retrieval::TwoRound);

        // Every config in examples/ still loads (cargo runs tests from the crate root).
        for dir in std::fs::read_dir("examples").unwrap() {
            for file in std::fs::read_dir(dir.unwrap().path()).unwrap() {
                let path = file.unwrap().path();
                if path.extension().is_some_and(|e| e == "toml") {
                    crate::config::Config::load(path.to_str().unwrap())
                        .unwrap_or_else(|e| panic!("{} no longer loads: {}", path.display(), e));
                }
            }
        }

        // A config from before M8 (no directory settings) still loads, switched off.
        let older = load_config("older", &example.replace("directory = true", "")).unwrap();
        assert!(!older.rag.directory);

        let with_mode = example.replace("directory = true", "directory = true\nretrieval = \"directory\"");
        assert_eq!(load_config("mode", &with_mode).unwrap().rag.retrieval, Retrieval::Directory);

        let without_switch = example.replace("directory = true", "retrieval = \"directory\"");
        let error = load_config("no-switch", &without_switch).unwrap_err().to_string();
        assert!(error.contains("rag.directory = true"), "{}", error);
    }

    #[test]
    fn scope_defaults_to_all_and_parses_lowercase() {
        let request: QueryRequest = serde_json::from_str(r#"{"query": "q"}"#).unwrap();
        assert_eq!(request.scope, Scope::All);
        let request: QueryRequest = serde_json::from_str(r#"{"query": "q", "scope": "local"}"#).unwrap();
        assert_eq!(request.scope, Scope::Local);
    }
}
