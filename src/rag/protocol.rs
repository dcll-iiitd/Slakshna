//! Peer-to-peer retrieval: one node asks another for chunks matching a query.
//!
//! This is a second protocol on the node's Iroh endpoint, next to gossip. The
//! asker opens a QUIC connection and one two-way stream, writes a JSON
//! `RagRequest`, and reads a JSON `RagResponse` back. Unlike gossip, only the
//! two nodes involved see the query and the chunks.
//!
//! Version 2 has these kinds of request:
//! `hello` fetches the peer's signed Merkle root, `centroids` and `chunks` are
//! the two rounds of a cluster-routed query, `search` is the flat
//! one-round query, and `directory` fetches every centroid at once for
//! directory mode. Everything a peer returns comes with proofs against its
//! signed root.

use super::merkle::{ ClusterEntry, ProofStep, SignedRoot };
use super::sidecar::ClusterSearch;
use super::{ Chunk, RagService };
use crate::history::BoxError;
use iroh::endpoint::Connection;
use iroh::protocol::{ AcceptError, ProtocolHandler };
use iroh::{ Endpoint, EndpointId };
use serde::{ Deserialize, Serialize };
use std::sync::Arc;
use std::time::Instant;
use tracing::{ info, warn };

/// Protocol identifier. Bump the trailing number on any incompatible change to
/// the request or response, so mismatched builds refuse to connect instead of
/// misreading each other.
pub const RAG_ALPN: &[u8] = b"/slakshna/rag/2";

pub const WIRE_VERSION: u32 = 2;
/// Upper bounds a peer applies, whatever the asker requests.
const MAX_TOP_K: usize = 50;
const MAX_CENTROIDS: usize = 10;
const MAX_CLUSTERS: usize = 10;
const MAX_PER_CLUSTER: usize = 10;

/// One request: common fields plus one of the five kinds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagRequest {
    pub v: u32,
    pub query_id: String,
    #[serde(flatten)]
    pub body: RequestBody,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RequestBody {
    /// Fetch the peer's signed root.
    Hello,
    /// Round 1: the peer's `n` clusters nearest to `query`.
    Centroids {
        query: String,
        n: usize,
    },
    /// Round 2: the best `per_cluster` chunks of each listed cluster, under
    /// the root version the clusters were chosen from.
    Chunks {
        query: String,
        root_version: u64,
        clusters: Vec<u32>,
        per_cluster: usize,
    },
    /// Flat mode: the peer's best `top_k` chunks overall.
    Search {
        query: String,
        top_k: usize,
    },
    /// Directory mode: every cluster entry of the peer's current root.
    /// Answered only by nodes with
    /// `[rag] directory = true`.
    Directory,
}

impl RequestBody {
    pub fn kind(&self) -> &'static str {
        match self {
            RequestBody::Hello => "hello",
            RequestBody::Centroids { .. } => "centroids",
            RequestBody::Chunks { .. } => "chunks",
            RequestBody::Search { .. } => "search",
            RequestBody::Directory => "directory",
        }
    }
}

impl RagRequest {
    pub fn new(body: RequestBody) -> Self {
        RagRequest {
            v: WIRE_VERSION,
            query_id: uuid::Uuid::new_v4().to_string(),
            body,
        }
    }
}

/// A chunk as it travels between peers: with its cluster and the proof that
/// leads from it to that cluster's subtree root. Only used on the wire; once
/// checked it becomes a plain `Chunk`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenChunk {
    pub chunk_id: String,
    pub source: String,
    pub text: String,
    pub score: f64,
    pub cluster_id: u32,
    pub chunk_proof: Vec<ProofStep>,
}

impl ProvenChunk {
    pub fn into_chunk(self, node_id: &str) -> Chunk {
        Chunk {
            chunk_id: self.chunk_id,
            source: self.source,
            text: self.text,
            score: self.score,
            node_id: Some(node_id.to_string()),
            ..Default::default()
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagResponse {
    pub v: u32,
    pub query_id: String,
    pub node_id: String,
    pub embedding_model: String,
    pub error: Option<String>,
    #[serde(flatten)]
    pub body: ResponseBody,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ResponseBody {
    Hello {
        root: SignedRoot,
    },
    Centroids {
        root_version: u64,
        centroids: Vec<ClusterEntry>,
        retrieve_ms: u64,
    },
    Chunks {
        root_version: u64,
        clusters: Vec<ClusterEntry>,
        chunks: Vec<ProvenChunk>,
        retrieve_ms: u64,
        /// Set with `error: "stale_root"`: the peer's current signed root.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        root: Option<SignedRoot>,
    },
    Search {
        root_version: u64,
        clusters: Vec<ClusterEntry>,
        chunks: Vec<ProvenChunk>,
        retrieve_ms: u64,
    },
    /// Every cluster entry of the peer's root at `root_version`.
    Directory {
        root_version: u64,
        clusters: Vec<ClusterEntry>,
    },
    /// The request failed; `error` says why.
    Error,
}

impl ResponseBody {
    pub fn kind(&self) -> &'static str {
        match self {
            ResponseBody::Hello { .. } => "hello",
            ResponseBody::Centroids { .. } => "centroids",
            ResponseBody::Chunks { .. } => "chunks",
            ResponseBody::Search { .. } => "search",
            ResponseBody::Directory { .. } => "directory",
            ResponseBody::Error => "error",
        }
    }

    /// How many items this response carries, for the log.
    fn item_count(&self) -> usize {
        match self {
            ResponseBody::Hello { .. } | ResponseBody::Error => 0,
            ResponseBody::Centroids { centroids, .. } => centroids.len(),
            ResponseBody::Directory { clusters, .. } => clusters.len(),
            ResponseBody::Chunks { chunks, .. } | ResponseBody::Search { chunks, .. } => chunks.len(),
        }
    }
}

/// A `chunks` request named a root version the peer no longer serves. Carries
/// the peer's current signed root, so the asker can update what it knows.
#[derive(Debug)]
pub struct StaleRoot(pub SignedRoot);

impl std::fmt::Display for StaleRoot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "peer's root has moved on to version {}", self.0.version)
    }
}

impl std::error::Error for StaleRoot {}

/// The answering side: serves this node's index to peers.
pub struct RagProtocol {
    service: Arc<RagService>,
    allowed_peers: Option<Vec<String>>,
}

impl std::fmt::Debug for RagProtocol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RagProtocol").finish_non_exhaustive()
    }
}

impl RagProtocol {
    pub fn new(service: Arc<RagService>, allowed_peers: Option<Vec<String>>) -> Self {
        RagProtocol { service, allowed_peers }
    }

    async fn answer(&self, request_bytes: &[u8]) -> RagResponse {
        let mut response = RagResponse {
            v: WIRE_VERSION,
            query_id: String::new(),
            node_id: self.service.node_id().to_string(),
            embedding_model: self.service.config.embedding_model.clone(),
            error: None,
            body: ResponseBody::Error,
        };

        let request: RagRequest = match serde_json::from_slice(request_bytes) {
            Ok(request) => request,
            Err(e) => {
                response.error = Some(format!("malformed request: {}", e));
                return response;
            }
        };
        response.query_id = request.query_id;
        if request.v != WIRE_VERSION {
            response.error = Some(format!("unsupported request version {}", request.v));
            return response;
        }

        let request_body = request.body.clone();
        match self.respond(request.body).await {
            Ok((mut body, error)) => {
                // Experiments only: corrupt the honest answer ([rag] adversary).
                self.service.misbehave(&request_body, &mut body).await;
                response.body = body;
                response.error = error;
            }
            Err(e) => response.error = Some(e.to_string()),
        }
        response
    }

    /// Serve one request from the local sidecar. Returns the body, plus an
    /// error string for answers that are refusals but still carry data
    /// (`stale_root`).
    async fn respond(&self, body: RequestBody) -> Result<(ResponseBody, Option<String>), BoxError> {
        let started = Instant::now();
        let service = &self.service;
        let elapsed_ms = || started.elapsed().as_millis() as u64;

        match body {
            RequestBody::Hello => Ok((ResponseBody::Hello { root: service.current_root().await? }, None)),
            RequestBody::Centroids { query, n } => {
                let result = service.sidecar.centroids(&query, n.min(MAX_CENTROIDS)).await?;
                service.ensure_root_version(result.root_version).await;
                Ok((
                    ResponseBody::Centroids {
                        root_version: result.root_version,
                        centroids: result.centroids,
                        retrieve_ms: elapsed_ms(),
                    },
                    None,
                ))
            }
            RequestBody::Chunks { query, root_version, mut clusters, per_cluster } => {
                clusters.truncate(MAX_CLUSTERS);
                let search = service.sidecar.search_clusters(
                    &query,
                    root_version,
                    &clusters,
                    per_cluster.min(MAX_PER_CLUSTER),
                ).await?;
                match search {
                    ClusterSearch::Found(result) => {
                        service.ensure_root_version(result.root_version).await;
                        Ok((
                            ResponseBody::Chunks {
                                root_version: result.root_version,
                                clusters: result.clusters,
                                chunks: result.chunks,
                                retrieve_ms: elapsed_ms(),
                                root: None,
                            },
                            None,
                        ))
                    }
                    ClusterSearch::StaleRoot(commitment) => {
                        let root = service.adopt_commitment(&commitment).await?;
                        Ok((
                            ResponseBody::Chunks {
                                root_version: root.version,
                                clusters: Vec::new(),
                                chunks: Vec::new(),
                                retrieve_ms: elapsed_ms(),
                                root: Some(root),
                            },
                            Some("stale_root".to_string()),
                        ))
                    }
                }
            }
            RequestBody::Search { query, top_k } => {
                let result = service.sidecar.retrieve_with_proofs(&query, top_k.min(MAX_TOP_K)).await?;
                service.ensure_root_version(result.root_version).await;
                Ok((
                    ResponseBody::Search {
                        root_version: result.root_version,
                        clusters: result.clusters,
                        chunks: result.chunks,
                        retrieve_ms: elapsed_ms(),
                    },
                    None,
                ))
            }
            RequestBody::Directory => {
                // The directory reveals every topic this node holds, so
                // it is served only when switched on.
                if !service.config.directory {
                    return Err("this node does not serve a centroid directory ([rag] directory = false)".into());
                }
                let result = service.sidecar.clusters().await?;
                service.ensure_root_version(result.root_version).await;
                Ok((ResponseBody::Directory { root_version: result.root_version, clusters: result.clusters }, None))
            }
        }
    }
}

impl ProtocolHandler for RagProtocol {
    async fn accept(&self, connection: Connection) -> Result<(), AcceptError> {
        let remote = connection.remote_id();

        if let Some(allowed) = &self.allowed_peers {
            if !allowed.is_empty() && !allowed.contains(&remote.to_string()) {
                warn!("🚫 Refused RAG query from unauthorized peer: {}", remote.fmt_short());
                connection.close(1u32.into(), b"not allowed");
                return Ok(());
            }
        }

        let (mut send, mut recv) = connection.accept_bi().await?;
        let request_bytes = recv
            .read_to_end(self.service.config.max_message_bytes)
            .await
            .map_err(AcceptError::from_err)?;

        let started = Instant::now();
        let response = self.answer(&request_bytes).await;
        match &response.error {
            None => info!(
                "📚 RAG {} request from peer {}: returned {} item(s) in {} ms",
                response.body.kind(),
                remote.fmt_short(),
                response.body.item_count(),
                started.elapsed().as_millis()
            ),
            Some(error) => warn!("📚 RAG request from peer {} failed: {}", remote.fmt_short(), error),
        }

        let response_bytes = serde_json::to_vec(&response).map_err(AcceptError::from_err)?;
        send.write_all(&response_bytes).await.map_err(AcceptError::from_err)?;
        send.finish()?;

        // The asker closes the connection once it has read the response.
        connection.closed().await;
        Ok(())
    }
}

/// Bytes exchanged with one peer: the JSON request written and the JSON
/// response read (QUIC and TLS overhead not included).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct ByteCount {
    pub sent: u64,
    pub received: u64,
}

impl ByteCount {
    pub fn add(&mut self, other: ByteCount) {
        self.sent += other.sent;
        self.received += other.received;
    }
}

/// The asking side: send one request to one peer and read its response.
///
/// Fails if the peer reports an error, answers another query, or answers with
/// a different kind. A `stale_root` refusal fails with a `StaleRoot` error
/// (find it with `downcast_ref`), which carries the peer's current root.
/// The bytes written and read are added to `bytes` as they happen, so they
/// are counted even when the request fails or is cut off by a timeout.
pub async fn request_peer(
    endpoint: &Endpoint,
    peer: EndpointId,
    request: &RagRequest,
    max_message_bytes: usize,
    bytes: &mut ByteCount,
) -> Result<RagResponse, BoxError> {
    let request_bytes = serde_json::to_vec(request)?;
    let connection = endpoint.connect(peer, RAG_ALPN).await?;
    let (mut send, mut recv) = connection.open_bi().await?;
    send.write_all(&request_bytes).await?;
    bytes.sent += request_bytes.len() as u64;
    send.finish()?;
    let response_bytes = recv.read_to_end(max_message_bytes).await?;
    bytes.received += response_bytes.len() as u64;
    connection.close(0u32.into(), b"done");

    let response: RagResponse = serde_json::from_slice(&response_bytes)?;
    check_response(request, response)
}

fn check_response(request: &RagRequest, response: RagResponse) -> Result<RagResponse, BoxError> {
    if response.query_id != request.query_id && response.error.is_none() {
        return Err("peer answered a different query".into());
    }
    match response.error.as_deref() {
        None => {}
        Some("stale_root") => {
            return match response.body {
                ResponseBody::Chunks { root: Some(root), .. } => Err(Box::new(StaleRoot(root))),
                _ => Err("peer reported stale_root without its current root".into()),
            };
        }
        Some(error) => {
            return Err(format!("peer reported: {}", error).into());
        }
    }
    if response.body.kind() != request.body.kind() {
        return Err(
            format!(
                "peer answered a {} request with {}",
                request.body.kind(),
                response.body.kind()
            ).into()
        );
    }
    Ok(response)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn signed_root() -> SignedRoot {
        SignedRoot {
            root: "ab".repeat(32),
            version: 1792328689018,
            num_chunks: 161,
            num_clusters: 13,
            dim: 384,
            embedding_model: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
            signature: "cd".repeat(64),
        }
    }

    fn response(query_id: &str, error: Option<&str>, body: ResponseBody) -> RagResponse {
        RagResponse {
            v: WIRE_VERSION,
            query_id: query_id.to_string(),
            node_id: "slakshna1peer".to_string(),
            embedding_model: "m".to_string(),
            error: error.map(str::to_string),
            body,
        }
    }

    #[test]
    fn requests_match_the_documented_json() {
        // The wire shapes other nodes rely on; changing them breaks compatibility.
        let cases = [
            (RequestBody::Hello, json!({"v": 2, "kind": "hello", "query_id": "q"})),
            (
                RequestBody::Centroids { query: "x".into(), n: 3 },
                json!({"v": 2, "kind": "centroids", "query_id": "q", "query": "x", "n": 3}),
            ),
            (
                RequestBody::Chunks { query: "x".into(), root_version: 7, clusters: vec![4, 7], per_cluster: 3 },
                json!({"v": 2, "kind": "chunks", "query_id": "q", "query": "x", "root_version": 7,
                       "clusters": [4, 7], "per_cluster": 3}),
            ),
            (
                RequestBody::Search { query: "x".into(), top_k: 5 },
                json!({"v": 2, "kind": "search", "query_id": "q", "query": "x", "top_k": 5}),
            ),
            (RequestBody::Directory, json!({"v": 2, "kind": "directory", "query_id": "q"})),
        ];
        for (body, expected) in cases {
            let request = RagRequest { v: 2, query_id: "q".into(), body: body.clone() };
            assert_eq!(serde_json::to_value(&request).unwrap(), expected);
            let parsed: RagRequest = serde_json::from_value(expected).unwrap();
            assert_eq!(parsed.body, body);
        }
    }

    #[test]
    fn responses_round_trip_with_large_versions() {
        let hello = response("q", None, ResponseBody::Hello { root: signed_root() });
        let value = serde_json::to_value(&hello).unwrap();
        assert_eq!(value["kind"], "hello");
        assert_eq!(value["error"], serde_json::Value::Null);
        assert_eq!(value["root"]["version"], 1792328689018u64);
        let parsed: RagResponse = serde_json::from_value(value).unwrap();
        assert!(matches!(parsed.body, ResponseBody::Hello { root } if root == signed_root()));

        let chunks = response("q", None, ResponseBody::Chunks {
            root_version: u64::MAX,
            clusters: Vec::new(),
            chunks: Vec::new(),
            retrieve_ms: 3,
            root: None,
        });
        let value = serde_json::to_value(&chunks).unwrap();
        assert!(value.get("root").is_none());
        let parsed: RagResponse = serde_json::from_value(value).unwrap();
        assert!(matches!(parsed.body, ResponseBody::Chunks { root_version: u64::MAX, .. }));
    }

    #[test]
    fn directory_answers_match_the_documented_json() {
        // The wire shape other nodes rely on.
        let request = RagRequest::new(RequestBody::Directory);
        let answer = response(&request.query_id, None, ResponseBody::Directory {
            root_version: 7,
            clusters: Vec::new(),
        });
        let value = serde_json::to_value(&answer).unwrap();
        assert_eq!(value["kind"], "directory");
        assert_eq!(value["root_version"], 7);
        assert!(value["clusters"].as_array().unwrap().is_empty());
        let parsed: RagResponse = serde_json::from_value(value).unwrap();
        assert!(check_response(&request, parsed).is_ok());

        // An older build answers the new kind with an error, which the asker
        // turns into "no directory from this peer".
        let refused = response(&request.query_id, Some("unknown variant `directory`"), ResponseBody::Error);
        assert!(check_response(&request, refused).is_err());
    }

    #[test]
    fn check_response_accepts_matching_kind_and_id() {
        let request = RagRequest::new(RequestBody::Hello);
        let ok = response(&request.query_id, None, ResponseBody::Hello { root: signed_root() });
        assert!(check_response(&request, ok).is_ok());
    }

    #[test]
    fn check_response_rejects_mismatches_and_errors() {
        let request = RagRequest::new(RequestBody::Hello);
        let other_id = response("other", None, ResponseBody::Hello { root: signed_root() });
        assert!(check_response(&request, other_id).unwrap_err().to_string().contains("different query"));

        let wrong_kind = response(&request.query_id, None, ResponseBody::Error);
        assert!(check_response(&request, wrong_kind).unwrap_err().to_string().contains("with error"));

        let failed = response("", Some("malformed request: x"), ResponseBody::Error);
        assert!(check_response(&request, failed).unwrap_err().to_string().contains("malformed"));
    }

    #[test]
    fn stale_root_carries_the_peers_current_root() {
        let request = RagRequest::new(RequestBody::Chunks {
            query: "x".into(),
            root_version: 1,
            clusters: vec![0],
            per_cluster: 3,
        });
        let stale = response(&request.query_id, Some("stale_root"), ResponseBody::Chunks {
            root_version: signed_root().version,
            clusters: Vec::new(),
            chunks: Vec::new(),
            retrieve_ms: 1,
            root: Some(signed_root()),
        });
        let error = check_response(&request, stale).unwrap_err();
        let StaleRoot(root) = error.downcast_ref::<StaleRoot>().expect("a StaleRoot error");
        assert_eq!(*root, signed_root());
    }
}
