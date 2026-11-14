//! Client for, and launcher of, the local Python RAG sidecar.

use super::merkle::{ decode_f32le, ClusterEntry, Commitment };
use super::protocol::ProvenChunk;
use super::Chunk;
use crate::config::{ NodeConfig, RagConfig };
use crate::history::BoxError;
use serde::{ de::DeserializeOwned, Deserialize, Serialize };
use std::time::Duration;
use tracing::info;

/// How long to wait for a freshly started sidecar to load its embedding model.
const STARTUP_TIMEOUT: Duration = Duration::from_secs(180);
const RETRIEVE_TIMEOUT: Duration = Duration::from_secs(30);
/// Generation runs a language model on CPU and can take most of a minute.
const GENERATE_TIMEOUT: Duration = Duration::from_secs(600);
const INGEST_TIMEOUT: Duration = Duration::from_secs(1800);

#[derive(Debug, Deserialize)]
pub struct RetrieveResponse {
    pub chunks: Vec<Chunk>,
}

#[derive(Debug, Deserialize)]
pub struct GenerateResponse {
    pub answer: String,
}

/// Round 1 of a peer query: this node's clusters nearest to a query.
#[derive(Debug, Deserialize)]
pub struct CentroidsResponse {
    pub root_version: u64,
    pub centroids: Vec<ClusterEntry>,
}

/// Chunks with proofs, plus the entries of the clusters they belong to.
#[derive(Debug, Deserialize)]
pub struct ProvenChunks {
    pub root_version: u64,
    pub clusters: Vec<ClusterEntry>,
    pub chunks: Vec<ProvenChunk>,
}

/// A node's whole centroid directory (`GET /clusters`).
#[derive(Debug, Deserialize)]
pub struct ClustersResponse {
    pub root_version: u64,
    pub clusters: Vec<ClusterEntry>,
}

/// Result of `/search_clusters`. Cluster ids only mean something under one
/// root, so the sidecar refuses a request made for another version.
#[derive(Debug)]
pub enum ClusterSearch {
    Found(ProvenChunks),
    /// The asker's version is not current; this is the commitment that is.
    StaleRoot(Commitment),
}

/// Decode a sidecar reply, turning an HTTP error status into an error.
async fn read_json<R: DeserializeOwned>(path: &str, response: reqwest::Response) -> Result<R, BoxError> {
    let status = response.status();
    if !status.is_success() {
        let detail = response.text().await.unwrap_or_default();
        return Err(format!("RAG sidecar {} returned {}: {}", path, status, detail).into());
    }
    Ok(response.json::<R>().await?)
}

#[derive(Clone)]
pub struct SidecarClient {
    base_url: String,
    http: reqwest::Client,
}

impl SidecarClient {
    pub fn new(port: u16) -> Self {
        SidecarClient {
            base_url: format!("http://127.0.0.1:{}", port),
            http: reqwest::Client::new(),
        }
    }

    async fn send<B: Serialize>(
        &self,
        path: &str,
        body: &B,
        timeout: Duration,
    ) -> Result<reqwest::Response, BoxError> {
        let response = self
            .http
            .post(format!("{}{}", self.base_url, path))
            .json(body)
            .timeout(timeout)
            .send()
            .await
            .map_err(|e| format!("RAG sidecar unreachable at {}: {}", self.base_url, e))?;
        Ok(response)
    }

    async fn post<B: Serialize, R: DeserializeOwned>(
        &self,
        path: &str,
        body: &B,
        timeout: Duration,
    ) -> Result<R, BoxError> {
        let response = self.send(path, body, timeout).await?;
        read_json(path, response).await
    }

    pub async fn health(&self) -> Result<serde_json::Value, BoxError> {
        let response = self
            .http
            .get(format!("{}/health", self.base_url))
            .timeout(Duration::from_secs(5))
            .send()
            .await?
            .error_for_status()?;
        Ok(response.json().await?)
    }

    pub async fn retrieve(&self, query: &str, top_k: usize) -> Result<RetrieveResponse, BoxError> {
        self.post(
            "/retrieve",
            &serde_json::json!({ "query": query, "top_k": top_k }),
            RETRIEVE_TIMEOUT,
        ).await
    }

    pub async fn generate(&self, query: &str, chunks: &[Chunk]) -> Result<GenerateResponse, BoxError> {
        self.post(
            "/generate",
            &serde_json::json!({ "query": query, "chunks": chunks }),
            GENERATE_TIMEOUT,
        ).await
    }

    /// Embed texts with this node's model: unit vectors, one per text.
    pub async fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, BoxError> {
        #[derive(Deserialize)]
        struct Embedded {
            dim: u32,
            vectors: Vec<String>,
        }
        let embedded: Embedded = self.post(
            "/embed",
            &serde_json::json!({ "texts": texts }),
            RETRIEVE_TIMEOUT,
        ).await?;
        embedded.vectors
            .iter()
            .map(|v| decode_f32le(v, embedded.dim))
            .collect()
    }

    /// This node's unsigned Merkle commitment; the node signs it.
    pub async fn commitment(&self) -> Result<Commitment, BoxError> {
        let response = self
            .http
            .get(format!("{}/commitment", self.base_url))
            .timeout(RETRIEVE_TIMEOUT)
            .send()
            .await
            .map_err(|e| format!("RAG sidecar unreachable at {}: {}", self.base_url, e))?;
        read_json("/commitment", response).await
    }

    /// Every cluster entry of the current root, for peers' directory mode.
    pub async fn clusters(&self) -> Result<ClustersResponse, BoxError> {
        let response = self
            .http
            .get(format!("{}/clusters", self.base_url))
            .timeout(RETRIEVE_TIMEOUT)
            .send()
            .await
            .map_err(|e| format!("RAG sidecar unreachable at {}: {}", self.base_url, e))?;
        read_json("/clusters", response).await
    }

    pub async fn centroids(&self, query: &str, n: usize) -> Result<CentroidsResponse, BoxError> {
        self.post(
            "/centroids",
            &serde_json::json!({ "query": query, "n": n }),
            RETRIEVE_TIMEOUT,
        ).await
    }

    pub async fn search_clusters(
        &self,
        query: &str,
        root_version: u64,
        clusters: &[u32],
        per_cluster: usize,
    ) -> Result<ClusterSearch, BoxError> {
        let body = serde_json::json!({
            "query": query,
            "root_version": root_version,
            "clusters": clusters,
            "per_cluster": per_cluster,
        });
        let response = self.send("/search_clusters", &body, RETRIEVE_TIMEOUT).await?;
        if response.status() == reqwest::StatusCode::CONFLICT {
            #[derive(Deserialize)]
            struct Stale {
                root: Commitment,
            }
            #[derive(Deserialize)]
            struct Conflict {
                detail: Stale,
            }
            let conflict: Conflict = response.json().await?;
            return Ok(ClusterSearch::StaleRoot(conflict.detail.root));
        }
        Ok(ClusterSearch::Found(read_json("/search_clusters", response).await?))
    }

    pub async fn retrieve_with_proofs(&self, query: &str, top_k: usize) -> Result<ProvenChunks, BoxError> {
        self.post(
            "/retrieve",
            &serde_json::json!({ "query": query, "top_k": top_k, "with_proofs": true }),
            RETRIEVE_TIMEOUT,
        ).await
    }

    /// Forward an ingest request body (`{path}` or `{documents}`) unchanged.
    pub async fn ingest(&self, body: &serde_json::Value) -> Result<serde_json::Value, BoxError> {
        self.post("/ingest", body, INGEST_TIMEOUT).await
    }
}

/// Launch `rag_engine.py serve` for this node and wait until it answers.
///
/// The returned child must be kept alive for as long as the node runs. The
/// sidecar is also told to exit when this process does, so a node that is
/// killed outright does not leave one behind holding the port.
pub async fn spawn_sidecar(
    rag: &RagConfig,
    node: &NodeConfig,
    client: &SidecarClient,
) -> Result<tokio::process::Child, BoxError> {
    if client.health().await.is_ok() {
        return Err(format!(
            "port {} already answers as a RAG sidecar; stop it, or set [rag] spawn_sidecar = false to use it",
            rag.sidecar_port
        ).into());
    }

    info!("📚 Starting RAG sidecar on 127.0.0.1:{} ({})", rag.sidecar_port, rag.python);

    let mut child = tokio::process::Command::new(&rag.python)
        .arg("rag_engine.py")
        .arg("serve")
        .args(["--data-dir", &node.data_dir])
        .args(["--port", &rag.sidecar_port.to_string()])
        .args(["--embedding-model", &rag.embedding_model])
        .args(["--n-clusters", &rag.n_clusters.to_string()])
        .args(["--generator", &rag.generator])
        .args(["--generator-model", &rag.generator_model])
        .args(["--generator-url", &rag.generator_url])
        .arg("--exit-with-parent")
        .current_dir(".")
        .kill_on_drop(true)
        .spawn()
        .map_err(|e| format!("Failed to start RAG sidecar with '{}': {}", rag.python, e))?;

    let deadline = tokio::time::Instant::now() + STARTUP_TIMEOUT;
    loop {
        if client.health().await.is_ok() {
            return Ok(child);
        }
        if let Some(status) = child.try_wait()? {
            return Err(format!("RAG sidecar exited during startup ({})", status).into());
        }
        if tokio::time::Instant::now() >= deadline {
            return Err("RAG sidecar did not become ready in time".into());
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
}
