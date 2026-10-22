//! Federated inference.
//!
//! Publish (after each round): hash → blob → signed gossip announcement.
//! Query: select from local registry → fetch by hash if not cached → verify
//! SHA-256 → attach to the local model and generate on-device through the
//! Python sidecar (Bhaskera `serve` branch + PEFT).

pub mod selector;

use crate::config::Config;
use crate::history::{BoxError, UpdateHistory};
use crate::network::mesh::MeshNetwork;
use crate::network::Network;
use crate::registry::{AdapterMeta, AdapterRegistry};
use crate::state::State;
use selector::{rank, SelectionContext, Strategy};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};
use tracing::{info, warn};

/// The caller's request is invalid (HTTP 400), as opposed to the node failing to serve it.
#[derive(Debug)]
pub struct InvalidRequest(pub String);

impl std::fmt::Display for InvalidRequest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for InvalidRequest {}

fn bad(msg: impl Into<String>) -> BoxError {
    Box::new(InvalidRequest(msg.into()))
}

const WEIGHTS_FILE: &str = "adapter_model.safetensors";
const CONFIG_FILE: &str = "adapter_config.json";

/// Adapter fields that `ml_engine.py` adds to its final JSON line when
/// federated inference is enabled.
#[derive(Debug, Clone, Deserialize)]
pub struct AdapterOutput {
    pub adapter_hash: String,
    pub adapter_path: String,
    pub adapter_size: u64,
    pub adapter_config: String,
    #[serde(default)]
    pub adapter_base_model: String,
    #[serde(default)]
    pub adapter_rank: u32,
    #[serde(default)]
    pub adapter_target_modules: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct InferRequest {
    pub prompt: String,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f64>,
    /// Overrides `[inference] selection` for this query.
    #[serde(default)]
    pub selection: Option<String>,
    /// Force the adapter advertised by this node (simulation / testing).
    /// Falls through to normal selection if that node has none.
    #[serde(default)]
    pub from_node: Option<String>,
    /// Optional system message for the chat template.
    #[serde(default)]
    pub system: Option<String>,
    /// Send the prompt as-is, without the chat template.
    #[serde(default)]
    pub raw: bool,
}

#[derive(Debug, Serialize)]
pub struct AdapterInfo {
    pub node_id: String,
    pub round: u64,
    pub sha256: String,
    pub cache_hit: bool,
}

#[derive(Debug, Serialize)]
pub struct InferResponse {
    pub text: String,
    /// `None` when the base model answered.
    pub adapter: Option<AdapterInfo>,
    pub note: Option<String>,
}

#[derive(Serialize)]
struct SidecarRequest<'a> {
    prompt: &'a str,
    max_new_tokens: u32,
    temperature: f64,
    adapter_id: Option<&'a str>,
    adapter_dir: Option<&'a str>,
    chat: bool,
    system: Option<&'a str>,
}

#[derive(Deserialize)]
struct SidecarResponse {
    text: String,
}

pub struct InferenceService {
    config: Config,
    own_node_id: String,
    history: Arc<RwLock<UpdateHistory>>,
    state: Arc<RwLock<State>>,
    network: Arc<RwLock<MeshNetwork>>,
    http: reqwest::Client,
    /// One lock per adapter hash: two queries for the same adapter fetch it once, while
    /// different adapters (and unrelated queries) are never stuck behind a slow download.
    fetch_locks: Mutex<HashMap<String, Arc<Mutex<()>>>>,
    sidecar: Mutex<Option<tokio::process::Child>>,
}

impl InferenceService {
    pub fn new(
        config: Config,
        own_node_id: String,
        history: Arc<RwLock<UpdateHistory>>,
        state: Arc<RwLock<State>>,
        network: Arc<RwLock<MeshNetwork>>,
    ) -> Self {
        InferenceService {
            config,
            own_node_id,
            history,
            state,
            network,
            // A stuck sidecar must not hang the request forever.
            http: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(config.inference.request_timeout_secs))
                .build()
                .unwrap_or_default(),
            fetch_locks: Mutex::new(HashMap::new()),
            sidecar: Mutex::new(None),
        }
    }

    fn adapters_dir(&self) -> PathBuf {
        Path::new(&self.config.node.data_dir).join("adapters")
    }

    // ------------------------------------------------------------------
    // Sidecar
    // ------------------------------------------------------------------

    /// Launch `inference_engine.py` (runs under the serve venv's python).
    pub async fn start_sidecar(&self) -> Result<(), BoxError> {
        let inf = &self.config.inference;
        let mut cmd = tokio::process::Command::new(&inf.sidecar_python);
        cmd.arg("inference_engine.py")
            .current_dir(".")
            .env("SLAKSHNA_INFER_PORT", inf.sidecar_port.to_string())
            .env("SLAKSHNA_INFER_BASE_MODEL", &inf.base_model)
            .env("SLAKSHNA_SERVE_PATH", &inf.serve_bhaskera_path)
            .kill_on_drop(true);
        if let Some(gid) = self.config.node.gpu_id {
            cmd.env("CUDA_VISIBLE_DEVICES", gid.to_string());
        }
        let child = cmd.spawn().map_err(|e| format!("failed to start inference sidecar: {e}"))?;
        *self.sidecar.lock().await = Some(child);
        info!("🧠 Inference sidecar starting on 127.0.0.1:{}", inf.sidecar_port);
        Ok(())
    }

    // ------------------------------------------------------------------
    // Step 1-2: publish our adapter
    // ------------------------------------------------------------------

    /// Register the freshly exported adapter in iroh-blobs and gossip its
    /// signed metadata. Failures are logged and never affect training.
    pub async fn publish_adapter(&self, round: u64, out: &AdapterOutput) {
        if let Err(e) = self.try_publish(round, out).await {
            warn!("⚠️ Could not publish adapter for round {}: {}", round, e);
        }
    }

    /// Publish the newest adapter already on disk that this node exported
    /// (e.g. after a restart, or if an earlier publish failed). No-op if none.
    pub async fn publish_latest_existing(&self) {
        let Ok(rd) = std::fs::read_dir(self.adapters_dir()) else { return };
        let mut best: Option<(u64, AdapterOutput)> = None;
        for e in rd.flatten() {
            let dir = e.path();
            let read = |f: &str| std::fs::read_to_string(dir.join(f)).ok();
            let (Some(meta), Some(cfg)) = (read("adapter_meta.json"), read(CONFIG_FILE)) else { continue };
            let Ok(m) = serde_json::from_str::<serde_json::Value>(&meta) else { continue };
            if m["node_id"].as_str() != Some(self.own_node_id.as_str()) { continue }
            let round = m["round"].as_u64().unwrap_or(0);
            if best.as_ref().map_or(false, |(r, _)| *r >= round) { continue }
            let Some(hash) = m["adapter_model_safetensors_sha256"].as_str() else { continue };
            let Ok(size) = std::fs::metadata(dir.join(WEIGHTS_FILE)).map(|x| x.len()) else { continue };
            best = Some((round, AdapterOutput {
                adapter_hash: hash.to_string(),
                adapter_path: dir.to_string_lossy().to_string(),
                adapter_size: size,
                adapter_config: cfg,
                adapter_base_model: m["base_model"].as_str().unwrap_or("").to_string(),
                adapter_rank: m["rank"].as_u64().unwrap_or(0) as u32,
                adapter_target_modules: m["target_modules"].as_array().map(|a|
                    a.iter().filter_map(|v| v.as_str().map(String::from)).collect()).unwrap_or_default(),
            }));
        }
        if let Some((round, out)) = best {
            info!("♻️ Re-publishing newest on-disk adapter (round {})", round);
            self.publish_adapter(round, &out).await;
        }
    }

    async fn try_publish(&self, round: u64, out: &AdapterOutput) -> Result<(), BoxError> {
        let weights = Path::new(&out.adapter_path).join(WEIGHTS_FILE);

        // Independent recompute: never advertise a hash we did not verify ourselves.
        let sha256 = sha256_file(&weights).await?;
        if sha256 != out.adapter_hash {
            return Err(format!("engine hash {} != file hash {}", out.adapter_hash, sha256).into());
        }

        let (blobs, endpoint_id) = {
            let net = self.network.read().await;
            (net.blobs(), net.endpoint_id().await)
        };
        let blobs = blobs.ok_or("blob service not running")?;
        let endpoint_id = endpoint_id.ok_or("endpoint not bound")?;
        let blake3 = blobs.add_file(&weights).await?;

        let trust_score = {
            let history = self.history.read().await;
            history
                .trust_rankings(usize::MAX)
                .into_iter()
                .find(|(n, _)| n == &self.own_node_id)
                .map(|(_, t)| t)
                .unwrap_or(0.0)
        };
        let keypair = {
            let st = self.state.read().await;
            st.get_keypair().cloned().ok_or("node keypair not loaded")?
        };

        let mut meta = AdapterMeta {
            node_id: self.own_node_id.clone(),
            endpoint_id,
            pubkey: String::new(),
            round,
            sha256,
            blake3,
            size: out.adapter_size,
            base_model: out.adapter_base_model.clone(),
            rank: out.adapter_rank,
            target_modules: out.adapter_target_modules.clone(),
            adapter_config: out.adapter_config.clone(),
            trust_score,
            timestamp: chrono::Utc::now().timestamp(),
            signature: String::new(),
        };
        meta.sign(&keypair);

        info!("🧩 Publishing adapter round {} sha256 {}", round, &meta.sha256[..12]);
        self.network.read().await.announce_adapter(meta).await
    }

    // ------------------------------------------------------------------
    // Steps 3-6: answer a query
    // ------------------------------------------------------------------

    pub async fn infer(&self, req: InferRequest) -> Result<InferResponse, BoxError> {
        let inf = &self.config.inference;

        // Validate before any work: the API is reachable by anyone who can reach the port.
        if req.prompt.trim().is_empty() {
            return Err(bad("prompt is empty"));
        }
        if req.prompt.chars().count() > inf.max_prompt_chars {
            return Err(bad(format!("prompt is longer than {} characters", inf.max_prompt_chars)));
        }
        let temperature = req.temperature.unwrap_or(0.7);
        if !temperature.is_finite() || !(0.0..=2.0).contains(&temperature) {
            return Err(bad("temperature must be between 0 and 2"));
        }
        let mut notes: Vec<String> = Vec::new();
        let asked = req.max_tokens.unwrap_or(256).max(1);
        let max_tokens = asked.min(inf.max_new_tokens_limit);
        if max_tokens < asked {
            notes.push(format!("max_tokens reduced from {asked} to the limit of {max_tokens}"));
        }

        let strategy: Strategy = req
            .selection
            .as_deref()
            .unwrap_or(&inf.selection)
            .parse()
            .map_err(|e: String| bad(e))?;

        // 3. Rank from the local registry; no network involved.
        let (candidates, registry): (Vec<AdapterMeta>, Arc<RwLock<AdapterRegistry>>) = {
            let registry = self.network.read().await.registry();
            let reg = registry.read().await;
            (reg.latest_per_node().into_iter().cloned().collect(), registry.clone())
        };
        let local_trust: HashMap<String, f64> = {
            let history = self.history.read().await;
            history.trust_rankings(usize::MAX).into_iter().collect()
        };
        let mut refs: Vec<&AdapterMeta> = candidates.iter().collect();
        if let Some(n) = req.from_node.as_deref() {
            if refs.iter().any(|m| m.node_id == n) {
                refs.retain(|m| m.node_id == n);
            } else {
                warn!("from_node {} has no adapter in the registry; using normal selection", n);
            }
        }
        let ranked = rank(
            &refs,
            strategy,
            &SelectionContext {
                own_node_id: &self.own_node_id,
                local_trust: &local_trust,
                min_trust: inf.min_trust,
                base_model: &inf.base_model,
            },
        );
        // Be honest when "trust" is not actually deciding anything.
        if strategy == Strategy::Trust && req.from_node.is_none() && ranked.len() > 1 {
            let t0 = local_trust.get(&ranked[0].node_id).copied().unwrap_or(0.0);
            if ranked.iter().all(|m| local_trust.get(&m.node_id).copied().unwrap_or(0.0) == t0) {
                notes.push("no trust scores separate the adapters yet; chose by newest round, then own adapter".into());
            }
        }

        // 4-5. Fetch (if needed) and verify, walking down the ranking: one dead or
        // corrupt adapter falls through to the next best before the base model.
        const MAX_ATTEMPTS: usize = 3;
        let deadline = std::time::Duration::from_secs(inf.fetch_timeout_secs);
        let mut used: Option<(AdapterMeta, PathBuf, bool)> = None;
        for meta in ranked.iter().take(MAX_ATTEMPTS) {
            match tokio::time::timeout(deadline, self.ensure_adapter(meta, &registry)).await {
                Ok(Ok((dir, hit))) => {
                    used = Some(((*meta).clone(), dir, hit));
                    break;
                }
                Ok(Err(e)) => {
                    warn!("⚠️ Adapter {} unavailable: {}", &meta.sha256[..12], e);
                    notes.push(format!("adapter {} unavailable ({e})", &meta.sha256[..12]));
                }
                Err(_) => {
                    warn!("⚠️ Adapter {} timed out after {:?}", &meta.sha256[..12], deadline);
                    notes.push(format!("adapter {} timed out", &meta.sha256[..12]));
                }
            }
        }
        if used.is_none() {
            notes.push(if ranked.is_empty() {
                "no suitable adapter in the registry; answered with the base model".into()
            } else {
                "no adapter could be fetched; answered with the base model".into()
            });
        }

        // 6. Local forward pass.
        let adapter_id = used.as_ref().map(|(m, _, _)| m.sha256.clone());
        let adapter_dir = used.as_ref().map(|(_, d, _)| d.to_string_lossy().to_string());
        let body = SidecarRequest {
            prompt: &req.prompt,
            max_new_tokens: max_tokens,
            temperature,
            adapter_id: adapter_id.as_deref(),
            adapter_dir: adapter_dir.as_deref(),
            chat: inf.chat_template && !req.raw,
            system: req.system.as_deref(),
        };
        let resp = self
            .http
            .post(format!("http://127.0.0.1:{}/generate", inf.sidecar_port))
            .json(&body)
            .send()
            .await
            .map_err(|e| format!("inference sidecar unreachable: {e}"))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(format!("inference sidecar error {status}: {text}").into());
        }
        let parsed: SidecarResponse = resp.json().await.map_err(|e| format!("bad sidecar reply: {e}"))?;

        Ok(InferResponse {
            text: parsed.text,
            adapter: used.map(|(m, _, hit)| AdapterInfo {
                node_id: m.node_id,
                round: m.round,
                sha256: m.sha256,
                cache_hit: hit,
            }),
            note: if notes.is_empty() { None } else { Some(notes.join("; ")) },
        })
    }

    /// Return the local directory of `meta`'s adapter, downloading it by hash
    /// from the nodes that hold it if it is not cached. The bool is true on a cache hit.
    async fn ensure_adapter(
        &self,
        meta: &AdapterMeta,
        registry: &Arc<RwLock<AdapterRegistry>>,
    ) -> Result<(PathBuf, bool), BoxError> {
        let final_dir = self.adapters_dir().join(&meta.sha256);
        if final_dir.join(WEIGHTS_FILE).is_file() && final_dir.join(CONFIG_FILE).is_file() {
            touch(&final_dir);
            return Ok((final_dir, true));
        }

        let hash_lock = {
            let mut locks = self.fetch_locks.lock().await;
            locks.entry(meta.sha256.clone()).or_default().clone()
        };
        let _guard = hash_lock.lock().await;
        // Another query may have finished the download while we waited.
        if final_dir.join(WEIGHTS_FILE).is_file() && final_dir.join(CONFIG_FILE).is_file() {
            return Ok((final_dir, true));
        }

        if meta.size > self.config.inference.max_adapter_bytes {
            return Err(format!("adapter is {} bytes, over the configured limit", meta.size).into());
        }
        let blobs = self.network.read().await.blobs().ok_or("blob service not running")?;

        // The advertiser first, then any other node advertising the same hash.
        let mut sources: Vec<AdapterMeta> = vec![meta.clone()];
        {
            let reg = registry.read().await;
            for h in reg.holders_of(&meta.sha256) {
                if !sources.iter().any(|s| s.node_id == h.node_id) {
                    sources.push(h.clone());
                }
            }
        }

        let adapters_dir = self.adapters_dir();
        std::fs::create_dir_all(&adapters_dir)?;
        let tmp_dir = adapters_dir.join(format!(".fetch_{}", meta.sha256));
        let mut last_err: BoxError = "no sources".into();

        for src in &sources {
            let _ = std::fs::remove_dir_all(&tmp_dir);
            std::fs::create_dir_all(&tmp_dir)?;
            let tmp_weights = tmp_dir.join(WEIGHTS_FILE);

            info!("⬇️ Fetching adapter {} from {}", &meta.sha256[..12], src.node_id);
            if let Err(e) = blobs.fetch(&src.blake3, &src.endpoint_id, &tmp_weights).await {
                warn!("⚠️ Fetch from {} failed: {}", src.node_id, e);
                last_err = e;
                continue;
            }

            // 5. Verify against the advertised hash before the file is ever used.
            let actual = sha256_file(&tmp_weights).await?;
            if actual != meta.sha256 {
                warn!(
                    "🚫 Hash mismatch from {}: advertised {} got {}",
                    src.node_id, &meta.sha256[..12], &actual[..12]
                );
                registry.write().await.mark_rejected(&src.node_id, &meta.sha256);
                last_err = format!("hash mismatch from {}", src.node_id).into();
                continue;
            }

            std::fs::write(tmp_dir.join(CONFIG_FILE), &meta.adapter_config)?;
            let _ = std::fs::remove_dir_all(&final_dir); // clear any partial leftovers
            std::fs::rename(&tmp_dir, &final_dir)?;
            info!("✅ Adapter {} verified and cached", &meta.sha256[..12]);
            self.evict_cache(registry).await;
            return Ok((final_dir, false));
        }

        let _ = std::fs::remove_dir_all(&tmp_dir);
        Err(last_err)
    }

    /// Keep at most `cache_max_adapters` fetched adapters; our own are never evicted.
    async fn evict_cache(&self, registry: &Arc<RwLock<AdapterRegistry>>) {
        let own_hashes: Vec<String> = {
            let reg = registry.read().await;
            reg.latest_per_node()
                .into_iter()
                .filter(|m| m.node_id == self.own_node_id)
                .map(|m| m.sha256.clone())
                .collect()
        };
        let Ok(rd) = std::fs::read_dir(self.adapters_dir()) else { return };
        let mut entries: Vec<(std::time::SystemTime, PathBuf)> = rd
            .flatten()
            .filter(|e| {
                let name = e.file_name().to_string_lossy().to_string();
                !name.starts_with('.') && !own_hashes.contains(&name)
            })
            .filter_map(|e| Some((e.metadata().ok()?.modified().ok()?, e.path())))
            .collect();
        entries.sort_by_key(|(t, _)| *t);
        let excess = entries.len().saturating_sub(self.config.inference.cache_max_adapters);
        for (_, path) in entries.into_iter().take(excess) {
            info!("🧹 Evicting cached adapter {}", path.display());
            let _ = std::fs::remove_dir_all(path);
        }
    }

    /// Registry contents for `GET /adapters`.
    pub async fn list_adapters(&self) -> Vec<serde_json::Value> {
        let registry = self.network.read().await.registry();
        let reg = registry.read().await;
        let trust: HashMap<String, f64> = {
            let history = self.history.read().await;
            history.trust_rankings(usize::MAX).into_iter().collect()
        };
        reg.latest_per_node()
            .into_iter()
            .map(|m| {
                serde_json::json!({
                    "node_id": m.node_id,
                    "round": m.round,
                    "sha256": m.sha256,
                    "size": m.size,
                    "base_model": m.base_model,
                    "rank": m.rank,
                    "advertised_trust": m.trust_score,
                    "local_trust": trust.get(&m.node_id).copied().unwrap_or(0.0),
                    "cached": self.adapters_dir().join(&m.sha256).join(WEIGHTS_FILE).is_file(),
                    "is_self": m.node_id == self.own_node_id,
                })
            })
            .collect()
    }
}

/// Bump a directory's mtime so LRU eviction sees it as recently used.
fn touch(dir: &Path) {
    let _ = std::fs::File::open(dir).and_then(|f| f.set_modified(std::time::SystemTime::now()));
}

async fn sha256_file(path: &Path) -> Result<String, BoxError> {
    let path = path.to_path_buf();
    tokio::task::spawn_blocking(move || -> Result<String, BoxError> {
        let mut f = std::fs::File::open(&path)?;
        let mut hasher = Sha256::new();
        let mut buf = vec![0u8; 1024 * 1024];
        loop {
            let n = f.read(&mut buf)?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
        }
        Ok(hex::encode(hasher.finalize()))
    })
    .await?
}
