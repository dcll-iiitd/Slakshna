use serde::{Deserialize, Serialize};
use std::fs;

type BoxError = Box<dyn std::error::Error + Send + Sync>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub federation: FederationConfig,
    #[serde(default)]
    pub training: TrainingConfig,
    #[serde(default)]
    pub compression: CompressionConfig,
    pub node: NodeConfig,
    pub network: NetworkConfig,
    #[serde(default)]
    pub discovery: DiscoveryConfig,
    #[serde(default)]
    pub inference: InferenceConfig,
    pub logging: LoggingConfig,
}

/// Federated inference. Disabled by default so training-only configs keep
/// working unchanged.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    #[serde(default)]
    pub enabled: bool,
    /// Adapter selection strategy: "trust" (highest local trust score),
    /// "self" (this node's own personalised adapter) or "latest" (newest round).
    #[serde(default = "default_selection")]
    pub selection: String,
    /// Adapters from nodes whose locally-computed trust is below this are ignored.
    #[serde(default)]
    pub min_trust: f64,
    /// How many fetched adapters are kept on disk (own adapters are never evicted).
    #[serde(default = "default_cache_max_adapters")]
    pub cache_max_adapters: usize,
    /// Refuse to fetch adapters larger than this.
    #[serde(default = "default_max_adapter_bytes")]
    pub max_adapter_bytes: u64,
    /// Base model the local inference sidecar loads (HF name or local path).
    /// Adapters advertising a different base model are never selected.
    #[serde(default)]
    pub base_model: String,
    /// Local port of the Python inference sidecar (Bhaskera `serve` branch).
    #[serde(default = "default_sidecar_port")]
    pub sidecar_port: u16,
    /// Path to the Bhaskera `serve` submodule checkout.
    #[serde(default = "default_serve_path")]
    pub serve_bhaskera_path: String,
    /// Python interpreter for the sidecar (the serve venv's python).
    #[serde(default = "default_sidecar_python")]
    pub sidecar_python: String,
    /// Spawn the sidecar automatically. Disable to run it yourself.
    #[serde(default = "default_true")]
    pub spawn_sidecar: bool,
    /// Skip the federated training loop: the node only gossips, serves
    /// adapters and answers queries. For testing inference on saved adapters.
    #[serde(default)]
    pub serve_only: bool,
    /// System One decisions: how many nodes' adapters are mixed.
    #[serde(default = "default_decision_k")]
    pub decision_k: usize,
    /// How adapters' probabilities are mixed: "prob_mean" (default; a bad
    /// member's influence is bounded by its weight) or "logit_mean".
    #[serde(default = "default_combine")]
    pub combine: String,
    /// Softmax temperature turning local trust into mixing weights.
    #[serde(default = "default_weight_temperature")]
    pub weight_temperature: f64,
    /// Re-score the options under this many cyclic re-orderings and average,
    /// which cancels letter/position bias. 0 = auto (every shift for up to 6
    /// options, 6 evenly spread shifts beyond that), 1 = a single, undebiased
    /// pass. Cost is this many forward passes per adapter.
    #[serde(default = "default_permutations")]
    pub permutations: u32,
    /// Divides the label logits before the softmax. Fit offline on held-out
    /// decisions (>1 softens an over-confident model). 1.0 = no scaling.
    #[serde(default = "default_calibration_temperature")]
    pub calibration_temperature: f64,
    /// Max time to wait for missing adapters; slow ones are dropped.
    #[serde(default = "default_fetch_timeout")]
    pub fetch_timeout_secs: u64,
}

fn default_decision_k() -> usize { 3 }
fn default_combine() -> String { "prob_mean".to_string() }
// Trust scores are per-round averages on a ~0..1 scale (see `rank_trust`), so
// differences between peers are small; a low temperature keeps them meaningful.
fn default_weight_temperature() -> f64 { 0.1 }
fn default_permutations() -> u32 { 0 }
fn default_calibration_temperature() -> f64 { 1.0 }
fn default_fetch_timeout() -> u64 { 30 }
fn default_selection() -> String { "trust".to_string() }
fn default_cache_max_adapters() -> usize { 16 }
fn default_max_adapter_bytes() -> u64 { 2 * 1024 * 1024 * 1024 }
fn default_sidecar_port() -> u16 { 8765 }
fn default_serve_path() -> String { "Bhaskera-serve".to_string() }
fn default_sidecar_python() -> String { "python".to_string() }

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            selection: default_selection(),
            min_trust: 0.0,
            cache_max_adapters: default_cache_max_adapters(),
            max_adapter_bytes: default_max_adapter_bytes(),
            base_model: String::new(),
            sidecar_port: default_sidecar_port(),
            serve_bhaskera_path: default_serve_path(),
            sidecar_python: default_sidecar_python(),
            spawn_sidecar: true,
            serve_only: false,
            decision_k: default_decision_k(),
            combine: default_combine(),
            weight_temperature: default_weight_temperature(),
            permutations: default_permutations(),
            calibration_temperature: default_calibration_temperature(),
            fetch_timeout_secs: default_fetch_timeout(),
        }
    }
}

/// Identifies the federation a node participates in. Every node sharing an `id`
/// derives the same gossip topic and therefore trains together.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederationConfig {
    pub id: String,
    pub name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Length of a federated epoch. Nodes align to this on the wall clock so
    /// that independently started peers land on the same boundary.
    #[serde(default = "default_epoch_duration")]
    pub epoch_duration_secs: u64,
    /// How far into the epoch a node waits for peer updates to arrive before
    /// aggregating with whatever it has.
    #[serde(default = "default_sync_deadline")]
    pub sync_deadline_secs: u64,
    /// Number of participants expected in the federation. Once updates from
    /// this many nodes are on record, the sync barrier is released early.
    #[serde(default = "default_expected_peers")]
    pub expected_peers: usize,
}

fn default_epoch_duration() -> u64 {
    600
}
fn default_sync_deadline() -> u64 {
    300
}
fn default_expected_peers() -> usize {
    6
}

impl Default for TrainingConfig {
    fn default() -> Self {
        TrainingConfig {
            epoch_duration_secs: default_epoch_duration(),
            sync_deadline_secs: default_sync_deadline(),
            expected_peers: default_expected_peers(),
        }
    }
}

/// Settings for model-delta transport.  The payload itself remains opaque to
/// the Rust gossip layer; these values are passed to the Python ML engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionConfig {
    #[serde(default = "default_compression_enabled")]
    pub enabled: bool,
    #[serde(default = "default_sparsity")]
    pub sparsity: f64,
    #[serde(default = "default_quantization")]
    pub quantization: String,
    #[serde(default)]
    pub allow_legacy_delta_format: bool,
    #[serde(default = "default_max_payload_bytes")]
    pub max_payload_bytes: usize,
    #[serde(default = "default_max_tensor_elements")]
    pub max_tensor_elements: usize,
}

fn default_compression_enabled() -> bool { true }
fn default_sparsity() -> f64 { 0.1 }
fn default_quantization() -> String { "symmetric_int8".to_string() }
// Base64 expands by roughly 4/3; retain room under gossip's 10 MiB message cap.
fn default_max_payload_bytes() -> usize { 7 * 1024 * 1024 }
fn default_max_tensor_elements() -> usize { 10_000_000 }

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            enabled: default_compression_enabled(),
            sparsity: default_sparsity(),
            quantization: default_quantization(),
            allow_legacy_delta_format: false,
            max_payload_bytes: default_max_payload_bytes(),
            max_tensor_elements: default_max_tensor_elements(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeConfig {
    pub id: String,
    pub data_dir: String,
    #[serde(default)]
    pub gpu_id: Option<u32>,
    #[serde(default)]
    pub num_gpus: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    pub host: String,
    pub p2p_port: u16,
    pub ws_port: u16,
    pub api_port: u16,
    /// Optional seed peers, as Iroh EndpointId strings, each optionally suffixed
    /// with `@host:port` to pin a direct address on closed networks.
    ///
    /// Every entry is an equal member of the federation — there is no privileged
    /// node here. The list may be empty: a node that finds no seeds still starts,
    /// advertises itself over mDNS and the mainline DHT, and merges into the swarm
    /// as soon as any peer shows up. Peers learned at runtime are remembered in
    /// `{data_dir}/rocksdb` and redialled automatically on the next start.
    #[serde(default, alias = "boot_nodes")]
    pub peers: Option<Vec<String>>,
    /// Optional list of allowed Iroh EndpointId strings.
    /// If non-empty, only these peers may connect (whitelisting).
    #[serde(default)]
    pub allowed_peers: Option<Vec<String>>,
    /// Reject model updates and peer reviews that are not signed by the key
    /// behind their `node_id`. Turn off only while migrating a federation whose
    /// nodes still run builds that send placeholder signatures.
    #[serde(default = "default_true")]
    pub require_signed_records: bool,
}

/// Which serverless discovery mechanisms this node participates in. All of them
/// are symmetric: a node both advertises itself and looks others up.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryConfig {
    /// mDNS on the local network. Zero-config discovery of federation members on
    /// the same LAN or the same host, scoped to the federation id.
    #[serde(default = "default_true")]
    pub mdns: bool,
    /// BitTorrent mainline DHT. Resolves an EndpointId to its current address
    /// anywhere on the internet without contacting any server we operate.
    #[serde(default = "default_true")]
    pub dht: bool,
    /// Number 0's public pkarr/DNS address lookup. Fastest resolution path, but
    /// it is third-party infrastructure — turn it off for a fully self-contained
    /// federation and rely on `mdns` + `dht`.
    #[serde(default = "default_true")]
    pub dns: bool,
    /// Use public relay servers as a fallback transport when direct hole punching
    /// fails. Relays only forward encrypted QUIC; they cannot read traffic.
    #[serde(default = "default_true")]
    pub relay: bool,
}

fn default_true() -> bool {
    true
}

impl Default for DiscoveryConfig {
    fn default() -> Self {
        DiscoveryConfig { mdns: true, dht: true, dns: true, relay: true }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    pub level: String,
}

impl Config {
    pub fn load(path: &str) -> Result<Self, BoxError> {
        let content = fs::read_to_string(path)?;
        let config: Config = toml::from_str(&content)?;
        if !config.compression.sparsity.is_finite()
            || config.compression.sparsity <= 0.0
            || config.compression.sparsity > 1.0
        {
            return Err("compression.sparsity must be finite and in (0, 1]".into());
        }
        if config.compression.quantization != "symmetric_int8" {
            return Err("compression.quantization must be symmetric_int8".into());
        }
        if config.compression.max_payload_bytes == 0
            || config.compression.max_payload_bytes > 2000000000
        {
            return Err("compression.max_payload_bytes must be between 1 and 2000000000".into());
        }
        if config.compression.max_tensor_elements == 0 {
            return Err("compression.max_tensor_elements must be positive".into());
        }
        if !matches!(config.inference.selection.as_str(), "trust" | "self" | "latest") {
            return Err("inference.selection must be one of: trust, self, latest".into());
        }
        if !matches!(config.inference.combine.as_str(), "prob_mean" | "logit_mean") {
            return Err("inference.combine must be one of: prob_mean, logit_mean".into());
        }
        if config.inference.decision_k == 0 || config.inference.decision_k > 16 {
            return Err("inference.decision_k must be between 1 and 16".into());
        }
        if config.inference.permutations > 8 {
            return Err("inference.permutations must be between 0 (auto) and 8".into());
        }
        for (name, v) in [
            ("weight_temperature", config.inference.weight_temperature),
            ("calibration_temperature", config.inference.calibration_temperature),
        ] {
            if !v.is_finite() || v <= 0.0 {
                return Err(format!("inference.{name} must be finite and positive").into());
            }
        }
        if config.inference.enabled && config.inference.base_model.is_empty() {
            return Err("inference.base_model must be set when inference.enabled = true".into());
        }
        Ok(config)
    }

    pub fn save(&self, path: &str) -> Result<(), BoxError> {
        let content = toml::to_string_pretty(self)?;
        fs::write(path, content)?;
        Ok(())
    }
}
