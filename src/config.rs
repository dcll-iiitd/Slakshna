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
    pub logging: LoggingConfig,
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
            || config.compression.max_payload_bytes > 7 * 1024 * 1024
        {
            return Err("compression.max_payload_bytes must be between 1 and 7340032".into());
        }
        if config.compression.max_tensor_elements == 0 {
            return Err("compression.max_tensor_elements must be positive".into());
        }
        Ok(config)
    }

    pub fn save(&self, path: &str) -> Result<(), BoxError> {
        let content = toml::to_string_pretty(self)?;
        fs::write(path, content)?;
        Ok(())
    }
}
