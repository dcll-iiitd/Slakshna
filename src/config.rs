use serde::{Deserialize, Serialize};
use std::fs;

type BoxError = Box<dyn std::error::Error + Send + Sync>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub federation: FederationConfig,
    #[serde(default)]
    pub training: TrainingConfig,
    pub node: NodeConfig,
    pub network: NetworkConfig,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeConfig {
    pub id: String,
    /// "bootstrap" for a node peers dial into first, "peer" for everyone else.
    #[serde(rename = "type")]
    pub node_type: String,
    pub data_dir: String,
    #[serde(default)]
    pub gpu_id: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    pub topology: String,
    pub host: String,
    pub p2p_port: u16,
    pub ws_port: u16,
    pub api_port: u16,
    /// Iroh NodeId strings of peers to connect to on startup,
    /// optionally suffixed with `@host:port` for a direct dial.
    #[serde(default)]
    pub boot_nodes: Option<Vec<String>>,
    /// Optional list of allowed Iroh NodeId strings.
    /// If non-empty, only these peers may connect (whitelisting).
    #[serde(default)]
    pub allowed_peers: Option<Vec<String>>,
    // Keep the nested table last: TOML requires values to precede sub-tables,
    // so `Config::save` would otherwise emit a file it cannot read back.
    #[serde(default)]
    pub star: Option<StarConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StarConfig {
    pub master_url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    pub level: String,
}

impl Config {
    pub fn load(path: &str) -> Result<Self, BoxError> {
        let content = fs::read_to_string(path)?;
        let config: Config = toml::from_str(&content)?;
        Ok(config)
    }

    pub fn save(&self, path: &str) -> Result<(), BoxError> {
        let content = toml::to_string_pretty(self)?;
        fs::write(path, content)?;
        Ok(())
    }
}
