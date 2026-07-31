use serde::{Deserialize, Serialize};
use std::fs;

type BoxError = Box<dyn std::error::Error + Send + Sync>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub chain: ChainConfig,
    pub block: BlockConfig,
    pub rewards: RewardsConfig,
    pub genesis: GenesisConfig,
    pub faucet: FaucetConfig,
    pub token: TokenConfig,
    pub node: NodeConfig,
    pub network: NetworkConfig,
    pub validators: ValidatorsConfig,
    pub pruning: PruningConfig,
    pub logging: LoggingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainConfig {
    pub chain_id: String,
    pub chain_name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockConfig {
    pub block_time: u64,
    pub gas_limit: u64,
    pub max_txs_per_block: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RewardsConfig {
    pub block_reward: u64,
    pub validator_percent: u64,
    pub service_pool_percent: u64,
    pub top_nodes: usize,
    pub rank_1_percent: u64,
    pub rank_2_percent: u64,
    pub rank_3_percent: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenesisConfig {
    pub master_address: String,
    pub master_balance: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaucetConfig {
    pub enabled: bool,
    pub amount: u64,
    pub cooldown: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenConfig {
    pub name: String,
    pub symbol: String,
    pub decimals: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeConfig {
    pub id: String,
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
    /// Iroh NodeId strings of peers to connect to on startup.
    /// Replaces the old libp2p multiaddress boot_nodes.
    #[serde(default)]
    pub boot_nodes: Option<Vec<String>>,
    #[serde(default)]
    pub star: Option<StarConfig>,
    /// Optional list of allowed Iroh NodeId strings.
    /// If non-empty, only these peers may connect (whitelisting).
    #[serde(default)]
    pub allowed_peers: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StarConfig {
    pub master_url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatorsConfig {
    pub addresses: Vec<String>,
    pub max_validators: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PruningConfig {
    pub keep_blocks: u64,
    pub keep_txs: u64,
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
