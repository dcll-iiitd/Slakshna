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
    /// Steps 4-5: federated agents and A2A. Off unless `[agents] enabled = true`.
    #[serde(default)]
    pub agents: AgentsConfig,
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
        Ok(config)
    }

    pub fn save(&self, path: &str) -> Result<(), BoxError> {
        let content = toml::to_string_pretty(self)?;
        fs::write(path, content)?;
        Ok(())
    }
}

/// Per-task spending ceiling for agent work (A1).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskBudgetConfig {
    #[serde(default = "d_tokens")]
    pub tokens: u64,
    #[serde(default)]
    pub gpu_seconds: u64,
    #[serde(default = "d_wall")]
    pub wall_clock_secs: u64,
}
fn d_tokens() -> u64 { 20_000 }
fn d_wall() -> u64 { 600 }
impl Default for TaskBudgetConfig {
    fn default() -> Self {
        TaskBudgetConfig { tokens: d_tokens(), gpu_seconds: 0, wall_clock_secs: d_wall() }
    }
}

/// A bearer key allowed to call the A2A HTTP gateway. Only the SHA-256 of the key
/// is stored, so the config file does not hold a usable secret.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiKeyConfig {
    pub principal: String,
    pub key_sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetConfig {
    pub name: String,
    /// JSON-lines file; each line `{"input": "<base64>", "label": "<base64>"}`.
    pub path: String,
    #[serde(default = "d_metric")]
    pub metric: String,
    #[serde(default = "d_queries")]
    pub queries_per_requester: u32,
    #[serde(default)]
    pub epsilon_total: Option<f64>,
    #[serde(default)]
    pub auditors: Vec<String>,
}
fn d_metric() -> String { "accuracy".into() }
fn d_queries() -> u32 { 20 }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolsConfig {
    /// JSON manifest: `{"servers": [{"name", "command", "args", "allow": [...]}]}`. Unset disables MCP tools.
    #[serde(default)]
    pub manifest: Option<String>,
    /// Max model/tool steps in one `agent` task.
    #[serde(default = "d_steps")]
    pub max_steps: usize,
}
fn d_steps() -> usize { 6 }
impl Default for ToolsConfig {
    fn default() -> Self {
        ToolsConfig { manifest: None, max_steps: d_steps() }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProviderConfig {
    /// Hosts the agent may call. Empty disables the plugin.
    #[serde(default)]
    pub allowed_hosts: Vec<String>,
    #[serde(default)]
    pub allow_insecure_loopback: bool,
}

/// This node's part in research programs: which roles it plays and how it runs candidates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchConfig {
    #[serde(default)]
    pub enabled: bool,
    /// Any of: planner, executor, evaluator.
    #[serde(default)]
    pub roles: Vec<String>,
    /// Base project directory candidates are applied to (executor and evaluator).
    #[serde(default)]
    pub project: Option<String>,
    /// Files a candidate may change (exact paths); anything else is rejected. Empty = unrestricted.
    #[serde(default)]
    pub allowed_paths: Vec<String>,
    /// Text file describing the project for the model-backed planner and maker (trusted operator notes).
    #[serde(default)]
    pub guidance_file: Option<String>,
    /// Container runtime: auto (podman, then docker, then apptainer), podman, docker or apptainer.
    #[serde(default = "d_runtime")]
    pub runtime: String,
    /// Give run/evaluation containers GPU access (Docker: `--gpus`; Podman: CDI device).
    #[serde(default)]
    pub gpu: bool,
    /// Restrict the containers to these GPU indices, e.g. "0,1" (all when unset).
    #[serde(default)]
    pub gpu_devices: Option<String>,
    /// Read-only mounts, `host_path:name` -> `/inputs/name` (models, datasets).
    #[serde(default)]
    pub mounts: Vec<String>,
    /// Environment for the containers, `KEY=VALUE`.
    #[serde(default)]
    pub env: Vec<String>,
    #[serde(default = "d_cpus")]
    pub cpus: u32,
    #[serde(default = "d_mem")]
    pub memory_mb: u64,
    /// Container image for runs and for evaluation (must be on `allowed_images` for outside requests; here the operator chooses it).
    #[serde(default)]
    pub image: Option<String>,
    /// Command run in the sandbox to train/execute a candidate (executor).
    #[serde(default)]
    pub run_command: Vec<String>,
    /// Command run in the sandbox to predict `/work/in.jsonl` -> `/work/out.jsonl` (evaluator).
    #[serde(default)]
    pub predict_command: Vec<String>,
    /// Name of the `[[agents.datasets]]` entry the evaluator role scores on.
    #[serde(default)]
    pub dataset: Option<String>,
    #[serde(default)]
    pub gpu_class: Option<String>,
    #[serde(default)]
    pub data_access: Vec<String>,
    #[serde(default = "d_gpu_secs")]
    pub gpu_seconds: u64,
    /// Total GPU-seconds this node may spend on runs before a hard stop.
    #[serde(default = "d_gpu_budget")]
    pub gpu_budget_seconds: u64,
    #[serde(default = "d_committee")]
    pub committee_size: usize,
    #[serde(default = "d_lease")]
    pub lease_secs: u64,
    #[serde(default = "d_capacity")]
    pub capacity: usize,
    #[serde(default = "d_tick")]
    pub tick_secs: u64,
    /// Ticks to wait for proposals/evaluations before proceeding with what exists.
    #[serde(default = "d_wait")]
    pub wait_ticks: u32,
    /// Planners that must independently propose an item for it to join the agenda (2 = agreement required).
    #[serde(default = "d_supporters")]
    pub min_supporters: usize,
    /// With min_supporters = 1: items one planner may get adopted alone.
    #[serde(default = "d_solo")]
    pub solo_per_planner: usize,
    /// An item failed by this many distinct nodes is abandoned so the generation can proceed.
    #[serde(default = "d_failures")]
    pub max_item_failures: usize,
    /// Most items on one agenda.
    #[serde(default = "d_agenda_items")]
    pub max_agenda_items: usize,
    #[serde(default = "d_maxgen")]
    pub max_generations: u64,
    #[serde(default = "d_run_wall")]
    pub run_wall_clock_secs: u64,
}
fn d_runtime() -> String { "auto".into() }
fn d_cpus() -> u32 { 4 }
fn d_mem() -> u64 { 16 * 1024 }
fn d_gpu_secs() -> u64 { 3600 }
fn d_gpu_budget() -> u64 { 86_400 }
fn d_committee() -> usize { 5 }
fn d_lease() -> u64 { 120 }
fn d_capacity() -> usize { 1 }
fn d_tick() -> u64 { 10 }
fn d_wait() -> u32 { 12 }
fn d_failures() -> usize { 3 }
fn d_supporters() -> usize { 2 }
fn d_solo() -> usize { 2 }
fn d_agenda_items() -> usize { 8 }
fn d_maxgen() -> u64 { 5 }
fn d_run_wall() -> u64 { 1800 }
impl Default for ResearchConfig {
    fn default() -> Self {
        toml::from_str("").expect("all ResearchConfig fields have defaults")
    }
}

/// Per-node data-egress policy (A7). A node holding private data should set
/// `allow = ["finding", "metric"]` so only typed findings can leave.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EgressConfig {
    /// Export kinds that may leave: finding, metric, artifact.
    #[serde(default = "d_egress_allow")]
    pub allow: Vec<String>,
    #[serde(default = "d_egress_bytes")]
    pub max_bytes: usize,
    #[serde(default = "d_egress_blocked")]
    pub blocked_classes: Vec<String>,
}
fn d_egress_allow() -> Vec<String> { vec!["finding".into(), "metric".into(), "artifact".into()] }
fn d_egress_bytes() -> usize { 1024 * 1024 }
fn d_egress_blocked() -> Vec<String> { vec!["pii".into(), "raw".into()] }
impl Default for EgressConfig {
    fn default() -> Self {
        EgressConfig { allow: d_egress_allow(), max_bytes: d_egress_bytes(), blocked_classes: d_egress_blocked() }
    }
}

/// An outside A2A agent this node can delegate to. The bearer token is read from the
/// environment variable named here, so no secret sits in the config file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteAgentConfig {
    /// Skill id this node exposes locally.
    pub skill_id: String,
    /// Base URL of the remote A2A gateway.
    pub url: String,
    /// Skill id on the remote agent.
    pub remote_skill: String,
    #[serde(default)]
    pub bearer_env: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PolicyConfig {
    #[serde(default)]
    pub allowed_counterparties: Option<Vec<String>>,
    #[serde(default)]
    pub denied_counterparties: Vec<String>,
    #[serde(default)]
    pub allowed_data_classes: Vec<String>,
    #[serde(default)]
    pub max_spend_per_task: u64,
    #[serde(default)]
    pub max_spend_per_day: u64,
    #[serde(default)]
    pub human_approval_actions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentsConfig {
    #[serde(default)]
    pub enabled: bool,
    /// Where the A2A JSON-RPC gateway listens; 0 disables the gateway.
    #[serde(default)]
    pub gateway_port: u16,
    #[serde(default = "d_bind")]
    pub gateway_bind: String,
    /// PEM paths; with both set the gateway serves HTTPS, otherwise plain HTTP
    /// (put it behind a TLS terminator or keep it on loopback).
    #[serde(default)]
    pub tls_cert: Option<String>,
    #[serde(default)]
    pub tls_key: Option<String>,
    /// Public URL advertised in the Agent Card.
    #[serde(default)]
    pub public_url: Option<String>,
    #[serde(default)]
    pub api_keys: Vec<ApiKeyConfig>,
    /// OpenAI-compatible federation serving endpoint used by the `chat` skill.
    #[serde(default)]
    pub serve_url: Option<String>,
    /// Model name to request from the serving endpoint; unset = the first model it lists.
    #[serde(default)]
    pub serve_model: Option<String>,
    #[serde(default = "d_python")]
    pub python: String,
    #[serde(default = "d_conc")]
    pub max_concurrent_tasks: usize,
    #[serde(default = "d_task_timeout")]
    pub task_timeout_secs: u64,
    #[serde(default)]
    pub task_budget: TaskBudgetConfig,
    /// Container images candidates may be evaluated in (exact match or `prefix*`).
    #[serde(default)]
    pub allowed_images: Vec<String>,
    #[serde(default)]
    pub datasets: Vec<DatasetConfig>,
    /// Hex Ed25519 public keys allowed to broadcast a halt.
    #[serde(default)]
    pub halt_keys: Vec<String>,
    #[serde(default)]
    pub policy: PolicyConfig,
    /// SHA-256 (hex) of the admin bearer key for `/agents/*` management endpoints
    /// (tasks, approvals, directory). Unset disables them.
    #[serde(default)]
    pub admin_key_sha256: Option<String>,
    /// Tools the `agent` skill may use (MCP servers from a per-agent manifest).
    #[serde(default)]
    pub tools: ToolsConfig,
    /// Outside HTTP APIs the `agent` skill may call (policy-gated plugin).
    #[serde(default)]
    pub provider: ProviderConfig,
    /// Participation in federated research programs (Step 4). Off by default.
    #[serde(default)]
    pub research: ResearchConfig,
    /// Price per skill announced in the Agent Card (settlement units).
    #[serde(default)]
    pub prices: std::collections::BTreeMap<String, u64>,
    /// What may leave this node in task results.
    #[serde(default)]
    pub egress: EgressConfig,
    /// Outside A2A agents exposed as local skills.
    #[serde(default)]
    pub remote: Vec<RemoteAgentConfig>,
    #[serde(default = "d_card_name")]
    pub card_name: String,
    #[serde(default)]
    pub card_description: String,
}
fn d_bind() -> String { "127.0.0.1".into() }
fn d_python() -> String { "python3".into() }
fn d_conc() -> usize { 4 }
fn d_task_timeout() -> u64 { 600 }
fn d_card_name() -> String { "slakshna-agent".into() }

impl Default for AgentsConfig {
    fn default() -> Self {
        toml::from_str("").expect("all AgentsConfig fields have defaults")
    }
}
