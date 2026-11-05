//! A2A v1.0 data model, task lifecycle and persistent task store (M1, M2).
//!
//! Field names and enum values follow the A2A v1.0 specification
//! (a2a-protocol.org/latest/specification): camelCase JSON, `TASK_STATE_*`
//! and `ROLE_*` enum strings, AgentCard with `supportedInterfaces`,
//! `protocolBinding` values `JSONRPC` / `GRPC` / `HTTP+JSON`, and Parts that
//! carry exactly one of `text`, `raw`, `url`, `data`.
//!
//! Every Agent Card version, task contract and lifecycle transition is also
//! written as a provenance record through a [`ProvenanceSink`], so work done
//! for outside agents is as attributable as work done inside the federation.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tokio::sync::broadcast;

// ---------------------------------------------------------------------------
// Wire types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TaskState {
    #[serde(rename = "TASK_STATE_SUBMITTED")]
    Submitted,
    #[serde(rename = "TASK_STATE_WORKING")]
    Working,
    #[serde(rename = "TASK_STATE_INPUT_REQUIRED")]
    InputRequired,
    #[serde(rename = "TASK_STATE_AUTH_REQUIRED")]
    AuthRequired,
    #[serde(rename = "TASK_STATE_COMPLETED")]
    Completed,
    #[serde(rename = "TASK_STATE_FAILED")]
    Failed,
    #[serde(rename = "TASK_STATE_CANCELED")]
    Canceled,
    #[serde(rename = "TASK_STATE_REJECTED")]
    Rejected,
}

impl TaskState {
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Canceled | Self::Rejected)
    }

    pub fn can_transition_to(self, next: TaskState) -> bool {
        use TaskState::*;
        match self {
            Submitted => matches!(next, Working | Rejected | Canceled | Failed),
            Working => matches!(next, Completed | Failed | Canceled | InputRequired | AuthRequired),
            InputRequired | AuthRequired => matches!(next, Working | Canceled | Failed | Rejected),
            Completed | Failed | Canceled | Rejected => false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Role {
    #[serde(rename = "ROLE_USER")]
    User,
    #[serde(rename = "ROLE_AGENT")]
    Agent,
}

/// Exactly one of `text`, `raw` (base64), `url`, `data` must be set.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Part {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filename: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub media_type: Option<String>,
}

impl Part {
    pub fn text(s: impl Into<String>) -> Self {
        Part { text: Some(s.into()), ..Default::default() }
    }
    pub fn data(v: Value) -> Self {
        Part { data: Some(v), ..Default::default() }
    }
    pub fn validate(&self) -> Result<(), String> {
        let n = [self.text.is_some(), self.raw.is_some(), self.url.is_some(), self.data.is_some()]
            .iter()
            .filter(|b| **b)
            .count();
        if n == 1 { Ok(()) } else { Err(format!("a Part must carry exactly one of text/raw/url/data, found {n}")) }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Message {
    pub message_id: String,
    pub role: Role,
    pub parts: Vec<Part>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub extensions: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub reference_task_ids: Vec<String>,
}

impl Message {
    pub fn validate(&self) -> Result<(), String> {
        if self.message_id.trim().is_empty() {
            return Err("messageId must not be empty".into());
        }
        if self.parts.is_empty() {
            return Err("a Message needs at least one Part".into());
        }
        self.parts.iter().try_for_each(Part::validate)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Artifact {
    pub artifact_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub parts: Vec<Part>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
}

impl Artifact {
    /// SHA-256 over the canonical JSON of the artifact (what `a2a.acceptance` commits to).
    pub fn hash(&self) -> String {
        sha256_hex(&serde_json::to_vec(self).expect("artifact serializes"))
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TaskStatus {
    pub state: TaskState,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<Message>,
    pub timestamp: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Task {
    pub id: String,
    pub context_id: String,
    pub status: TaskStatus,
    #[serde(default)]
    pub artifacts: Vec<Artifact>,
    #[serde(default)]
    pub history: Vec<Message>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
}

// -- Agent Card -------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentInterface {
    pub url: String,
    /// `JSONRPC`, `GRPC`, `HTTP+JSON`, or `SLAKSHNA-QUIC` for the native binding.
    pub protocol_binding: String,
    pub protocol_version: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tenant: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentCapabilities {
    #[serde(default)]
    pub streaming: bool,
    #[serde(default)]
    pub push_notifications: bool,
    #[serde(default)]
    pub extended_agent_card: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentSkill {
    pub id: String,
    pub name: String,
    pub description: String,
    pub tags: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub examples: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub input_modes: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub output_modes: Vec<String>,
}

/// Detached signature over the card (JWS-style envelope, Ed25519 by the
/// agent key delegated from the owning node).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CardSignature {
    /// Hex Ed25519 public key of the signing agent key.
    pub key_id: String,
    pub signature: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentCard {
    pub name: String,
    pub description: String,
    pub supported_interfaces: Vec<AgentInterface>,
    pub version: String,
    pub capabilities: AgentCapabilities,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub security_schemes: Option<Value>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub security_requirements: Vec<Value>,
    pub default_input_modes: Vec<String>,
    pub default_output_modes: Vec<String>,
    pub skills: Vec<AgentSkill>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub signatures: Vec<CardSignature>,
    /// Slakshna extension: the owning node, so outside clients can check the delegation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub slakshna_node_id: Option<String>,
    /// Slakshna extension: the node's QUIC endpoint id, so federation peers can dial the agent directly.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub slakshna_endpoint_id: Option<String>,
    /// Slakshna extension: price per skill (service announcement), in settlement units.
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub slakshna_prices: std::collections::BTreeMap<String, u64>,
}

const CARD_DOMAIN: &[u8] = b"slakshna/agent-card/v1\0";

impl AgentCard {
    /// Hash of the card with `signatures` removed (what is signed and what
    /// the `a2a.card` record commits to).
    pub fn content_hash(&self) -> String {
        let mut c = self.clone();
        c.signatures.clear();
        sha256_hex(&serde_json::to_vec(&c).expect("card serializes"))
    }

    pub fn sign(&mut self, agent_key: &crate::identity::Keypair) {
        let mut m = CARD_DOMAIN.to_vec();
        m.extend_from_slice(self.content_hash().as_bytes());
        self.signatures = vec![CardSignature {
            key_id: agent_key.public_key_hex(),
            signature: agent_key.sign_hex(&m),
        }];
    }

    /// Verify at least one signature and return the verified key id.
    pub fn verified_key(&self) -> Option<String> {
        let mut m = CARD_DOMAIN.to_vec();
        m.extend_from_slice(self.content_hash().as_bytes());
        self.signatures
            .iter()
            .find(|s| crate::identity::verify_hex(&s.key_id, &m, &s.signature))
            .map(|s| s.key_id.clone())
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.name.trim().is_empty() || self.version.trim().is_empty() {
            return Err("name and version are required".into());
        }
        if self.supported_interfaces.is_empty() {
            return Err("at least one supportedInterface is required".into());
        }
        if self.skills.iter().any(|s| s.id.is_empty()) {
            return Err("every skill needs an id".into());
        }
        Ok(())
    }
}

pub fn sha256_hex(b: &[u8]) -> String {
    hex::encode(Sha256::digest(b))
}

pub fn now_rfc3339() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
}

// ---------------------------------------------------------------------------
// Provenance sink
// ---------------------------------------------------------------------------

/// Where lifecycle records go. The node implements this by signing the record
/// and appending it to its provenance log; tests use a collecting sink.
pub trait ProvenanceSink: Send + Sync {
    fn emit(&self, kind: &str, body: Value);
}

#[allow(dead_code)] // library surface: used by tests and integrators, or pending the settlement-chain decision
pub struct NullSink;
impl ProvenanceSink for NullSink {
    fn emit(&self, _: &str, _: Value) {}
}

// ---------------------------------------------------------------------------
// Task store
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum TaskError {
    #[error("task {0} not found")]
    NotFound(String),
    #[error("invalid transition {0:?} -> {1:?}")]
    BadTransition(TaskState, TaskState),
    #[error("invalid message: {0}")]
    InvalidMessage(String),
    #[error("store io: {0}")]
    Io(String),
}

#[derive(Debug, Clone)]
pub enum TaskEvent {
    Status { task_id: String, status: TaskStatus, is_final: bool },
    Artifact { task_id: String, artifact: Artifact },
}

#[derive(Serialize, Deserialize, Default)]
struct Persisted {
    tasks: HashMap<String, Task>,
    /// (sender, messageId) -> task id, for idempotent submission.
    idempotency: HashMap<String, String>,
}

/// Durable task store. All state changes are written to disk (atomically via
/// rename) before they are acknowledged, so a provider restart loses nothing.
pub struct TaskStore {
    path: Option<PathBuf>,
    inner: Mutex<Persisted>,
    events: broadcast::Sender<TaskEvent>,
    sink: Arc<dyn ProvenanceSink>,
}

impl TaskStore {
    /// `path = None` keeps the store in memory only.
    pub fn open(path: Option<PathBuf>, sink: Arc<dyn ProvenanceSink>) -> Result<Self, TaskError> {
        let inner = match &path {
            Some(p) if p.exists() => {
                let b = std::fs::read(p).map_err(|e| TaskError::Io(e.to_string()))?;
                serde_json::from_slice(&b).map_err(|e| TaskError::Io(format!("corrupt task store: {e}")))?
            }
            _ => Persisted::default(),
        };
        let (events, _) = broadcast::channel(256);
        Ok(TaskStore { path, inner: Mutex::new(inner), events, sink })
    }

    fn persist(&self, g: &Persisted) -> Result<(), TaskError> {
        let Some(p) = &self.path else { return Ok(()) };
        if let Some(dir) = p.parent() {
            std::fs::create_dir_all(dir).map_err(|e| TaskError::Io(e.to_string()))?;
        }
        let tmp = p.with_extension("tmp");
        std::fs::write(&tmp, serde_json::to_vec(g).expect("store serializes")).map_err(|e| TaskError::Io(e.to_string()))?;
        std::fs::rename(&tmp, p).map_err(|e| TaskError::Io(e.to_string()))
    }

    pub fn subscribe(&self) -> broadcast::Receiver<TaskEvent> {
        self.events.subscribe()
    }

    fn record_transition(&self, task: &Task, contract_hash: &str) {
        self.sink.emit(
            crate::agents::records::KIND_A2A_TASK,
            serde_json::json!({
                "task_id": task.id,
                "contract_hash": contract_hash,
                "state": task.status.state,
            }),
        );
    }

    /// Create a task from an incoming message. Idempotent: the same
    /// (sender, messageId) returns the existing task without side effects.
    /// Returns (task, created).
    pub fn submit(&self, sender: &str, mut msg: Message, skill: Option<&str>) -> Result<(Task, bool), TaskError> {
        msg.validate().map_err(TaskError::InvalidMessage)?;
        let mut g = self.inner.lock().unwrap();
        let key = format!("{sender}\0{}", msg.message_id);
        if let Some(id) = g.idempotency.get(&key) {
            return Ok((g.tasks[id].clone(), false));
        }
        // A message addressed to an existing task continues it rather than creating one.
        let id = hex::encode(&Sha256::digest(key.as_bytes())[..16]);
        let context_id = msg.context_id.clone().unwrap_or_else(|| id.clone());
        msg.task_id = Some(id.clone());
        msg.context_id = Some(context_id.clone());
        let contract_hash = sha256_hex(&serde_json::to_vec(&(sender, skill, &msg)).unwrap());
        let task = Task {
            id: id.clone(),
            context_id,
            status: TaskStatus { state: TaskState::Submitted, message: None, timestamp: now_rfc3339() },
            artifacts: vec![],
            history: vec![msg],
            metadata: Some(serde_json::json!({
                "sender": sender,
                "skill": skill,
                "contractHash": contract_hash,
            })),
        };
        g.tasks.insert(id.clone(), task.clone());
        g.idempotency.insert(key, id);
        self.persist(&g)?;
        drop(g);
        self.record_transition(&task, &contract_hash);
        let _ = self.events.send(TaskEvent::Status { task_id: task.id.clone(), status: task.status.clone(), is_final: false });
        Ok((task, true))
    }

    pub fn get(&self, id: &str) -> Result<Task, TaskError> {
        self.inner.lock().unwrap().tasks.get(id).cloned().ok_or_else(|| TaskError::NotFound(id.into()))
    }

    pub fn list(&self) -> Vec<Task> {
        let mut v: Vec<Task> = self.inner.lock().unwrap().tasks.values().cloned().collect();
        v.sort_by(|a, b| a.status.timestamp.cmp(&b.status.timestamp).then(a.id.cmp(&b.id)));
        v
    }

    pub fn transition(&self, id: &str, next: TaskState, message: Option<Message>) -> Result<Task, TaskError> {
        let mut g = self.inner.lock().unwrap();
        let t = g.tasks.get_mut(id).ok_or_else(|| TaskError::NotFound(id.into()))?;
        if !t.status.state.can_transition_to(next) {
            return Err(TaskError::BadTransition(t.status.state, next));
        }
        t.status = TaskStatus { state: next, message: message.clone(), timestamp: now_rfc3339() };
        if let Some(m) = message {
            t.history.push(m);
        }
        let task = t.clone();
        self.persist(&g)?;
        drop(g);
        let ch = task.metadata.as_ref().and_then(|m| m["contractHash"].as_str()).unwrap_or("").to_string();
        self.record_transition(&task, &ch);
        let _ = self.events.send(TaskEvent::Status { task_id: task.id.clone(), status: task.status.clone(), is_final: next.is_terminal() });
        Ok(task)
    }

    pub fn add_artifact(&self, id: &str, artifact: Artifact) -> Result<(), TaskError> {
        let mut g = self.inner.lock().unwrap();
        let t = g.tasks.get_mut(id).ok_or_else(|| TaskError::NotFound(id.into()))?;
        if t.status.state.is_terminal() {
            return Err(TaskError::BadTransition(t.status.state, t.status.state));
        }
        artifact.parts.iter().try_for_each(Part::validate).map_err(TaskError::InvalidMessage)?;
        t.artifacts.push(artifact.clone());
        self.persist(&g)?;
        drop(g);
        let _ = self.events.send(TaskEvent::Artifact { task_id: id.into(), artifact });
        Ok(())
    }

    pub fn cancel(&self, id: &str) -> Result<Task, TaskError> {
        self.transition(id, TaskState::Canceled, None)
    }

    /// Tasks that were in flight when the process stopped.
    pub fn unfinished(&self) -> Vec<Task> {
        self.list().into_iter().filter(|t| matches!(t.status.state, TaskState::Submitted | TaskState::Working)).collect()
    }
}

// ---------------------------------------------------------------------------
// Mailboxes: store-and-forward for offline agents
// ---------------------------------------------------------------------------

/// A message waiting for an agent that is currently unreachable.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Parcel {
    pub to: String,
    pub skill: Option<String>,
    pub message: Message,
    pub enqueued_at: String,
    pub attempts: u32,
    /// Unix seconds before which no retry is made (exponential backoff).
    #[serde(default)]
    pub next_attempt: i64,
}

/// Durable outbound queue per destination. The sender keeps a parcel until the
/// recipient acknowledges it, so delivery survives restarts on either side.
pub struct Mailbox {
    path: Option<PathBuf>,
    q: Mutex<Vec<Parcel>>,
    sent: Mutex<HashMap<String, String>>,
}

impl Mailbox {
    pub fn open(path: Option<PathBuf>) -> Result<Self, TaskError> {
        let q = match &path {
            Some(p) if p.exists() => serde_json::from_slice(&std::fs::read(p).map_err(|e| TaskError::Io(e.to_string()))?)
                .map_err(|e| TaskError::Io(format!("corrupt mailbox: {e}")))?,
            _ => vec![],
        };
        let sent = match &path {
            Some(p) => std::fs::read(p.with_extension("sent.json")).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default(),
            None => HashMap::new(),
        };
        Ok(Mailbox { path, q: Mutex::new(q), sent: Mutex::new(sent) })
    }

    fn persist(&self, q: &[Parcel]) -> Result<(), TaskError> {
        let Some(p) = &self.path else { return Ok(()) };
        if let Some(d) = p.parent() {
            std::fs::create_dir_all(d).map_err(|e| TaskError::Io(e.to_string()))?;
        }
        let tmp = p.with_extension("tmp");
        std::fs::write(&tmp, serde_json::to_vec(q).unwrap()).map_err(|e| TaskError::Io(e.to_string()))?;
        std::fs::rename(&tmp, p).map_err(|e| TaskError::Io(e.to_string()))
    }

    /// Enqueue; a repeat of the same (to, messageId) is ignored.
    pub fn enqueue(&self, to: &str, skill: Option<&str>, message: Message) -> Result<bool, TaskError> {
        message.validate().map_err(TaskError::InvalidMessage)?;
        let mut q = self.q.lock().unwrap();
        if q.iter().any(|p| p.to == to && p.message.message_id == message.message_id) {
            return Ok(false);
        }
        q.push(Parcel { to: to.into(), skill: skill.map(String::from), message, enqueued_at: now_rfc3339(), attempts: 0, next_attempt: 0 });
        self.persist(&q)?;
        Ok(true)
    }

    pub fn pending(&self) -> Vec<Parcel> {
        self.q.lock().unwrap().clone()
    }

    #[allow(dead_code)] // library surface: used by tests and integrators, or pending the settlement-chain decision
    pub fn pending_for(&self, to: &str) -> Vec<Parcel> {
        self.q.lock().unwrap().iter().filter(|p| p.to == to).cloned().collect()
    }

    pub fn note_attempt(&self, to: &str, message_id: &str) {
        let mut q = self.q.lock().unwrap();
        if let Some(p) = q.iter_mut().find(|p| p.to == to && p.message.message_id == message_id) {
            p.attempts += 1;
            let backoff = (2i64.saturating_pow(p.attempts.min(10))).min(300);
            p.next_attempt = chrono::Utc::now().timestamp() + backoff;
        }
        let _ = self.persist(&q);
    }

    /// Parcels whose backoff has elapsed.
    pub fn due(&self, now: i64) -> Vec<Parcel> {
        self.q.lock().unwrap().iter().filter(|p| p.next_attempt <= now).cloned().collect()
    }

    /// Remember which remote task a delivered message became.
    pub fn record_sent(&self, to: &str, message_id: &str, task_id: &str) {
        let mut g = self.sent.lock().unwrap();
        g.insert(format!("{to}\0{message_id}"), task_id.to_string());
        if let Some(p) = &self.path {
            let _ = std::fs::write(p.with_extension("sent.json"), serde_json::to_vec(&*g).unwrap());
        }
    }

    pub fn sent_task(&self, to: &str, message_id: &str) -> Option<String> {
        self.sent.lock().unwrap().get(&format!("{to}\0{message_id}")).cloned()
    }

    /// Remove an acknowledged parcel.
    pub fn ack(&self, to: &str, message_id: &str) -> Result<(), TaskError> {
        let mut q = self.q.lock().unwrap();
        q.retain(|p| !(p.to == to && p.message.message_id == message_id));
        self.persist(&q)
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;

    #[derive(Default)]
    pub struct CollectSink(pub Mutex<Vec<(String, Value)>>);
    impl ProvenanceSink for CollectSink {
        fn emit(&self, kind: &str, body: Value) {
            self.0.lock().unwrap().push((kind.into(), body));
        }
    }

    pub fn user_msg(id: &str, text: &str) -> Message {
        Message { message_id: id.into(), role: Role::User, parts: vec![Part::text(text)], context_id: None, task_id: None, metadata: None, extensions: vec![], reference_task_ids: vec![] }
    }

    fn tmpfile(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("slk-a2a-{}-{}", rand::random::<u64>(), name))
    }

    #[test]
    fn wire_names_match_spec() {
        let m = user_msg("m1", "hi");
        let v = serde_json::to_value(&m).unwrap();
        assert_eq!(v["messageId"], "m1");
        assert_eq!(v["role"], "ROLE_USER");
        assert_eq!(serde_json::to_value(TaskState::InputRequired).unwrap(), "TASK_STATE_INPUT_REQUIRED");
        let t = Task { id: "t".into(), context_id: "c".into(), status: TaskStatus { state: TaskState::Working, message: None, timestamp: now_rfc3339() }, artifacts: vec![], history: vec![], metadata: None };
        let v = serde_json::to_value(&t).unwrap();
        assert!(v.get("contextId").is_some());
        assert_eq!(v["status"]["state"], "TASK_STATE_WORKING");
    }

    #[test]
    fn part_requires_exactly_one_content_field() {
        assert!(Part::text("x").validate().is_ok());
        assert!(Part::default().validate().is_err());
        let both = Part { text: Some("a".into()), url: Some("b".into()), ..Default::default() };
        assert!(both.validate().is_err());
    }

    #[test]
    fn lifecycle_rules() {
        use TaskState::*;
        assert!(Submitted.can_transition_to(Working));
        assert!(Working.can_transition_to(Completed));
        assert!(InputRequired.can_transition_to(Working));
        assert!(!Completed.can_transition_to(Working));
        assert!(!Submitted.can_transition_to(Completed));
        assert!(Canceled.is_terminal() && !Working.is_terminal());
    }

    #[test]
    fn submit_is_idempotent_and_records_every_transition() {
        let sink = Arc::new(CollectSink::default());
        let s = TaskStore::open(None, sink.clone()).unwrap();
        let (t1, c1) = s.submit("peerA", user_msg("m1", "do it"), Some("chat")).unwrap();
        let (t2, c2) = s.submit("peerA", user_msg("m1", "do it"), Some("chat")).unwrap();
        assert!(c1 && !c2);
        assert_eq!(t1.id, t2.id);
        // another sender reusing the id gets its own task
        let (t3, _) = s.submit("peerB", user_msg("m1", "do it"), None).unwrap();
        assert_ne!(t1.id, t3.id);
        s.transition(&t1.id, TaskState::Working, None).unwrap();
        s.transition(&t1.id, TaskState::Completed, None).unwrap();
        assert_eq!(s.transition(&t1.id, TaskState::Working, None), Err(TaskError::BadTransition(TaskState::Completed, TaskState::Working)));
        let log = sink.0.lock().unwrap();
        let states: Vec<_> = log.iter().filter(|(_, b)| b["task_id"] == t1.id.as_str()).map(|(_, b)| b["state"].as_str().unwrap().to_string()).collect();
        assert_eq!(states, ["TASK_STATE_SUBMITTED", "TASK_STATE_WORKING", "TASK_STATE_COMPLETED"]);
        // the emitted body is a valid a2a.task record body
        assert!(crate::agents::records::validate_body(crate::agents::records::KIND_A2A_TASK, &log[0].1).is_ok());
    }

    #[test]
    fn tasks_survive_restart() {
        let p = tmpfile("tasks.json");
        let id = {
            let s = TaskStore::open(Some(p.clone()), Arc::new(NullSink)).unwrap();
            let (t, _) = s.submit("p", user_msg("m", "long job"), None).unwrap();
            s.transition(&t.id, TaskState::Working, None).unwrap();
            s.add_artifact(&t.id, Artifact { artifact_id: "a1".into(), name: None, description: None, parts: vec![Part::text("partial")], metadata: None }).unwrap();
            t.id
        };
        let s = TaskStore::open(Some(p), Arc::new(NullSink)).unwrap();
        let t = s.get(&id).unwrap();
        assert_eq!(t.status.state, TaskState::Working);
        assert_eq!(t.artifacts.len(), 1);
        assert_eq!(s.unfinished().len(), 1);
        // idempotency also survives
        assert!(!s.submit("p", user_msg("m", "long job"), None).unwrap().1);
        s.transition(&id, TaskState::Completed, None).unwrap();
        assert!(s.unfinished().is_empty());
    }

    #[tokio::test]
    async fn streaming_events_arrive_in_order_with_final_flag() {
        let s = TaskStore::open(None, Arc::new(NullSink)).unwrap();
        let mut rx = s.subscribe();
        let (t, _) = s.submit("p", user_msg("m", "x"), None).unwrap();
        s.transition(&t.id, TaskState::Working, None).unwrap();
        s.add_artifact(&t.id, Artifact { artifact_id: "a".into(), name: None, description: None, parts: vec![Part::text("r")], metadata: None }).unwrap();
        s.transition(&t.id, TaskState::Completed, None).unwrap();
        let mut seen = vec![];
        for _ in 0..4 {
            seen.push(rx.recv().await.unwrap());
        }
        assert!(matches!(&seen[0], TaskEvent::Status { is_final: false, .. }));
        assert!(matches!(&seen[2], TaskEvent::Artifact { .. }));
        assert!(matches!(&seen[3], TaskEvent::Status { is_final: true, status, .. } if status.state == TaskState::Completed));
    }

    #[test]
    fn cannot_add_artifact_to_finished_task_or_invalid_message() {
        let s = TaskStore::open(None, Arc::new(NullSink)).unwrap();
        let (t, _) = s.submit("p", user_msg("m", "x"), None).unwrap();
        s.cancel(&t.id).unwrap();
        let a = Artifact { artifact_id: "a".into(), name: None, description: None, parts: vec![Part::text("r")], metadata: None };
        assert!(s.add_artifact(&t.id, a).is_err());
        let mut bad = user_msg("m2", "x");
        bad.parts.clear();
        assert!(matches!(s.submit("p", bad, None), Err(TaskError::InvalidMessage(_))));
    }

    #[test]
    fn agent_card_signature_roundtrip_and_tamper() {
        let k = crate::identity::Keypair::generate();
        let mut c = AgentCard {
            name: "evaluator".into(), description: "d".into(), version: "1.0.0".into(),
            supported_interfaces: vec![AgentInterface { url: "https://x/a2a".into(), protocol_binding: "JSONRPC".into(), protocol_version: "1.0".into(), tenant: None }],
            capabilities: AgentCapabilities { streaming: true, ..Default::default() },
            security_schemes: None, security_requirements: vec![],
            default_input_modes: vec!["text/plain".into()], default_output_modes: vec!["application/json".into()],
            skills: vec![AgentSkill { id: "chat".into(), name: "Chat".into(), description: "d".into(), tags: vec![], examples: vec![], input_modes: vec![], output_modes: vec![] }],
            signatures: vec![], slakshna_node_id: Some("n".into()), slakshna_endpoint_id: None, slakshna_prices: Default::default(),
        };
        assert!(c.validate().is_ok());
        c.sign(&k);
        assert_eq!(c.verified_key(), Some(k.public_key_hex()));
        let h = c.content_hash();
        c.signatures.clear();
        assert_eq!(c.content_hash(), h, "hash excludes signatures");
        c.sign(&k);
        c.description = "changed".into();
        assert_eq!(c.verified_key(), None);
        let v = serde_json::to_value(&c).unwrap();
        assert!(v.get("supportedInterfaces").is_some() && v["supportedInterfaces"][0].get("protocolBinding").is_some());
    }

    #[test]
    fn mailbox_is_durable_deduped_and_acked() {
        let p = tmpfile("mb.json");
        let mb = Mailbox::open(Some(p.clone())).unwrap();
        assert!(mb.enqueue("peer", Some("chat"), user_msg("m1", "a")).unwrap());
        assert!(!mb.enqueue("peer", Some("chat"), user_msg("m1", "a")).unwrap());
        assert!(mb.enqueue("other", None, user_msg("m1", "a")).unwrap());
        mb.note_attempt("peer", "m1");
        drop(mb);
        let mb = Mailbox::open(Some(p)).unwrap();
        assert_eq!(mb.pending_for("peer").len(), 1);
        assert_eq!(mb.pending_for("peer")[0].attempts, 1);
        mb.ack("peer", "m1").unwrap();
        assert!(mb.pending_for("peer").is_empty());
        assert_eq!(mb.pending().len(), 1);
    }
}

// ---------------------------------------------------------------------------
// Delegation chains (M4)
// ---------------------------------------------------------------------------

/// One verified hop of a delegation chain: who handed which task to whom, under what scope.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Hop {
    /// Node that held `parent_task` and delegated it.
    pub delegator: String,
    pub parent_task: String,
    pub sub_task: String,
    /// Node that holds `sub_task`, if its task record has been seen.
    pub delegate: Option<String>,
    pub scope: std::collections::BTreeSet<String>,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ChainError {
    #[error("unknown root task {0}")]
    UnknownRoot(String),
    #[error("{author} delegated task {parent} but does not hold it")]
    Unauthorized { author: String, parent: String },
    #[error("sub-task {sub} was granted scope {extra:?} that its parent did not have")]
    ScopeEscalation { sub: String, extra: Vec<String> },
    #[error("delegation cycle at task {0}")]
    Cycle(String),
    #[error("delegation chain deeper than {0}")]
    TooDeep(usize),
}

pub const MAX_CHAIN_DEPTH: usize = 16;

/// Reconstruct and check the delegation tree under `root_task` from signed records.
/// * the author of each `a2a.delegation` must be the node holding the parent task (the
///   author of that task's `a2a.task` records);
/// * a sub-task's scope must be a subset of its parent's (the root's is `root_scope`);
/// * no cycles, bounded depth.
/// Hops are returned breadth-first. Because every record is signed, "who asked whom, under
/// what scope" is attributable end to end.
pub fn verify_delegation_chain<'a>(
    records: impl IntoIterator<Item = &'a crate::history::UpdateRecord>,
    root_task: &str,
    root_scope: &std::collections::BTreeSet<String>,
) -> Result<Vec<Hop>, ChainError> {
    use crate::agents::records::{A2aDelegation, A2aTask, KIND_A2A_DELEGATION, KIND_A2A_TASK};
    use crate::history::RecordKind;
    use std::collections::{BTreeMap, BTreeSet, VecDeque};

    let mut holder: BTreeMap<String, String> = BTreeMap::new();
    let mut delegations: Vec<(String, A2aDelegation)> = vec![];
    for r in records {
        let RecordKind::Provenance { kind, body } = &r.kind else { continue };
        if r.verify().is_err() {
            continue;
        }
        if kind == KIND_A2A_TASK {
            if let Ok(t) = serde_json::from_value::<A2aTask>(body.clone()) {
                holder.entry(t.task_id).or_insert_with(|| r.node_id.clone());
            }
        } else if kind == KIND_A2A_DELEGATION {
            if let Ok(d) = serde_json::from_value::<A2aDelegation>(body.clone()) {
                delegations.push((r.node_id.clone(), d));
            }
        }
    }
    if !holder.contains_key(root_task) {
        return Err(ChainError::UnknownRoot(root_task.into()));
    }
    let mut scopes: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    scopes.insert(root_task.to_string(), root_scope.clone());
    let mut depth: BTreeMap<String, usize> = BTreeMap::new();
    depth.insert(root_task.to_string(), 0);
    let mut hops = vec![];
    let mut queue = VecDeque::from([root_task.to_string()]);
    let mut visited: BTreeSet<String> = BTreeSet::from([root_task.to_string()]);
    while let Some(parent) = queue.pop_front() {
        let mut children: Vec<&(String, A2aDelegation)> = delegations.iter().filter(|(_, d)| d.parent_task == parent).collect();
        children.sort_by(|a, b| a.1.sub_task.cmp(&b.1.sub_task));
        for (author, d) in children {
            if holder.get(&parent) != Some(author) {
                return Err(ChainError::Unauthorized { author: author.clone(), parent: parent.clone() });
            }
            let parent_scope = &scopes[&parent];
            let scope: BTreeSet<String> = d.scope.iter().cloned().collect();
            let extra: Vec<String> = scope.difference(parent_scope).cloned().collect();
            if !extra.is_empty() {
                return Err(ChainError::ScopeEscalation { sub: d.sub_task.clone(), extra });
            }
            if !visited.insert(d.sub_task.clone()) {
                return Err(ChainError::Cycle(d.sub_task.clone()));
            }
            let dpt = depth[&parent] + 1;
            if dpt > MAX_CHAIN_DEPTH {
                return Err(ChainError::TooDeep(MAX_CHAIN_DEPTH));
            }
            depth.insert(d.sub_task.clone(), dpt);
            scopes.insert(d.sub_task.clone(), scope.clone());
            hops.push(Hop { delegator: author.clone(), parent_task: parent.clone(), sub_task: d.sub_task.clone(), delegate: holder.get(&d.sub_task).cloned(), scope });
            queue.push_back(d.sub_task.clone());
        }
    }
    Ok(hops)
}

#[cfg(test)]
mod chain_tests {
    use super::*;
    use crate::agents::graph::make_record;
    use crate::agents::records::{KIND_A2A_DELEGATION, KIND_A2A_TASK};
    use crate::history::UpdateRecord;
    use crate::identity::Keypair;
    use std::collections::BTreeSet;

    struct N { kp: Keypair, id: String }
    fn n() -> N { let kp = Keypair::generate(); let id = kp.node_id().to_string(); N { kp, id } }
    fn task(who: &N, id: &str) -> UpdateRecord {
        let mut r = make_record(&who.id, &format!("{:064x}", id.len() as u64 + rand::random::<u32>() as u64), KIND_A2A_TASK, &serde_json::json!({"task_id": id, "contract_hash": "a".repeat(64), "state": "TASK_STATE_WORKING"}));
        r.sign(&who.kp);
        r
    }
    fn deleg(who: &N, parent: &str, sub: &str, scope: &[&str]) -> UpdateRecord {
        let mut r = make_record(&who.id, &format!("{:064x}", rand::random::<u64>()), KIND_A2A_DELEGATION, &serde_json::json!({"parent_task": parent, "sub_task": sub, "scope": scope}));
        r.sign(&who.kp);
        r
    }
    fn sc(v: &[&str]) -> BTreeSet<String> { v.iter().map(|s| s.to_string()).collect() }

    #[test]
    fn three_level_chain_is_attributed_hop_by_hop_with_narrowing_scope() {
        let (a, b, c) = (n(), n(), n());
        let recs = vec![task(&a, "t1"), task(&b, "t2"), task(&c, "t3"), deleg(&a, "t1", "t2", &["read", "eval"]), deleg(&b, "t2", "t3", &["eval"])];
        let hops = verify_delegation_chain(&recs, "t1", &sc(&["read", "eval", "pay"])).unwrap();
        assert_eq!(hops.len(), 2);
        assert_eq!((hops[0].delegator.as_str(), hops[0].delegate.as_deref(), hops[0].scope.len()), (a.id.as_str(), Some(b.id.as_str()), 2));
        assert_eq!((hops[1].delegator.as_str(), hops[1].delegate.as_deref(), hops[1].scope.len()), (b.id.as_str(), Some(c.id.as_str()), 1));
        // order of records does not matter
        let mut rev = recs.clone();
        rev.reverse();
        assert_eq!(verify_delegation_chain(&rev, "t1", &sc(&["read", "eval", "pay"])).unwrap(), hops);
    }

    #[test]
    fn escalation_impersonation_cycles_and_unknown_roots_are_rejected() {
        let (a, b, c) = (n(), n(), n());
        let base = vec![task(&a, "t1"), task(&b, "t2"), task(&c, "t3")];
        // b grants c more than b itself received
        let mut r = base.clone();
        r.push(deleg(&a, "t1", "t2", &["eval"]));
        r.push(deleg(&b, "t2", "t3", &["eval", "pay"]));
        assert!(matches!(verify_delegation_chain(&r, "t1", &sc(&["eval", "pay"])), Err(ChainError::ScopeEscalation { extra, .. }) if extra == ["pay"]));
        // c delegates t1 although only a holds it
        let mut r = base.clone();
        r.push(deleg(&c, "t1", "t2", &["eval"]));
        assert!(matches!(verify_delegation_chain(&r, "t1", &sc(&["eval"])), Err(ChainError::Unauthorized { .. })));
        // t1 -> t2 -> t1
        let mut r = base.clone();
        r.push(deleg(&a, "t1", "t2", &["eval"]));
        r.push(deleg(&b, "t2", "t1", &["eval"]));
        assert!(matches!(verify_delegation_chain(&r, "t1", &sc(&["eval"])), Err(ChainError::Cycle(_))));
        assert_eq!(verify_delegation_chain(&base, "nope", &sc(&[])), Err(ChainError::UnknownRoot("nope".into())));
        // forged (unsigned) delegation is ignored rather than trusted
        let mut forged = deleg(&c, "t1", "t3", &["eval"]);
        forged.signature = "00".repeat(64);
        let mut r = base.clone();
        r.push(forged);
        assert_eq!(verify_delegation_chain(&r, "t1", &sc(&["eval"])).unwrap().len(), 0);
    }

    #[test]
    fn sub_task_whose_holder_is_unknown_yet_is_still_a_hop() {
        let (a, b) = (n(), n());
        let r = vec![task(&a, "t1"), deleg(&a, "t1", "t2", &["eval"])];
        let hops = verify_delegation_chain(&r, "t1", &sc(&["eval"])).unwrap();
        assert_eq!(hops[0].delegate, None);
        let _ = b;
    }
}
