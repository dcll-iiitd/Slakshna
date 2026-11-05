//! The node's agent runtime: ties provenance authoring, the A2A task store,
//! skills, policy and the kill switch together (A1, M1, M2, M6).
//!
//! * Every lifecycle event is authored as a signed record in this node's log
//!   and gossiped, in order, by one background task.
//! * Incoming messages become durable tasks; an executor runs them under a
//!   per-task budget, concurrency limit, policy check and timeout. Tasks that
//!   were in flight when the process stopped are resumed on start.
//! * The node delegates a scoped, expiring key to its agent and publishes a
//!   signed Agent Card.

use crate::agents::a2a::*;
use crate::agents::budget::{Budget, BudgetTracker};
use crate::agents::delegation::delegation_record;
use crate::agents::records::*;
use crate::agents::safety::{KillSwitch, OrgPolicy, PolicyEngine, PolicyRequest, Verdict};
use crate::config::AgentsConfig;
use crate::history::{UpdateHistory, UpdateRecord};
use crate::identity::Keypair;
use async_trait::async_trait;
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::{mpsc, RwLock, Semaphore};

/// How records authored here reach the rest of the federation.
#[async_trait]
pub trait Broadcaster: Send + Sync {
    /// Federation-wide topic.
    async fn broadcast(&self, record: &UpdateRecord);
    /// A research program's own topic; falls back to the federation topic.
    async fn broadcast_program(&self, _program: &str, record: &UpdateRecord) {
        self.broadcast(record).await
    }
    async fn join_program(&self, _program: &str) {}
    async fn broadcast_halt(&self, _msg: crate::agents::safety::HaltMsg) {}
}

#[async_trait]
impl Broadcaster for RwLock<crate::network::mesh::MeshNetwork> {
    async fn broadcast(&self, record: &UpdateRecord) {
        use crate::network::Network;
        let _ = self.read().await.broadcast_update(record).await;
    }

    async fn broadcast_program(&self, program: &str, record: &UpdateRecord) {
        let topics = self.read().await.topics();
        match topics {
            Some(t) if t.broadcast(program, record).await.is_ok() => {}
            _ => self.broadcast(record).await,
        }
    }

    async fn join_program(&self, program: &str) {
        if let Some(t) = self.read().await.topics() {
            if let Err(e) = t.join(program).await {
                tracing::warn!("could not join program topic: {e}");
            }
        }
    }

    async fn broadcast_halt(&self, msg: crate::agents::safety::HaltMsg) {
        self.read().await.broadcast_halt(msg).await;
    }
}

/// Collects nothing; for nodes/tests that only need local provenance.
pub struct NoBroadcast;
#[async_trait]
impl Broadcaster for NoBroadcast {
    async fn broadcast(&self, _: &UpdateRecord) {}
}

#[derive(Debug, thiserror::Error)]
pub enum SkillError {
    #[error("{0}")]
    Failed(String),
    #[error("invalid input: {0}")]
    BadInput(String),
}

pub struct SkillCtx {
    pub budget: Arc<BudgetTracker>,
    #[allow(dead_code)] // library surface: used by tests and integrators, or pending the settlement-chain decision
    pub node_id: String,
    pub requester: String,
    pub sink: Arc<dyn ProvenanceSink>,
}

#[async_trait]
pub trait Skill: Send + Sync {
    fn descriptor(&self) -> AgentSkill;

    /// What kind of export this skill's results are, and the data class they carry,
    /// for the node's egress policy. Defaults to a general artifact.
    fn export(&self) -> (crate::agents::safety::ExportKind, &'static str) {
        (crate::agents::safety::ExportKind::Artifact, "general")
    }
    async fn run(&self, ctx: &SkillCtx, task: &Task, input: &Message) -> Result<Vec<Artifact>, SkillError>;
}

/// Sends records over a channel so authoring order equals emission order.
struct ChannelSink(mpsc::UnboundedSender<(String, Value)>);
impl ProvenanceSink for ChannelSink {
    fn emit(&self, kind: &str, body: Value) {
        let _ = self.0.send((kind.to_string(), body));
    }
}

pub struct AgentService {
    pub node_id: String,
    keypair: Keypair,
    agent_key: Keypair,
    pub history: Arc<RwLock<UpdateHistory>>,
    broadcaster: Arc<dyn Broadcaster>,
    pub tasks: Arc<TaskStore>,
    pub mailbox: Arc<Mailbox>,
    skills: Mutex<HashMap<String, Arc<dyn Skill>>>,
    pub policy: Arc<PolicyEngine>,
    pub kill: Arc<KillSwitch>,
    pub cfg: AgentsConfig,
    sink: Arc<ChannelSink>,
    permits: Arc<Semaphore>,
    running: Mutex<HashMap<String, tokio::task::AbortHandle>>,
    /// policy approval id -> task id (tasks parked in INPUT_REQUIRED)
    awaiting: Mutex<HashMap<u64, String>>,
    egress: crate::agents::safety::EgressPolicy,
    verifier: Arc<crate::agents::safety::TokenVerifier>,
    public_url: Mutex<Option<String>>,
    endpoint_id: Mutex<Option<String>>,
    card_dir: PathBuf,
    directory: Mutex<crate::agents::discovery::Directory>,
    fetched_cards: Mutex<HashMap<(String, String), AgentCard>>,
    joined_programs: Mutex<HashSet<String>>,
    delegator: std::sync::OnceLock<Arc<crate::agents::delegate::Delegator>>,
    research: std::sync::OnceLock<Arc<crate::agents::research::ResearchNode>>,
}

fn now() -> i64 {
    chrono::Utc::now().timestamp()
}

impl AgentService {
    pub async fn new(
        cfg: AgentsConfig,
        keypair: Keypair,
        history: Arc<RwLock<UpdateHistory>>,
        broadcaster: Arc<dyn Broadcaster>,
        data_dir: &str,
    ) -> Result<Arc<Self>, Box<dyn std::error::Error + Send + Sync>> {
        let dir = PathBuf::from(data_dir).join("agents");
        std::fs::create_dir_all(&dir)?;
        let node_id = keypair.node_id().to_string();
        let (tx, rx) = mpsc::unbounded_channel();
        let sink = Arc::new(ChannelSink(tx));
        let tasks = Arc::new(TaskStore::open(Some(dir.join("tasks.json")), sink.clone())?);
        let mailbox = Arc::new(Mailbox::open(Some(dir.join("mailbox.json")))?);
        let agent_key = load_or_create_agent_key(&dir.join("agent.key"))?;

        let mut p = OrgPolicy::default();
        p.allowed_counterparties = cfg.policy.allowed_counterparties.clone().map(|v| v.into_iter().collect());
        p.denied_counterparties = cfg.policy.denied_counterparties.iter().cloned().collect();
        p.allowed_data_classes = cfg.policy.allowed_data_classes.iter().cloned().collect();
        p.max_spend_per_task = cfg.policy.max_spend_per_task;
        p.max_spend_per_day = cfg.policy.max_spend_per_day;
        p.human_approval_actions = cfg.policy.human_approval_actions.iter().cloned().collect();
        let policy = Arc::new(PolicyEngine::new(p, sink.clone()));
        let kill = Arc::new(KillSwitch::new(cfg.halt_keys.clone()));

        let verifier = Arc::new(crate::agents::safety::TokenVerifier::new([node_id.clone()]));
        let svc = Arc::new(AgentService {
            node_id,
            keypair,
            agent_key,
            history,
            broadcaster,
            tasks,
            mailbox,
            skills: Mutex::new(HashMap::new()),
            policy,
            kill,
            permits: Arc::new(Semaphore::new(cfg.max_concurrent_tasks.max(1))),
            running: Mutex::new(HashMap::new()),
            awaiting: Mutex::new(HashMap::new()),
            verifier,
            egress: {
                use crate::agents::safety::{EgressPolicy, ExportKind};
                let mut e = EgressPolicy::default();
                e.allowed = cfg
                    .egress
                    .allow
                    .iter()
                    .filter_map(|k| match k.as_str() {
                        "finding" => Some(ExportKind::Finding),
                        "metric" => Some(ExportKind::Metric),
                        "artifact" => Some(ExportKind::Artifact),
                        _ => None, // raw data can never be enabled
                    })
                    .collect();
                e.max_bytes = cfg.egress.max_bytes;
                e.blocked_classes = cfg.egress.blocked_classes.iter().cloned().collect();
                e
            },
            public_url: Mutex::new(cfg.public_url.clone()),
            endpoint_id: Mutex::new(None),
            card_dir: dir,
            directory: Mutex::new(Default::default()),
            fetched_cards: Mutex::new(HashMap::new()),
            joined_programs: Mutex::new(HashSet::new()),
            delegator: std::sync::OnceLock::new(),
            research: std::sync::OnceLock::new(),
            sink,
            cfg,
        });
        svc.clone().spawn_author_loop(rx);
        Ok(svc)
    }

    pub fn sink(&self) -> Arc<dyn ProvenanceSink> {
        self.sink.clone()
    }

    pub fn emit(&self, kind: &str, body: Value) {
        self.sink.emit(kind, body);
    }

    /// Author, sign, append and broadcast a provenance record now, returning it
    /// (callers that need the record hash, e.g. to reference it later).
    pub async fn author(&self, kind: &str, body: Value) -> Result<UpdateRecord, crate::history::RecordError> {
        let rec = {
            let mut h = self.history.write().await;
            h.author_provenance(&self.keypair, kind, body)?
        };
        self.broadcaster.broadcast(&rec).await;
        Ok(rec)
    }

    /// Like `author`, but gossips on the research program's own topic.
    pub async fn author_in(&self, program: &str, kind: &str, body: Value) -> Result<UpdateRecord, crate::history::RecordError> {
        let rec = {
            let mut h = self.history.write().await;
            h.author_provenance(&self.keypair, kind, body)?
        };
        self.broadcaster.broadcast_program(program, &rec).await;
        Ok(rec)
    }

    pub async fn join_program(&self, program: &str) {
        self.broadcaster.join_program(program).await;
    }

    /// Apply a signed halt/resume locally and tell the federation.
    pub async fn halt(&self, msg: crate::agents::safety::HaltMsg) -> Result<bool, String> {
        let changed = self.kill.apply(&msg)?;
        self.broadcaster.broadcast_halt(msg).await;
        Ok(changed)
    }

    /// Seed of this node's Ed25519 key, used to evaluate the planning VRF.
    pub fn signing_seed(&self) -> [u8; 32] {
        self.keypair.to_bytes()
    }

    pub fn agent_public_key(&self) -> String {
        self.agent_key.public_key_hex()
    }

    fn spawn_author_loop(self: Arc<Self>, mut rx: mpsc::UnboundedReceiver<(String, Value)>) {
        tokio::spawn(async move {
            while let Some((kind, body)) = rx.recv().await {
                let rec = {
                    let mut h = self.history.write().await;
                    h.author_provenance(&self.keypair, &kind, body)
                };
                match rec {
                    Ok(r) => self.broadcaster.broadcast(&r).await,
                    Err(e) => tracing::error!("could not author {kind} record: {e}"),
                }
            }
        });
    }

    pub fn register_skill(&self, skill: Arc<dyn Skill>) {
        let d = skill.descriptor();
        self.skills.lock().unwrap().insert(d.id, skill);
    }

    pub fn skill_ids(&self) -> Vec<String> {
        let mut v: Vec<String> = self.skills.lock().unwrap().keys().cloned().collect();
        v.sort();
        v
    }

    /// This node's research-loop driver (None unless `[agents.research]` is enabled).
    pub fn research(&self) -> Option<Arc<crate::agents::research::ResearchNode>> {
        self.research.get().cloned()
    }

    pub fn set_research(&self, r: Arc<crate::agents::research::ResearchNode>) {
        let _ = self.research.set(r);
    }

    /// The cross-node delegator (None until the network endpoint exists).
    pub fn delegator(&self) -> Option<Arc<crate::agents::delegate::Delegator>> {
        self.delegator.get().cloned()
    }

    pub fn set_delegator(&self, d: Arc<crate::agents::delegate::Delegator>) {
        let _ = self.delegator.set(d);
    }

    /// Record this node's QUIC endpoint id so the card tells peers where to dial.
    pub fn set_endpoint_id(&self, id: &str) {
        *self.endpoint_id.lock().unwrap() = Some(id.to_string());
    }

    pub fn set_public_url(&self, url: &str) {
        *self.public_url.lock().unwrap() = Some(url.to_string());
    }

    // -- identity / card ------------------------------------------------------

    /// Delegate the agent key to scopes needed for cards and tasks, unless a live
    /// delegation with enough remaining life already exists in the log.
    pub async fn ensure_delegation(&self) {
        let pk = self.agent_public_key();
        let live = {
            let h = self.history.read().await;
            let reg = crate::agents::delegation::DelegationRegistry::from_records(h.provenance(KIND_KEY_DELEGATION));
            let t = now() + 7 * 86400;
            reg.check(&self.node_id, &pk, "a2a.card", t).is_ok() && reg.check(&self.node_id, &pk, "a2a.task", t).is_ok()
        };
        if live {
            return;
        }
        let r = delegation_record(&self.node_id, "", &pk, &["a2a.card", "a2a.task"], now() - 5, now() + 30 * 86400, false);
        if let RecordKind::Provenance { body, .. } = r.kind {
            self.emit(KIND_KEY_DELEGATION, body);
        }
    }

    pub fn card(&self) -> AgentCard {
        let skills: Vec<AgentSkill> = {
            let m = self.skills.lock().unwrap();
            let mut v: Vec<_> = m.values().map(|s| s.descriptor()).collect();
            v.sort_by(|a, b| a.id.cmp(&b.id));
            v
        };
        let mut interfaces = vec![AgentInterface {
            url: format!("slakshna://{}", self.node_id),
            protocol_binding: "SLAKSHNA-QUIC".into(),
            protocol_version: "1.0".into(),
            tenant: None,
        }];
        if let Some(u) = self.public_url.lock().unwrap().clone() {
            interfaces.push(AgentInterface { url: format!("{}/a2a", u.trim_end_matches('/')), protocol_binding: "JSONRPC".into(), protocol_version: "1.0".into(), tenant: None });
        }
        let mut c = AgentCard {
            name: self.cfg.card_name.clone(),
            description: if self.cfg.card_description.is_empty() { "Slakshna federation agent".into() } else { self.cfg.card_description.clone() },
            supported_interfaces: interfaces,
            version: "1.0.0".into(),
            capabilities: AgentCapabilities { streaming: true, push_notifications: false, extended_agent_card: false },
            security_schemes: Some(json!({"bearer": {"httpAuthSecurityScheme": {"scheme": "Bearer", "description": "Gateway API key"}}})),
            security_requirements: vec![json!({"bearer": []})],
            default_input_modes: vec!["text/plain".into(), "application/json".into()],
            default_output_modes: vec!["text/plain".into(), "application/json".into()],
            skills,
            signatures: vec![],
            slakshna_node_id: Some(self.node_id.clone()),
            slakshna_endpoint_id: self.endpoint_id.lock().unwrap().clone(),
            slakshna_prices: self.cfg.prices.clone(),
        };
        c.sign(&self.agent_key);
        c
    }

    /// Publish (or re-publish after a change) the card as an `a2a.card` record.
    pub async fn publish_card(&self) {
        let card = self.card();
        let hash = card.content_hash();
        let state = self.card_dir.join("card_state.json");
        let (mut version, last_hash): (u32, String) = std::fs::read(&state)
            .ok()
            .and_then(|b| serde_json::from_slice::<(u32, String)>(&b).ok())
            .unwrap_or((0, String::new()));
        if last_hash == hash {
            return;
        }
        version += 1;
        let _ = std::fs::write(&state, serde_json::to_vec(&(version, &hash)).unwrap());
        self.emit(KIND_A2A_CARD, json!({"agent_public_key": self.agent_public_key(), "card_hash": hash, "version": version, "revoked": false}));
    }

    #[allow(dead_code)] // library surface: used by tests and integrators, or pending the settlement-chain decision
    pub async fn revoke_card(&self) {
        let version = std::fs::read(self.card_dir.join("card_state.json"))
            .ok()
            .and_then(|b| serde_json::from_slice::<(u32, String)>(&b).ok())
            .map(|(v, _)| v + 1)
            .unwrap_or(1);
        self.emit(KIND_A2A_CARD, json!({"agent_public_key": self.agent_public_key(), "card_hash": self.card().content_hash(), "version": version, "revoked": true}));
    }

    // -- tasks ---------------------------------------------------------------

    /// Accept a message from `sender` as a new (or already known) task.
    pub fn send_message(self: &Arc<Self>, sender: &str, msg: Message) -> Result<Task, TaskError> {
        let skill = msg.metadata.as_ref().and_then(|m| m["skill"].as_str()).map(String::from);
        let (task, created) = self.tasks.submit(sender, msg, skill.as_deref())?;
        if created {
            self.clone().spawn_task(task.id.clone());
        }
        Ok(task)
    }

    pub fn cancel_task(&self, id: &str) -> Result<Task, TaskError> {
        let t = self.tasks.cancel(id)?;
        if let Some(h) = self.running.lock().unwrap().remove(id) {
            h.abort();
        }
        Ok(t)
    }

    /// Resume tasks that were in flight when the process last stopped.
    pub fn recover(self: &Arc<Self>) -> usize {
        let pending = self.tasks.unfinished();
        for t in &pending {
            self.clone().spawn_task(t.id.clone());
        }
        pending.len()
    }

    /// A human approved the policy request that parked a task; run it.
    pub fn approve(self: &Arc<Self>, approval_id: u64, approver: &str) -> Result<(), String> {
        let task_id = self.awaiting.lock().unwrap().remove(&approval_id).ok_or("no task awaits that approval")?;
        self.policy.approve(approval_id, approver, now())?;
        let _ = self.tasks.transition(&task_id, TaskState::Working, None);
        self.clone().spawn_task_inner(task_id, true);
        Ok(())
    }

    fn spawn_task(self: Arc<Self>, id: String) {
        self.spawn_task_inner(id, false)
    }

    fn spawn_task_inner(self: Arc<Self>, id: String, approved: bool) {
        let mut running = self.running.lock().unwrap();
        if running.contains_key(&id) {
            return;
        }
        let svc = self.clone();
        let tid = id.clone();
        let h = tokio::spawn(async move {
            svc.run_task(&tid, approved).await;
            svc.running.lock().unwrap().remove(&tid);
        });
        running.insert(id, h.abort_handle());
    }

    fn agent_message(&self, text: impl Into<String>, task: &Task) -> Message {
        Message {
            message_id: format!("{}-{}", task.id, chrono::Utc::now().timestamp_micros()),
            role: Role::Agent,
            parts: vec![Part::text(text)],
            context_id: Some(task.context_id.clone()),
            task_id: Some(task.id.clone()),
            metadata: None,
            extensions: vec![],
            reference_task_ids: vec![],
        }
    }

    async fn run_task(self: &Arc<Self>, id: &str, approved: bool) {
        let Ok(task) = self.tasks.get(id) else { return };
        if task.status.state.is_terminal() || task.status.state == TaskState::InputRequired {
            return;
        }
        let _permit = self.permits.clone().acquire_owned().await.expect("semaphore open");
        let fail = |state: TaskState, why: String| {
            let m = self.agent_message(why, &task);
            let _ = self.tasks.transition(id, state, Some(m));
        };
        let fresh = task.status.state == TaskState::Submitted;
        if self.kill.is_halted() {
            return fail(if fresh { TaskState::Rejected } else { TaskState::Failed }, "node is halted".into());
        }
        let sender = task.metadata.as_ref().and_then(|m| m["sender"].as_str()).unwrap_or("unknown").to_string();
        let skill_id = task.metadata.as_ref().and_then(|m| m["skill"].as_str()).map(String::from);
        let skill = {
            let m = self.skills.lock().unwrap();
            match &skill_id {
                Some(s) => m.get(s).cloned(),
                None if m.len() == 1 => m.values().next().cloned(),
                None => None,
            }
        };
        let Some(skill) = skill else {
            return fail(if fresh { TaskState::Rejected } else { TaskState::Failed }, match skill_id {
                Some(s) => format!("unknown skill '{s}'"),
                None => "a skill must be named in the message metadata".into(),
            });
        };
        // Policy applies when work is accepted, before it starts. A recovered
        // (already Working) task was admitted before the restart.
        let mut needs_park = None;
        if fresh && !approved {
            let req = PolicyRequest { actor: self.node_id.clone(), action: format!("task:{}", skill.descriptor().id), counterparty: Some(sender.clone()), data_class: None, spend: 0, task: Some(id.to_string()) };
            match self.policy.evaluate(req, now()) {
                Verdict::Allow => {}
                Verdict::Deny(why) => return fail(TaskState::Rejected, format!("policy: {why}")),
                Verdict::NeedsApproval(aid) => needs_park = Some(aid),
            }
        }
        if fresh {
            if let Err(e) = self.tasks.transition(id, TaskState::Working, None) {
                tracing::debug!("task {id} not startable: {e}");
                return;
            }
        }
        if let Some(aid) = needs_park {
            self.awaiting.lock().unwrap().insert(aid, id.to_string());
            let m = self.agent_message(format!("awaiting human approval (request {aid})"), &task);
            let _ = self.tasks.transition(id, TaskState::InputRequired, Some(m));
            return;
        }
        let b = &self.cfg.task_budget;
        let budget = Arc::new(BudgetTracker::new(Budget { tokens: b.tokens, gpu_seconds: b.gpu_seconds, wall_clock_secs: b.wall_clock_secs }));
        self.kill.register_budget(&budget);
        let ctx = SkillCtx { budget, node_id: self.node_id.clone(), requester: sender, sink: self.sink() };
        let input = task.history.first().cloned().expect("a submitted task has its first message");
        let timeout = Duration::from_secs(self.cfg.task_timeout_secs.max(1));
        match tokio::time::timeout(timeout, skill.run(&ctx, &task, &input)).await {
            Ok(Ok(artifacts)) if {
                // Egress policy: what kind of thing is leaving, how big, and what class of data.
                let (kind, class) = skill.export();
                let bytes: usize = artifacts.iter().map(|a| serde_json::to_vec(a).map(|b| b.len()).unwrap_or(usize::MAX)).sum();
                match self.egress.check(kind, class, bytes) {
                    Ok(()) => false,
                    Err(why) => {
                        self.emit(KIND_POLICY_AUDIT, json!({"action": format!("egress:{kind:?}"), "approved": false, "decided_by": "policy:egress"}));
                        fail(TaskState::Failed, format!("egress policy: {why}"));
                        true
                    }
                }
            } => {}
            Ok(Ok(mut artifacts)) => {
                // Output filter: no secret of this node leaves in a result.
                let secrets = vec![hex::encode(self.keypair.to_bytes()), self.agent_key.public_key_hex(), hex::encode(self.agent_key.to_bytes())];
                let mut fired = vec![];
                for a in artifacts.iter_mut() {
                    for p in a.parts.iter_mut() {
                        if let Some(t) = p.text.as_mut() {
                            let (c, f) = crate::agents::safety::scrub_output(t, &secrets);
                            if !f.is_empty() { *t = c; fired.extend(f); }
                        }
                        if let Some(d) = p.data.as_mut() {
                            crate::agents::safety::scrub_json(d, &secrets, &mut fired);
                        }
                    }
                }
                if !fired.is_empty() {
                    fired.sort();
                    fired.dedup();
                    self.emit(KIND_POLICY_AUDIT, json!({"action": format!("output.redacted:{}", fired.join(",")), "approved": false, "decided_by": "filter:output"}));
                }
                for a in artifacts {
                    let _ = self.tasks.add_artifact(id, a);
                }
                // A cancel that raced with completion wins: the transition simply fails.
                let _ = self.tasks.transition(id, TaskState::Completed, None);
            }
            Ok(Err(SkillError::BadInput(e))) => fail(TaskState::Failed, e),
            Ok(Err(SkillError::Failed(e))) => fail(TaskState::Failed, e),
            Err(_) => fail(TaskState::Failed, format!("timed out after {}s", timeout.as_secs())),
        }
    }

    // -- discovery -------------------------------------------------------------

    /// Anti-entropy: pull `peer`'s provenance records of every agent-layer kind and add
    /// the ones we lack (each is verified on insert). Gossip only delivers records
    /// published while we are connected; this is how a node that joins later, or was
    /// offline, catches up on delegations, cards and research programs.
    /// Returns the number of new records.
    pub async fn sync_from_peer(&self, endpoint: &iroh::Endpoint, peer: iroh::EndpointId) -> usize {
        use crate::agents::protocol::{request, AgentRequest, AgentResponse};
        let mut added = 0;
        for kind in ALL_KINDS {
            // research.* kinds only if we take part in research; cards/keys/tasks always.
            let resp = request(endpoint, peer, &AgentRequest::Records { kind: (*kind).to_string(), limit: 1000 }).await;
            let Ok(AgentResponse::Records { records }) = resp else { continue };
            let mut h = self.history.write().await;
            for r in records {
                let known = h.peer_updates.get(&r.node_id).map_or(false, |l| l.iter().any(|x| x.hash == r.hash));
                if !known && h.record_update(r).is_ok() {
                    added += 1;
                }
            }
        }
        added
    }

    /// Rebuild the directory from the provenance log, then fetch the cards of `peers`
    /// over the agent ALPN and attach those that verify against their announcements.
    pub async fn refresh_directory(&self, endpoint: &iroh::Endpoint, peers: Vec<iroh::EndpointId>) {
        use crate::agents::protocol::{request, AgentRequest, AgentResponse};
        for peer in peers {
            self.sync_from_peer(endpoint, peer).await;
            if let Ok(AgentResponse::Card { card }) = request(endpoint, peer, &AgentRequest::GetCard).await {
                if let (Some(nid), Some(key)) = (card.slakshna_node_id.clone(), card.verified_key()) {
                    self.fetched_cards.lock().unwrap().insert((nid, key), card);
                }
            }
        }
        let recs: Vec<UpdateRecord> = {
            let h = self.history.read().await;
            h.peer_updates.values().flatten().cloned().collect()
        };
        let mut dir = crate::agents::discovery::Directory::from_records(&recs);
        for ((nid, _), card) in self.fetched_cards.lock().unwrap().iter() {
            if let Err(e) = dir.attach_card(nid, card.clone(), now()) {
                tracing::debug!("card of {nid} not listed: {e}");
            }
        }
        *self.directory.lock().unwrap() = dir;
    }

    /// Verifier that trusts this node's own tokens (the node is the only issuer for its agent).
    pub fn token_verifier(&self) -> Arc<crate::agents::safety::TokenVerifier> {
        self.verifier.clone()
    }

    /// Issue a short-lived, single-resource capability token to this node's agent key. This is how
    /// a task gets exactly the tool access it needs and nothing else.
    pub fn issue_token(&self, resource: &str, max_amount: u64, ttl_secs: i64) -> crate::agents::safety::CapabilityToken {
        let t = now();
        crate::agents::safety::CapabilityToken::issue(&self.keypair, &self.agent_key.public_key_hex(), resource, max_amount, t - 1, t + ttl_secs)
    }

    /// Fill a team template from the verified directory, preferring agents on nodes the team does
    /// not use yet and, within that, higher reputation (as this node has learned it).
    pub fn form_team(&self, template: &crate::agents::coordination::TeamTemplate) -> Result<Vec<crate::agents::coordination::Member>, String> {
        let dir = self.directory.lock().unwrap();
        let d = self.delegator();
        crate::agents::coordination::form_team(template, &dir, &|n, k| d.as_ref().map_or(0.5, |d| d.reputation(n, k)), now())
    }

    /// The collaboration's blackboard as this node currently sees it.
    pub async fn blackboard(&self, collab: &str) -> crate::agents::coordination::Blackboard {
        let h = self.history.read().await;
        crate::agents::coordination::Blackboard::from_records(collab, h.peer_updates.values().flatten())
    }

    /// Post to a collaboration's blackboard (a signed `coord.post` record). Returns the post id.
    pub async fn post(&self, collab: &str, kind: PostKind, reply_to: Option<&str>, text: &str) -> Result<String, String> {
        let body = serde_json::to_value(Post { collaboration: collab.into(), kind, reply_to: reply_to.map(String::from), content_hash: sha256_hex(text.as_bytes()), text: text.into() }).map_err(|e| e.to_string())?;
        crate::agents::records::validate_body(KIND_POST, &body)?;
        self.author(KIND_POST, body).await.map(|r| r.hash).map_err(|e| e.to_string())
    }

    /// Verified delegation chain under `root_task` from the records this node holds.
    pub async fn delegation_chain(&self, root_task: &str, scope: &[String]) -> Result<Vec<crate::agents::a2a::Hop>, crate::agents::a2a::ChainError> {
        let h = self.history.read().await;
        let scope: std::collections::BTreeSet<String> = scope.iter().cloned().collect();
        crate::agents::a2a::verify_delegation_chain(h.peer_updates.values().flatten(), root_task, &scope)
    }

    /// QUIC endpoint id of a federation node, from its verified card.
    pub fn endpoint_of(&self, node_id: &str) -> Option<String> {
        self.directory.lock().unwrap().endpoint_of(node_id, now())
    }

    /// Verified agents offering `query` (a skill id or tag).
    pub fn search_agents(&self, query: &str) -> Vec<(String, String, AgentCard)> {
        self.directory.lock().unwrap().search(query, now()).into_iter().map(|e| (e.node_id.clone(), e.agent_key.clone(), e.card.clone())).collect()
    }

    pub fn all_agents(&self) -> Vec<(String, String, AgentCard)> {
        self.directory.lock().unwrap().entries(now()).into_iter().map(|e| (e.node_id.clone(), e.agent_key.clone(), e.card.clone())).collect()
    }

    /// Join the topic of every research program announced in the log.
    pub fn spawn_program_watcher(self: &Arc<Self>) {
        let svc = self.clone();
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(Duration::from_secs(5));
            loop {
                tick.tick().await;
                let programs: Vec<String> = {
                    let h = svc.history.read().await;
                    h.provenance(KIND_OBJECTIVE).into_iter().map(|r| r.hash.clone()).collect()
                };
                for p in programs {
                    let fresh = svc.joined_programs.lock().unwrap().insert(p.clone());
                    if fresh {
                        svc.join_program(&p).await;
                    }
                }
            }
        });
    }

    /// Queue `message` for delivery to a peer (`endpoint:<id>` or a bare endpoint id) and try now.
    /// If the peer is offline the parcel stays in the durable mailbox and is retried with
    /// backoff; the remote side deduplicates on `messageId`, so a retry never duplicates work.
    pub async fn send_to_peer(&self, endpoint: &iroh::Endpoint, to: &str, message: Message) -> Result<(), TaskError> {
        self.mailbox.enqueue(to, None, message)?;
        self.deliver_pending(endpoint).await;
        Ok(())
    }

    /// One sweep over the mailbox. Returns how many parcels were delivered.
    pub async fn deliver_pending(&self, endpoint: &iroh::Endpoint) -> usize {
        use crate::agents::protocol::{request, AgentRequest, AgentResponse};
        let mut delivered = 0;
        for p in self.mailbox.due(now()) {
            let Ok(peer) = p.to.strip_prefix("endpoint:").unwrap_or(&p.to).parse::<iroh::EndpointId>() else {
                // Unroutable destination: drop it rather than retry forever.
                let _ = self.mailbox.ack(&p.to, &p.message.message_id);
                continue;
            };
            match request(endpoint, peer, &AgentRequest::SendMessage { message: p.message.clone() }).await {
                Ok(AgentResponse::Task { task }) => {
                    self.mailbox.record_sent(&p.to, &p.message.message_id, &task.id);
                    let _ = self.mailbox.ack(&p.to, &p.message.message_id);
                    delivered += 1;
                }
                Ok(AgentResponse::Error { message }) => {
                    tracing::warn!("peer {} refused parcel {}: {message}", p.to, p.message.message_id);
                    // A refusal is an answer: retrying the same message cannot change it.
                    let _ = self.mailbox.ack(&p.to, &p.message.message_id);
                }
                _ => self.mailbox.note_attempt(&p.to, &p.message.message_id),
            }
        }
        delivered
    }

    /// Background retry loop for the mailbox.
    pub fn spawn_mail_loop(self: &Arc<Self>, endpoint: iroh::Endpoint) {
        let svc = self.clone();
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(Duration::from_secs(5));
            loop {
                tick.tick().await;
                svc.deliver_pending(&endpoint).await;
            }
        });
    }

    /// Record the delegator's verdict on a result (`a2a.acceptance`).
    pub fn record_acceptance(&self, task_id: &str, artifact: &Artifact, accepted: bool) {
        self.emit(KIND_A2A_ACCEPTANCE, json!({"task_id": task_id, "result_hash": artifact.hash(), "accepted": accepted}));
    }

    /// Record that `parent` handed `sub` to another agent (`a2a.delegation`).
    pub fn record_delegation(&self, parent: &str, sub: &str, scope: &[&str]) {
        self.emit(KIND_A2A_DELEGATION, json!({"parent_task": parent, "sub_task": sub, "scope": scope}));
    }
}

fn load_or_create_agent_key(path: &std::path::Path) -> Result<Keypair, Box<dyn std::error::Error + Send + Sync>> {
    if let Ok(s) = std::fs::read_to_string(path) {
        return Keypair::from_hex(s.trim());
    }
    let k = Keypair::generate();
    std::fs::write(path, hex::encode(k.to_bytes()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))?;
    }
    Ok(k)
}

use crate::history::RecordKind;

/// Tracks which approvals/tasks exist; exposed to the HTTP API.
pub fn pending_approvals(svc: &AgentService) -> Vec<(u64, String)> {
    let mut v: Vec<_> = svc.awaiting.lock().unwrap().iter().map(|(k, v)| (*k, v.clone())).collect();
    v.sort();
    v
}

/// Distinct sender principals seen, for tests and the API.
#[allow(dead_code)] // library surface: used by tests and integrators, or pending the settlement-chain decision
pub fn senders(svc: &AgentService) -> HashSet<String> {
    svc.tasks.list().iter().filter_map(|t| t.metadata.as_ref().and_then(|m| m["sender"].as_str()).map(String::from)).collect()
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::agents::discovery::Directory;
    use crate::config::Config;
    use crate::state::State;
    use std::sync::atomic::{AtomicUsize, Ordering};

    pub fn tmpdir(tag: &str) -> String {
        let p = std::env::temp_dir().join(format!("slk-svc-{tag}-{}", rand::random::<u64>()));
        std::fs::create_dir_all(&p).unwrap();
        p.to_string_lossy().into_owned()
    }

    pub async fn make_service(dir: &str, kp: Keypair, mutate: impl FnOnce(&mut AgentsConfig)) -> Arc<AgentService> {
        let cfg: Config = toml::from_str(&format!(
            "[federation]\nid=\"t\"\nname=\"t\"\n[node]\nid=\"x\"\ndata_dir=\"{dir}\"\n[network]\nhost=\"127.0.0.1\"\napi_port=0\nws_port=0\np2p_port=0\n[logging]\nlevel=\"info\"\n"
        ))
        .unwrap();
        let state = Arc::new(RwLock::new(State::new(&cfg.node.data_dir).unwrap()));
        let hist = Arc::new(RwLock::new(UpdateHistory::new(cfg.clone(), state, kp.node_id()).await.unwrap()));
        let mut acfg = AgentsConfig::default();
        acfg.enabled = true;
        mutate(&mut acfg);
        AgentService::new(acfg, kp, hist, Arc::new(NoBroadcast), dir).await.unwrap()
    }

    pub fn user_msg(id: &str, skill: &str, text: &str) -> Message {
        Message { message_id: id.into(), role: Role::User, parts: vec![Part::text(text)], context_id: None, task_id: None, metadata: Some(json!({"skill": skill})), extensions: vec![], reference_task_ids: vec![] }
    }

    pub struct TestSkill {
        pub id: &'static str,
        pub delay_ms: u64,
        pub fail: bool,
        pub runs: Arc<AtomicUsize>,
        pub active: Arc<AtomicUsize>,
        pub max_active: Arc<AtomicUsize>,
    }
    impl TestSkill {
        pub fn new(id: &'static str) -> Self {
            TestSkill { id, delay_ms: 0, fail: false, runs: Arc::new(0.into()), active: Arc::new(0.into()), max_active: Arc::new(0.into()) }
        }
    }
    #[async_trait]
    impl Skill for TestSkill {
        fn descriptor(&self) -> AgentSkill {
            AgentSkill { id: self.id.into(), name: self.id.into(), description: "test".into(), tags: vec!["test".into()], examples: vec![], input_modes: vec![], output_modes: vec![] }
        }
        async fn run(&self, _: &SkillCtx, _: &Task, input: &Message) -> Result<Vec<Artifact>, SkillError> {
            self.runs.fetch_add(1, Ordering::SeqCst);
            let a = self.active.fetch_add(1, Ordering::SeqCst) + 1;
            self.max_active.fetch_max(a, Ordering::SeqCst);
            tokio::time::sleep(Duration::from_millis(self.delay_ms)).await;
            self.active.fetch_sub(1, Ordering::SeqCst);
            if self.fail {
                return Err(SkillError::Failed("boom".into()));
            }
            let text = input.parts[0].text.clone().unwrap_or_default();
            Ok(vec![Artifact { artifact_id: "out".into(), name: None, description: None, parts: vec![Part::text(text.to_uppercase())], metadata: None }])
        }
    }

    pub async fn wait_state(svc: &AgentService, id: &str, want: TaskState) -> Task {
        for _ in 0..400 {
            let t = svc.tasks.get(id).unwrap();
            if t.status.state == want {
                return t;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        panic!("task {id} never reached {want:?}; is {:?}", svc.tasks.get(id).unwrap().status.state);
    }

    #[tokio::test]
    async fn task_runs_to_completion_and_leaves_signed_ordered_records() {
        let dir = tmpdir("a");
        let kp = Keypair::generate();
        let svc = make_service(&dir, kp.clone(), |_| {}).await;
        svc.register_skill(Arc::new(TestSkill::new("shout")));
        let t = svc.send_message("peer:1", user_msg("m1", "shout", "hello")).unwrap();
        let done = wait_state(&svc, &t.id, TaskState::Completed).await;
        assert_eq!(done.artifacts[0].parts[0].text.as_deref(), Some("HELLO"));
        tokio::time::sleep(Duration::from_millis(100)).await; // author loop drains
        let h = svc.history.read().await;
        let recs = h.provenance(KIND_A2A_TASK);
        let states: Vec<String> = recs.iter().map(|r| match &r.kind { RecordKind::Provenance { body, .. } => body["state"].as_str().unwrap().to_string(), _ => unreachable!() }).collect();
        assert_eq!(states, ["TASK_STATE_SUBMITTED", "TASK_STATE_WORKING", "TASK_STATE_COMPLETED"]);
        assert!(recs.iter().all(|r| r.verify().is_ok() && r.node_id == kp.node_id().to_string()));
        // hash chain: each record links to the previous one of this node
        let log = &h.peer_updates[&kp.node_id().to_string()];
        for w in log.windows(2) { assert_eq!(w[1].prev_hash, w[0].hash); }
    }

    #[tokio::test]
    async fn resending_the_same_message_does_not_run_it_twice() {
        let svc = make_service(&tmpdir("b"), Keypair::generate(), |_| {}).await;
        let skill = TestSkill::new("s");
        let runs = skill.runs.clone();
        svc.register_skill(Arc::new(skill));
        let a = svc.send_message("p", user_msg("same", "s", "x")).unwrap();
        let b = svc.send_message("p", user_msg("same", "s", "x")).unwrap();
        assert_eq!(a.id, b.id);
        wait_state(&svc, &a.id, TaskState::Completed).await;
        let _ = svc.send_message("p", user_msg("same", "s", "x")).unwrap();
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert_eq!(runs.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn unknown_skill_failing_skill_and_timeout_end_in_the_right_states() {
        let svc = make_service(&tmpdir("c"), Keypair::generate(), |c| c.task_timeout_secs = 1).await;
        let mut failing = TestSkill::new("bad");
        failing.fail = true;
        svc.register_skill(Arc::new(failing));
        let mut slow = TestSkill::new("slow");
        slow.delay_ms = 5000;
        svc.register_skill(Arc::new(slow));
        let t = svc.send_message("p", user_msg("1", "nope", "x")).unwrap();
        let r = wait_state(&svc, &t.id, TaskState::Rejected).await;
        assert!(r.status.message.unwrap().parts[0].text.clone().unwrap().contains("unknown skill"));
        let t = svc.send_message("p", user_msg("2", "bad", "x")).unwrap();
        let f = wait_state(&svc, &t.id, TaskState::Failed).await;
        assert!(f.status.message.unwrap().parts[0].text.clone().unwrap().contains("boom"));
        let t = svc.send_message("p", user_msg("3", "slow", "x")).unwrap();
        let f = wait_state(&svc, &t.id, TaskState::Failed).await;
        assert!(f.status.message.unwrap().parts[0].text.clone().unwrap().contains("timed out"));
        // ambiguous: no skill named and more than one registered
        let mut m = user_msg("4", "", "x");
        m.metadata = None;
        let t = svc.send_message("p", m).unwrap();
        wait_state(&svc, &t.id, TaskState::Rejected).await;
    }

    #[tokio::test]
    async fn cancel_stops_a_running_task() {
        let svc = make_service(&tmpdir("d"), Keypair::generate(), |_| {}).await;
        let mut s = TestSkill::new("slow");
        s.delay_ms = 10_000;
        let active = s.active.clone();
        svc.register_skill(Arc::new(s));
        let t = svc.send_message("p", user_msg("1", "slow", "x")).unwrap();
        wait_state(&svc, &t.id, TaskState::Working).await;
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(active.load(Ordering::SeqCst), 1);
        svc.cancel_task(&t.id).unwrap();
        assert_eq!(svc.tasks.get(&t.id).unwrap().status.state, TaskState::Canceled);
        tokio::time::sleep(Duration::from_millis(100)).await;
        // the aborted skill future was dropped mid-sleep: it never decremented, so it is still counted
        // as active only if it kept running; assert it did not complete.
        assert_eq!(svc.tasks.get(&t.id).unwrap().status.state, TaskState::Canceled);
        assert!(svc.cancel_task(&t.id).is_err(), "cancelling a finished task is an error");
    }

    #[tokio::test]
    async fn concurrency_limit_is_respected() {
        let svc = make_service(&tmpdir("e"), Keypair::generate(), |c| c.max_concurrent_tasks = 2).await;
        let mut s = TestSkill::new("s");
        s.delay_ms = 80;
        let max = s.max_active.clone();
        svc.register_skill(Arc::new(s));
        let ids: Vec<_> = (0..6).map(|i| svc.send_message("p", user_msg(&format!("m{i}"), "s", "x")).unwrap().id).collect();
        for id in &ids { wait_state(&svc, id, TaskState::Completed).await; }
        assert_eq!(max.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn policy_denies_and_parks_for_human_approval() {
        let svc = make_service(&tmpdir("f"), Keypair::generate(), |c| {
            c.policy.denied_counterparties = vec!["peer:evil".into()];
            c.policy.human_approval_actions = vec!["task:risky".into()];
        }).await;
        svc.register_skill(Arc::new(TestSkill::new("safe")));
        svc.register_skill(Arc::new(TestSkill::new("risky")));
        let t = svc.send_message("peer:evil", user_msg("1", "safe", "x")).unwrap();
        let r = wait_state(&svc, &t.id, TaskState::Rejected).await;
        assert!(r.status.message.unwrap().parts[0].text.clone().unwrap().contains("denied"));
        // risky task parks, then a human approves and it completes
        let t = svc.send_message("peer:ok", user_msg("2", "risky", "go")).unwrap();
        wait_state(&svc, &t.id, TaskState::InputRequired).await;
        let pending = pending_approvals(&svc);
        assert_eq!(pending.len(), 1);
        svc.approve(pending[0].0, "alice").unwrap();
        let done = wait_state(&svc, &t.id, TaskState::Completed).await;
        assert_eq!(done.artifacts[0].parts[0].text.as_deref(), Some("GO"));
        assert!(svc.approve(pending[0].0, "alice").is_err());
        tokio::time::sleep(Duration::from_millis(100)).await;
        let h = svc.history.read().await;
        let audits = h.provenance(KIND_POLICY_AUDIT);
        assert!(audits.iter().any(|r| matches!(&r.kind, RecordKind::Provenance { body, .. } if body["decided_by"] == "human:alice" && body["approved"] == true)));
        assert!(audits.iter().any(|r| matches!(&r.kind, RecordKind::Provenance { body, .. } if body["approved"] == false)));
    }

    /// A skill that tries to return the node's own secrets has them removed, and the redaction is audited.
    #[tokio::test]
    async fn results_never_carry_the_nodes_own_secrets() {
        struct Leaky(String);
        #[async_trait]
        impl Skill for Leaky {
            fn descriptor(&self) -> AgentSkill {
                AgentSkill { id: "leaky".into(), name: "leaky".into(), description: "x".into(), tags: vec![], examples: vec![], input_modes: vec![], output_modes: vec![] }
            }
            async fn run(&self, _: &SkillCtx, _: &Task, _: &Message) -> Result<Vec<Artifact>, SkillError> {
                Ok(vec![Artifact { artifact_id: "a".into(), name: None, description: None, parts: vec![Part::text(format!("my key is {}", self.0)), Part::data(json!({"k": self.0}))], metadata: None }])
            }
        }
        let kp = Keypair::generate();
        let svc = make_service(&tmpdir("scrub"), kp.clone(), |_| {}).await;
        svc.register_skill(Arc::new(Leaky(hex::encode(kp.to_bytes()))));
        let t = svc.send_message("p", user_msg("1", "leaky", "x")).unwrap();
        let done = wait_state(&svc, &t.id, TaskState::Completed).await;
        assert!(!serde_json::to_string(&done).unwrap().contains(&hex::encode(kp.to_bytes())));
        assert!(done.artifacts[0].parts[0].text.clone().unwrap().contains("[REDACTED"));
        tokio::time::sleep(Duration::from_millis(100)).await;
        let h = svc.history.read().await;
        assert!(h.provenance(KIND_POLICY_AUDIT).iter().any(|r| matches!(&r.kind, RecordKind::Provenance { body, .. } if body["action"].as_str().map_or(false, |a| a.starts_with("output.redacted")))));
    }

    /// A node that holds private data allows only findings out: an artifact-kind result is blocked and audited.
    #[tokio::test]
    async fn egress_policy_blocks_disallowed_exports_and_oversize_results() {
        struct Dump;
        #[async_trait]
        impl Skill for Dump {
            fn descriptor(&self) -> AgentSkill { AgentSkill { id: "dump".into(), name: "d".into(), description: "x".into(), tags: vec![], examples: vec![], input_modes: vec![], output_modes: vec![] } }
            async fn run(&self, _: &SkillCtx, _: &Task, _: &Message) -> Result<Vec<Artifact>, SkillError> {
                Ok(vec![Artifact { artifact_id: "a".into(), name: None, description: None, parts: vec![Part::text("rows rows rows")], metadata: None }])
            }
        }
        struct Score;
        #[async_trait]
        impl Skill for Score {
            fn descriptor(&self) -> AgentSkill { AgentSkill { id: "score".into(), name: "s".into(), description: "x".into(), tags: vec![], examples: vec![], input_modes: vec![], output_modes: vec![] } }
            fn export(&self) -> (crate::agents::safety::ExportKind, &'static str) { (crate::agents::safety::ExportKind::Finding, "metrics") }
            async fn run(&self, _: &SkillCtx, _: &Task, _: &Message) -> Result<Vec<Artifact>, SkillError> {
                Ok(vec![Artifact { artifact_id: "a".into(), name: None, description: None, parts: vec![Part::data(json!({"accuracy": 0.9}))], metadata: None }])
            }
        }
        struct Big;
        #[async_trait]
        impl Skill for Big {
            fn descriptor(&self) -> AgentSkill { AgentSkill { id: "big".into(), name: "b".into(), description: "x".into(), tags: vec![], examples: vec![], input_modes: vec![], output_modes: vec![] } }
            fn export(&self) -> (crate::agents::safety::ExportKind, &'static str) { (crate::agents::safety::ExportKind::Finding, "metrics") }
            async fn run(&self, _: &SkillCtx, _: &Task, _: &Message) -> Result<Vec<Artifact>, SkillError> {
                Ok(vec![Artifact { artifact_id: "a".into(), name: None, description: None, parts: vec![Part::text("x".repeat(5000))], metadata: None }])
            }
        }
        let svc = make_service(&tmpdir("egress"), Keypair::generate(), |c| { c.egress.allow = vec!["finding".into()]; c.egress.max_bytes = 1000; }).await;
        svc.register_skill(Arc::new(Dump));
        svc.register_skill(Arc::new(Score));
        svc.register_skill(Arc::new(Big));
        let blocked = svc.send_message("p", user_msg("1", "dump", "x")).unwrap();
        let f = wait_state(&svc, &blocked.id, TaskState::Failed).await;
        assert!(f.status.message.unwrap().parts[0].text.clone().unwrap().contains("egress policy"));
        assert!(f.artifacts.is_empty(), "nothing leaked into the task");
        let ok = svc.send_message("p", user_msg("2", "score", "x")).unwrap();
        assert_eq!(wait_state(&svc, &ok.id, TaskState::Completed).await.artifacts.len(), 1);
        let big = svc.send_message("p", user_msg("3", "big", "x")).unwrap();
        let f = wait_state(&svc, &big.id, TaskState::Failed).await;
        assert!(f.status.message.unwrap().parts[0].text.clone().unwrap().contains("exceeds"));
        tokio::time::sleep(Duration::from_millis(100)).await;
        let h = svc.history.read().await;
        assert!(h.provenance(KIND_POLICY_AUDIT).iter().any(|r| matches!(&r.kind, RecordKind::Provenance { body, .. } if body["decided_by"] == "policy:egress")));
    }

    #[tokio::test]
    async fn halted_node_refuses_work() {
        use crate::agents::safety::{HaltAction, HaltMsg};
        let op = Keypair::generate();
        let pk = op.public_key_hex();
        let svc = make_service(&tmpdir("g"), Keypair::generate(), |c| c.halt_keys = vec![pk]).await;
        svc.register_skill(Arc::new(TestSkill::new("s")));
        svc.kill.apply(&HaltMsg::sign(&op, HaltAction::Halt, 1, "drill")).unwrap();
        let t = svc.send_message("p", user_msg("1", "s", "x")).unwrap();
        let f = wait_state(&svc, &t.id, TaskState::Rejected).await;
        assert!(f.status.message.unwrap().parts[0].text.clone().unwrap().contains("halted"));
    }

    #[test]
    fn in_flight_task_survives_a_process_restart_and_completes() {
        let dir = tmpdir("h");
        let kp = Keypair::generate();
        let id = {
            // "Process 1": its own runtime, killed while the task is working.
            let rt = tokio::runtime::Runtime::new().unwrap();
            let id = rt.block_on(async {
                let svc = make_service(&dir, kp.clone(), |_| {}).await;
                let mut s = TestSkill::new("long");
                s.delay_ms = 60_000;
                svc.register_skill(Arc::new(s));
                let t = svc.send_message("peer", user_msg("job", "long", "payload")).unwrap();
                wait_state(&svc, &t.id, TaskState::Working).await;
                t.id
            });
            rt.shutdown_background(); // the "crash"
            id
        };
        // "Process 2": same data dir, a skill that finishes quickly.
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let svc = make_service(&dir, kp.clone(), |_| {}).await;
            assert_eq!(svc.tasks.get(&id).unwrap().status.state, TaskState::Working, "state persisted across the crash");
            svc.register_skill(Arc::new(TestSkill::new("long")));
            assert_eq!(svc.recover(), 1);
            let done = wait_state(&svc, &id, TaskState::Completed).await;
            assert_eq!(done.artifacts[0].parts[0].text.as_deref(), Some("PAYLOAD"));
            // the same message is still deduplicated after restart
            let again = svc.send_message("peer", user_msg("job", "long", "payload")).unwrap();
            assert_eq!(again.id, id);
        });
    }

    #[tokio::test]
    async fn delegation_and_card_records_make_the_agent_discoverable() {
        let kp = Keypair::generate();
        let svc = make_service(&tmpdir("i"), kp.clone(), |_| {}).await;
        svc.register_skill(Arc::new(TestSkill::new("evaluate")));
        svc.ensure_delegation().await;
        svc.publish_card().await;
        svc.publish_card().await; // unchanged card: no second record
        tokio::time::sleep(Duration::from_millis(150)).await;
        let h = svc.history.read().await;
        assert_eq!(h.provenance(KIND_A2A_CARD).len(), 1);
        let all: Vec<&UpdateRecord> = h.peer_updates.values().flatten().collect();
        let mut dir = Directory::from_records(all);
        dir.attach_card(&svc.node_id, svc.card(), now()).expect("card verifies against its record and delegation");
        assert_eq!(dir.search("evaluate", now()).len(), 1);
        drop(h);
        // a changed skill set is a new card version; revocation removes the agent
        svc.register_skill(Arc::new(TestSkill::new("chat")));
        svc.publish_card().await;
        svc.revoke_card().await;
        tokio::time::sleep(Duration::from_millis(150)).await;
        let h = svc.history.read().await;
        let all: Vec<&UpdateRecord> = h.peer_updates.values().flatten().collect();
        let dir = Directory::from_records(all);
        assert!(dir.is_revoked(&svc.node_id, &svc.agent_public_key()));
        assert!(dir.entries(now()).is_empty());
    }
}
