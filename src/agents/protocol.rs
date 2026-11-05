//! `slakshna/agent/1` ALPN (week 0 prerequisite, M2 transport).
//!
//! The endpoint now serves two protocols, routed by ALPN: iroh-gossip for the
//! training mesh and this one for agent traffic. Each request is one QUIC
//! bidirectional stream: the client writes a JSON request and finishes the
//! stream, the server answers with one JSON response. Peers are authenticated
//! by the QUIC handshake against their endpoint key, and the optional
//! `allowed_peers` list from the config applies here exactly as it does to
//! gossip.
//!
//! This is the transport the A2A task protocol (M2) will carry its messages
//! on. What it serves today is real and useful for A2/A3: a peer can fetch any
//! node's verified provenance records of a given kind.

use crate::agents::a2a::{AgentCard, Artifact, Message, Task, TaskEvent, TaskStatus};
use crate::agents::service::AgentService;
use crate::history::{UpdateHistory, UpdateRecord};
use iroh::endpoint::Connection;
use iroh::protocol::{AcceptError, ProtocolHandler};
use iroh::{Endpoint, EndpointAddr};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;

pub const AGENT_ALPN: &[u8] = b"slakshna/agent/1";
const MAX_REQUEST_BYTES: usize = 64 * 1024;
const MAX_RESPONSE_BYTES: usize = 16 * 1024 * 1024;
const MAX_RECORDS: usize = 1000;
const IO_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentRequest {
    Hello,
    /// Verified provenance records of one kind held by the responder.
    Records { kind: String, limit: u32 },
    /// A2A SendMessage. Idempotent on `message.messageId`, so a retry (or a
    /// mailbox redelivery) never creates a second task.
    SendMessage { message: Message },
    GetTask { id: String },
    ListTasks,
    CancelTask { id: String },
    /// The responder's signed Agent Card.
    GetCard,
    /// Streams `WireEvent` JSON lines until the task reaches a final state.
    SubscribeTask { id: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentResponse {
    Hello { node_id: String, version: u32 },
    Records { records: Vec<UpdateRecord> },
    Task { task: Task },
    Tasks { tasks: Vec<Task> },
    Card { card: AgentCard },
    Error { message: String },
}

/// One line of a task subscription stream.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WireEvent {
    Status { task_id: String, status: TaskStatus, is_final: bool },
    Artifact { task_id: String, artifact: Artifact },
    Error { message: String },
}

impl WireEvent {
    fn from_event(e: TaskEvent) -> Self {
        match e {
            TaskEvent::Status { task_id, status, is_final } => WireEvent::Status { task_id, status, is_final },
            TaskEvent::Artifact { task_id, artifact } => WireEvent::Artifact { task_id, artifact },
        }
    }
}

/// Filled once the node's agent service exists (it needs the network, which
/// needs this handler, so the service is attached afterwards).
pub type AgentSlot = Arc<std::sync::OnceLock<Arc<AgentService>>>;

#[derive(Clone)]
pub struct AgentProtocol {
    history: Arc<RwLock<UpdateHistory>>,
    node_id: String,
    allowed_peers: Option<Vec<String>>,
    agent: AgentSlot,
}

impl std::fmt::Debug for AgentProtocol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentProtocol").field("node_id", &self.node_id).finish()
    }
}

impl AgentProtocol {
    pub fn new(history: Arc<RwLock<UpdateHistory>>, node_id: String, allowed_peers: Option<Vec<String>>) -> Self {
        AgentProtocol { history, node_id, allowed_peers, agent: Arc::new(std::sync::OnceLock::new()) }
    }

    pub fn with_agent_slot(mut self, slot: AgentSlot) -> Self {
        self.agent = slot;
        self
    }

    fn service(&self) -> Result<&Arc<AgentService>, AgentResponse> {
        self.agent.get().ok_or_else(|| AgentResponse::Error { message: "this node does not run an agent service".into() })
    }

    async fn respond(&self, remote: &str, req: AgentRequest) -> AgentResponse {
        let sender = format!("endpoint:{remote}");
        macro_rules! svc {
            () => {
                match self.service() { Ok(s) => s, Err(e) => return e }
            };
        }
        let task_err = |e: crate::agents::a2a::TaskError| AgentResponse::Error { message: e.to_string() };
        match req {
            AgentRequest::SendMessage { message } => match svc!().send_message(&sender, message) {
                Ok(task) => AgentResponse::Task { task },
                Err(e) => task_err(e),
            },
            AgentRequest::GetTask { id } => match svc!().tasks.get(&id) {
                Ok(task) => AgentResponse::Task { task },
                Err(e) => task_err(e),
            },
            AgentRequest::ListTasks => AgentResponse::Tasks { tasks: svc!().tasks.list() },
            AgentRequest::CancelTask { id } => match svc!().cancel_task(&id) {
                Ok(task) => AgentResponse::Task { task },
                Err(e) => task_err(e),
            },
            AgentRequest::GetCard => AgentResponse::Card { card: svc!().card() },
            AgentRequest::SubscribeTask { .. } => AgentResponse::Error { message: "subscriptions are streamed".into() },
            AgentRequest::Hello => AgentResponse::Hello { node_id: self.node_id.clone(), version: 1 },
            AgentRequest::Records { kind, limit } => {
                if !crate::agents::records::ALL_KINDS.contains(&kind.as_str()) {
                    return AgentResponse::Error { message: format!("unknown kind '{kind}'") };
                }
                let limit = (limit as usize).min(MAX_RECORDS);
                let h = self.history.read().await;
                let records = h.provenance(&kind).into_iter().take(limit).cloned().collect();
                AgentResponse::Records { records }
            }
        }
    }
}

impl AgentProtocol {
    /// Write the task's current status, then live events, one JSON line each,
    /// until a final status (or the stream ends / times out).
    async fn stream_task(&self, id: &str, send: &mut iroh::endpoint::SendStream) -> Result<(), AcceptError> {
        async fn line(send: &mut iroh::endpoint::SendStream, ev: &WireEvent) -> Result<(), AcceptError> {
            let mut b = serde_json::to_vec(ev).map_err(user_err)?;
            b.push(b'\n');
            send.write_all(&b).await.map_err(user_err)
        }
        let svc = match self.service() {
            Ok(s) => s,
            Err(_) => {
                line(send, &WireEvent::Error { message: "this node does not run an agent service".into() }).await?;
                return send.finish().map_err(user_err);
            }
        };
        // Subscribe before reading the current state so no transition is missed.
        let mut rx = svc.tasks.subscribe();
        let task = match svc.tasks.get(id) {
            Ok(t) => t,
            Err(e) => {
                line(send, &WireEvent::Error { message: e.to_string() }).await?;
                return send.finish().map_err(user_err);
            }
        };
        let done = task.status.state.is_terminal();
        for a in &task.artifacts {
            line(send, &WireEvent::Artifact { task_id: task.id.clone(), artifact: a.clone() }).await?;
        }
        line(send, &WireEvent::Status { task_id: task.id.clone(), status: task.status.clone(), is_final: done }).await?;
        if !done {
            let deadline = tokio::time::Instant::now() + Duration::from_secs(svc.cfg.task_timeout_secs + 60);
            loop {
                match tokio::time::timeout_at(deadline, rx.recv()).await {
                    Ok(Ok(ev)) => {
                        let belongs = match &ev {
                            TaskEvent::Status { task_id, .. } | TaskEvent::Artifact { task_id, .. } => task_id == id,
                        };
                        if !belongs {
                            continue;
                        }
                        let fin = matches!(&ev, TaskEvent::Status { is_final: true, .. });
                        line(send, &WireEvent::from_event(ev)).await?;
                        if fin {
                            break;
                        }
                    }
                    Ok(Err(tokio::sync::broadcast::error::RecvError::Lagged(_))) => {
                        // Missed events: resend the authoritative state.
                        if let Ok(t) = svc.tasks.get(id) {
                            let fin = t.status.state.is_terminal();
                            line(send, &WireEvent::Status { task_id: t.id, status: t.status, is_final: fin }).await?;
                            if fin { break; }
                        }
                    }
                    _ => break,
                }
            }
        }
        send.finish().map_err(user_err)
    }
}

fn user_err<E: std::error::Error + Send + Sync + 'static>(e: E) -> AcceptError {
    AcceptError::from_err(e)
}

#[derive(Debug, thiserror::Error)]
#[error("{0}")]
struct ProtoError(String);

impl ProtocolHandler for AgentProtocol {
    async fn accept(&self, conn: Connection) -> Result<(), AcceptError> {
        let remote = conn.remote_id().to_string();
        if let Some(allowed) = &self.allowed_peers {
            if !allowed.is_empty() && !allowed.contains(&remote) {
                tracing::warn!("🚫 Blocked agent connection from unauthorized peer: {}", remote);
                conn.close(1u32.into(), b"not allowed");
                return Ok(());
            }
        }
        // Serve streams until the peer closes the connection.
        loop {
            let (mut send, mut recv) = match conn.accept_bi().await {
                Ok(s) => s,
                Err(_) => return Ok(()), // closed by peer
            };
            let body = tokio::time::timeout(IO_TIMEOUT, recv.read_to_end(MAX_REQUEST_BYTES))
                .await
                .map_err(|_| user_err(ProtoError("request timed out".into())))?
                .map_err(user_err)?;
            let parsed = serde_json::from_slice::<AgentRequest>(&body);
            if let Ok(AgentRequest::SubscribeTask { id }) = &parsed {
                self.stream_task(id, &mut send).await?;
                continue;
            }
            let resp = match parsed {
                Ok(req) => self.respond(&remote, req).await,
                Err(e) => AgentResponse::Error { message: format!("bad request: {e}") },
            };
            let out = serde_json::to_vec(&resp).map_err(user_err)?;
            send.write_all(&out).await.map_err(user_err)?;
            send.finish().map_err(user_err)?;
        }
    }
}

/// Subscribe to a task on `peer`; events arrive on the returned channel as they happen.
pub async fn subscribe(
    endpoint: &Endpoint,
    peer: impl Into<EndpointAddr>,
    task_id: &str,
) -> Result<tokio::sync::mpsc::Receiver<WireEvent>, Box<dyn std::error::Error + Send + Sync>> {
    let conn = endpoint.connect(peer, AGENT_ALPN).await?;
    let (mut send, mut recv) = conn.open_bi().await?;
    send.write_all(&serde_json::to_vec(&AgentRequest::SubscribeTask { id: task_id.into() })?).await?;
    send.finish()?;
    let (tx, rx) = tokio::sync::mpsc::channel(64);
    tokio::spawn(async move {
        let _conn = conn; // keep the connection open while streaming
        let mut buf = Vec::new();
        let mut chunk = vec![0u8; 16 * 1024];
        loop {
            match recv.read(&mut chunk).await {
                Ok(Some(n)) => {
                    buf.extend_from_slice(&chunk[..n]);
                    if buf.len() > MAX_RESPONSE_BYTES {
                        return;
                    }
                    while let Some(pos) = buf.iter().position(|b| *b == b'\n') {
                        let line: Vec<u8> = buf.drain(..=pos).collect();
                        if let Ok(ev) = serde_json::from_slice::<WireEvent>(&line) {
                            if tx.send(ev).await.is_err() {
                                return;
                            }
                        }
                    }
                }
                _ => return,
            }
        }
    });
    Ok(rx)
}

/// Send one request to `peer` over the agent ALPN.
pub async fn request(
    endpoint: &Endpoint,
    peer: impl Into<EndpointAddr>,
    req: &AgentRequest,
) -> Result<AgentResponse, Box<dyn std::error::Error + Send + Sync>> {
    tokio::time::timeout(IO_TIMEOUT, async {
        let conn = endpoint.connect(peer, AGENT_ALPN).await?;
        let (mut send, mut recv) = conn.open_bi().await?;
        send.write_all(&serde_json::to_vec(req)?).await?;
        send.finish()?;
        let body = recv.read_to_end(MAX_RESPONSE_BYTES).await?;
        conn.close(0u32.into(), b"done");
        Ok::<_, Box<dyn std::error::Error + Send + Sync>>(serde_json::from_slice(&body)?)
    })
    .await
    .map_err(|_| -> Box<dyn std::error::Error + Send + Sync> { "agent request timed out".into() })?
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agents::graph::make_record;
    use crate::history::RecordKind;
    use crate::agents::records::*;
    use crate::config::Config;
    use crate::identity::Keypair;
    use crate::state::State;
    use iroh::endpoint::presets;
    use iroh::protocol::Router;
    use iroh::RelayMode;

    async fn endpoint(alpns: Vec<Vec<u8>>) -> Endpoint {
        Endpoint::builder(presets::Minimal)
            .relay_mode(RelayMode::Disabled)
            .alpns(alpns)
            .bind()
            .await
            .unwrap()
    }

    async fn history_with(records: Vec<UpdateRecord>, kp: &Keypair) -> Arc<RwLock<UpdateHistory>> {
        let dir = std::env::temp_dir().join(format!("slk-proto-{}", rand::random::<u64>()));
        let cfg: Config = toml::from_str(&format!(
            r#"
            [federation]
            id = "t"
            name = "t"
            [node]
            id = "x"
            data_dir = "{}"
            [network]
            host = "127.0.0.1"
            api_port = 0
            ws_port = 0
            p2p_port = 0
            [logging]
            level = "info"
            "#,
            dir.display()
        ))
        .expect("test config parses");
        let state = Arc::new(RwLock::new(State::new(&cfg.node.data_dir).unwrap()));
        let mut h = UpdateHistory::new(cfg, state, kp.node_id()).await.unwrap();
        for r in records {
            h.record_update(r).unwrap();
        }
        Arc::new(RwLock::new(h))
    }

    fn signed_candidate(kp: &Keypair, n: u8) -> UpdateRecord {
        let body = Candidate {
            program: "a".repeat(64),
            diff_hash: format!("{:064x}", n),
            parents: vec![],
            agenda_item: None,
        };
        let mut r = make_record(kp.node_id().as_str(), &"0".repeat(64), KIND_CANDIDATE, &body);
        r.sign(kp);
        r
    }

    #[tokio::test]
    async fn serves_hello_and_records_over_second_alpn_alongside_gossip() {
        let kp = Keypair::generate();
        let recs = vec![signed_candidate(&kp, 1), signed_candidate(&kp, 2)];
        let hist = history_with(recs.clone(), &kp).await;

        let server = endpoint(vec![iroh_gossip::ALPN.to_vec(), AGENT_ALPN.to_vec()]).await;
        let gossip = iroh_gossip::net::Gossip::builder().spawn(server.clone());
        let router = Router::builder(server.clone())
            .accept(iroh_gossip::ALPN, gossip)
            .accept(AGENT_ALPN, AgentProtocol::new(hist, kp.node_id().to_string(), None))
            .spawn();

        let client = endpoint(vec![]).await;
        let addr = server.addr();

        match request(&client, addr.clone(), &AgentRequest::Hello).await.unwrap() {
            AgentResponse::Hello { node_id, version } => {
                assert_eq!(node_id, kp.node_id().to_string());
                assert_eq!(version, 1);
            }
            other => panic!("unexpected {other:?}"),
        }
        match request(&client, addr.clone(), &AgentRequest::Records { kind: KIND_CANDIDATE.into(), limit: 10 }).await.unwrap() {
            AgentResponse::Records { records } => {
                assert_eq!(records.len(), 2);
                assert!(records.iter().all(|r| r.verify().is_ok()));
            }
            other => panic!("unexpected {other:?}"),
        }
        // limit honoured, unknown kind refused
        match request(&client, addr.clone(), &AgentRequest::Records { kind: KIND_CANDIDATE.into(), limit: 1 }).await.unwrap() {
            AgentResponse::Records { records } => assert_eq!(records.len(), 1),
            other => panic!("unexpected {other:?}"),
        }
        assert!(matches!(
            request(&client, addr.clone(), &AgentRequest::Records { kind: "bogus".into(), limit: 1 }).await.unwrap(),
            AgentResponse::Error { .. }
        ));
        // The gossip ALPN is still routed on the same endpoint.
        let g = client.connect(addr, iroh_gossip::ALPN).await;
        assert!(g.is_ok(), "gossip ALPN must still be accepted");
        router.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn unauthorized_peer_is_refused() {
        let kp = Keypair::generate();
        let hist = history_with(vec![signed_candidate(&kp, 1)], &kp).await;
        let server = endpoint(vec![AGENT_ALPN.to_vec()]).await;
        let router = Router::builder(server.clone())
            .accept(AGENT_ALPN, AgentProtocol::new(hist, kp.node_id().to_string(), Some(vec!["someone-else".into()])))
            .spawn();
        let client = endpoint(vec![]).await;
        let r = request(&client, server.addr(), &AgentRequest::Hello).await;
        assert!(r.is_err(), "peer not on allowed_peers must not get a response");
        router.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn garbage_request_gets_error_not_crash() {
        let kp = Keypair::generate();
        let hist = history_with(vec![], &kp).await;
        let server = endpoint(vec![AGENT_ALPN.to_vec()]).await;
        let router = Router::builder(server.clone())
            .accept(AGENT_ALPN, AgentProtocol::new(hist, kp.node_id().to_string(), None))
            .spawn();
        let client = endpoint(vec![]).await;
        let conn = client.connect(server.addr(), AGENT_ALPN).await.unwrap();
        let (mut s, mut r) = conn.open_bi().await.unwrap();
        s.write_all(b"{not json").await.unwrap();
        s.finish().unwrap();
        let out = r.read_to_end(1 << 20).await.unwrap();
        let resp: AgentResponse = serde_json::from_slice(&out).unwrap();
        assert!(matches!(resp, AgentResponse::Error { .. }));
        router.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn two_nodes_fetching_the_same_evidence_compute_identical_lanes() {
        use crate::agents::graph::EvidenceGraph;
        // Node A holds a research program authored by three nodes.
        let (owner, ev1, ev2) = (Keypair::generate(), Keypair::generate(), Keypair::generate());
        let mut all = vec![];
        let emit = |all: &mut Vec<UpdateRecord>, kp: &Keypair, kind: &str, body: &dyn erased::Ser| -> String {
            let id = kp.node_id().to_string();
            let prev = all.iter().rev().find(|r| r.node_id == id).map(|r| r.hash.clone()).unwrap_or("0".repeat(64));
            let mut r = UpdateRecord {
                node_id: id, prev_hash: prev,
                kind: RecordKind::Provenance { kind: kind.into(), body: body.to_value() },
                signature: String::new(), hash: String::new(), public_key: String::new(),
            };
            r.sign(kp);
            let h = r.hash.clone();
            all.push(r);
            h
        };
        let prog = emit(&mut all, &owner, KIND_OBJECTIVE, &Objective {
            goal: "g".into(), metric: "acc".into(), direction: Direction::Maximize,
            maturity: MaturityRules { min_evaluators: 2, frontier_size: 2, gem_threshold: 0.9 } });
        for (i, (a, b)) in [(0.6, 0.7), (0.95, 0.92)].iter().enumerate() {
            let c = emit(&mut all, &owner, KIND_CANDIDATE, &Candidate { program: prog.clone(), diff_hash: format!("{:064x}", i), parents: vec![], agenda_item: None });
            let run = emit(&mut all, &owner, KIND_RUN, &Run { candidate: c.clone(), container_hash: "c".repeat(64), input_hashes: vec![], seed: i as u64 });
            for (kp, v) in [(&ev1, a), (&ev2, b)] {
                emit(&mut all, kp, KIND_FINDING, &Finding { run: run.clone(), candidate: c.clone(), metric: "acc".into(), value: *v,
                    evaluator_code_hash: "e".repeat(64), data_merkle_root: "d".repeat(64) });
            }
        }
        let local = EvidenceGraph::from_records(&all);
        let want = local.lanes(&prog).unwrap();
        assert_eq!(want.gems.len(), 1);

        // Serve every record from node A (grouped by author, as the log is).
        let kp_a = Keypair::generate();
        let hist = history_with(vec![], &kp_a).await;
        {
            let mut h = hist.write().await;
            for r in &all {
                h.record_update(r.clone()).unwrap();
            }
        }
        let server = endpoint(vec![AGENT_ALPN.to_vec()]).await;
        let router = Router::builder(server.clone())
            .accept(AGENT_ALPN, AgentProtocol::new(hist, kp_a.node_id().to_string(), None))
            .spawn();

        // Node B pulls each research kind over QUIC and builds its own graph.
        let client = endpoint(vec![]).await;
        let mut fetched = vec![];
        for kind in [KIND_OBJECTIVE, KIND_CANDIDATE, KIND_RUN, KIND_FINDING, KIND_DECISION] {
            match request(&client, server.addr(), &AgentRequest::Records { kind: kind.into(), limit: 1000 }).await.unwrap() {
                AgentResponse::Records { records } => fetched.extend(records),
                other => panic!("unexpected {other:?}"),
            }
        }
        assert_eq!(fetched.len(), all.len());
        let remote = EvidenceGraph::from_records(&fetched);
        assert_eq!(remote.lanes(&prog).unwrap(), want);
        assert_eq!(remote.evidence_root(&prog), local.evidence_root(&prog));
        router.shutdown().await.unwrap();
    }

    // ---- A2A over the agent ALPN (M2) ----

    use crate::agents::a2a::{Part, Role, TaskState};
    use crate::agents::service::tests::{make_service, tmpdir, TestSkill};
    use std::time::Duration;

    fn msg(id: &str, skill: &str, text: &str) -> Message {
        Message { message_id: id.into(), role: Role::User, parts: vec![Part::text(text)], context_id: None, task_id: None, metadata: Some(serde_json::json!({"skill": skill})), extensions: vec![], reference_task_ids: vec![] }
    }

    async fn unwrap_task(r: AgentResponse) -> Task {
        match r { AgentResponse::Task { task } => task, other => panic!("expected task, got {other:?}") }
    }

    async fn serve_agent(dir: &str, kp: Keypair, secret: iroh::SecretKey, delay_ms: u64) -> (Router, Endpoint, Arc<AgentService>) {
        let svc = make_service(dir, kp.clone(), |c| c.task_timeout_secs = 120).await;
        let mut skill = TestSkill::new("shout");
        skill.delay_ms = delay_ms;
        svc.register_skill(Arc::new(skill));
        let ep = Endpoint::builder(presets::Minimal).secret_key(secret).relay_mode(RelayMode::Disabled).alpns(vec![AGENT_ALPN.to_vec()]).bind().await.unwrap();
        let slot: AgentSlot = Default::default();
        slot.set(svc.clone()).ok();
        let proto = AgentProtocol::new(svc.history.clone(), kp.node_id().to_string(), None).with_agent_slot(slot);
        let router = Router::builder(ep.clone()).accept(AGENT_ALPN, proto).spawn();
        (router, ep, svc)
    }

    #[tokio::test]
    async fn a2a_send_get_list_cancel_card_and_idempotency_over_quic() {
        let kp = Keypair::generate();
        let (router, server, svc) = serve_agent(&tmpdir("q1"), kp, iroh::SecretKey::generate(), 0).await;
        svc.ensure_delegation().await;
        let client = endpoint(vec![]).await;
        let addr = server.addr();

        let t = unwrap_task(request(&client, addr.clone(), &AgentRequest::SendMessage { message: msg("m1", "shout", "hi there") }).await.unwrap()).await;
        let again = unwrap_task(request(&client, addr.clone(), &AgentRequest::SendMessage { message: msg("m1", "shout", "hi there") }).await.unwrap()).await;
        assert_eq!(t.id, again.id, "retry must not create a second task");
        // sender is the authenticated QUIC peer, not a claim made in the message
        assert_eq!(t.metadata.as_ref().unwrap()["sender"], format!("endpoint:{}", client.id()));

        let mut done = None;
        for _ in 0..100 {
            let g = unwrap_task(request(&client, addr.clone(), &AgentRequest::GetTask { id: t.id.clone() }).await.unwrap()).await;
            if g.status.state == TaskState::Completed { done = Some(g); break; }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        assert_eq!(done.unwrap().artifacts[0].parts[0].text.as_deref(), Some("HI THERE"));

        match request(&client, addr.clone(), &AgentRequest::ListTasks).await.unwrap() {
            AgentResponse::Tasks { tasks } => assert_eq!(tasks.len(), 1),
            o => panic!("{o:?}"),
        }
        match request(&client, addr.clone(), &AgentRequest::GetCard).await.unwrap() {
            AgentResponse::Card { card } => {
                assert!(card.verified_key().is_some(), "card must be signed");
                assert_eq!(card.skills[0].id, "shout");
            }
            o => panic!("{o:?}"),
        }
        assert!(matches!(request(&client, addr.clone(), &AgentRequest::GetTask { id: "nope".into() }).await.unwrap(), AgentResponse::Error { .. }));
        assert!(matches!(request(&client, addr, &AgentRequest::CancelTask { id: t.id }).await.unwrap(), AgentResponse::Error { .. }), "completed task cannot be cancelled");
        router.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn task_updates_stream_to_the_client_until_final() {
        let kp = Keypair::generate();
        let (router, server, _svc) = serve_agent(&tmpdir("q2"), kp, iroh::SecretKey::generate(), 300).await;
        let client = endpoint(vec![]).await;
        let t = unwrap_task(request(&client, server.addr(), &AgentRequest::SendMessage { message: msg("s1", "shout", "stream me") }).await.unwrap()).await;
        let mut rx = subscribe(&client, server.addr(), &t.id).await.unwrap();
        let mut states = vec![];
        let mut artifacts = 0;
        while let Some(ev) = tokio::time::timeout(Duration::from_secs(10), rx.recv()).await.expect("stream stalled") {
            match ev {
                WireEvent::Status { status, is_final, .. } => { states.push(status.state); if is_final { break; } }
                WireEvent::Artifact { .. } => artifacts += 1,
                WireEvent::Error { message } => panic!("{message}"),
            }
        }
        assert_eq!(states.last(), Some(&TaskState::Completed));
        assert_eq!(artifacts, 1);
        // subscribing to a finished task yields its final state immediately
        let mut rx = subscribe(&client, server.addr(), &t.id).await.unwrap();
        let mut got_final = false;
        while let Some(ev) = rx.recv().await { if let WireEvent::Status { is_final: true, .. } = ev { got_final = true; } }
        assert!(got_final);
        let mut rx = subscribe(&client, server.addr(), "missing").await.unwrap();
        assert!(matches!(rx.recv().await, Some(WireEvent::Error { .. })));
        router.shutdown().await.unwrap();
    }

    /// M2 done-when: a long-running task survives a provider restart and still completes.
    #[test]
    fn long_running_task_survives_a_provider_restart_and_still_completes() {
        let dir = tmpdir("q3");
        let kp = Keypair::generate();
        let secret = iroh::SecretKey::generate();
        let client_rt = tokio::runtime::Runtime::new().unwrap();
        let client = client_rt.block_on(endpoint(vec![]));

        // Provider, first life: accepts the task, which is far from done when it dies.
        let rt1 = tokio::runtime::Runtime::new().unwrap();
        let (addr1, task_id) = rt1.block_on(async {
            let (router, ep, _svc) = serve_agent(&dir, kp.clone(), secret.clone(), 60_000).await;
            std::mem::forget(router); // dies with the runtime, like a crashed process
            let addr = ep.addr();
            let t = unwrap_task(request(&ep_client().await, addr.clone(), &AgentRequest::SendMessage { message: msg("long", "shout", "survive") }).await.unwrap()).await;
            (addr, t.id)
        });
        let _ = addr1;
        rt1.shutdown_background();

        // Second life: same data dir and identity; the skill is quick now. recover() resumes the task.
        let rt2 = tokio::runtime::Runtime::new().unwrap();
        let addr2 = rt2.block_on(async {
            let (router, ep, svc) = serve_agent(&dir, kp.clone(), secret.clone(), 0).await;
            std::mem::forget(router);
            assert_eq!(svc.tasks.get(&task_id).unwrap().status.state, TaskState::Working);
            assert_eq!(svc.recover(), 1);
            ep.addr()
        });
        let done = client_rt.block_on(async {
            for _ in 0..200 {
                let g = unwrap_task(request(&client, addr2.clone(), &AgentRequest::GetTask { id: task_id.clone() }).await.unwrap()).await;
                if g.status.state == TaskState::Completed { return Some(g); }
                tokio::time::sleep(Duration::from_millis(25)).await;
            }
            None
        });
        assert_eq!(done.expect("task must complete after the restart").artifacts[0].parts[0].text.as_deref(), Some("SURVIVE"));
        rt2.shutdown_background();
    }

    async fn ep_client() -> Endpoint { endpoint(vec![]).await }

    /// Store-and-forward: a message to an offline agent waits in the durable mailbox
    /// and is delivered (once) when the agent comes back.
    #[tokio::test]
    async fn offline_agent_gets_its_mail_when_it_returns_and_redelivery_is_idempotent() {
        use iroh::address_lookup::MemoryLookup;
        let kp = Keypair::generate();
        let secret = iroh::SecretKey::generate();
        let provider_id = secret.public();
        let lookup = MemoryLookup::new();
        let sender_ep = Endpoint::builder(presets::Minimal).relay_mode(RelayMode::Disabled).address_lookup(lookup.clone()).bind().await.unwrap();
        let sender = make_service(&tmpdir("mail-s"), Keypair::generate(), |_| {}).await;
        let to = format!("endpoint:{provider_id}");

        // Provider is down: nothing listens, no address known.
        sender.send_to_peer(&sender_ep, &to, msg("mail-1", "shout", "wake up")).await.unwrap();
        assert_eq!(sender.mailbox.pending_for(&to).len(), 1);
        assert_eq!(sender.mailbox.pending_for(&to)[0].attempts, 1);
        assert_eq!(sender.deliver_pending(&sender_ep).await, 0, "backoff: not due yet");
        assert_eq!(sender.mailbox.pending_for(&to)[0].attempts, 1);

        // Provider comes up (same identity). Make the parcel due and sweep.
        let (router, ep, psvc) = serve_agent(&tmpdir("mail-p"), kp, secret, 0).await;
        lookup.add_endpoint_info(ep.addr());
        let dir = sender.mailbox.pending_for(&to);
        assert_eq!(dir.len(), 1);
        // fast-forward the backoff by re-opening the mailbox view as "due"
        let mut parcel = dir[0].clone();
        parcel.next_attempt = 0;
        sender.mailbox.ack(&to, "mail-1").unwrap();
        sender.mailbox.enqueue(&to, None, parcel.message.clone()).unwrap();
        assert_eq!(sender.deliver_pending(&sender_ep).await, 1);
        assert!(sender.mailbox.pending_for(&to).is_empty(), "acked after delivery");
        let remote_task = sender.mailbox.sent_task(&to, "mail-1").expect("remote task id remembered");
        let mut done = None;
        for _ in 0..100 {
            let t = psvc.tasks.get(&remote_task).unwrap();
            if t.status.state == TaskState::Completed { done = Some(t); break; }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        assert_eq!(done.unwrap().artifacts[0].parts[0].text.as_deref(), Some("WAKE UP"));
        // The ack was "lost": the same parcel is delivered again and creates no second task.
        sender.mailbox.enqueue(&to, None, msg("mail-1", "shout", "wake up")).unwrap();
        assert_eq!(sender.deliver_pending(&sender_ep).await, 1);
        assert_eq!(psvc.tasks.list().len(), 1);
        router.shutdown().await.unwrap();
    }

    mod erased {
        pub trait Ser { fn to_value(&self) -> serde_json::Value; }
        impl<T: serde::Serialize> Ser for T {
            fn to_value(&self) -> serde_json::Value { serde_json::to_value(self).unwrap() }
        }
    }
}
