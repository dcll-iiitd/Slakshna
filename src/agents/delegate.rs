//! Cross-node delegation (M3, M4): this node hands a task to agents elsewhere in the
//! federation and decides what to believe, with no coordinator.
//!
//! 1. Find verified agents offering the skill (directory), excluding ourselves.
//! 2. Bid: each agent's announced price and our learned reputation of it rank the
//!    candidates (contract-net award); the best `redundancy` are awarded the task.
//! 3. Each awarded agent runs it as a normal A2A task over `slakshna/agent/1`.
//! 4. Their answers are pooled, weighted by reputation; a quorum is required.
//! 5. Reputation is updated (agreement with the consensus, or ground truth on a
//!    probe), every hop and verdict is written as signed provenance
//!    (`a2a.task`, `a2a.delegation`, `a2a.acceptance`), and the reputation book is saved.
//!
//! Vote-only reputation can be captured by cheap colluders (see testbed.rs); callers
//! that can supply a task with a known answer pass it as `probe_truth`.

use crate::agents::a2a::{Message, Part, Role, TaskState};
use crate::agents::coordination::{pool, rank_bids, Bid, TaskSpec};
use crate::agents::protocol::{request, subscribe, AgentRequest, AgentResponse, WireEvent};
use crate::agents::reputation::{AgentKey, Outcome, ReputationBook, Role as RepRole};
use crate::agents::service::AgentService;
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct DelegateRequest {
    pub skill: String,
    /// Sent to each agent as one data part.
    pub input: Value,
    pub redundancy: usize,
    pub max_price: u64,
    pub timeout: Duration,
    pub quorum: f64,
    /// If the correct answer is known (a probe), reputation is updated from it instead of from agreement.
    pub probe_truth: Option<Value>,
}

impl DelegateRequest {
    pub fn new(skill: &str, input: Value) -> Self {
        DelegateRequest { skill: skill.into(), input, redundancy: 3, max_price: u64::MAX / 4, timeout: Duration::from_secs(60), quorum: 0.5, probe_truth: None }
    }
}

#[derive(Debug, Clone)]
pub struct AnswerRow {
    pub node_id: String,
    pub price: u64,
    pub reputation_before: f64,
    pub state: Option<TaskState>,
    pub remote_task: Option<String>,
    pub answer: Option<Value>,
    pub agreed: bool,
}

#[derive(Debug, Clone)]
pub struct Delegation {
    pub root_task: String,
    pub consensus: Option<Value>,
    pub share: f64,
    pub rows: Vec<AnswerRow>,
    pub cost: u64,
}

#[derive(Debug, thiserror::Error)]
pub enum DelegateError {
    #[error("no verified agent offers skill '{0}'")]
    NoAgents(String),
    #[error("no agent could be reached or priced within budget")]
    NoBids,
}

pub struct Delegator {
    svc: Arc<AgentService>,
    endpoint: iroh::Endpoint,
    book: Mutex<ReputationBook>,
    book_path: Option<PathBuf>,
}

fn answer_of(a: &crate::agents::a2a::Artifact) -> Option<Value> {
    a.parts.first().and_then(|p| p.data.clone().or_else(|| p.text.clone().map(Value::String)))
}

impl Delegator {
    pub fn new(svc: Arc<AgentService>, endpoint: iroh::Endpoint, data_dir: Option<PathBuf>) -> std::io::Result<Arc<Self>> {
        let book_path = data_dir.map(|d| d.join("reputation.json"));
        let book = match &book_path {
            Some(p) => ReputationBook::load(p)?,
            None => ReputationBook::default(),
        };
        Ok(Arc::new(Delegator { svc, endpoint, book: Mutex::new(book), book_path }))
    }

    /// An audit of a finding by `evaluator_node` found it consistent (or not): feeds that node's
    /// evaluator reputation.
    pub fn record_evaluator_audit(&self, evaluator_node: &str, consistent: bool) {
        let mut book = self.book.lock().unwrap();
        book.record_outcome(Outcome { agent: (evaluator_node.to_string(), "node".to_string()), role: RepRole::Evaluator, success: consistent });
        if let Some(p) = &self.book_path {
            let _ = book.save(p);
        }
    }

    #[allow(dead_code)] // library surface: used by tests and integrators, or pending the settlement-chain decision
    pub fn evaluator_reputation(&self, evaluator_node: &str) -> f64 {
        self.book.lock().unwrap().reputation(&(evaluator_node.to_string(), "node".to_string()), RepRole::Evaluator)
    }

    pub fn reputation(&self, node: &str, agent_key: &str) -> f64 {
        self.book.lock().unwrap().reputation(&(node.to_string(), agent_key.to_string()), RepRole::Executor)
    }

    pub async fn delegate(&self, req: DelegateRequest) -> Result<Delegation, DelegateError> {
        let candidates: Vec<_> = self.svc.search_agents(&req.skill).into_iter().filter(|(n, _, c)| *n != self.svc.node_id && c.slakshna_endpoint_id.is_some()).collect();
        if candidates.is_empty() {
            return Err(DelegateError::NoAgents(req.skill));
        }
        let spec = TaskSpec { id: "delegation".into(), skill: req.skill.clone(), input: req.input.clone(), max_price: req.max_price, deadline_secs: req.timeout.as_secs().max(1) };
        let bids: Vec<Bid> = {
            let book = self.book.lock().unwrap();
            candidates
                .iter()
                .map(|(n, k, c)| Bid {
                    bidder: n.clone(),
                    price: c.slakshna_prices.get(&req.skill).copied().unwrap_or(0),
                    eta_secs: 0,
                    reputation: book.reputation(&(n.clone(), k.clone()), RepRole::Executor),
                })
                .collect()
        };
        let ranked = rank_bids(&spec, &bids);
        if ranked.is_empty() {
            return Err(DelegateError::NoBids);
        }
        let winners: Vec<_> = ranked.iter().take(req.redundancy.max(1)).filter_map(|b| candidates.iter().find(|(n, _, _)| *n == b.bidder).map(|c| (b.clone(), c.clone()))).collect();

        // Our own record of the delegated job: the root of the chain.
        let uuid = format!("{:032x}", rand::random::<u128>());
        let root_msg = Message { message_id: format!("delegation-{uuid}"), role: Role::User, parts: vec![Part::data(req.input.clone())], context_id: None, task_id: None, metadata: Some(json!({"skill": req.skill})), extensions: vec![], reference_task_ids: vec![] };
        let root = self.svc.tasks.submit("local", root_msg, Some(&req.skill)).map(|(t, _)| t.id).unwrap_or_else(|_| format!("delegation-{uuid}"));
        let _ = self.svc.tasks.transition(&root, TaskState::Working, None);

        // Run the awarded agents concurrently.
        let futs = winners.iter().map(|(bid, (node, key, card))| {
            let msg = Message { message_id: format!("{uuid}-{node}"), role: Role::User, parts: vec![Part::data(req.input.clone())], context_id: None, task_id: None, metadata: Some(json!({"skill": req.skill})), extensions: vec![], reference_task_ids: vec![root.clone()] };
            let ep = self.endpoint.clone();
            let eid = card.slakshna_endpoint_id.clone().unwrap();
            let timeout = req.timeout;
            let (node, key, price, rep) = (node.clone(), key.clone(), bid.price, bid.reputation);
            async move {
                let run = async {
                    let peer: iroh::EndpointId = eid.parse().ok()?;
                    let task = match request(&ep, peer, &AgentRequest::SendMessage { message: msg }).await.ok()? {
                        AgentResponse::Task { task } => task,
                        _ => return None,
                    };
                    let mut rx = subscribe(&ep, peer, &task.id).await.ok()?;
                    let mut art = None;
                    let mut state = None;
                    while let Some(ev) = rx.recv().await {
                        match ev {
                            WireEvent::Artifact { artifact, .. } => art = Some(artifact),
                            WireEvent::Status { status, is_final, .. } => {
                                state = Some(status.state);
                                if is_final {
                                    break;
                                }
                            }
                            WireEvent::Error { .. } => break,
                        }
                    }
                    Some((task.id, state, art))
                };
                let got = tokio::time::timeout(timeout, run).await.ok().flatten();
                (node, key, price, rep, got)
            }
        });
        let results = futures::future::join_all(futs).await;

        // Record each hop (who asked whom, under what scope) and gather answers.
        let mut rows: Vec<(AnswerRow, Option<crate::agents::a2a::Artifact>, AgentKey)> = vec![];
        for (node, key, price, rep, got) in results {
            let (remote_task, state, art) = match got {
                Some((t, s, a)) => (Some(t), s, a),
                None => (None, None, None),
            };
            if let Some(t) = &remote_task {
                self.svc.record_delegation(&root, t, &[&req.skill]);
            }
            let answer = if state == Some(TaskState::Completed) { art.as_ref().and_then(answer_of) } else { None };
            rows.push((AnswerRow { node_id: node.clone(), price, reputation_before: rep, state, remote_task, answer, agreed: false }, art, (node, key)));
        }

        // Pool the answers, weighted by reputation.
        let votes: Vec<(String, String)> = rows.iter().filter_map(|(r, _, _)| r.answer.as_ref().map(|a| (r.node_id.clone(), a.to_string()))).collect();
        let weight = |id: &str| rows.iter().find(|(r, _, _)| r.node_id == id).map(|(r, _, _)| r.reputation_before).unwrap_or(0.0);
        let pooled = pool(&votes, &weight, req.quorum);
        let consensus: Option<Value> = pooled.as_ref().and_then(|p| serde_json::from_str(&p.answer).ok());

        // Learn and record verdicts.
        let truth = req.probe_truth.as_ref().map(|t| t.to_string());
        {
            let mut book = self.book.lock().unwrap();
            for (row, art, key) in rows.iter_mut() {
                let mine = row.answer.as_ref().map(|a| a.to_string());
                row.agreed = match (&truth, &pooled, &mine) {
                    (Some(t), _, Some(m)) => m == t,
                    (None, Some(p), Some(m)) => *m == p.answer,
                    _ => false,
                };
                book.record_outcome(Outcome { agent: key.clone(), role: RepRole::Executor, success: row.agreed });
                if let (Some(a), Some(t)) = (art.as_ref(), row.remote_task.as_ref()) {
                    self.svc.record_acceptance(t, a, row.agreed);
                }
            }
            if let Some(p) = &self.book_path {
                let _ = book.save(p);
            }
        }
        let ok = consensus.is_some();
        let _ = self.svc.tasks.transition(&root, if ok { TaskState::Completed } else { TaskState::Failed }, None);
        let cost = rows.iter().map(|(r, _, _)| r.price).sum();
        Ok(Delegation { root_task: root, share: pooled.map(|p| p.share).unwrap_or(0.0), consensus, rows: rows.into_iter().map(|(r, _, _)| r).collect(), cost })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agents::a2a::{AgentSkill, Artifact, Task};
    use crate::agents::protocol::{AgentProtocol, AgentSlot, AGENT_ALPN};
    use crate::agents::records::{KIND_A2A_ACCEPTANCE, KIND_A2A_DELEGATION};
    use crate::agents::service::tests::{make_service, tmpdir};
    use crate::agents::service::{Skill, SkillCtx, SkillError};
    use crate::history::RecordKind;
    use crate::identity::Keypair;
    use async_trait::async_trait;
    use iroh::address_lookup::MemoryLookup;
    use iroh::endpoint::presets;
    use iroh::protocol::Router;
    use iroh::RelayMode;

    struct Doubler(bool); // true = honest
    #[async_trait]
    impl Skill for Doubler {
        fn descriptor(&self) -> AgentSkill { AgentSkill { id: "double".into(), name: "double".into(), description: "x2".into(), tags: vec![], examples: vec![], input_modes: vec![], output_modes: vec![] } }
        async fn run(&self, _: &SkillCtx, _: &Task, input: &Message) -> Result<Vec<Artifact>, SkillError> {
            let n = crate::agents::skills::input_json(input)?["n"].as_i64().unwrap_or(0);
            let out = if self.0 { n * 2 } else { n * 2 + 1 };
            Ok(vec![Artifact { artifact_id: "r".into(), name: None, description: None, parts: vec![Part::data(json!({"result": out}))], metadata: None }])
        }
    }

    struct Node { svc: Arc<AgentService>, ep: iroh::Endpoint, _router: Router }

    async fn node(lookup: &MemoryLookup, honest: Option<bool>, price: u64) -> Node {
        let kp = Keypair::generate();
        let dir = tmpdir("deleg");
        let svc = make_service(&dir, kp.clone(), |c| { c.prices.insert("double".into(), price); }).await;
        if let Some(h) = honest { svc.register_skill(Arc::new(Doubler(h))); }
        let ep = iroh::Endpoint::builder(presets::Minimal).relay_mode(RelayMode::Disabled).alpns(vec![AGENT_ALPN.to_vec()]).address_lookup(lookup.clone()).bind().await.unwrap();
        svc.set_endpoint_id(&ep.id().to_string());
        svc.ensure_delegation().await;
        svc.publish_card().await;
        let slot: AgentSlot = Default::default();
        let _ = slot.set(svc.clone());
        let router = Router::builder(ep.clone()).accept(AGENT_ALPN, AgentProtocol::new(svc.history.clone(), kp.node_id().to_string(), None).with_agent_slot(slot)).spawn();
        lookup.add_endpoint_info(ep.addr());
        Node { svc, ep, _router: router }
    }

    #[tokio::test]
    async fn a_node_delegates_across_the_federation_pools_answers_and_learns_whom_to_trust() {
        let lookup = MemoryLookup::new();
        let me = node(&lookup, None, 0).await;
        // three workers: two honest (cheap-ish), one liar that undercuts
        let workers = vec![node(&lookup, Some(true), 12).await, node(&lookup, Some(true), 11).await, node(&lookup, Some(false), 9).await];
        tokio::time::sleep(Duration::from_millis(200)).await; // let the author loops drain
        let peers: Vec<iroh::EndpointId> = workers.iter().map(|w| w.ep.id()).collect();
        me.svc.refresh_directory(&me.ep, peers.clone()).await;
        assert_eq!(me.svc.search_agents("double").len(), 3, "directory lists the verified workers");

        let book_dir = PathBuf::from(tmpdir("book"));
        let d = Delegator::new(me.svc.clone(), me.ep.clone(), Some(book_dir.clone())).unwrap();
        let mut req = DelegateRequest::new("double", json!({"n": 21}));
        req.probe_truth = None;
        let r = d.delegate(req).await.unwrap();
        assert_eq!(r.consensus, Some(json!({"result": 42})), "two honest answers outvote the liar");
        assert_eq!(r.rows.len(), 3);
        assert_eq!(r.rows.iter().filter(|x| x.agreed).count(), 2);
        assert_eq!(r.cost, 12 + 11 + 9);
        let liar = r.rows.iter().find(|x| x.price == 9).unwrap();
        assert!(!liar.agreed && liar.answer == Some(json!({"result": 43})));

        // Reputation moved the right way and was saved.
        let rep = |price: u64| { let w = workers.iter().find(|w| w.svc.card().slakshna_prices["double"] == price).unwrap(); d.reputation(&w.svc.node_id, &w.svc.agent_public_key()) };
        assert!(rep(11) > 0.5 && rep(12) > 0.5 && rep(9) < 0.5, "honest up, liar down: {} {} {}", rep(11), rep(12), rep(9));
        assert!(book_dir.join("reputation.json").exists());
        let reloaded = Delegator::new(me.svc.clone(), me.ep.clone(), Some(book_dir)).unwrap();
        assert!((reloaded.reputation(&workers[2].svc.node_id, &workers[2].svc.agent_public_key()) - rep(9)).abs() < 1e-9);

        // Provenance: three delegation hops and three verdicts, signed by us, forming a verifiable chain.
        tokio::time::sleep(Duration::from_millis(300)).await;
        let h = me.svc.history.read().await;
        assert_eq!(h.provenance(KIND_A2A_DELEGATION).len(), 3);
        let verdicts: Vec<bool> = h.provenance(KIND_A2A_ACCEPTANCE).iter().filter_map(|x| match &x.kind { RecordKind::Provenance { body, .. } => body["accepted"].as_bool(), _ => None }).collect();
        assert_eq!((verdicts.iter().filter(|v| **v).count(), verdicts.iter().filter(|v| !**v).count()), (2, 1));
        let scope: std::collections::BTreeSet<String> = ["double".to_string()].into();
        // remote task records live in the workers' logs; pull them so the chain can name each delegate
        drop(h);
        for w in &workers { me.svc.sync_from_peer(&me.ep, w.ep.id()).await; }
        let h = me.svc.history.read().await;
        let recs2: Vec<&crate::history::UpdateRecord> = h.peer_updates.values().flatten().collect();
        let hops = crate::agents::a2a::verify_delegation_chain(recs2, &r.root_task, &scope).unwrap();
        assert_eq!(hops.len(), 3);
        assert!(hops.iter().all(|x| x.delegator == me.svc.node_id && x.delegate.is_some()));
    }

    #[tokio::test]
    async fn the_liar_is_dropped_from_later_awards_once_known() {
        let lookup = MemoryLookup::new();
        let me = node(&lookup, None, 0).await;
        // four workers; the liar is the cheapest so it is awarded first
        let workers = vec![node(&lookup, Some(true), 12).await, node(&lookup, Some(true), 11).await, node(&lookup, Some(true), 10).await, node(&lookup, Some(false), 9).await];
        tokio::time::sleep(Duration::from_millis(200)).await;
        me.svc.refresh_directory(&me.ep, workers.iter().map(|w| w.ep.id()).collect()).await;
        let d = Delegator::new(me.svc.clone(), me.ep.clone(), None).unwrap();
        // First round: liar (9) + two cheapest honest (10, 11) are awarded; with ground truth the liar is exposed.
        let mut first = DelegateRequest::new("double", json!({"n": 5}));
        first.probe_truth = Some(json!({"result": 10}));
        let r1 = d.delegate(first).await.unwrap();
        assert!(r1.rows.iter().any(|x| x.price == 9 && !x.agreed));
        // Second round: the liar's reputation is now lower than the honest agent it displaced, so it is no longer awarded.
        let r2 = d.delegate(DelegateRequest::new("double", json!({"n": 8}))).await.unwrap();
        assert!(r2.rows.iter().all(|x| x.price != 9), "the exposed liar is not awarded again: {:?}", r2.rows.iter().map(|x| x.price).collect::<Vec<_>>());
        assert_eq!(r2.consensus, Some(json!({"result": 16})));
    }

    #[tokio::test]
    async fn errors_when_nobody_offers_the_skill_and_dead_agents_do_not_block_consensus() {
        let lookup = MemoryLookup::new();
        let me = node(&lookup, None, 0).await;
        let d = Delegator::new(me.svc.clone(), me.ep.clone(), None).unwrap();
        assert!(matches!(d.delegate(DelegateRequest::new("double", json!({"n": 1}))).await, Err(DelegateError::NoAgents(_))));
        let w1 = node(&lookup, Some(true), 10).await;
        let w2 = node(&lookup, Some(true), 11).await;
        let w3 = node(&lookup, Some(true), 12).await;
        tokio::time::sleep(Duration::from_millis(200)).await;
        me.svc.refresh_directory(&me.ep, vec![w1.ep.id(), w2.ep.id(), w3.ep.id()]).await;
        // w3 goes away after being listed
        w3._router.shutdown().await.unwrap();
        let mut req = DelegateRequest::new("double", json!({"n": 4}));
        req.timeout = Duration::from_secs(3);
        let r = d.delegate(req).await.unwrap();
        assert_eq!(r.consensus, Some(json!({"result": 8})), "two live honest agents are a quorum of the answers received");
        assert!(r.rows.iter().any(|x| x.answer.is_none()), "the dead agent produced no answer");
    }
}
