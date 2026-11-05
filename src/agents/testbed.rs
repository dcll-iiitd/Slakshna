//! Testbed and evaluation (M7).
//!
//! Runs a population of agents over the real transport (one iroh endpoint per
//! agent, speaking `slakshna/agent/1`) on a benchmark with a known answer, with
//! `k` of them malicious, and compares two ways of getting work done:
//!
//! * **Centralized baseline**: an orchestrator hands each task to one worker
//!   (round robin) and trusts the answer. No verification, no learning.
//! * **Decentralized (the plan's design)**: contract-net (announce -> bid ->
//!   award), redundant execution by the best `redundancy` bidders, trust-weighted
//!   pooling of their answers, and reputation updates from agreement with the
//!   pooled consensus. No coordinator holds state: the requester is just another agent.
//! * **Decentralized + probes**: the same, with known-answer probe tasks mixed in.
//!   Measured finding: without probes, cheap colluders that win the first awards
//!   become the pooled consensus, their reputation *rises* for agreeing with each
//!   other, and the design locks in the lie (0% success at k=2 of 9). Anchoring
//!   reputation to ground truth on probes removes the lock-in.
//!
//! Metrics per run: task success (answer equals ground truth), mean latency,
//! A2A messages per task, and cost (price paid to awarded workers).
//! Adversaries: `Colluding` agents return one shared wrong answer (the hardest
//! case for voting), `Lying` agents return independent wrong answers, `Silent`
//! agents never answer.

use crate::agents::a2a::{AgentSkill, Artifact, Message, Part, Role, Task, TaskState};
use crate::agents::coordination::{pool, rank_bids, Bid, TaskSpec};
use crate::agents::protocol::{request, subscribe, AgentProtocol, AgentRequest, AgentResponse, AgentSlot, WireEvent, AGENT_ALPN};
use crate::agents::reputation::{AgentKey, Outcome, ReputationBook, Role as RepRole};
use crate::agents::service::{AgentService, NoBroadcast, Skill, SkillCtx, SkillError};
use async_trait::async_trait;
use iroh::address_lookup::MemoryLookup;
use iroh::endpoint::presets;
use iroh::protocol::Router;
use iroh::{Endpoint, EndpointAddr, RelayMode};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use serde_json::{json, Value};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Behavior {
    Honest,
    /// Independent wrong answers.
    Lying,
    /// All colluders give the same wrong answer.
    Colluding,
    /// Never answers.
    Silent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Every task to one fixed worker (the best single agent, for reference).
    Single(usize),
    Centralized,
    /// Contract-net + redundancy + pooling; reputation learned from agreement alone.
    Decentralized,
    /// As above, plus probe tasks whose answer the requester knows (a calibration burst
    /// first, then a steady trickle), so reputation is anchored to ground truth.
    DecentralizedProbed,
}

/// One benchmark task: an input every agent receives and the answer it should reach.
#[derive(Debug, Clone)]
pub struct TaskItem {
    pub input: Value,
    pub truth: String,
}

#[derive(Debug, Clone)]
pub struct Solved {
    pub answer: String,
    /// Tokens the agent's model generated/consumed for this answer (0 for non-LLM solvers).
    pub tokens: u64,
    /// Time the agent spent computing the answer.
    pub compute_ms: u64,
}

/// How an honest agent produces an answer.
#[async_trait]
pub trait Solver: Send + Sync {
    async fn solve(&self, input: &Value) -> Result<Solved, String>;
}

/// Sum of `input.numbers`.
pub struct SumSolver;

#[async_trait]
impl Solver for SumSolver {
    async fn solve(&self, input: &Value) -> Result<Solved, String> {
        let nums = input["numbers"].as_array().ok_or("numbers required")?;
        let sum: i64 = nums.iter().map(|n| n.as_i64().ok_or("integers only")).sum::<Result<i64, _>>()?;
        Ok(Solved { answer: sum.to_string(), tokens: 0, compute_ms: 0 })
    }
}

pub fn sum_tasks(n: usize, seed: u64) -> Vec<TaskItem> {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let nums: Vec<i64> = (0..8).map(|_| rng.gen_range(-100..100)).collect();
            TaskItem { truth: nums.iter().sum::<i64>().to_string(), input: json!({"numbers": nums}) }
        })
        .collect()
}

fn input_hash(input: &Value, salt: u64) -> u64 {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(serde_json::to_vec(input).unwrap_or_default());
    h.update(salt.to_le_bytes());
    u64::from_le_bytes(h.finalize()[..8].try_into().unwrap())
}

/// The lie every colluder tells: a function of the task alone, so colluders agree without
/// communicating and without knowing the truth (it is just a number far from any real answer).
fn shared_lie(input: &Value) -> String {
    (1_000_000 + input_hash(input, 0) % 1_000_000).to_string()
}

/// A lie that differs per agent.
fn independent_lie(input: &Value, agent_seed: u64) -> String {
    (1_000_000 + input_hash(input, agent_seed) % 1_000_000).to_string()
}

#[derive(Debug, Clone)]
pub struct Metrics {
    pub mode: Mode,
    pub adversary: Behavior,
    pub malicious: usize,
    pub tasks: usize,
    pub successes: usize,
    /// Per real task: did the final answer equal the truth.
    pub correct: Vec<bool>,
    pub mean_latency_ms: f64,
    pub messages_per_task: f64,
    pub cost_per_task: f64,
    /// Model tokens spent by awarded agents per real task (probes included).
    pub tokens_per_task: f64,
    /// Agent compute time per real task (probes included).
    pub compute_ms_per_task: f64,
}

impl Metrics {
    pub fn success_rate(&self) -> f64 {
        self.successes as f64 / self.tasks as f64
    }
}

struct SolveSkill {
    behavior: Behavior,
    seed: u64,
    solver: Arc<dyn Solver>,
}

#[async_trait]
impl Skill for SolveSkill {
    fn descriptor(&self) -> AgentSkill {
        AgentSkill { id: "solve".into(), name: "solve".into(), description: "solve a benchmark task".into(), tags: vec![], examples: vec![], input_modes: vec![], output_modes: vec![] }
    }

    async fn run(&self, _: &SkillCtx, _task: &Task, input: &Message) -> Result<Vec<Artifact>, SkillError> {
        let req = crate::agents::skills::input_json(input)?;
        let solved = match self.behavior {
            Behavior::Honest => self.solver.solve(&req).await.map_err(SkillError::Failed)?,
            Behavior::Colluding => Solved { answer: shared_lie(&req), tokens: 0, compute_ms: 0 },
            Behavior::Lying => Solved { answer: independent_lie(&req, self.seed), tokens: 0, compute_ms: 0 },
            Behavior::Silent => {
                tokio::time::sleep(Duration::from_secs(3600)).await;
                return Err(SkillError::Failed("unreachable".into()));
            }
        };
        Ok(vec![Artifact { artifact_id: "a".into(), name: None, description: None, parts: vec![Part::data(json!({"answer": solved.answer, "tokens": solved.tokens, "compute_ms": solved.compute_ms}))], metadata: None }])
    }
}

struct BidSkill {
    price: u64,
}

#[async_trait]
impl Skill for BidSkill {
    fn descriptor(&self) -> AgentSkill {
        AgentSkill { id: "bid".into(), name: "bid".into(), description: "quote a price".into(), tags: vec![], examples: vec![], input_modes: vec![], output_modes: vec![] }
    }
    async fn run(&self, _: &SkillCtx, _: &Task, _: &Message) -> Result<Vec<Artifact>, SkillError> {
        Ok(vec![Artifact { artifact_id: "bid".into(), name: None, description: None, parts: vec![Part::data(json!({"price": self.price, "eta_ms": 5}))], metadata: None }])
    }
}

struct Worker {
    id: String,
    addr: EndpointAddr,
    _router: Router,
    _svc: Arc<AgentService>,
    price: u64,
}

pub struct Testbed {
    client: Endpoint,
    workers: Vec<Worker>,
    messages: Arc<AtomicU64>,
    call_timeout: Duration,
    /// Share of the (reputation-weighted) vote the winning answer needs. 0 = plurality (self-consistency
    /// style, right for model answers that are often all different); 0.5 = majority.
    pub quorum: f64,
    /// Known-answer probes issued before the first real task in `DecentralizedProbed` mode.
    pub calibration: usize,
}

fn tmp(tag: &str) -> String {
    let p = std::env::temp_dir().join(format!("slk-testbed-{tag}-{}", rand::random::<u64>()));
    std::fs::create_dir_all(&p).unwrap();
    p.to_string_lossy().into_owned()
}

impl Testbed {
    /// Sum-benchmark population (no model involved).
    pub async fn start(behaviors: &[Behavior]) -> Result<Testbed, Box<dyn std::error::Error + Send + Sync>> {
        let solvers: Vec<Arc<dyn Solver>> = behaviors.iter().map(|_| Arc::new(SumSolver) as Arc<dyn Solver>).collect();
        Self::start_with(behaviors, solvers, Duration::from_millis(800)).await
    }

    /// `behaviors[i]` is worker i's behavior and `solvers[i]` how it answers when honest.
    pub async fn start_with(behaviors: &[Behavior], solvers: Vec<Arc<dyn Solver>>, call_timeout: Duration) -> Result<Testbed, Box<dyn std::error::Error + Send + Sync>> {
        let lookup = MemoryLookup::new();
        let mut workers = vec![];
        for (i, b) in behaviors.iter().enumerate() {
            let kp = crate::identity::Keypair::generate();
            let dir = tmp(&format!("w{i}"));
            let cfg: crate::config::Config = toml::from_str(&format!(
                "[federation]\nid=\"tb\"\nname=\"tb\"\n[node]\nid=\"x\"\ndata_dir=\"{dir}\"\n[network]\nhost=\"127.0.0.1\"\napi_port=0\nws_port=0\np2p_port=0\n[logging]\nlevel=\"error\"\n"
            ))?;
            let state = Arc::new(tokio::sync::RwLock::new(crate::state::State::new(&cfg.node.data_dir)?));
            let hist = Arc::new(tokio::sync::RwLock::new(crate::history::UpdateHistory::new(cfg, state, kp.node_id()).await?));
            let mut acfg = crate::config::AgentsConfig::default();
            acfg.enabled = true;
            acfg.max_concurrent_tasks = 64;
            acfg.task_timeout_secs = call_timeout.as_secs() + 3600;
            let svc = AgentService::new(acfg, kp.clone(), hist, Arc::new(NoBroadcast), &dir).await?;
            // Attackers undercut: the cheapest bidders are the malicious ones, so price alone never protects the requester.
            let price = if *b == Behavior::Honest { 10 + (i as u64 % 3) } else { 9 };
            svc.register_skill(Arc::new(SolveSkill { behavior: *b, seed: i as u64 + 1, solver: solvers[i].clone() }));
            svc.register_skill(Arc::new(BidSkill { price }));
            let ep = Endpoint::builder(presets::Minimal).relay_mode(RelayMode::Disabled).alpns(vec![AGENT_ALPN.to_vec()]).bind().await?;
            let slot: AgentSlot = Default::default();
            let _ = slot.set(svc.clone());
            let proto = AgentProtocol::new(svc.history.clone(), kp.node_id().to_string(), None).with_agent_slot(slot);
            let router = Router::builder(ep.clone()).accept(AGENT_ALPN, proto).spawn();
            lookup.add_endpoint_info(ep.addr());
            workers.push(Worker { id: format!("w{i}"), addr: ep.addr(), _router: router, _svc: svc, price });
        }
        let client = Endpoint::builder(presets::Minimal).relay_mode(RelayMode::Disabled).address_lookup(lookup).bind().await?;
        Ok(Testbed { client, workers, messages: Arc::new(AtomicU64::new(0)), call_timeout, quorum: 0.5, calibration: 8 })
    }

    /// One A2A exchange: send, stream until final. Two messages on the wire.
    async fn call(&self, w: &Worker, skill: &str, data: Value, id: String) -> Option<Value> {
        let msg = Message { message_id: id, role: Role::User, parts: vec![Part::data(data)], context_id: None, task_id: None, metadata: Some(json!({"skill": skill})), extensions: vec![], reference_task_ids: vec![] };
        self.messages.fetch_add(1, Ordering::Relaxed);
        let task = match request(&self.client, w.addr.clone(), &AgentRequest::SendMessage { message: msg }).await.ok()? {
            AgentResponse::Task { task } => task,
            _ => return None,
        };
        self.messages.fetch_add(1, Ordering::Relaxed);
        let mut rx = subscribe(&self.client, w.addr.clone(), &task.id).await.ok()?;
        let fut = async {
            let mut art = None;
            while let Some(ev) = rx.recv().await {
                match ev {
                    WireEvent::Artifact { artifact, .. } => art = artifact.parts.first().and_then(|p| p.data.clone()),
                    WireEvent::Status { status, is_final: true, .. } => return (status.state == TaskState::Completed).then_some(art).flatten(),
                    _ => {}
                }
            }
            None
        };
        tokio::time::timeout(self.call_timeout, fut).await.ok().flatten()
    }

    /// Run `tasks` in order under `mode`. `probes` (known-answer tasks, cycled) are mixed in for
    /// `DecentralizedProbed`. `Single(i)` sends every task to worker i.
    pub async fn run(&self, mode: Mode, adversary: Behavior, malicious: usize, tasks: &[TaskItem], probes: &[TaskItem], redundancy: usize, seed: u64) -> Metrics {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        self.messages.store(0, Ordering::Relaxed);
        let mut book = ReputationBook::default();
        let key = |w: &Worker| -> AgentKey { (w.id.clone(), "agent".to_string()) };
        for w in &self.workers {
            book.set_node_trust(&w.id, 0.5);
        }
        let (mut cost, mut lat, mut tokens, mut compute) = (0u64, 0f64, 0u64, 0u64);
        let mut correct = vec![];
        let run_tag = rand::random::<u32>();
        let probed = mode == Mode::DecentralizedProbed && !probes.is_empty();
        let calibration = if probed { self.calibration } else { 0 };
        let mut executed = 0usize;
        let mut t = 0usize; // real tasks completed
        let mut seq = 0usize; // all tasks issued, probes included
        let mut probe_cursor = 0usize;
        let usage = |v: &Value| (v["tokens"].as_u64().unwrap_or(0), v["compute_ms"].as_u64().unwrap_or(0));
        while t < tasks.len() {
            let is_probe = probed && (seq < calibration || rng.gen_bool(0.1));
            seq += 1;
            executed += 1;
            let item = if is_probe {
                probe_cursor += 1;
                &probes[(probe_cursor - 1) % probes.len()]
            } else {
                &tasks[t]
            };
            let started = Instant::now();
            let answer: Option<String> = match mode {
                Mode::Centralized | Mode::Single(_) => {
                    let w = match mode {
                        Mode::Single(i) => &self.workers[i],
                        _ => &self.workers[t % self.workers.len()],
                    };
                    cost += w.price;
                    let r = self.call(w, "solve", item.input.clone(), format!("c{run_tag}-{t}")).await;
                    if let Some(v) = &r {
                        let (tk, c) = usage(v);
                        tokens += tk;
                        compute += c;
                    }
                    r.and_then(|v| v["answer"].as_str().map(String::from))
                }
                Mode::Decentralized | Mode::DecentralizedProbed => {
                    // announce -> bids
                    let spec = TaskSpec { id: format!("d{run_tag}-{seq}"), skill: "solve".into(), input: item.input.clone(), max_price: 100, deadline_secs: 10 };
                    let mut bids = vec![];
                    let calls: Vec<_> = self.workers.iter().map(|w| self.call(w, "bid", json!({}), format!("b{run_tag}-{seq}-{}", w.id))).collect();
                    for (w, r) in self.workers.iter().zip(futures::future::join_all(calls).await) {
                        if let Some(v) = r {
                            bids.push(Bid { bidder: w.id.clone(), price: v["price"].as_u64().unwrap_or(u64::MAX), eta_secs: 0, reputation: book.reputation(&key(w), RepRole::Executor) });
                        }
                    }
                    // award the best `redundancy` bidders
                    let ranked = rank_bids(&spec, &bids);
                    let winners: Vec<&Worker> = ranked.iter().take(redundancy).filter_map(|b| self.workers.iter().find(|w| w.id == b.bidder)).collect();
                    cost += winners.iter().map(|w| w.price).sum::<u64>();
                    let solves: Vec<_> = winners.iter().map(|w| self.call(w, "solve", item.input.clone(), format!("s{run_tag}-{seq}-{}", w.id))).collect();
                    let results = futures::future::join_all(solves).await;
                    for v in results.iter().flatten() {
                        let (tk, c) = usage(v);
                        tokens += tk;
                        compute += c;
                    }
                    let votes: Vec<(String, String)> = winners.iter().zip(&results).filter_map(|(w, r)| r.as_ref().and_then(|v| v["answer"].as_str()).filter(|a| *a != "none").map(|a| (w.id.clone(), a.to_string()))).collect();
                    let weight = |id: &str| self.workers.iter().find(|w| w.id == id).map(|w| book.reputation(&key(w), RepRole::Executor)).unwrap_or(0.0);
                    let pooled = pool(&votes, &weight, self.quorum);
                    // Learn. Votes-only mode learns from agreement with the consensus. Probed mode learns ONLY
                    // from ground truth: agreement rewards consistency, not correctness, and with noisy honest
                    // agents (real models disagree with each other) perfectly consistent colluders look more
                    // trustworthy than honest agents and win their awards back.
                    for w in &winners {
                        let mine = votes.iter().find(|(id, _)| *id == w.id).map(|(_, a)| a.as_str());
                        if is_probe {
                            book.record_outcome(Outcome { agent: key(w), role: RepRole::Executor, success: mine == Some(item.truth.as_str()) });
                        } else if !probed {
                            let good = matches!((&pooled, mine), (Some(p), Some(a)) if a == p.answer);
                            book.record_outcome(Outcome { agent: key(w), role: RepRole::Executor, success: good });
                        }
                    }
                    pooled.map(|p| p.answer)
                }
            };
            lat += started.elapsed().as_secs_f64() * 1000.0;
            if !is_probe {
                correct.push(answer.as_deref() == Some(item.truth.as_str()));
                t += 1;
            }
        }
        let n = tasks.len().max(1) as f64;
        Metrics {
            mode, adversary, malicious, tasks: tasks.len(), successes: correct.iter().filter(|c| **c).count(), correct,
            mean_latency_ms: lat / executed as f64,
            messages_per_task: self.messages.load(Ordering::Relaxed) as f64 / n,
            cost_per_task: cost as f64 / n,
            tokens_per_task: tokens as f64 / n,
            compute_ms_per_task: compute as f64 / n,
        }
    }
}

/// Run the full matrix: for each adversary type and each k, both modes on a fresh population.
pub async fn run_matrix(workers: usize, ks: &[usize], adversaries: &[Behavior], tasks: usize, redundancy: usize) -> Result<Vec<Metrics>, Box<dyn std::error::Error + Send + Sync>> {
    let mut out = vec![];
    for adv in adversaries {
        for &k in ks {
            let mut behaviors = vec![Behavior::Honest; workers];
            // spread malicious workers through the population so round-robin meets them evenly
            for j in 0..k {
                behaviors[(j * workers) / k.max(1) + (workers / (2 * k.max(1)))] = *adv;
            }
            let tb = Testbed::start(&behaviors).await?;
            let items = sum_tasks(tasks, 42 + k as u64);
            let probes = sum_tasks(32, 7_000 + k as u64);
            for mode in [Mode::Centralized, Mode::Decentralized, Mode::DecentralizedProbed] {
                out.push(tb.run(mode, *adv, k, &items, &probes, redundancy, 42 + k as u64).await);
            }
        }
    }
    Ok(out)
}

pub fn to_markdown(rows: &[Metrics], workers: usize) -> String {
    let mut s = format!("| adversary | k of {workers} | mode | success | latency ms | msgs/task | cost/task |\n|---|---|---|---|---|---|---|\n");
    for m in rows {
        s += &format!(
            "| {:?} | {} | {} | {:.0}% | {:.1} | {:.1} | {:.1} |\n",
            m.adversary, m.malicious,
            match m.mode { Mode::Single(_) => "single agent", Mode::Centralized => "centralized", Mode::Decentralized => "decentralized (votes only)", Mode::DecentralizedProbed => "decentralized + probes" },
            m.success_rate() * 100.0, m.mean_latency_ms, m.messages_per_task, m.cost_per_task,
        );
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn find(rows: &[Metrics], mode: Mode, adv: Behavior, k: usize) -> &Metrics {
        rows.iter().find(|m| m.mode == mode && m.adversary == adv && m.malicious == k).unwrap()
    }

    /// The numbers behind the pilot go/no-go: measured, and the assertions state what the data supports.
    /// * no adversary: every mode is exact;
    /// * the unverified centralized baseline loses about k/9 of tasks;
    /// * vote-only pooling is NOT enough against cheap colluders (it can lock in the lie);
    /// * pooling plus ground-truth probes is robust, at several times the message cost.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn measured_robustness_of_the_decentralized_design_against_the_centralized_baseline() {
        let advs = [Behavior::Colluding, Behavior::Lying, Behavior::Silent];
        let rows = run_matrix(9, &[0, 1, 2, 3], &advs, 20, 3).await.unwrap();
        println!("\n{}", to_markdown(&rows, 9));
        for adv in advs {
            for mode in [Mode::Centralized, Mode::Decentralized, Mode::DecentralizedProbed] {
                let m = find(&rows, mode, adv, 0);
                assert_eq!(m.successes, m.tasks, "{adv:?}/{mode:?} with no adversary must be exact");
            }
        }
        for k in [1usize, 2, 3] {
            for adv in advs {
                let c = find(&rows, Mode::Centralized, adv, k);
                let p = find(&rows, Mode::DecentralizedProbed, adv, k);
                assert!(c.success_rate() < 0.95, "{adv:?} k={k}: baseline should be hurt, got {:.2}", c.success_rate());
                assert!(p.success_rate() >= 0.9, "{adv:?} k={k}: pooling+probes got only {:.2}", p.success_rate());
                assert!(p.success_rate() > c.success_rate(), "{adv:?} k={k}");
            }
        }
        // The documented failure of vote-only reputation: cheap colluders capture the consensus.
        let vote_only = find(&rows, Mode::Decentralized, Behavior::Colluding, 2);
        let probed = find(&rows, Mode::DecentralizedProbed, Behavior::Colluding, 2);
        assert!(vote_only.success_rate() < 0.5, "expected the lock-in failure mode, got {:.2}", vote_only.success_rate());
        assert!(probed.success_rate() > vote_only.success_rate() + 0.4);
        // Verification costs messages.
        assert!(find(&rows, Mode::Decentralized, Behavior::Colluding, 0).messages_per_task > find(&rows, Mode::Centralized, Behavior::Colluding, 0).messages_per_task);
    }

    // ---- M3 done-when: a five-agent team from three institutions, no coordinator ----

    use crate::agents::coordination::{form_team, Blackboard, TeamTemplate};
    use crate::agents::discovery::{Directory, SCOPE_CARD};
    use crate::agents::records::{KIND_POST, Post, PostKind};
    use crate::history::{RecordKind, UpdateRecord};
    use std::collections::BTreeMap;

    struct TeamNode {
        svc: Arc<AgentService>,
        ep: Endpoint,
        router: Router,
        institution: &'static str,
        role_skill: &'static str,
    }

    struct Noop(&'static str);
    #[async_trait]
    impl Skill for Noop {
        fn descriptor(&self) -> AgentSkill {
            AgentSkill { id: self.0.into(), name: self.0.into(), description: format!("{} role", self.0), tags: vec![], examples: vec![], input_modes: vec![], output_modes: vec![] }
        }
        async fn run(&self, _: &SkillCtx, _: &Task, _: &Message) -> Result<Vec<Artifact>, SkillError> {
            Ok(vec![])
        }
    }

    async fn team_node(institution: &'static str, role_skill: &'static str, lookup: &MemoryLookup) -> TeamNode {
        let kp = crate::identity::Keypair::generate();
        let dir = tmp("team");
        let cfg: crate::config::Config = toml::from_str(&format!("[federation]\nid=\"t\"\nname=\"t\"\n[node]\nid=\"x\"\ndata_dir=\"{dir}\"\n[network]\nhost=\"127.0.0.1\"\napi_port=0\nws_port=0\np2p_port=0\n[logging]\nlevel=\"error\"\n")).unwrap();
        let state = Arc::new(tokio::sync::RwLock::new(crate::state::State::new(&cfg.node.data_dir).unwrap()));
        let hist = Arc::new(tokio::sync::RwLock::new(crate::history::UpdateHistory::new(cfg, state, kp.node_id()).await.unwrap()));
        let mut acfg = crate::config::AgentsConfig::default();
        acfg.enabled = true;
        let svc = AgentService::new(acfg, kp.clone(), hist, Arc::new(NoBroadcast), &dir).await.unwrap();
        svc.register_skill(Arc::new(Noop(role_skill)));
        svc.ensure_delegation().await;
        svc.publish_card().await;
        let ep = Endpoint::builder(presets::Minimal).relay_mode(RelayMode::Disabled).alpns(vec![AGENT_ALPN.to_vec()]).address_lookup(lookup.clone()).bind().await.unwrap();
        let slot: AgentSlot = Default::default();
        let _ = slot.set(svc.clone());
        let router = Router::builder(ep.clone()).accept(AGENT_ALPN, AgentProtocol::new(svc.history.clone(), kp.node_id().to_string(), None).with_agent_slot(slot)).spawn();
        lookup.add_endpoint_info(ep.addr());
        TeamNode { svc, ep, router, institution, role_skill }
    }

    /// Everyone pulls everyone's records over the agent ALPN (what the mesh's anti-entropy does).
    async fn exchange(nodes: &[TeamNode]) {
        for a in nodes {
            for b in nodes {
                if a.svc.node_id != b.svc.node_id {
                    a.svc.sync_from_peer(&a.ep, b.ep.id()).await;
                }
            }
        }
    }

    async fn board(n: &TeamNode, collab: &str) -> Blackboard {
        let h = n.svc.history.read().await;
        Blackboard::from_records(collab, h.peer_updates.values().flatten())
    }

    async fn post(n: &TeamNode, collab: &str, kind: PostKind, reply_to: Option<&str>, text: &str) -> String {
        let body = serde_json::to_value(Post { collaboration: collab.into(), kind, reply_to: reply_to.map(String::from), content_hash: crate::agents::a2a::sha256_hex(text.as_bytes()), text: text.into() }).unwrap();
        n.svc.author(KIND_POST, body).await.unwrap().hash
    }

    fn data() -> Vec<i64> { (1..=20).collect() }
    fn sum_squares_independent(v: &[i64]) -> i64 { v.iter().fold(0, |a, x| a + x * x) }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn five_agents_from_three_institutions_complete_a_benchmark_task_with_no_coordinator() {
        let lookup = MemoryLookup::new();
        let roles = [("inst-A", "research"), ("inst-A", "review"), ("inst-B", "code"), ("inst-B", "review"), ("inst-C", "evaluate")];
        let mut nodes = vec![];
        for (inst, skill) in roles { nodes.push(team_node(inst, skill, &lookup).await); }
        exchange(&nodes).await;

        // --- team formation from the verified directory (cards announced via records, fetched over QUIC) ---
        let mut dir = {
            let h = nodes[0].svc.history.read().await;
            let recs: Vec<&UpdateRecord> = h.peer_updates.values().flatten().collect();
            Directory::from_records(recs)
        };
        let now = chrono::Utc::now().timestamp();
        for n in &nodes { dir.attach_card(&n.svc.node_id, n.svc.card(), now).expect("card verifies"); }
        let _ = SCOPE_CARD;
        let team = form_team(&TeamTemplate::research_team(), &dir, &|_, _| 0.5, now).unwrap();
        assert_eq!(team.len(), 5);
        let inst_of: BTreeMap<String, &str> = nodes.iter().map(|n| (n.svc.node_id.clone(), n.institution)).collect();
        let institutions: std::collections::BTreeSet<_> = team.iter().map(|m| inst_of[&m.node_id]).collect();
        assert_eq!(institutions.len(), 3, "the team spans three institutions");

        // --- the collaboration: every agent acts on its role when it sees the right post ---
        let collab = "bench-1";
        let truth = sum_squares_independent(&data());
        let by_role = |skill: &str| -> Vec<&TeamNode> { nodes.iter().filter(|n| n.role_skill == skill).collect() };
        let (researcher, coder, evaluator) = (by_role("research")[0], by_role("code")[0], by_role("evaluate")[0]);
        let reviewers = by_role("review");

        for round in 0..6 {
            // researcher: states the problem once
            if board(researcher, collab).await.proposals().is_empty() {
                post(researcher, collab, PostKind::Proposal, None, "task: sum of squares of 1..=20, answer as an integer").await;
            }
            exchange(&nodes).await;
            // coder: first attempt is wrong (sums instead of squares); after criticism it fixes the bug
            {
                let b = board(coder, collab).await;
                if let Some(prop) = b.proposals().first().map(|p| p.id.clone()) {
                    let mine: Vec<_> = b.results().into_iter().filter(|r| r.author == coder.svc.node_id).collect();
                    match mine.last() {
                        None => { post(coder, collab, PostKind::Result, Some(&prop), &format!("answer={}", data().iter().sum::<i64>())).await; }
                        Some(last) => {
                            let criticised = b.replies(&last.id).iter().any(|r| r.post.kind == PostKind::Critique);
                            if criticised && mine.len() == 1 {
                                post(coder, collab, PostKind::Result, Some(&prop), &format!("answer={}", data().iter().map(|x| x * x).sum::<i64>())).await;
                            }
                        }
                    }
                }
            }
            exchange(&nodes).await;
            // reviewers and evaluator: check every result they have not yet judged
            for judge in reviewers.iter().copied().chain(std::iter::once(evaluator)) {
                let b = board(judge, collab).await;
                for r in b.results() {
                    if b.replies(&r.id).iter().any(|x| x.author == judge.svc.node_id) { continue; }
                    let claimed: Option<i64> = r.post.text.strip_prefix("answer=").and_then(|t| t.parse().ok());
                    let ok = claimed == Some(sum_squares_independent(&data()));
                    let kind = if ok { PostKind::Support } else { PostKind::Critique };
                    post(judge, collab, kind, Some(&r.id), if ok { "verified" } else { "does not match my computation" }).await;
                }
            }
            exchange(&nodes).await;
            // acceptance is computed independently by every node from the shared records
            let verdicts: Vec<Option<String>> = {
                let mut v = vec![];
                for n in &nodes {
                    let b = board(n, collab).await;
                    let accepted = b.results().into_iter().filter(|r| b.score(&r.id, &|_| 1.0) >= 4.0).map(|r| r.post.text.clone()).next();
                    v.push(accepted);
                }
                v
            };
            if verdicts.iter().all(|v| v.is_some()) {
                assert!(verdicts.iter().all(|v| v == &verdicts[0]), "all nodes agree on the accepted result");
                assert_eq!(verdicts[0].as_deref(), Some(format!("answer={truth}").as_str()));
                assert!(round >= 1, "the first attempt was wrong, so at least one critique round was needed");
                // the wrong first attempt stayed on the board, criticised
                let b = board(researcher, collab).await;
                assert_eq!(b.results().len(), 2);
                assert!(b.results().iter().any(|r| b.score(&r.id, &|_| 1.0) < 0.0));
                // every post on the board is a signed provenance record
                let h = researcher.svc.history.read().await;
                assert!(h.peer_updates.values().flatten().filter(|r| matches!(&r.kind, RecordKind::Provenance { kind, .. } if kind == KIND_POST)).all(|r| r.verify().is_ok()));
                drop(h);
                for n in nodes { n.router.shutdown().await.unwrap(); }
                return;
            }
        }
        panic!("the team did not converge within 6 rounds");
    }
}
