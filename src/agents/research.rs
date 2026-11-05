//! The research loop (Step 4): one node's participation in a federated research
//! program, composed from the protocol pieces.
//!
//! Per generation: planners publish proposals (each carrying a VRF ticket) ->
//! the committee's trust-weighted merge becomes the agenda -> the agenda's items
//! are claimed through the task market -> executors turn an item into a candidate
//! (patch -> worktree -> sandboxed run) -> evaluators score candidates on their own
//! data and publish findings -> lanes, credit and payouts are computed from the
//! evidence graph by anyone.
//!
//! Everything a node does is a signed provenance record; the model-facing and
//! container-facing parts sit behind traits (`CandidateMaker`, `Runner`,
//! `CandidateEvaluator`, `Proposer`) so they can be backed by the federation's
//! model, Podman/Apptainer, or the data owner's evaluator service.

use crate::agents::credit::{self, CreditRule, CreditTable};
use crate::agents::evaluator::Attestation;
use crate::agents::graph::{EvidenceGraph, Lanes};
use crate::agents::market::{CapabilityAd, ItemStatus, Market};
use crate::agents::planning::{self, MergeConfig, MergedAgenda, Ticket};
use crate::agents::records::*;
use crate::agents::sandbox;
use crate::agents::service::AgentService;
use crate::history::{RecordKind, UpdateRecord};
use async_trait::async_trait;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

type BoxError = Box<dyn std::error::Error + Send + Sync>;

pub fn sha256_hex(b: &[u8]) -> String {
    hex::encode(Sha256::digest(b))
}

/// Artifact content address: BLAKE3, the hash iroh-blobs uses, so a hash in a
/// provenance record can be fetched directly over the blob protocol.
pub fn blake3_hex(b: &[u8]) -> String {
    blake3::hash(b).to_hex().to_string()
}

/// Where artifacts (candidate diffs, container images, ...) live. `get` returns
/// the bytes if they are available locally or can be fetched from peers.
#[async_trait]
pub trait ArtifactStore: Send + Sync {
    async fn put(&self, bytes: &[u8]) -> Result<String, String>;
    async fn get(&self, hash: &str) -> Option<Vec<u8>>;
}

/// A candidate's change: file path -> new content. Its canonical serialization
/// is the diff blob; `hash()` is the `diff_hash` recorded in the candidate.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Patch {
    pub files: BTreeMap<String, Vec<u8>>,
}

impl Patch {
    pub fn to_bytes(&self) -> Vec<u8> {
        use base64::Engine;
        let m: BTreeMap<&String, String> = self.files.iter().map(|(k, v)| (k, base64::engine::general_purpose::STANDARD.encode(v))).collect();
        serde_json::to_vec(&m).expect("patch serializes")
    }

    pub fn from_bytes(b: &[u8]) -> Result<Self, String> {
        use base64::Engine;
        let m: BTreeMap<String, String> = serde_json::from_slice(b).map_err(|e| e.to_string())?;
        let mut files = BTreeMap::new();
        for (k, v) in m {
            files.insert(k, base64::engine::general_purpose::STANDARD.decode(v).map_err(|e| e.to_string())?);
        }
        Ok(Patch { files })
    }

    #[allow(dead_code)] // library surface: used by tests and integrators, or pending the settlement-chain decision
    pub fn hash(&self) -> String {
        blake3_hex(&self.to_bytes())
    }

    /// True if every path is one of `allowed` (exact match). Programs whose candidates may only change
    /// specific files (e.g. a prompt file) enforce this before running or scoring anyone's patch.
    pub fn within(&self, allowed: &[String]) -> bool {
        self.files.keys().all(|k| allowed.iter().any(|a| a == k))
    }

    /// Write the files into `dir`, refusing paths that escape it.
    pub fn apply(&self, dir: &std::path::Path) -> Result<(), String> {
        for (path, content) in &self.files {
            let rel = std::path::Path::new(path);
            if rel.is_absolute() || rel.components().any(|c| matches!(c, std::path::Component::ParentDir | std::path::Component::Prefix(_) | std::path::Component::RootDir)) {
                return Err(format!("patch path '{path}' escapes the worktree"));
            }
            let dest = dir.join(rel);
            if let Some(parent) = dest.parent() {
                std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
            }
            std::fs::write(&dest, content).map_err(|e| e.to_string())?;
        }
        Ok(())
    }
}

/// Content-addressed local blob store. Artifacts are put here under their
/// SHA-256 and announced by hash; peers fetch them by hash (see `blobs.rs`).
#[allow(dead_code)] // library surface: used by tests and integrators, or pending the settlement-chain decision
pub struct LocalBlobs {
    dir: PathBuf,
}

impl LocalBlobs {
    #[allow(dead_code)] // library surface: used by tests and integrators, or pending the settlement-chain decision
    pub fn open(dir: PathBuf) -> std::io::Result<Self> {
        std::fs::create_dir_all(&dir)?;
        Ok(LocalBlobs { dir })
    }

    pub fn put(&self, bytes: &[u8]) -> std::io::Result<String> {
        let h = blake3_hex(bytes);
        let p = self.dir.join(&h);
        if !p.exists() {
            let tmp = self.dir.join(format!("{h}.tmp"));
            std::fs::write(&tmp, bytes)?;
            std::fs::rename(tmp, p)?;
        }
        Ok(h)
    }

    /// Returns the bytes only if they still hash to `hash` (detects corruption).
    pub fn get(&self, hash: &str) -> Option<Vec<u8>> {
        if !is_hash(hash) {
            return None;
        }
        let b = std::fs::read(self.dir.join(hash)).ok()?;
        (blake3_hex(&b) == hash).then_some(b)
    }

    #[allow(dead_code)] // library surface: used by tests and integrators, or pending the settlement-chain decision
    pub fn has(&self, hash: &str) -> bool {
        self.get(hash).is_some()
    }
}

#[async_trait]
impl ArtifactStore for LocalBlobs {
    async fn put(&self, bytes: &[u8]) -> Result<String, String> {
        LocalBlobs::put(self, bytes).map_err(|e| e.to_string())
    }
    async fn get(&self, hash: &str) -> Option<Vec<u8>> {
        LocalBlobs::get(self, hash)
    }
}

pub struct RunReport {
    pub container_hash: String,
    pub input_hashes: Vec<String>,
}

#[async_trait]
pub trait Proposer: Send + Sync {
    /// Items this planner wants on the next agenda, given the current lanes.
    async fn propose_items(&self, program: &str, generation: u64, lanes: &Lanes) -> Vec<ProposedItem>;
}

#[async_trait]
pub trait CandidateMaker: Send + Sync {
    /// Turn an agenda item into a patch on top of the parent candidates' patches.
    async fn make(&self, item: &AgendaItem, parents: &[Patch]) -> Result<Patch, String>;
}

#[async_trait]
pub trait Runner: Send + Sync {
    /// Execute the candidate in its worktree; must be deterministic given `seed`.
    async fn run(&self, worktree: &std::path::Path, seed: u64) -> Result<RunReport, String>;
}

#[async_trait]
pub trait CandidateEvaluator: Send + Sync {
    /// Score a candidate on this node's data; None = cannot evaluate it.
    async fn evaluate(&self, patch: &Patch, run: &RunReport) -> Option<Attestation>;
}

/// Runs a candidate in the container sandbox (Podman/Apptainer): the image is
/// fixed by the operator, the worktree is the only writable mount, no network.
pub struct SandboxRunner {
    pub profile: sandbox::SandboxProfile,
    pub command: Vec<String>,
    pub wall_clock: Duration,
}

#[async_trait]
impl Runner for SandboxRunner {
    async fn run(&self, worktree: &std::path::Path, seed: u64) -> Result<RunReport, String> {
        let mut spec = self.profile.spec(worktree, self.command.clone(), self.wall_clock);
        spec.env.push(("SLAKSHNA_SEED".into(), seed.to_string()));
        let name = format!("slk-run-{}", rand::random::<u32>());
        let r = sandbox::run(&spec, &name).await.map_err(|e| e.to_string())?;
        if r.timed_out {
            return Err("run exceeded its wall-clock budget".into());
        }
        if r.flooded {
            return Err("run was killed for flooding stdout/stderr".into());
        }
        if r.exit_code != Some(0) {
            return Err(format!("run failed (exit {:?}): {}", r.exit_code, r.stderr.chars().take(500).collect::<String>()));
        }
        Ok(RunReport { container_hash: sha256_hex(self.profile.image.as_bytes()), input_hashes: vec![] })
    }
}

#[async_trait]
pub trait FindingAuditor: Send + Sync {
    /// Spot-check one finding against its data owner. Some(true) = consistent, Some(false) = not,
    /// None = could not be checked (owner unreachable, not an authorized auditor, ...).
    async fn audit(&self, finding: &Finding, finding_author: &str, patch: &Patch) -> Option<bool>;
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedAgenda {
    pub evidence_root: String,
    pub committee: BTreeMap<String, f64>,
    pub merged: MergedAgenda,
}

struct Prop {
    planner: String,
    body: crate::agents::records::Proposal,
}

fn proposals_for(records: &[UpdateRecord], program: &str, generation: u64) -> Vec<Prop> {
    let mut v = vec![];
    for r in records {
        let RecordKind::Provenance { kind, body } = &r.kind else { continue };
        if kind != KIND_PROPOSAL || r.verify().is_err() {
            continue;
        }
        if let Ok(p) = serde_json::from_value::<crate::agents::records::Proposal>(body.clone()) {
            if p.program == program && p.generation == generation {
                v.push(Prop { planner: r.node_id.clone(), body: p });
            }
        }
    }
    // One proposal per planner: the lowest record order is irrelevant, so keep the
    // lexicographically smallest serialization for determinism.
    v.sort_by(|a, b| {
        a.planner
            .cmp(&b.planner)
            // a planner that retried after an empty proposal is judged by the non-empty one
            .then_with(|| b.body.items.len().cmp(&a.body.items.len()))
            .then_with(|| serde_json::to_string(&a.body).unwrap().cmp(&serde_json::to_string(&b.body).unwrap()))
    });
    v.dedup_by(|a, b| a.planner == b.planner);
    v
}

/// Deterministically derive the generation's committee and agenda from the
/// proposals on record. Any node with the same records gets the same answer.
pub fn resolve_agenda(
    records: &[UpdateRecord],
    program: &str,
    generation: u64,
    committee_size: usize,
    weight: &dyn Fn(&str) -> f64,
    cfg: &MergeConfig,
) -> Option<ResolvedAgenda> {
    let props = proposals_for(records, program, generation);
    if props.is_empty() {
        return None;
    }
    // Canonical evidence root: the one most planners saw (ties: smallest).
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    for p in &props {
        *counts.entry(p.body.evidence_root.as_str()).or_default() += 1;
    }
    let root = counts.iter().max_by(|a, b| a.1.cmp(b.1).then_with(|| b.0.cmp(a.0))).map(|(r, _)| r.to_string())?;

    let tickets: Vec<Ticket> = props.iter().map(|p| Ticket { node_id: p.planner.clone(), public_key: p.body.ticket_public_key.clone(), proof: p.body.ticket_proof.clone() }).collect();
    let weights: BTreeMap<String, f64> = props.iter().map(|p| (p.planner.clone(), weight(&p.planner))).collect();
    let members = planning::select_committee(&tickets, &weights, program, generation, &root, committee_size);
    if members.is_empty() {
        return None;
    }
    let committee: BTreeMap<String, f64> = members.iter().map(|m| (m.clone(), weights[m])).collect();
    let proposals: Vec<planning::Proposal> = props
        .iter()
        .filter(|p| committee.contains_key(&p.planner))
        .map(|p| planning::Proposal { planner: p.planner.clone(), evidence_root: p.body.evidence_root.clone(), items: p.body.items.clone() })
        .collect();
    let merged = planning::merge(&root, &committee, &proposals, cfg);
    Some(ResolvedAgenda { evidence_root: root, committee, merged })
}

/// Check that an `research.agenda` record is what the proposals on record produce.
pub fn verify_agenda(records: &[UpdateRecord], agenda: &UpdateRecord, committee_size: usize, weight: &dyn Fn(&str) -> f64, cfg: &MergeConfig) -> bool {
    let RecordKind::Provenance { kind, body } = &agenda.kind else { return false };
    if kind != KIND_AGENDA || agenda.verify().is_err() {
        return false;
    }
    let Ok(a) = serde_json::from_value::<Agenda>(body.clone()) else { return false };
    let Some(r) = resolve_agenda(records, &a.program, a.generation, committee_size, weight, cfg) else { return false };
    let committee: BTreeSet<&String> = r.committee.keys().collect();
    a.evidence_root == r.evidence_root
        && a.committee.iter().collect::<BTreeSet<_>>() == committee
        && a.committee.contains(&agenda.node_id)
        && a.items == r.merged.items
}

pub struct ResearchNode {
    pub svc: Arc<AgentService>,
    pub caps: CapabilityAd,
    pub lease_secs: u64,
    /// Max candidates this node builds per `step_execute` call.
    pub capacity: usize,
    pub blobs: Arc<dyn ArtifactStore>,
    pub base_project: PathBuf,
    pub work_root: PathBuf,
    pub proposer: Option<Arc<dyn Proposer>>,
    pub maker: Option<Arc<dyn CandidateMaker>>,
    pub runner: Option<Arc<dyn Runner>>,
    pub evaluator: Option<Arc<dyn CandidateEvaluator>>,
    pub merge_cfg: MergeConfig,
    pub committee_size: usize,
    pub planning_weight: Arc<dyn Fn(&str) -> f64 + Send + Sync>,
    /// Do not publish an agenda before this many proposals are on record (until the wait runs out).
    pub min_proposals: usize,
    /// Ticks to wait for proposals / evaluations before proceeding with what exists.
    pub wait_ticks: u32,
    /// Programs stop after this many generations (bounds spend).
    pub max_generations: u64,
    pub auditor: Option<Arc<dyn FindingAuditor>>,
    /// An item failed by this many distinct nodes is given up on so the generation can proceed.
    pub max_item_failures: usize,
    /// Items this node already failed (it does not retry them).
    failed_here: std::sync::Mutex<BTreeSet<(String, String)>>,
    /// If set, candidates may only change these files; others' patches that touch anything else are ignored.
    pub allowed_paths: Option<Vec<String>>,
    /// Hard ceiling on GPU-seconds this node spends on runs; charged per item before it starts.
    pub gpu_budget: Option<Arc<crate::agents::budget::BudgetTracker>>,
    /// Flag findings that disagree with the other evaluators of the same shared benchmark.
    pub check_consensus: bool,
    /// Told the outcome of every audit: (evaluator node, was it consistent).
    pub on_audit: Option<Arc<dyn Fn(&str, bool) + Send + Sync>>,
    /// Fraction of unremarkable findings audited at random, besides the suspicious ones.
    pub audit_rate: f64,
    waiting: std::sync::Mutex<BTreeMap<String, u32>>,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct TickReport {
    pub proposed: usize,
    pub agendas: usize,
    pub executed: usize,
    pub evaluated: usize,
    pub audited: usize,
}

impl ResearchNode {
    pub fn new(svc: Arc<AgentService>, caps: CapabilityAd, blobs: Arc<dyn ArtifactStore>, base_project: PathBuf, work_root: PathBuf) -> Self {
        ResearchNode {
            svc, caps, lease_secs: 60, capacity: 2, blobs, base_project, work_root,
            proposer: None, maker: None, runner: None, evaluator: None,
            merge_cfg: MergeConfig::default(), committee_size: 5,
            planning_weight: Arc::new(|_| 1.0),
            min_proposals: 5, wait_ticks: 3, max_generations: 5,
            auditor: None, max_item_failures: 3, failed_here: Default::default(), allowed_paths: None, gpu_budget: None, check_consensus: false, on_audit: None, audit_rate: 0.1,
            waiting: Default::default(),
        }
    }

    fn bump_wait(&self, key: &str) -> u32 {
        let mut w = self.waiting.lock().unwrap();
        let n = w.entry(key.to_string()).or_default();
        *n += 1;
        *n
    }

    /// The project's current content of the files candidates may change, as context for the first
    /// generation (None if no allow-list is configured or nothing readable).
    fn baseline_patch(&self) -> Option<Patch> {
        let allowed = self.allowed_paths.as_ref()?;
        let mut p = Patch::default();
        for path in allowed {
            if let Ok(b) = std::fs::read(self.base_project.join(path)) {
                p.files.insert(path.clone(), b);
            }
        }
        (!p.files.is_empty()).then_some(p)
    }

    /// Hashes of the agenda records that really are what the proposals on record produce.
    /// An agenda published by anyone who was not drawn into the committee, or whose items are
    /// not the merge of the committee's proposals, is not in this set and is never acted on.
    fn valid_agendas(&self, recs: &[UpdateRecord]) -> BTreeSet<String> {
        let w = self.planning_weight.clone();
        recs.iter()
            .filter(|r| matches!(&r.kind, RecordKind::Provenance { kind, .. } if kind == KIND_AGENDA))
            .filter(|r| verify_agenda(recs, r, self.committee_size, &*w, &self.merge_cfg))
            .map(|r| r.hash.clone())
            .collect()
    }

    /// What this node sees of a program's planning and market state (for operators and debugging).
    pub async fn debug_state(&self, program: &str, now: i64) -> serde_json::Value {
        let recs = self.records().await;
        let valid = self.valid_agendas(&recs);
        let market = Market::from_records(&recs);
        let mut props: BTreeMap<u64, Vec<String>> = BTreeMap::new();
        let mut agendas = vec![];
        for r in &recs {
            let RecordKind::Provenance { kind, body } = &r.kind else { continue };
            if kind == KIND_PROPOSAL {
                if let Ok(p) = serde_json::from_value::<crate::agents::records::Proposal>(body.clone()) {
                    if p.program == program {
                        props.entry(p.generation).or_default().push(format!("{}…:{}items:root={}", &r.node_id[..12.min(r.node_id.len())], p.items.len(), &p.evidence_root[..8]));
                    }
                }
            } else if kind == KIND_AGENDA {
                if let Ok(a) = serde_json::from_value::<Agenda>(body.clone()) {
                    if a.program == program {
                        let statuses: Vec<String> = a.items.iter().map(|i| format!("{}={:?}", i.id, market.status(&r.hash, &i.id, now))).collect();
                        agendas.push(serde_json::json!({"hash": &r.hash[..12], "generation": a.generation, "author": &r.node_id[..12.min(r.node_id.len())], "valid_here": valid.contains(&r.hash), "committee": a.committee.len(), "items": statuses}));
                    }
                }
            }
        }
        serde_json::json!({"node": self.svc.node_id, "records": recs.len(), "proposals": props, "agendas": agendas, "gpu_budget_left": self.gpu_budget.as_ref().map(|b| b.remaining().1)})
    }

    /// Latest generation with a *valid* agenda for `program`, and that agenda's record hash.
    fn latest_agenda(recs: &[UpdateRecord], program: &str, valid: &BTreeSet<String>) -> Option<(u64, String)> {
        let mut best: Option<(u64, String)> = None;
        for r in recs {
            let RecordKind::Provenance { kind, body } = &r.kind else { continue };
            if kind != KIND_AGENDA || !valid.contains(&r.hash) {
                continue;
            }
            let Ok(a) = serde_json::from_value::<Agenda>(body.clone()) else { continue };
            if a.program == program && best.as_ref().map_or(true, |(g, h)| (a.generation, &r.hash) > (*g, h)) {
                best = Some((a.generation, r.hash.clone()));
            }
        }
        best
    }

    /// One scheduling pass over every program this node knows. Drives generations forward:
    /// when the latest agenda is fully done and its candidates are evaluated (or the grace
    /// period passes), planners propose the next generation and committee members publish it.
    pub async fn tick(&self, now: i64) -> Result<TickReport, BoxError> {
        let mut rep = TickReport::default();
        let recs = self.records().await;
        let programs: Vec<(String, u64)> = recs
            .iter()
            .filter(|r| matches!(&r.kind, RecordKind::Provenance { kind, .. } if kind == KIND_OBJECTIVE) && r.verify().is_ok())
            .map(|r| (r.hash.clone(), 0))
            .collect();
        let valid = self.valid_agendas(&recs);
        for (program, _) in programs {
            let (gen, agenda) = Self::latest_agenda(&recs, &program, &valid).map(|(g, h)| (g, Some(h))).unwrap_or((0, None));
            let next = gen + 1;
            if next > self.max_generations {
                continue;
            }
            let ready = match &agenda {
                None => true,
                Some(h) => {
                    let done = Market::from_records(&recs).all_settled(h, now, self.max_item_failures);
                    let evaluated = EvidenceGraph::from_records(&recs).lanes(&program).map_or(true, |l| l.incubator.is_empty());
                    done && (evaluated || self.bump_wait(&format!("eval:{program}:{gen}")) > self.wait_ticks)
                }
            };
            if !ready {
                continue;
            }
            if self.proposer.is_some() && self.propose(&program, next).await?.is_some() {
                rep.proposed += 1;
            }
            // Wait for enough proposals before merging, so members do not publish different agendas.
            let n_props = proposals_for(&recs, &program, next).len();
            if n_props > 0 && (n_props >= self.min_proposals || self.bump_wait(&format!("prop:{program}:{next}")) > self.wait_ticks) {
                if self.finalize_agenda(&program, next).await?.is_some() {
                    rep.agendas += 1;
                }
            }
        }
        rep.executed = self.step_execute(now).await?;
        rep.evaluated = self.step_evaluate().await?;
        rep.audited = self.step_audit().await?;
        if self.check_consensus {
            rep.audited += self.step_consensus(0.02).await?;
        }
        Ok(rep)
    }

    pub async fn records(&self) -> Vec<UpdateRecord> {
        let h = self.svc.history.read().await;
        let mut nodes: Vec<&String> = h.peer_updates.keys().collect();
        nodes.sort();
        nodes.into_iter().flat_map(|n| h.peer_updates[n].iter().cloned()).collect()
    }

    pub async fn start_program(&self, objective: &Objective) -> Result<String, BoxError> {
        // The objective's own hash is the program id, known only once it is authored: it goes out on
        // the federation topic so every node learns the program exists, then both sides join its topic.
        let rec = self.svc.author(KIND_OBJECTIVE, serde_json::to_value(objective)?).await?;
        self.svc.join_program(&rec.hash).await;
        Ok(rec.hash)
    }

    /// Planner role: publish this node's proposal (once per generation).
    pub async fn propose(&self, program: &str, generation: u64) -> Result<Option<String>, BoxError> {
        let Some(proposer) = &self.proposer else { return Ok(None) };
        let recs = self.records().await;
        // One proposal per generation, unless the earlier one was empty (e.g. the model was briefly unusable).
        if proposals_for(&recs, program, generation).iter().any(|p| p.planner == self.svc.node_id && !p.body.items.is_empty()) {
            return Ok(None);
        }
        let graph = EvidenceGraph::from_records(&recs);
        let root = graph.evidence_root(program);
        let lanes = graph.lanes(program).ok_or("unknown program")?;
        let items = proposer.propose_items(program, generation, &lanes).await;
        if items.is_empty() {
            return Ok(None); // nothing usable yet: try again next tick rather than publish an empty proposal
        }
        let ticket = Ticket::create(&self.svc.node_id, &self.svc.signing_seed(), program, generation, &root);
        let body = crate::agents::records::Proposal { program: program.into(), generation, evidence_root: root, ticket_public_key: ticket.public_key, ticket_proof: ticket.proof, items };
        Ok(Some(self.svc.author_in(program, KIND_PROPOSAL, serde_json::to_value(&body)?).await?.hash))
    }

    /// Committee member: publish the merged agenda if the proposals on record yield one
    /// that is not already published. Returns the agenda record hash.
    pub async fn finalize_agenda(&self, program: &str, generation: u64) -> Result<Option<String>, BoxError> {
        let recs = self.records().await;
        let w = self.planning_weight.clone();
        let Some(r) = resolve_agenda(&recs, program, generation, self.committee_size, &*w, &self.merge_cfg) else { return Ok(None) };
        if !r.committee.contains_key(&self.svc.node_id) {
            return Ok(None);
        }
        if r.merged.items.is_empty() {
            tracing::warn!(
                "generation {generation} of program {}: the committee's proposals produced no adoptable item ({} dissenting); \
                 lower agents.research.min_supporters if planners propose different ideas",
                &program[..12.min(program.len())], r.merged.dissent.len()
            );
            return Ok(None);
        }
        let want = Agenda {
            program: program.into(), generation, evidence_root: r.evidence_root,
            committee: r.committee.keys().cloned().collect(),
            items: r.merged.items,
        };
        let already = recs.iter().any(|x| matches!(&x.kind, RecordKind::Provenance { kind, body } if kind == KIND_AGENDA
            && serde_json::from_value::<Agenda>(body.clone()).map_or(false, |a| a == want)));
        if already {
            return Ok(None);
        }
        Ok(Some(self.svc.author_in(program, KIND_AGENDA, serde_json::to_value(&want)?).await?.hash))
    }

    /// Executor role: claim what this node can do, build candidates, complete claims.
    /// Returns the number of candidates produced.
    pub async fn step_execute(&self, now: i64) -> Result<usize, BoxError> {
        let (Some(maker), Some(runner)) = (&self.maker, &self.runner) else { return Ok(0) };
        let recs = self.records().await;
        let market = Market::from_records(&recs);
        let graph = EvidenceGraph::from_records(&recs);
        let valid = self.valid_agendas(&recs);
        let mut made = 0;
        for (agenda_hash, item_id) in market.claimable(&self.caps, now) {
            if made >= self.capacity {
                break;
            }
            if !valid.contains(&agenda_hash) {
                continue; // not the committee's agenda: do not spend resources on it
            }
            if self.failed_here.lock().unwrap().contains(&(agenda_hash.clone(), item_id.clone())) || market.failures(&agenda_hash, &item_id) >= self.max_item_failures {
                continue; // already failed it here, or enough nodes have: not worth another try
            }
            // The agenda's program decides which lineage to build on.
            let Some(agenda) = recs.iter().find(|r| r.hash == agenda_hash) else { continue };
            let RecordKind::Provenance { body, .. } = &agenda.kind else { continue };
            let Ok(a) = serde_json::from_value::<Agenda>(body.clone()) else { continue };
            let Some(item) = a.items.iter().find(|i| i.id == item_id).cloned() else { continue };

            let claim = self.svc.author_in(&a.program, KIND_CLAIM, json!({"agenda": agenda_hash, "item_id": item_id, "action": "claim", "at": now, "lease_secs": self.lease_secs})).await?;
            // Did we win? Another node may have claimed the same item concurrently.
            let recs2 = self.records().await;
            match Market::from_records(&recs2).status(&agenda_hash, &item_id, now) {
                Some(ItemStatus::Claimed { claim: c, .. }) if c == claim.hash => {}
                _ => continue,
            }

            // Parents: the best existing candidates (gems first, then frontier).
            let lanes = graph.lanes(&a.program).unwrap_or_default();
            let parent_ids: Vec<String> = lanes.gems.iter().chain(&lanes.frontier).take(2).map(|e| e.candidate.clone()).collect();
            let mut parent_patches: Vec<Patch> = vec![];
            if parent_ids.is_empty() {
                // First generation: no lineage yet, but the maker still needs to see what it is changing.
                if let Some(b) = self.baseline_patch() {
                    parent_patches.push(b);
                }
            }
            for id in &parent_ids {
                if let Some(c) = graph.candidate(id) {
                    if let Some(b) = self.blobs.get(&c.body.diff_hash).await {
                        if let Ok(p) = Patch::from_bytes(&b) {
                            parent_patches.push(p);
                        }
                    }
                }
            }

            // GPU-seconds are charged up front against the node's hard ceiling; an item that does not
            // fit is handed back so another node can take it.
            if let Some(b) = &self.gpu_budget {
                if let Err(e) = b.charge_gpu_seconds(item.requirements.gpu_seconds.max(1)) {
                    tracing::warn!("item {item_id} not run: {e}");
                    self.svc.author_in(&a.program, KIND_CLAIM, json!({"agenda": agenda_hash, "item_id": item_id, "action": "release", "at": now, "claim": claim.hash})).await?;
                    continue;
                }
            }
            let hb = self.heartbeat_loop(&a.program, &agenda_hash, &item_id, &claim.hash);
            let outcome = tokio::select! {
                r = self.build_candidate(maker.as_ref(), runner.as_ref(), &item, &a.program, &parent_ids, &parent_patches, &claim.hash) => Some(r),
                _ = hb => None,
            };
            match outcome {
                Some(Ok((run_rec, _cand_hash))) => {
                    let still_mine = matches!(Market::from_records(&self.records().await).status(&agenda_hash, &item_id, now + 1), Some(ItemStatus::Claimed { claim: c, .. }) if c == claim.hash);
                    if still_mine {
                        self.svc.author_in(&a.program, KIND_CLAIM, json!({"agenda": agenda_hash, "item_id": item_id, "action": "complete", "at": now + 1, "claim": claim.hash, "result": run_rec})).await?;
                        made += 1;
                    }
                }
                Some(Err(e)) => {
                    tracing::warn!("item {item_id}: {e}");
                    self.failed_here.lock().unwrap().insert((agenda_hash.clone(), item_id.clone()));
                    self.svc.author_in(&a.program, KIND_CLAIM, json!({"agenda": agenda_hash, "item_id": item_id, "action": "release", "at": now + 1, "claim": claim.hash})).await?;
                }
                None => {}
            }
        }
        Ok(made)
    }

    /// Resolves only when the claim can no longer be kept alive (never in normal operation),
    /// emitting heartbeats while it is polled.
    async fn heartbeat_loop(&self, program: &str, agenda: &str, item: &str, claim: &str) {
        let mut tick = tokio::time::interval(Duration::from_secs((self.lease_secs / 3).max(1)));
        tick.tick().await;
        loop {
            tick.tick().await;
            let at = chrono::Utc::now().timestamp();
            if self.svc.author_in(program, KIND_CLAIM, json!({"agenda": agenda, "item_id": item, "action": "heartbeat", "at": at, "claim": claim})).await.is_err() {
                return;
            }
        }
    }

    async fn build_candidate(
        &self,
        maker: &dyn CandidateMaker,
        runner: &dyn Runner,
        item: &AgendaItem,
        program: &str,
        parents: &[String],
        parent_patches: &[Patch],
        claim: &str,
    ) -> Result<(String, String), String> {
        let patch = maker.make(item, parent_patches).await?;
        if let Some(allowed) = &self.allowed_paths {
            if !patch.within(allowed) {
                return Err(format!("patch touches files outside {allowed:?}"));
            }
        }
        let diff_blob = patch.to_bytes();
        let diff_hash = self.blobs.put(&diff_blob).await?;
        let wt = sandbox::create_worktree(&self.base_project, &self.work_root, &claim[..32]).map_err(|e| e.to_string())?;
        patch.apply(&wt)?;
        let seed = u64::from_le_bytes(Sha256::digest([claim.as_bytes(), b"seed"].concat())[..8].try_into().unwrap());
        let report = runner.run(&wt, seed).await?;
        let cand = self
            .svc
            .author_in(program, KIND_CANDIDATE, json!({"program": program, "diff_hash": diff_hash, "parents": parents, "agenda_item": item.id}))
            .await
            .map_err(|e| e.to_string())?;
        let run = self
            .svc
            .author_in(program, KIND_RUN, json!({"candidate": cand.hash, "container_hash": report.container_hash, "input_hashes": report.input_hashes, "seed": seed}))
            .await
            .map_err(|e| e.to_string())?;
        let _ = std::fs::remove_dir_all(&wt);
        Ok((run.hash, cand.hash))
    }

    /// Evaluator role: score every candidate this node has not yet scored.
    pub async fn step_evaluate(&self) -> Result<usize, BoxError> {
        let Some(ev) = &self.evaluator else { return Ok(0) };
        let recs = self.records().await;
        let graph = EvidenceGraph::from_records(&recs);
        let me = &self.svc.node_id;
        let mut done = 0;
        let programs: Vec<String> = recs
            .iter()
            .filter(|r| matches!(&r.kind, RecordKind::Provenance { kind, .. } if kind == KIND_OBJECTIVE))
            .map(|r| r.hash.clone())
            .collect();
        for program in programs {
            let metric = graph.objective(&program).map(|o| o.body.metric.clone()).unwrap_or_default();
            for c in graph.candidates_of(&program) {
                if graph.findings_of(&c.id).iter().any(|f| &f.author == me) {
                    continue;
                }
                let Some(run) = graph.runs_of(&c.id).into_iter().next() else { continue };
                let Some(blob) = self.blobs.get(&c.body.diff_hash).await else { continue };
                let Ok(patch) = Patch::from_bytes(&blob) else { continue };
                if self.allowed_paths.as_ref().map_or(false, |a| !patch.within(a)) {
                    continue; // someone's patch reaches beyond what this program permits: do not run it here
                }
                let report = RunReport { container_hash: run.body.container_hash.clone(), input_hashes: run.body.input_hashes.clone() };
                let Some(att) = ev.evaluate(&patch, &report).await else { continue };
                if att.metric != metric {
                    continue;
                }
                self.svc
                    .author_in(&program, KIND_FINDING, json!({"run": run.id, "candidate": c.id, "metric": att.metric, "value": att.value, "evaluator_code_hash": att.evaluator_code_hash, "data_merkle_root": att.data_merkle_root}))
                    .await?;
                done += 1;
            }
        }
        Ok(done)
    }

    /// Record a human gate decision (objective, budget or innovation-gate changes). Only the
    /// program owner's decisions count; see `EvidenceGraph::gate_approved`.
    pub async fn decide_gate(&self, program: &str, generation: u64, gate: &str, approved: bool) -> Result<String, BoxError> {
        let body = json!({"program": program, "generation": generation, "action": {"type": "gate", "gate": gate, "approved": approved}});
        Ok(self.svc.author_in(program, KIND_DECISION, body).await?.hash)
    }

    /// Has the owner approved this gate? Anything that changes the objective, raises a budget
    /// or opens an innovation gate must check this first.
    pub async fn gate_open(&self, program: &str, gate: &str) -> bool {
        EvidenceGraph::from_records(&self.records().await).gate_approved(program, gate)
    }

    /// Auditor role: triage every other evaluator's findings (outliers among the findings for
    /// the same candidate, suspiciously perfect scores), spot-check those plus a random
    /// fraction, and write each verdict to the log as a `policy.audit` record. Returns the
    /// number of audits completed.
    pub async fn step_audit(&self) -> Result<usize, BoxError> {
        use crate::agents::evaluator::{OutlierTriage, Triage};
        let Some(auditor) = &self.auditor else { return Ok(0) };
        let recs = self.records().await;
        let graph = EvidenceGraph::from_records(&recs);
        let done: BTreeSet<String> = recs
            .iter()
            .filter_map(|r| match &r.kind {
                RecordKind::Provenance { kind, body } if kind == KIND_POLICY_AUDIT && r.node_id == self.svc.node_id => body["action"].as_str().and_then(|a| a.strip_prefix("audit:")).and_then(|a| a.split(':').next()).map(String::from),
                _ => None,
            })
            .collect();
        let triage = OutlierTriage::default();
        let mut n = 0;
        let programs: Vec<String> = recs
            .iter()
            .filter(|r| matches!(&r.kind, RecordKind::Provenance { kind, .. } if kind == KIND_OBJECTIVE))
            .map(|r| r.hash.clone())
            .collect();
        for program in programs {
            for c in graph.candidates_of(&program) {
                let findings = graph.findings_of(&c.id);
                let Some(blob) = self.blobs.get(&c.body.diff_hash).await else { continue };
                let Ok(patch) = Patch::from_bytes(&blob) else { continue };
                if self.allowed_paths.as_ref().map_or(false, |a| !patch.within(a)) {
                    continue;
                }
                for f in &findings {
                    if f.author == self.svc.node_id || done.contains(&f.id) {
                        continue;
                    }
                    let peers: Vec<f64> = findings.iter().filter(|g| g.id != f.id && g.author != f.author).map(|g| g.body.value).collect();
                    // deterministic "random" pick so retries do not change the decision
                    let lucky = (u64::from_le_bytes(Sha256::digest(f.id.as_bytes())[..8].try_into().unwrap()) as f64 / u64::MAX as f64) < self.audit_rate;
                    if !(triage.is_suspect(f.body.value, &peers) || lucky) {
                        continue;
                    }
                    let Some(ok) = auditor.audit(&f.body, &f.author, &patch).await else { continue };
                    self.svc
                        .author_in(&program, KIND_POLICY_AUDIT, json!({"action": format!("audit:{}:{}", f.id, if ok { "consistent" } else { "inconsistent" }), "approved": ok, "decided_by": format!("auditor:{}", self.svc.node_id)}))
                        .await?;
                    if let Some(cb) = &self.on_audit {
                        cb(&f.author, ok);
                    }
                    n += 1;
                }
            }
        }
        Ok(n)
    }

    /// Redundant evaluation on shared benchmarks (exact class): when at least three evaluators scored
    /// the same candidate on the same data commitment with the same scoring code, their median is the
    /// consensus and anyone further than `tolerance` from it is flagged with a `policy.audit` record.
    /// No sandbox or data access is needed, only the findings on record.
    pub async fn step_consensus(&self, tolerance: f64) -> Result<usize, BoxError> {
        use crate::agents::evaluator::{redundant_consensus, Attestation};
        let recs = self.records().await;
        let graph = EvidenceGraph::from_records(&recs);
        let done: BTreeSet<String> = recs
            .iter()
            .filter_map(|r| match &r.kind {
                RecordKind::Provenance { kind, body } if kind == KIND_POLICY_AUDIT && r.node_id == self.svc.node_id => body["action"].as_str().and_then(|a| a.strip_prefix("consensus:")).and_then(|a| a.split(':').next()).map(String::from),
                _ => None,
            })
            .collect();
        let programs: Vec<String> = recs.iter().filter(|r| matches!(&r.kind, RecordKind::Provenance { kind, .. } if kind == KIND_OBJECTIVE)).map(|r| r.hash.clone()).collect();
        let mut n = 0;
        for program in programs {
            for c in graph.candidates_of(&program) {
                let mut groups: BTreeMap<(String, String), Vec<&crate::agents::graph::Node<Finding>>> = BTreeMap::new();
                for f in graph.findings_of(&c.id) {
                    groups.entry((f.body.data_merkle_root.clone(), f.body.evaluator_code_hash.clone())).or_default().push(f);
                }
                for fs in groups.values() {
                    let authors: BTreeSet<&String> = fs.iter().map(|f| &f.author).collect();
                    if authors.len() < 3 {
                        continue;
                    }
                    let atts: Vec<Attestation> = fs.iter().map(|f| Attestation { metric: f.body.metric.clone(), value: f.body.value, evaluator_code_hash: f.body.evaluator_code_hash.clone(), data_merkle_root: f.body.data_merkle_root.clone(), n_examples: 0, noised: false }).collect();
                    let Ok(cons) = redundant_consensus(&atts, 3, tolerance) else { continue };
                    for i in cons.outliers {
                        let f = fs[i];
                        if done.contains(&f.id) || f.author == self.svc.node_id {
                            continue;
                        }
                        self.svc
                            .author_in(&program, KIND_POLICY_AUDIT, json!({"action": format!("consensus:{}:outlier", f.id), "approved": false, "decided_by": format!("consensus:{}", self.svc.node_id)}))
                            .await?;
                        if let Some(cb) = &self.on_audit {
                            cb(&f.author, false);
                        }
                        n += 1;
                    }
                }
            }
        }
        Ok(n)
    }

    #[allow(dead_code)] // library surface: used by tests and integrators, or pending the settlement-chain decision
    pub async fn lanes(&self, program: &str) -> Option<Lanes> {
        EvidenceGraph::from_records(&self.records().await).lanes(program)
    }

    /// The credit table every contributor can recompute from the graph.
    #[allow(dead_code)] // library surface: used by tests and integrators, or pending the settlement-chain decision
    pub async fn credit_table(&self, program: &str, rule: &CreditRule) -> CreditTable {
        let graph = EvidenceGraph::from_records(&self.records().await);
        credit::compute(&graph.lineage(program), &graph.gem_targets(program), rule, &graph.evidence_root(program))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::agents::evaluator::{EvaluatorConfig, EvaluatorService, Metric, Row};
    use crate::agents::market::Market;
    use crate::agents::service::tests::{make_service, tmpdir};
    use crate::agents::settlement::{MemoryLedger, SettlementChain};
    use crate::identity::Keypair;

    // ---- a small simulated federation of 12 peers; the bus plays the gossip layer ----

    struct Peer { node: ResearchNode, id: String }

    /// Deliver every record to every peer (verified on receipt, as gossip would).
    async fn sync(peers: &[Peer]) {
        let mut all: Vec<UpdateRecord> = vec![];
        for p in peers { all.extend(p.node.records().await); }
        for p in peers {
            let mut h = p.node.svc.history.write().await;
            for r in &all { let _ = h.record_update(r.clone()); }
        }
    }

    /// Item descriptions look like "lr=0.05"; the patch carries that config.
    struct LrMaker;
    #[async_trait]
    impl CandidateMaker for LrMaker {
        async fn make(&self, item: &AgendaItem, parents: &[Patch]) -> Result<Patch, String> {
            let mut p = Patch::default();
            p.files.insert("config.json".into(), json!({"lr": item.description.trim_start_matches("lr=")}).to_string().into_bytes());
            p.files.insert("parents.txt".into(), parents.len().to_string().into_bytes());
            Ok(p)
        }
    }

    /// Checks the patch really reached the worktree before "training".
    struct FileRunner;
    #[async_trait]
    impl Runner for FileRunner {
        async fn run(&self, wt: &std::path::Path, seed: u64) -> Result<RunReport, String> {
            std::fs::read(wt.join("config.json")).map_err(|e| format!("patch not applied: {e}"))?;
            std::fs::read(wt.join("base.txt")).map_err(|e| format!("base project missing: {e}"))?;
            Ok(RunReport { container_hash: sha256_hex(b"test-image"), input_hashes: vec![sha256_hex(&seed.to_le_bytes())] })
        }
    }

    fn lr_of(patch: &Patch) -> f64 {
        let v: serde_json::Value = serde_json::from_slice(&patch.files["config.json"]).unwrap();
        v["lr"].as_str().unwrap().parse().unwrap()
    }

    /// The "model" behind a candidate: right on a fraction of rows that depends on its lr.
    fn accuracy_for(lr: f64) -> usize {
        match lr { x if (x - 0.01).abs() < 1e-9 => 62, x if (x - 0.05).abs() < 1e-9 => 81, x if (x - 0.1).abs() < 1e-9 => 93, _ => 40 }
    }

    struct PrivateEval { svc: EvaluatorService }
    #[async_trait]
    impl CandidateEvaluator for PrivateEval {
        async fn evaluate(&self, patch: &Patch, _run: &RunReport) -> Option<Attestation> {
            let pct = accuracy_for(lr_of(patch));
            let pred = move |inp: &[u8]| {
                let i: usize = std::str::from_utf8(inp).unwrap().parse().unwrap();
                if i % 100 < pct { format!("{}", i % 3).into_bytes() } else { b"wrong".to_vec() }
            };
            self.svc.evaluate("research-loop", &pred).ok()
        }
    }

    fn dataset(n: usize, salt: usize) -> EvaluatorService {
        let rows: Vec<Row> = (0..n).map(|i| Row { input: format!("{i}").into_bytes(), label: format!("{}", i % 3).into_bytes() }).collect();
        let mut cfg = EvaluatorConfig::new(Metric::Accuracy);
        cfg.queries_per_requester = 1000;
        let _ = salt;
        EvaluatorService::new(rows, cfg).unwrap()
    }

    struct FixedProposer(Vec<(&'static str, &'static str)>);
    #[async_trait]
    impl Proposer for FixedProposer {
        async fn propose_items(&self, _p: &str, _g: u64, _l: &Lanes) -> Vec<ProposedItem> {
            self.0.iter().map(|(id, d)| ProposedItem { id: id.to_string(), description: d.to_string(), tag: id.to_string(), requirements: Requirements::default() }).collect()
        }
    }

    async fn federation(n: usize) -> (Vec<Peer>, PathBuf) {
        let base = PathBuf::from(tmpdir("proj"));
        std::fs::write(base.join("base.txt"), "base").unwrap();
        let mut peers = vec![];
        for i in 0..n {
            let dir = tmpdir(&format!("peer{i}"));
            let svc = make_service(&dir, Keypair::generate(), |_| {}).await;
            let id = svc.node_id.clone();
            let caps = CapabilityAd { node_id: id.clone(), gpu_seconds: 1000, ..Default::default() };
            let blobs: Arc<dyn ArtifactStore> = Arc::new(LocalBlobs::open(PathBuf::from(&dir).join("blobs")).unwrap());
            let mut node = ResearchNode::new(svc, caps, blobs, base.clone(), PathBuf::from(&dir).join("work"));
            node.committee_size = 5;
            peers.push(Peer { node, id });
        }
        (peers, base)
    }

    /// Everything the plan promises for a research generation, end to end over 12 peers:
    /// signed records, VRF committee, verifiable agenda, task market, sandbox-style build,
    /// private evaluation at three data owners, identical lanes everywhere, credit and payout.
    #[tokio::test]
    async fn a_full_generation_over_twelve_peers() {
        let (mut peers, _base) = federation(12).await;
        // Roles: 0-8 planners+executors, 9-11 data-owning evaluators. Blob stores are shared out of band here
        // (production fetches them over iroh-blobs by hash).
        let shared: Arc<dyn ArtifactStore> = Arc::new(LocalBlobs::open(PathBuf::from(tmpdir("shared-blobs"))).unwrap());
        for (i, p) in peers.iter_mut().enumerate() {
            p.node.blobs = shared.clone();
            p.node.proposer = Some(Arc::new(FixedProposer(vec![("a", "lr=0.01"), ("b", "lr=0.05"), ("c", "lr=0.1"), ("d", "lr=0.5"), ("noise", "lr=0.99")].into_iter().take(if i == 0 { 5 } else { 4 }).collect())));
            if i < 9 {
                p.node.maker = Some(Arc::new(LrMaker));
                p.node.runner = Some(Arc::new(FileRunner));
                p.node.capacity = 1;
            } else {
                p.node.evaluator = Some(Arc::new(PrivateEval { svc: dataset(300, i) }));
            }
        }
        let owner = &peers[0];
        let program = owner.node.start_program(&Objective {
            goal: "find the best learning rate".into(), metric: "accuracy".into(), direction: Direction::Maximize,
            maturity: MaturityRules { min_evaluators: 2, frontier_size: 2, gem_threshold: 0.9 },
        }).await.unwrap();
        sync(&peers).await;

        // --- planning: everyone proposes, the VRF committee merges ---
        for p in &peers { p.node.propose(&program, 1).await.unwrap(); }
        sync(&peers).await;
        for p in &peers { p.node.finalize_agenda(&program, 1).await.unwrap(); sync(&peers).await; }
        let recs = peers[3].node.records().await;
        let agendas: Vec<&UpdateRecord> = recs.iter().filter(|r| matches!(&r.kind, RecordKind::Provenance { kind, .. } if kind == KIND_AGENDA)).collect();
        assert_eq!(agendas.len(), 1, "committee members converge on one identical agenda");
        let w = |_: &str| 1.0;
        for p in &peers {
            assert!(verify_agenda(&p.node.records().await, agendas[0], 5, &w, &MergeConfig::default()), "every node recomputes the same agenda");
        }
        // a doctored agenda does not verify
        let RecordKind::Provenance { body, .. } = &agendas[0].kind else { unreachable!() };
        let mut forged = body.clone();
        forged["items"].as_array_mut().unwrap().push(json!({"id": "evil", "description": "lr=9", "requirements": {}}));
        let mut fr = agendas[0].clone();
        fr.kind = RecordKind::Provenance { kind: KIND_AGENDA.into(), body: forged };
        fr.sign(&Keypair::generate());
        assert!(!verify_agenda(&recs, &fr, 5, &w, &MergeConfig::default()));
        let agenda: Agenda = serde_json::from_value(body.clone()).unwrap();
        assert_eq!(agenda.committee.len(), 5);
        let n_items = agenda.items.len();
        assert!(n_items >= 4, "items proposed by a majority make the agenda (got {n_items})");
        assert!(agenda.items.iter().all(|i| i.id != "noise"), "the item only planner 0 proposed is not adopted");

        // --- execution through the market ---
        let now = chrono::Utc::now().timestamp();
        for round in 0..4 {
            for p in peers.iter().take(9) {
                p.node.step_execute(now + round).await.unwrap();
                sync(&peers).await;
            }
        }
        let recs = peers[5].node.records().await;
        let market = Market::from_records(&recs);
        let agenda_hash = agendas[0].hash.clone();
        assert!(market.all_done(&agenda_hash, now + 5), "every agenda item was completed");
        let graph = EvidenceGraph::from_records(&recs);
        assert_eq!(graph.candidates_of(&program).len(), n_items, "one candidate per item: no duplicate work when claims are visible");

        // --- private evaluation at three data owners ---
        for p in peers.iter().skip(9) { assert!(p.node.step_evaluate().await.unwrap() > 0); sync(&peers).await; }
        for p in peers.iter().skip(9) { assert_eq!(p.node.step_evaluate().await.unwrap(), 0, "nothing is scored twice"); }

        // --- everyone derives the same lanes, evidence root and credit ---
        let lanes0 = peers[0].node.lanes(&program).await.unwrap();
        for p in &peers { assert_eq!(p.node.lanes(&program).await.unwrap(), lanes0); }
        assert_eq!(lanes0.gems.len(), 1, "lr=0.1 (93%) is the only Gem");
        assert!(lanes0.gems[0].score.unwrap() > 0.9);
        assert!(lanes0.incubator.is_empty());
        assert!(!lanes0.frontier.is_empty());
        let rule = CreditRule::standard();
        let t0 = peers[0].node.credit_table(&program, &rule).await;
        for p in &peers { assert_eq!(p.node.credit_table(&program, &rule).await, t0); }
        assert!(t0.per_node.values().sum::<f64>() > 0.9 - 1e-9);

        // --- milestone settlement ---
        let ledger = MemoryLedger::default();
        let ev = vec![program.clone()];
        ledger.credit("sponsor", 1000, &ev).unwrap();
        let table = credit::payouts(&t0.per_node, 1000);
        credit::settle_milestone(&ledger, "milestone-1", "sponsor", 1000, &table, &ev).unwrap();
        assert_eq!(table.values().sum::<u64>(), 1000);
        for (node, amount) in &table { assert_eq!(ledger.balance(node), *amount); }
        assert_eq!(ledger.balance("sponsor"), 0);
        // evaluators were paid for evaluating, executors for building
        assert!(peers.iter().skip(9).all(|p| table.get(&p.id).copied().unwrap_or(0) > 0));

        // --- all records everywhere are valid signed provenance ---
        for p in &peers {
            let h = p.node.svc.history.read().await;
            assert!(h.peer_updates.values().flatten().all(|r| r.verify().is_ok()));
        }
    }

    /// An attacker who is not on the VRF committee publishes an agenda that names itself as the committee.
    /// The task market would list its items, but executors verify the agenda against the proposals first.
    #[tokio::test]
    async fn an_agenda_not_produced_by_the_committee_is_never_executed() {
        let (mut peers, _base) = federation(4).await;
        for p in peers.iter_mut() { p.node.maker = Some(Arc::new(LrMaker)); p.node.runner = Some(Arc::new(FileRunner)); p.node.committee_size = 1; p.node.capacity = 5; }
        peers[1].node.proposer = Some(Arc::new(FixedProposer(vec![("honest", "lr=0.1")])));
        let program = peers[1].node.start_program(&Objective { goal: "g".into(), metric: "accuracy".into(), direction: Direction::Maximize, maturity: MaturityRules { min_evaluators: 1, frontier_size: 1, gem_threshold: 0.9 } }).await.unwrap();
        sync(&peers).await;
        peers[1].node.propose(&program, 1).await.unwrap();
        sync(&peers).await;
        let real = peers[1].node.finalize_agenda(&program, 1).await.unwrap().unwrap();
        // peer 3 forges an agenda: committee = itself, items = whatever it likes
        let forged = peers[3].node.svc.author(KIND_AGENDA, json!({
            "program": program, "generation": 1, "evidence_root": "0".repeat(64),
            "committee": [peers[3].id], "items": [{"id": "steal", "description": "lr=9", "requirements": {}}],
        })).await.unwrap().hash;
        sync(&peers).await;
        let recs = peers[0].node.records().await;
        // the market lists the forged item (it only checks that the author declared itself on the committee)...
        assert!(Market::from_records(&recs).requirements(&forged, "steal").is_some());
        // ...but the node's own check rejects the forged agenda and accepts the real one
        let valid = peers[0].node.valid_agendas(&recs);
        assert!(valid.contains(&real) && !valid.contains(&forged));
        let now = chrono::Utc::now().timestamp();
        peers[0].node.step_execute(now).await.unwrap();
        sync(&peers).await;
        let g = EvidenceGraph::from_records(&peers[0].node.records().await);
        let items: Vec<String> = g.candidates_of(&program).iter().filter_map(|c| c.body.agenda_item.clone()).collect();
        assert_eq!(items, ["honest"], "only the committee's item was built; the forged one was ignored");
    }

    #[tokio::test]
    async fn a_claim_abandoned_by_a_dead_node_is_picked_up_after_the_lease() {
        let (mut peers, _base) = federation(3).await;
        for p in peers.iter_mut() { p.node.maker = Some(Arc::new(LrMaker)); p.node.runner = Some(Arc::new(FileRunner)); p.node.capacity = 5; p.node.lease_secs = 30; }
        // peer 2 is the planner/owner
        peers[2].node.proposer = Some(Arc::new(FixedProposer(vec![("only", "lr=0.1")])));
        peers[2].node.committee_size = 1;
        let program = peers[2].node.start_program(&Objective { goal: "g".into(), metric: "accuracy".into(), direction: Direction::Maximize, maturity: MaturityRules { min_evaluators: 1, frontier_size: 1, gem_threshold: 0.9 } }).await.unwrap();
        sync(&peers).await;
        peers[2].node.propose(&program, 1).await.unwrap();
        sync(&peers).await;
        let agenda_hash = peers[2].node.finalize_agenda(&program, 1).await.unwrap().unwrap();
        sync(&peers).await;
        let now = chrono::Utc::now().timestamp();
        // peer 0 claims the only item and then dies without heartbeats or completion
        peers[0].node.svc.author(KIND_CLAIM, json!({"agenda": agenda_hash, "item_id": "only", "action": "claim", "at": now, "lease_secs": 30})).await.unwrap();
        sync(&peers).await;
        assert_eq!(peers[1].node.step_execute(now + 5).await.unwrap(), 0, "still held by peer 0");
        sync(&peers).await;
        assert_eq!(peers[1].node.step_execute(now + 31).await.unwrap(), 1, "lease lapsed: peer 1 takes over");
        sync(&peers).await;
        let recs = peers[0].node.records().await;
        assert!(Market::from_records(&recs).all_done(&agenda_hash, now + 32));
        assert_eq!(EvidenceGraph::from_records(&recs).candidates_of(&program).len(), 1);
    }

    /// The scheduler alone (no hand-driven steps) carries a program through several generations,
    /// and later generations build on the best candidates found so far.
    #[tokio::test]
    async fn ticks_drive_a_program_through_multiple_generations() {
        let (mut peers, _base) = federation(8).await;
        let shared: Arc<dyn ArtifactStore> = Arc::new(LocalBlobs::open(PathBuf::from(tmpdir("tick-blobs"))).unwrap());
        for (i, p) in peers.iter_mut().enumerate() {
            p.node.blobs = shared.clone();
            p.node.committee_size = 3;
            p.node.min_proposals = 3;
            p.node.wait_ticks = 2;
            p.node.max_generations = 3;
            // generation-dependent proposals: each planner suggests the same two ideas per generation
            struct PerGen;
            #[async_trait]
            impl Proposer for PerGen {
                async fn propose_items(&self, _p: &str, g: u64, lanes: &Lanes) -> Vec<ProposedItem> {
                    let best = lanes.gems.first().or(lanes.frontier.first()).map(|e| e.candidate[..6].to_string()).unwrap_or_else(|| "none".into());
                    ["lr=0.05", "lr=0.1"].iter().enumerate().map(|(k, d)| ProposedItem { id: format!("g{g}-{k}-after-{best}"), description: d.to_string(), tag: format!("t{k}"), requirements: Requirements::default() }).collect()
                }
            }
            if i < 6 {
                p.node.proposer = Some(Arc::new(PerGen));
                p.node.maker = Some(Arc::new(LrMaker));
                p.node.runner = Some(Arc::new(FileRunner));
                p.node.capacity = 1;
            } else {
                p.node.evaluator = Some(Arc::new(PrivateEval { svc: dataset(200, i) }));
            }
        }
        let program = peers[0].node.start_program(&Objective { goal: "g".into(), metric: "accuracy".into(), direction: Direction::Maximize, maturity: MaturityRules { min_evaluators: 2, frontier_size: 2, gem_threshold: 0.9 } }).await.unwrap();
        sync(&peers).await;
        let mut now = chrono::Utc::now().timestamp();
        let mut totals = TickReport::default();
        for _ in 0..40 {
            for p in &peers {
                let r = p.node.tick(now).await.unwrap();
                totals.proposed += r.proposed; totals.agendas += r.agendas; totals.executed += r.executed; totals.evaluated += r.evaluated;
                sync(&peers).await;
            }
            now += 1;
        }
        let recs = peers[0].node.records().await;
        let gens: BTreeSet<u64> = recs.iter().filter_map(|r| match &r.kind {
            RecordKind::Provenance { kind, body } if kind == KIND_AGENDA => serde_json::from_value::<Agenda>(body.clone()).ok().map(|a| a.generation),
            _ => None,
        }).collect();
        assert_eq!(gens, BTreeSet::from([1, 2, 3]), "three generations ran, then max_generations stopped it");
        let g = EvidenceGraph::from_records(&recs);
        assert!(g.candidates_of(&program).len() >= 6, "each generation produced candidates ({})", g.candidates_of(&program).len());
        // later generations build on earlier winners: some candidate has parents
        assert!(g.candidates_of(&program).iter().any(|c| !c.body.parents.is_empty()), "lineage forms across generations");
        let lanes = peers[0].node.lanes(&program).await.unwrap();
        assert!(!lanes.gems.is_empty());
        assert_eq!(totals.evaluated > 0 && totals.executed > 0, true);
        // every node agrees on the outcome
        for p in &peers { assert_eq!(p.node.lanes(&program).await.unwrap(), lanes); }
    }

    /// Triage picks out the evaluator whose score is far from its peers', the auditor checks it, and the
    /// verdict is a signed record; honest findings are not audited unless sampled.
    #[tokio::test]
    async fn the_auditor_role_checks_suspicious_findings_and_records_verdicts() {
        struct Scripted(std::sync::Mutex<Vec<(String, f64)>>, bool);
        #[async_trait]
        impl FindingAuditor for Scripted {
            async fn audit(&self, f: &Finding, author: &str, _p: &Patch) -> Option<bool> {
                self.0.lock().unwrap().push((author.to_string(), f.value));
                Some(self.1 == (f.value < 0.99)) // "consistent" unless the score is the inflated one
            }
        }
        let (mut peers, _base) = federation(6).await;
        let shared: Arc<dyn ArtifactStore> = Arc::new(LocalBlobs::open(PathBuf::from(tmpdir("audit-blobs"))).unwrap());
        for p in peers.iter_mut() { p.node.blobs = shared.clone(); p.node.audit_rate = 0.0; }
        // peers 1..=4 evaluate honestly; peer 4 inflates; peer 5 audits
        peers[0].node.maker = Some(Arc::new(LrMaker));
        peers[0].node.runner = Some(Arc::new(FileRunner));
        for i in 1..=3 { peers[i].node.evaluator = Some(Arc::new(PrivateEval { svc: dataset(200, i) })); }
        struct Inflator;
        #[async_trait]
        impl CandidateEvaluator for Inflator {
            async fn evaluate(&self, patch: &Patch, run: &RunReport) -> Option<Attestation> {
                let mut a = PrivateEval { svc: dataset(200, 9) }.evaluate(patch, run).await?;
                a.value = 1.0;
                Some(a)
            }
        }
        peers[4].node.evaluator = Some(Arc::new(Inflator));
        let log = Arc::new(Scripted(Default::default(), true));
        let told: Arc<std::sync::Mutex<Vec<(String, bool)>>> = Default::default();
        let told2 = told.clone();
        peers[5].node.auditor = Some(log.clone());
        peers[5].node.on_audit = Some(Arc::new(move |n, ok| told2.lock().unwrap().push((n.to_string(), ok))));

        let program = peers[0].node.start_program(&Objective { goal: "g".into(), metric: "accuracy".into(), direction: Direction::Maximize, maturity: MaturityRules { min_evaluators: 2, frontier_size: 1, gem_threshold: 0.95 } }).await.unwrap();
        // one candidate built by peer 0 directly (agenda machinery is covered elsewhere)
        let patch = LrMaker.make(&AgendaItem { id: "x".into(), description: "lr=0.05".into(), requirements: Requirements::default() }, &[]).await.unwrap();
        let diff_hash = shared.put(&patch.to_bytes()).await.unwrap();
        let cand = peers[0].node.svc.author(KIND_CANDIDATE, json!({"program": program, "diff_hash": diff_hash, "parents": [], "agenda_item": "x"})).await.unwrap();
        peers[0].node.svc.author(KIND_RUN, json!({"candidate": cand.hash, "container_hash": sha256_hex(b"img"), "input_hashes": [], "seed": 1})).await.unwrap();
        sync(&peers).await;
        for p in peers.iter().take(5).skip(1) { p.node.step_evaluate().await.unwrap(); sync(&peers).await; }

        assert_eq!(peers[5].node.step_audit().await.unwrap(), 1, "only the outlier is triaged");
        let seen = log.0.lock().unwrap().clone();
        assert_eq!(seen.len(), 1);
        assert_eq!(seen[0].0, peers[4].id);
        assert_eq!(seen[0].1, 1.0);
        assert_eq!(told.lock().unwrap().as_slice(), [(peers[4].id.clone(), false)]);
        // the verdict is on the log, signed by the auditor, and is not repeated
        sync(&peers).await;
        let recs = peers[0].node.records().await;
        let verdicts: Vec<&UpdateRecord> = recs.iter().filter(|r| matches!(&r.kind, RecordKind::Provenance { kind, body } if kind == KIND_POLICY_AUDIT && body["action"].as_str().map_or(false, |a| a.starts_with("audit:")))).collect();
        assert_eq!(verdicts.len(), 1);
        assert!(verdicts[0].verify().is_ok() && verdicts[0].node_id == peers[5].id);
        assert_eq!(peers[5].node.step_audit().await.unwrap(), 0, "each finding is audited once");
    }

    #[tokio::test]
    async fn redundant_evaluation_flags_the_evaluator_that_disagrees_with_its_peers() {
        let (mut peers, _base) = federation(6).await;
        let shared: Arc<dyn ArtifactStore> = Arc::new(LocalBlobs::open(PathBuf::from(tmpdir("cons-blobs"))).unwrap());
        for p in peers.iter_mut() { p.node.blobs = shared.clone(); }
        for i in 1..=3 { peers[i].node.evaluator = Some(Arc::new(PrivateEval { svc: dataset(200, i) })); }
        struct Liar;
        #[async_trait]
        impl CandidateEvaluator for Liar {
            async fn evaluate(&self, patch: &Patch, run: &RunReport) -> Option<Attestation> {
                let mut a = PrivateEval { svc: dataset(200, 0) }.evaluate(patch, run).await?;
                a.value = (a.value - 0.3).max(0.0); // suppresses a competitor's score
                Some(a)
            }
        }
        peers[4].node.evaluator = Some(Arc::new(Liar));
        let flagged: Arc<std::sync::Mutex<Vec<String>>> = Default::default();
        let f2 = flagged.clone();
        peers[5].node.on_audit = Some(Arc::new(move |n, ok| { if !ok { f2.lock().unwrap().push(n.to_string()); } }));
        let program = peers[0].node.start_program(&Objective { goal: "g".into(), metric: "accuracy".into(), direction: Direction::Maximize, maturity: MaturityRules { min_evaluators: 2, frontier_size: 1, gem_threshold: 0.95 } }).await.unwrap();
        let patch = LrMaker.make(&AgendaItem { id: "x".into(), description: "lr=0.1".into(), requirements: Requirements::default() }, &[]).await.unwrap();
        let diff_hash = shared.put(&patch.to_bytes()).await.unwrap();
        let cand = peers[0].node.svc.author(KIND_CANDIDATE, json!({"program": program, "diff_hash": diff_hash, "parents": [], "agenda_item": "x"})).await.unwrap();
        peers[0].node.svc.author(KIND_RUN, json!({"candidate": cand.hash, "container_hash": sha256_hex(b"i"), "input_hashes": [], "seed": 1})).await.unwrap();
        sync(&peers).await;
        // fewer than three evaluators: nothing to compare
        for i in 1..=2 { peers[i].node.step_evaluate().await.unwrap(); sync(&peers).await; }
        assert_eq!(peers[5].node.step_consensus(0.02).await.unwrap(), 0);
        for i in [3usize, 4] { peers[i].node.step_evaluate().await.unwrap(); sync(&peers).await; }
        assert_eq!(peers[5].node.step_consensus(0.02).await.unwrap(), 1);
        assert_eq!(flagged.lock().unwrap().as_slice(), [peers[4].id.clone()], "only the dissenting evaluator is flagged");
        assert_eq!(peers[5].node.step_consensus(0.02).await.unwrap(), 0, "flagged once");
    }

    #[tokio::test]
    async fn the_gpu_budget_is_a_hard_ceiling_and_unaffordable_items_are_handed_back() {
        use crate::agents::budget::{Budget, BudgetTracker};
        let (mut peers, _base) = federation(2).await;
        peers[0].node.maker = Some(Arc::new(LrMaker));
        peers[0].node.runner = Some(Arc::new(FileRunner));
        peers[0].node.capacity = 10;
        peers[0].node.caps.gpu_seconds = 100;
        peers[0].node.gpu_budget = Some(Arc::new(BudgetTracker::new(Budget { tokens: u64::MAX / 4, gpu_seconds: 10, wall_clock_secs: 3600 })));
        // each item declares 6 GPU-seconds; the budget affords one of them
        struct GpuProposer;
        #[async_trait]
        impl Proposer for GpuProposer {
            async fn propose_items(&self, _: &str, _: u64, _: &Lanes) -> Vec<ProposedItem> {
                ["a", "b"].iter().map(|id| ProposedItem { id: id.to_string(), description: "lr=0.1".into(), tag: id.to_string(), requirements: Requirements { gpu_class: None, data_access: None, gpu_seconds: 6 } }).collect()
            }
        }
        peers[1].node.proposer = Some(Arc::new(GpuProposer));
        peers[1].node.committee_size = 1;
        let program = peers[1].node.start_program(&Objective { goal: "g".into(), metric: "accuracy".into(), direction: Direction::Maximize, maturity: MaturityRules { min_evaluators: 1, frontier_size: 1, gem_threshold: 0.9 } }).await.unwrap();
        sync(&peers).await;
        peers[1].node.propose(&program, 1).await.unwrap();
        sync(&peers).await;
        peers[1].node.finalize_agenda(&program, 1).await.unwrap().unwrap();
        sync(&peers).await;
        let now = chrono::Utc::now().timestamp();
        assert_eq!(peers[0].node.step_execute(now).await.unwrap(), 1, "one item fits the 10 GPU-second budget");
        assert!(peers[0].node.gpu_budget.as_ref().unwrap().remaining().1 == 4);
        sync(&peers).await;
        // the other item was released, not left claimed forever
        let recs = peers[1].node.records().await;
        let released = recs.iter().filter(|r| matches!(&r.kind, RecordKind::Provenance { kind, body } if kind == KIND_CLAIM && body["action"] == "release")).count();
        assert_eq!(released, 1);
        let g = EvidenceGraph::from_records(&recs);
        assert_eq!(g.candidates_of(&program).len(), 1);
        // and the budget stays exhausted: nothing more runs
        assert_eq!(peers[0].node.step_execute(now + 5).await.unwrap(), 0);
    }

    #[tokio::test]
    async fn patches_outside_the_allowed_paths_are_rejected_by_executors_and_ignored_by_evaluators() {
        let (mut peers, _base) = federation(2).await;
        let shared: Arc<dyn ArtifactStore> = Arc::new(LocalBlobs::open(PathBuf::from(tmpdir("allow-blobs"))).unwrap());
        for p in peers.iter_mut() { p.node.blobs = shared.clone(); p.node.allowed_paths = Some(vec!["config.json".into()]); }
        struct Sneaky;
        #[async_trait]
        impl CandidateMaker for Sneaky {
            async fn make(&self, _: &AgendaItem, _: &[Patch]) -> Result<Patch, String> {
                let mut p = Patch::default();
                p.files.insert("config.json".into(), br#"{"lr": "0.1"}"#.to_vec());
                p.files.insert("predict.py".into(), b"print('cheat')".to_vec());
                Ok(p)
            }
        }
        peers[0].node.maker = Some(Arc::new(Sneaky));
        peers[0].node.runner = Some(Arc::new(FileRunner));
        peers[1].node.proposer = Some(Arc::new(FixedProposer(vec![("x", "lr=0.1")])));
        peers[1].node.committee_size = 1;
        peers[1].node.evaluator = Some(Arc::new(PrivateEval { svc: dataset(100, 1) }));
        let program = peers[1].node.start_program(&Objective { goal: "g".into(), metric: "accuracy".into(), direction: Direction::Maximize, maturity: MaturityRules { min_evaluators: 1, frontier_size: 1, gem_threshold: 0.9 } }).await.unwrap();
        sync(&peers).await;
        peers[1].node.propose(&program, 1).await.unwrap();
        sync(&peers).await;
        peers[1].node.finalize_agenda(&program, 1).await.unwrap();
        sync(&peers).await;
        let now = chrono::Utc::now().timestamp();
        assert_eq!(peers[0].node.step_execute(now).await.unwrap(), 0, "the executor refuses a patch that reaches beyond the allowed files");
        // a patch published by someone who does not enforce the rule is not scored by those who do
        let patch = Sneaky.make(&AgendaItem { id: "x".into(), description: "".into(), requirements: Requirements::default() }, &[]).await.unwrap();
        let diff = shared.put(&patch.to_bytes()).await.unwrap();
        let cand = peers[0].node.svc.author(KIND_CANDIDATE, json!({"program": program, "diff_hash": diff, "parents": [], "agenda_item": "x"})).await.unwrap();
        peers[0].node.svc.author(KIND_RUN, json!({"candidate": cand.hash, "container_hash": sha256_hex(b"i"), "input_hashes": [], "seed": 1})).await.unwrap();
        sync(&peers).await;
        assert_eq!(peers[1].node.step_evaluate().await.unwrap(), 0);
    }

    /// An item nobody can build must not wedge the program: each node tries it once, the failures are on the
    /// log, and the next generation starts without it.
    #[tokio::test]
    async fn an_unbuildable_item_is_tried_once_per_node_and_the_generation_moves_on() {
        struct PickyMaker;
        #[async_trait]
        impl CandidateMaker for PickyMaker {
            async fn make(&self, item: &AgendaItem, p: &[Patch]) -> Result<Patch, String> {
                if item.id == "bad" { Err("model reply unusable".into()) } else { LrMaker.make(item, p).await }
            }
        }
        let (mut peers, _base) = federation(4).await;
        let shared: Arc<dyn ArtifactStore> = Arc::new(LocalBlobs::open(PathBuf::from(tmpdir("fail-blobs"))).unwrap());
        for (i, p) in peers.iter_mut().enumerate() {
            p.node.blobs = shared.clone();
            p.node.committee_size = 1;
            p.node.min_proposals = 1;
            p.node.wait_ticks = 1;
            p.node.max_generations = 2;
            p.node.max_item_failures = 2;
            p.node.capacity = 5;
            if i < 3 {
                p.node.proposer = Some(Arc::new(FixedProposer(vec![("good", "lr=0.1"), ("bad", "lr=0.05")])));
                p.node.maker = Some(Arc::new(PickyMaker));
                p.node.runner = Some(Arc::new(FileRunner));
            }
        }
        let program = peers[3].node.start_program(&Objective { goal: "g".into(), metric: "accuracy".into(), direction: Direction::Maximize, maturity: MaturityRules { min_evaluators: 1, frontier_size: 1, gem_threshold: 0.9 } }).await.unwrap();
        sync(&peers).await;
        let mut now = chrono::Utc::now().timestamp();
        for _ in 0..12 {
            for p in &peers { p.node.tick(now).await.unwrap(); sync(&peers).await; }
            now += 1;
        }
        let recs = peers[0].node.records().await;
        let releases = recs.iter().filter(|r| matches!(&r.kind, RecordKind::Provenance { kind, body } if kind == KIND_CLAIM && body["action"] == "release")).count();
        // generation 1: each of the 3 executors failed "bad" once; generation 2 may add up to 3 more. Never a retry loop.
        assert!(releases >= 2 && releases <= 6, "releases = {releases}");
        let gens: BTreeSet<u64> = recs.iter().filter_map(|r| match &r.kind {
            RecordKind::Provenance { kind, body } if kind == KIND_AGENDA => serde_json::from_value::<Agenda>(body.clone()).ok().map(|a| a.generation),
            _ => None,
        }).collect();
        assert!(gens.contains(&2), "the program moved on to generation 2 despite the unbuildable item: {gens:?}");
    }

    #[tokio::test]
    async fn human_gates_need_the_owners_approval() {
        let (peers, _base) = federation(2).await;
        let program = peers[0].node.start_program(&Objective { goal: "g".into(), metric: "accuracy".into(), direction: Direction::Maximize, maturity: MaturityRules { min_evaluators: 1, frontier_size: 1, gem_threshold: 0.9 } }).await.unwrap();
        sync(&peers).await;
        assert!(!peers[1].node.gate_open(&program, "budget").await);
        // a non-owner's "approval" counts for nothing
        peers[1].node.decide_gate(&program, 1, "budget", true).await.unwrap();
        sync(&peers).await;
        assert!(!peers[0].node.gate_open(&program, "budget").await);
        peers[0].node.decide_gate(&program, 1, "budget", true).await.unwrap();
        sync(&peers).await;
        for p in &peers { assert!(p.node.gate_open(&program, "budget").await); assert!(!p.node.gate_open(&program, "objective").await); }
        // the owner can close it again with a later decision
        peers[0].node.decide_gate(&program, 2, "budget", false).await.unwrap();
        sync(&peers).await;
        assert!(!peers[1].node.gate_open(&program, "budget").await);
    }

    #[tokio::test]
    async fn a_failing_run_releases_its_claim_so_another_node_can_try() {
        struct Broken;
        #[async_trait]
        impl Runner for Broken { async fn run(&self, _: &std::path::Path, _: u64) -> Result<RunReport, String> { Err("boom".into()) } }
        let (mut peers, _base) = federation(2).await;
        peers[0].node.maker = Some(Arc::new(LrMaker));
        peers[0].node.runner = Some(Arc::new(Broken));
        peers[1].node.maker = Some(Arc::new(LrMaker));
        peers[1].node.runner = Some(Arc::new(FileRunner));
        peers[1].node.proposer = Some(Arc::new(FixedProposer(vec![("x", "lr=0.1")])));
        peers[1].node.committee_size = 1;
        let program = peers[1].node.start_program(&Objective { goal: "g".into(), metric: "accuracy".into(), direction: Direction::Maximize, maturity: MaturityRules { min_evaluators: 1, frontier_size: 1, gem_threshold: 0.9 } }).await.unwrap();
        sync(&peers).await;
        peers[1].node.propose(&program, 1).await.unwrap();
        sync(&peers).await;
        let agenda_hash = peers[1].node.finalize_agenda(&program, 1).await.unwrap().unwrap();
        sync(&peers).await;
        let now = chrono::Utc::now().timestamp();
        assert_eq!(peers[0].node.step_execute(now).await.unwrap(), 0);
        sync(&peers).await;
        assert!(Market::from_records(&peers[1].node.records().await).status(&agenda_hash, "x", now + 1) == Some(ItemStatus::Open), "released, not stuck");
        assert_eq!(peers[1].node.step_execute(now + 1).await.unwrap(), 1);
    }

    #[test]
    fn patch_roundtrip_hash_and_path_safety() {
        let mut p = Patch::default();
        p.files.insert("train.py".into(), b"print(1)".to_vec());
        p.files.insert("cfg/a.yaml".into(), b"lr: 3e-4".to_vec());
        let q = Patch::from_bytes(&p.to_bytes()).unwrap();
        assert_eq!(p, q);
        assert_eq!(p.hash(), q.hash());
        let mut r = p.clone();
        r.files.insert("train.py".into(), b"print(2)".to_vec());
        assert_ne!(p.hash(), r.hash());
        let dir = std::env::temp_dir().join(format!("slk-patch-{}", rand::random::<u64>()));
        std::fs::create_dir_all(&dir).unwrap();
        p.apply(&dir).unwrap();
        assert_eq!(std::fs::read(dir.join("cfg/a.yaml")).unwrap(), b"lr: 3e-4");
        for bad in ["../evil", "/etc/passwd", "a/../../b"] {
            let mut e = Patch::default();
            e.files.insert(bad.into(), vec![1]);
            assert!(e.apply(&dir).is_err(), "{bad}");
        }
        assert!(Patch::from_bytes(b"not json").is_err());
    }

    #[test]
    fn blob_store_is_content_addressed_and_detects_corruption() {
        let dir = std::env::temp_dir().join(format!("slk-blobs-{}", rand::random::<u64>()));
        let b = LocalBlobs::open(dir.clone()).unwrap();
        let h = b.put(b"hello").unwrap();
        assert_eq!(h, blake3_hex(b"hello"));
        assert_eq!(b.put(b"hello").unwrap(), h, "idempotent");
        assert_eq!(b.get(&h).unwrap(), b"hello");
        assert!(b.get(&"0".repeat(64)).is_none());
        assert!(b.get("../etc/passwd").is_none(), "non-hash names are refused");
        std::fs::write(dir.join(&h), b"tampered").unwrap();
        assert!(b.get(&h).is_none() && !b.has(&h));
    }
}
