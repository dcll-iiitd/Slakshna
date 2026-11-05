//! Evidence graph and lanes (A2).
//!
//! The graph is a grow-only DAG built purely from a *set* of verified
//! provenance records, so any two nodes holding the same records compute the
//! same graph and the same lanes regardless of arrival order. It holds hashes,
//! metrics and lineage only, never raw data.
//!
//! Object ids are record hashes. A program is identified by the hash of its
//! Objective record.

use crate::agents::records::*;
use crate::history::{RecordKind, UpdateRecord};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashMap};

#[derive(Debug, Clone)]
pub struct Node<T> {
    pub id: String,
    pub author: String,
    pub body: T,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LaneEntry {
    pub candidate: String,
    /// Median over evaluators of each evaluator's mean metric; None while no findings.
    pub score: Option<f64>,
    pub evaluators: u32,
}

#[derive(Debug, Default, Clone, PartialEq)]
pub struct Lanes {
    pub incubator: Vec<LaneEntry>,
    pub frontier: Vec<LaneEntry>,
    pub gems: Vec<LaneEntry>,
}

#[derive(Debug, Default)]
pub struct EvidenceGraph {
    objectives: BTreeMap<String, Node<Objective>>,
    agendas: BTreeMap<String, Node<Agenda>>,
    candidates: BTreeMap<String, Node<Candidate>>,
    runs: BTreeMap<String, Node<Run>>,
    findings: BTreeMap<String, Node<Finding>>,
    decisions: BTreeMap<String, Node<Decision>>,
}

impl EvidenceGraph {
    /// Build from any collection of records. Invalid records (bad signature,
    /// bad schema) and non-research kinds are skipped; duplicates collapse.
    pub fn from_records<'a>(records: impl IntoIterator<Item = &'a UpdateRecord>) -> Self {
        let mut g = EvidenceGraph::default();
        for r in records {
            g.insert(r);
        }
        g
    }

    pub fn insert(&mut self, r: &UpdateRecord) -> bool {
        let RecordKind::Provenance { kind, body } = &r.kind else { return false };
        if !kind.starts_with("research.") || r.verify().is_err() {
            return false;
        }
        macro_rules! put {
            ($map:ident, $ty:ty) => {{
                let Ok(b) = serde_json::from_value::<$ty>(body.clone()) else { return false };
                self.$map
                    .insert(r.hash.clone(), Node { id: r.hash.clone(), author: r.node_id.clone(), body: b });
                true
            }};
        }
        match kind.as_str() {
            KIND_OBJECTIVE => put!(objectives, Objective),
            KIND_AGENDA => put!(agendas, Agenda),
            KIND_CANDIDATE => put!(candidates, Candidate),
            KIND_RUN => put!(runs, Run),
            KIND_FINDING => put!(findings, Finding),
            KIND_DECISION => put!(decisions, Decision),
            _ => false,
        }
    }

    pub fn objective(&self, program: &str) -> Option<&Node<Objective>> {
        self.objectives.get(program)
    }

    pub fn candidate(&self, id: &str) -> Option<&Node<Candidate>> {
        self.candidates.get(id)
    }

    /// All objects of a program, by id. Used to derive the evidence root.
    fn program_object_ids<'a>(&'a self, program: &str) -> BTreeSet<&'a str> {
        let mut ids = BTreeSet::new();
        if let Some((k, _)) = self.objectives.get_key_value(program) {
            ids.insert(k.as_str());
        }
        for c in self.candidates.values().filter(|c| c.body.program == program) {
            ids.insert(c.id.as_str());
        }
        for r in self.runs.values() {
            if self.candidates.get(&r.body.candidate).is_some_and(|c| c.body.program == program) {
                ids.insert(r.id.as_str());
            }
        }
        for f in self.findings.values() {
            if self.candidates.get(&f.body.candidate).is_some_and(|c| c.body.program == program) {
                ids.insert(f.id.as_str());
            }
        }
        for d in self.decisions.values().filter(|d| d.body.program == program) {
            ids.insert(d.id.as_str());
        }
        ids
    }

    /// Hash over the sorted ids of every candidate, run, finding and decision
    /// of a program. Planners cite it so any node can check that two agendas
    /// were drawn from the same evidence.
    pub fn evidence_root(&self, program: &str) -> String {
        let mut h = Sha256::new();
        h.update(b"slakshna/evidence-root/v1\0");
        for id in self.program_object_ids(program) {
            h.update(id.as_bytes());
        }
        hex::encode(h.finalize())
    }

    /// Candidates retired by an authorized decision. A decision is authorized
    /// when its author owns the program or sat on the committee of any agenda
    /// of that program (same generation or earlier).
    fn retired<'a>(&'a self, program: &str) -> BTreeSet<&'a str> {
        let owner = self.objectives.get(program).map(|o| o.author.as_str());
        let mut out = BTreeSet::new();
        for d in self.decisions.values().filter(|d| d.body.program == program) {
            let authorized = Some(d.author.as_str()) == owner
                || self.agendas.values().any(|a| {
                    a.body.program == program
                        && a.body.generation <= d.body.generation
                        && a.body.committee.contains(&d.author)
                });
            if !authorized {
                continue;
            }
            if let DecisionAction::Retire { candidate } = &d.body.action {
                out.insert(candidate.as_str());
            }
        }
        out
    }

    /// Whether a gate has been approved by the program owner.
    pub fn gate_approved(&self, program: &str, gate: &str) -> bool {
        let Some(owner) = self.objectives.get(program).map(|o| &o.author) else { return false };
        let mut verdict = false;
        // Latest owner decision wins; ties broken by record hash (BTreeMap order).
        let mut best: Option<(u64, &str)> = None;
        for d in self.decisions.values().filter(|d| d.body.program == program && &d.author == owner) {
            if let DecisionAction::Gate { gate: g, approved } = &d.body.action {
                if g == gate && best.map_or(true, |b| (d.body.generation, d.id.as_str()) > b) {
                    best = Some((d.body.generation, d.id.as_str()));
                    verdict = *approved;
                }
            }
        }
        verdict
    }

    /// Candidates of a program with lineage and the data owners that evaluated each
    /// (input to the credit rule). Sorted by id.
    pub fn lineage(&self, program: &str) -> Vec<crate::agents::credit::LineageCandidate> {
        self.candidates
            .values()
            .filter(|c| c.body.program == program)
            .map(|c| {
                let mut evals: Vec<String> = self.findings.values().filter(|f| f.body.candidate == c.id).map(|f| f.author.clone()).collect();
                evals.sort();
                evals.dedup();
                crate::agents::credit::LineageCandidate { id: c.id.clone(), author: c.author.clone(), parents: c.body.parents.clone(), evaluators: evals }
            })
            .collect()
    }

    /// The Gems lane as credit targets (score = the candidate's lane score).
    pub fn gem_targets(&self, program: &str) -> Vec<crate::agents::credit::Target> {
        self.lanes(program)
            .map(|l| l.gems.into_iter().filter_map(|e| e.score.map(|s| crate::agents::credit::Target { candidate: e.candidate, score: s })).collect())
            .unwrap_or_default()
    }

    /// Run records of a program's candidate, lowest id first.
    pub fn runs_of(&self, candidate: &str) -> Vec<&Node<Run>> {
        self.runs.values().filter(|r| r.body.candidate == candidate).collect()
    }

    pub fn candidates_of(&self, program: &str) -> Vec<&Node<Candidate>> {
        self.candidates.values().filter(|c| c.body.program == program).collect()
    }

    pub fn findings_of(&self, candidate: &str) -> Vec<&Node<Finding>> {
        self.findings.values().filter(|f| f.body.candidate == candidate).collect()
    }

    /// Lanes as a deterministic view of the graph under the program's
    /// preregistered maturity rules.
    ///
    /// * score: for each evaluator (node) the mean of its findings on the
    ///   objective metric; the candidate's score is the median across evaluators.
    /// * incubator: fewer than `min_evaluators` distinct evaluators (and not retired).
    /// * mature candidates are ranked by score (direction aware, ties by id);
    ///   those at or beyond `gem_threshold` are Gems, and the best
    ///   `frontier_size` of the rest form the frontier.
    pub fn lanes(&self, program: &str) -> Option<Lanes> {
        let obj = &self.objectives.get(program)?.body;
        let retired = self.retired(program);

        let mut per_cand: HashMap<&str, BTreeMap<&str, Vec<f64>>> = HashMap::new();
        for f in self.findings.values() {
            let Some(c) = self.candidates.get(&f.body.candidate) else { continue };
            if c.body.program != program || f.body.metric != obj.metric {
                continue;
            }
            // The finding must refer to a known run of the same candidate.
            match self.runs.get(&f.body.run) {
                Some(r) if r.body.candidate == f.body.candidate => {}
                _ => continue,
            }
            per_cand
                .entry(f.body.candidate.as_str())
                .or_default()
                .entry(f.author.as_str())
                .or_default()
                .push(f.body.value);
        }

        let mut lanes = Lanes::default();
        let mut mature: Vec<LaneEntry> = Vec::new();
        for c in self.candidates.values().filter(|c| c.body.program == program) {
            if retired.contains(c.id.as_str()) {
                continue;
            }
            let (score, evaluators) = match per_cand.get(c.id.as_str()) {
                None => (None, 0),
                Some(by_eval) => {
                    let means: Vec<f64> = by_eval
                        .values()
                        .map(|v| v.iter().sum::<f64>() / v.len() as f64)
                        .collect();
                    (Some(median(means.clone())), means.len() as u32)
                }
            };
            let entry = LaneEntry { candidate: c.id.clone(), score, evaluators };
            if evaluators < obj.maturity.min_evaluators {
                lanes.incubator.push(entry);
            } else {
                mature.push(entry);
            }
        }

        let sign = match obj.direction {
            Direction::Maximize => 1.0,
            Direction::Minimize => -1.0,
        };
        mature.sort_by(|a, b| {
            let (sa, sb) = (a.score.unwrap_or(f64::NEG_INFINITY) * sign, b.score.unwrap_or(f64::NEG_INFINITY) * sign);
            sb.partial_cmp(&sa).unwrap().then_with(|| a.candidate.cmp(&b.candidate))
        });
        // Gems are reported only in the Gems lane; the frontier ranks the
        // remaining mature candidates.
        let mut non_gems = 0u32;
        for e in mature {
            let is_gem = e.score.is_some_and(|s| s * sign >= obj.maturity.gem_threshold * sign);
            if is_gem {
                lanes.gems.push(e);
            } else if non_gems < obj.maturity.frontier_size {
                non_gems += 1;
                lanes.frontier.push(e);
            }
        }
        Some(lanes)
    }
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 { v[n / 2] } else { (v[n / 2 - 1] + v[n / 2]) / 2.0 }
}

/// Build an unsigned record carrying a typed research/agent body (test helper).
#[cfg(test)]
pub fn make_record<T: serde::Serialize>(node_id: &str, prev_hash: &str, kind: &str, body: &T) -> UpdateRecord {
    UpdateRecord {
        node_id: node_id.to_string(),
        prev_hash: prev_hash.to_string(),
        kind: provenance(kind, body),
        signature: String::new(),
        hash: String::new(),
        public_key: String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::Keypair;

    struct Net { nodes: Vec<(Keypair, String)>, records: Vec<UpdateRecord> }

    impl Net {
        fn new(n: usize) -> Self {
            Net { nodes: (0..n).map(|_| { let k = Keypair::generate(); let id = k.node_id().to_string(); (k, id) }).collect(), records: vec![] }
        }
        fn emit<T: serde::Serialize>(&mut self, who: usize, kind: &str, body: &T) -> String {
            let (k, id) = &self.nodes[who];
            let prev = self.records.iter().rev().find(|r| &r.node_id == id).map(|r| r.hash.clone()).unwrap_or("0".repeat(64));
            let mut r = make_record(id, &prev, kind, body);
            r.sign(k);
            let h = r.hash.clone();
            self.records.push(r);
            h
        }
    }

    fn setup() -> (Net, String, Vec<String>) {
        let mut n = Net::new(4); // 0 owner/executor, 1..3 evaluators
        let prog = n.emit(0, KIND_OBJECTIVE, &Objective {
            goal: "maximize acc".into(), metric: "acc".into(), direction: Direction::Maximize,
            maturity: MaturityRules { min_evaluators: 2, frontier_size: 1, gem_threshold: 0.9 },
        });
        let mut cands = vec![];
        for (i, vals) in [[0.5, 0.6, 0.55], [0.7, 0.8, 0.75], [0.95, 0.93, 0.92]].iter().enumerate() {
            let c = n.emit(0, KIND_CANDIDATE, &Candidate {
                program: prog.clone(), diff_hash: format!("{:064x}", i), parents: vec![], agenda_item: None });
            let run = n.emit(0, KIND_RUN, &Run { candidate: c.clone(), container_hash: "c".repeat(64), input_hashes: vec![], seed: i as u64 });
            for (e, v) in vals.iter().enumerate() {
                n.emit(e + 1, KIND_FINDING, &Finding {
                    run: run.clone(), candidate: c.clone(), metric: "acc".into(), value: *v,
                    evaluator_code_hash: "e".repeat(64), data_merkle_root: "d".repeat(64) });
            }
            cands.push(c);
        }
        (n, prog, cands)
    }

    #[test]
    fn lanes_follow_preregistered_rules() {
        let (n, prog, c) = setup();
        let g = EvidenceGraph::from_records(&n.records);
        let l = g.lanes(&prog).unwrap();
        assert!(l.incubator.is_empty());
        assert_eq!(l.gems.iter().map(|e| &e.candidate).collect::<Vec<_>>(), vec![&c[2]]);
        // Of the two non-gem mature candidates only the best fits the frontier of size 1.
        assert_eq!(l.frontier.iter().map(|e| &e.candidate).collect::<Vec<_>>(), vec![&c[1]]);
    }

    #[test]
    fn identical_evidence_in_any_order_gives_identical_lanes() {
        let (n, prog, _) = setup();
        let a = EvidenceGraph::from_records(&n.records);
        let mut rev = n.records.clone();
        rev.reverse();
        // Also duplicate records, as gossip re-delivery would.
        rev.extend(n.records.iter().cloned());
        let b = EvidenceGraph::from_records(&rev);
        assert_eq!(a.lanes(&prog), b.lanes(&prog));
        assert_eq!(a.evidence_root(&prog), b.evidence_root(&prog));
    }

    #[test]
    fn candidate_with_too_few_evaluators_stays_in_incubator() {
        let (mut n, prog, _) = setup();
        let c = n.emit(0, KIND_CANDIDATE, &Candidate { program: prog.clone(), diff_hash: "9".repeat(64), parents: vec![], agenda_item: None });
        let run = n.emit(0, KIND_RUN, &Run { candidate: c.clone(), container_hash: "c".repeat(64), input_hashes: vec![], seed: 9 });
        n.emit(1, KIND_FINDING, &Finding { run, candidate: c.clone(), metric: "acc".into(), value: 0.99,
            evaluator_code_hash: "e".repeat(64), data_merkle_root: "d".repeat(64) });
        let l = EvidenceGraph::from_records(&n.records).lanes(&prog).unwrap();
        assert_eq!(l.incubator.len(), 1);
        assert_eq!(l.incubator[0].candidate, c);
    }

    #[test]
    fn one_evaluator_cannot_inflate_by_repeating_findings() {
        let (mut n, prog, _) = setup();
        let c = n.emit(0, KIND_CANDIDATE, &Candidate { program: prog.clone(), diff_hash: "8".repeat(64), parents: vec![], agenda_item: None });
        let run = n.emit(0, KIND_RUN, &Run { candidate: c.clone(), container_hash: "c".repeat(64), input_hashes: vec![], seed: 8 });
        for _ in 0..5 {
            n.emit(1, KIND_FINDING, &Finding { run: run.clone(), candidate: c.clone(), metric: "acc".into(), value: 0.99,
                evaluator_code_hash: "e".repeat(64), data_merkle_root: "d".repeat(64) });
        }
        let l = EvidenceGraph::from_records(&n.records).lanes(&prog).unwrap();
        assert!(l.incubator.iter().any(|e| e.candidate == c && e.evaluators == 1));
    }

    #[test]
    fn minimize_direction_flips_ranking() {
        let mut n = Net::new(3);
        let prog = n.emit(0, KIND_OBJECTIVE, &Objective {
            goal: "min loss".into(), metric: "loss".into(), direction: Direction::Minimize,
            maturity: MaturityRules { min_evaluators: 1, frontier_size: 2, gem_threshold: 0.1 },
        });
        let mut ids = vec![];
        for (i, v) in [0.5, 0.2, 0.05].iter().enumerate() {
            let c = n.emit(0, KIND_CANDIDATE, &Candidate { program: prog.clone(), diff_hash: format!("{:064x}", i), parents: vec![], agenda_item: None });
            let run = n.emit(0, KIND_RUN, &Run { candidate: c.clone(), container_hash: "c".repeat(64), input_hashes: vec![], seed: 0 });
            n.emit(1, KIND_FINDING, &Finding { run, candidate: c.clone(), metric: "loss".into(), value: *v,
                evaluator_code_hash: "e".repeat(64), data_merkle_root: "d".repeat(64) });
            ids.push(c);
        }
        let l = EvidenceGraph::from_records(&n.records).lanes(&prog).unwrap();
        assert_eq!(l.gems[0].candidate, ids[2]);
        assert_eq!(l.frontier.iter().map(|e| e.candidate.clone()).collect::<Vec<_>>(), vec![ids[1].clone(), ids[0].clone()]);
    }

    #[test]
    fn only_authorized_decisions_retire_candidates() {
        let (mut n, prog, c) = setup();
        // Node 2 is a plain evaluator: its Retire is ignored.
        n.emit(2, KIND_DECISION, &Decision { program: prog.clone(), generation: 0, action: DecisionAction::Retire { candidate: c[2].clone() } });
        let l = EvidenceGraph::from_records(&n.records).lanes(&prog).unwrap();
        assert_eq!(l.gems.len(), 1);
        // The owner's Retire takes effect.
        n.emit(0, KIND_DECISION, &Decision { program: prog.clone(), generation: 0, action: DecisionAction::Retire { candidate: c[2].clone() } });
        let l = EvidenceGraph::from_records(&n.records).lanes(&prog).unwrap();
        assert!(l.gems.is_empty());
    }

    #[test]
    fn forged_or_unsigned_records_are_ignored() {
        let (mut n, prog, _) = setup();
        let before = EvidenceGraph::from_records(&n.records).evidence_root(&prog);
        let (k, id) = (n.nodes[1].0.clone(), n.nodes[1].1.clone());
        let mut forged = make_record(&id, &"0".repeat(64), KIND_CANDIDATE,
            &Candidate { program: prog.clone(), diff_hash: "7".repeat(64), parents: vec![], agenda_item: None });
        forged.sign(&k);
        forged.signature = "00".repeat(64);
        let mut unsigned = forged.clone();
        unsigned.signature.clear();
        unsigned.public_key.clear();
        n.records.push(forged);
        n.records.push(unsigned);
        assert_eq!(EvidenceGraph::from_records(&n.records).evidence_root(&prog), before);
    }

    #[test]
    fn gate_approval_requires_owner() {
        let (mut n, prog, _) = setup();
        n.emit(1, KIND_DECISION, &Decision { program: prog.clone(), generation: 1, action: DecisionAction::Gate { gate: "budget".into(), approved: true } });
        assert!(!EvidenceGraph::from_records(&n.records).gate_approved(&prog, "budget"));
        n.emit(0, KIND_DECISION, &Decision { program: prog.clone(), generation: 1, action: DecisionAction::Gate { gate: "budget".into(), approved: true } });
        assert!(EvidenceGraph::from_records(&n.records).gate_approved(&prog, "budget"));
    }
}
