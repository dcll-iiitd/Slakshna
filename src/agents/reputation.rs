//! Reputation and accountability (M4).
//!
//! * Reputation is per (agent, role): a Beta posterior over task outcomes and
//!   peer reviews. The owning node's trust is the *prior*, so a fresh agent of a
//!   trusted node starts from that trust, and evidence takes over as it accrues.
//! * Reputation belongs to the agent key under its node and cannot be moved to
//!   another key. Creating an agent costs node stake (`AgentRegistry`).
//! * Collusion signals - reciprocal-review rings and correlated outlier votes -
//!   multiply a flagged agent's review weight by `COLLUSION_PENALTY`.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap};

pub const PRIOR_STRENGTH: f64 = 4.0;
pub const COLLUSION_PENALTY: f64 = 0.1;

pub type AgentKey = (String, String); // (node id, agent public key hex)

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Role {
    Executor,
    Evaluator,
    Planner,
    Reviewer,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Outcome {
    pub agent: AgentKey,
    pub role: Role,
    pub success: bool,
}

/// A reviewer's score in [0,1] for a target agent's work on `task`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Review {
    pub task: String,
    pub reviewer: AgentKey,
    pub target: AgentKey,
    pub role: Role,
    pub score: f64,
}

#[derive(Debug, Clone)]
pub struct CollusionConfig {
    pub positive_score: f64,
    /// Reviews in each direction before a pair counts as mutual.
    pub min_mutual: usize,
    pub min_ring_size: usize,
    /// How far above the per-task consensus a ring rates its own members.
    pub insider_gap: f64,
    pub outlier_distance: f64,
    pub min_co_outlier_tasks: usize,
}

impl Default for CollusionConfig {
    fn default() -> Self {
        CollusionConfig { positive_score: 0.8, min_mutual: 2, min_ring_size: 3, insider_gap: 0.4, outlier_distance: 0.3, min_co_outlier_tasks: 5 }
    }
}

#[derive(Debug, Default, Clone, PartialEq)]
pub struct CollusionFlags {
    pub ring_members: BTreeSet<AgentKey>,
    pub correlated_outliers: BTreeSet<AgentKey>,
}

impl CollusionFlags {
    pub fn flagged(&self, a: &AgentKey) -> bool {
        self.ring_members.contains(a) || self.correlated_outliers.contains(a)
    }
}

pub fn detect_collusion(reviews: &[Review], cfg: &CollusionConfig) -> CollusionFlags {
    let mut flags = CollusionFlags::default();

    // Median score of each (task, target) over all its reviewers (needs >= 3).
    let consensus: BTreeMap<(&str, &AgentKey), f64> = {
        let mut g: BTreeMap<(&str, &AgentKey), Vec<f64>> = BTreeMap::new();
        for r in reviews {
            g.entry((r.task.as_str(), &r.target)).or_default().push(r.score);
        }
        g.into_iter()
            .filter(|(_, v)| v.len() >= 3)
            .map(|(k, mut v)| {
                v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                (k, v[v.len() / 2])
            })
            .collect()
    };

    // --- reciprocal-review rings ---
    let mut pos: BTreeMap<(&AgentKey, &AgentKey), usize> = BTreeMap::new();
    for r in reviews.iter().filter(|r| r.score >= cfg.positive_score && r.reviewer != r.target) {
        *pos.entry((&r.reviewer, &r.target)).or_default() += 1;
    }
    // undirected "mutual" edges
    let mut adj: BTreeMap<&AgentKey, BTreeSet<&AgentKey>> = BTreeMap::new();
    for ((a, b), n) in &pos {
        if *n >= cfg.min_mutual && pos.get(&(*b, *a)).copied().unwrap_or(0) >= cfg.min_mutual {
            adj.entry(a).or_default().insert(b);
            adj.entry(b).or_default().insert(a);
        }
    }
    let mut seen: BTreeSet<&AgentKey> = BTreeSet::new();
    for start in adj.keys() {
        if seen.contains(start) {
            continue;
        }
        let mut comp: BTreeSet<&AgentKey> = BTreeSet::new();
        let mut stack = vec![*start];
        while let Some(x) = stack.pop() {
            if comp.insert(x) {
                stack.extend(adj[x].iter().copied());
            }
        }
        seen.extend(comp.iter().copied());
        if comp.len() < cfg.min_ring_size {
            continue;
        }
        // A ring is a mutual-praise cluster that rates its own members far above
        // what the rest of the reviewers of the same work say. Measuring
        // inflation against the per-task consensus (not against "outsiders")
        // keeps an honest cluster that is being attacked from looking like a ring.
        let mut inflation = vec![];
        for r in reviews.iter().filter(|r| comp.contains(&r.reviewer) && comp.contains(&r.target) && r.reviewer != r.target) {
            if let Some(m) = consensus.get(&(r.task.as_str(), &r.target)) {
                inflation.push(r.score - m);
            }
        }
        if !inflation.is_empty() && inflation.iter().sum::<f64>() / inflation.len() as f64 >= cfg.insider_gap {
            flags.ring_members.extend(comp.iter().map(|a| (*a).clone()));
        }
    }

    // --- correlated outlier votes ---
    let mut by_task_target: BTreeMap<(&str, &AgentKey), Vec<&Review>> = BTreeMap::new();
    for r in reviews {
        by_task_target.entry((r.task.as_str(), &r.target)).or_default().push(r);
    }
    // (a,b) -> (co-outlier tasks, shared tasks)
    let mut co: BTreeMap<(&AgentKey, &AgentKey), (usize, usize)> = BTreeMap::new();
    for rs in by_task_target.values().filter(|rs| rs.len() >= 3) {
        let mut scores: Vec<f64> = rs.iter().map(|r| r.score).collect();
        scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = scores[scores.len() / 2];
        let sign = |r: &Review| {
            let d = r.score - med;
            if d.abs() >= cfg.outlier_distance { Some(d > 0.0) } else { None }
        };
        for (i, a) in rs.iter().enumerate() {
            for b in rs.iter().skip(i + 1) {
                let (x, y) = if a.reviewer <= b.reviewer { (&a.reviewer, &b.reviewer) } else { (&b.reviewer, &a.reviewer) };
                let e = co.entry((x, y)).or_default();
                e.1 += 1;
                if let (Some(sa), Some(sb)) = (sign(a), sign(b)) {
                    if sa == sb {
                        e.0 += 1;
                    }
                }
            }
        }
    }
    for ((a, b), (n, shared)) in co {
        if n >= cfg.min_co_outlier_tasks && n as f64 >= 0.6 * shared as f64 {
            flags.correlated_outliers.insert(a.clone());
            flags.correlated_outliers.insert(b.clone());
        }
    }
    flags
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
#[allow(dead_code)] // library surface: used by tests and integrators, or pending the settlement-chain decision
pub enum RegistryError {
    #[error("stake {0} below the minimum {1}")]
    StakeTooLow(u64, u64),
    #[error("node already runs the maximum of {0} agents")]
    QuotaReached(usize),
    #[error("agent already registered")]
    Duplicate,
    #[error("insufficient node stake: has {have}, needs {need}")]
    Insufficient { have: u64, need: u64 },
}

/// Creating agents costs node stake and is quota-limited (sybil resistance).
#[derive(Debug, Default)]
#[allow(dead_code)] // library surface: used by tests and integrators, or pending the settlement-chain decision
pub struct AgentRegistry {
    pub min_stake: u64,
    pub max_agents_per_node: usize,
    node_stake: HashMap<String, u64>,
    agents: BTreeMap<AgentKey, u64>, // agent -> stake locked
}

impl AgentRegistry {
    #[allow(dead_code)] // library surface: used by tests and integrators, or pending the settlement-chain decision
    pub fn new(min_stake: u64, max_agents_per_node: usize) -> Self {
        AgentRegistry { min_stake, max_agents_per_node, ..Default::default() }
    }

    #[allow(dead_code)] // library surface: used by tests and integrators, or pending the settlement-chain decision
    pub fn deposit(&mut self, node: &str, amount: u64) {
        *self.node_stake.entry(node.to_string()).or_default() += amount;
    }

    #[allow(dead_code)] // library surface: used by tests and integrators, or pending the settlement-chain decision
    pub fn register(&mut self, node: &str, agent_pk: &str) -> Result<(), RegistryError> {
        let key = (node.to_string(), agent_pk.to_string());
        if self.agents.contains_key(&key) {
            return Err(RegistryError::Duplicate);
        }
        if self.agents.keys().filter(|(n, _)| n == node).count() >= self.max_agents_per_node {
            return Err(RegistryError::QuotaReached(self.max_agents_per_node));
        }
        let have = self.node_stake.get(node).copied().unwrap_or(0);
        if have < self.min_stake {
            return Err(RegistryError::Insufficient { have, need: self.min_stake });
        }
        *self.node_stake.get_mut(node).unwrap() -= self.min_stake;
        self.agents.insert(key, self.min_stake);
        Ok(())
    }

    #[allow(dead_code)] // library surface: used by tests and integrators, or pending the settlement-chain decision
    pub fn is_registered(&self, node: &str, agent_pk: &str) -> bool {
        self.agents.contains_key(&(node.to_string(), agent_pk.to_string()))
    }

    /// Burn an agent's locked stake and deregister it. Returns the amount slashed.
    #[allow(dead_code)] // library surface: used by tests and integrators, or pending the settlement-chain decision
    pub fn slash(&mut self, node: &str, agent_pk: &str) -> u64 {
        self.agents.remove(&(node.to_string(), agent_pk.to_string())).unwrap_or(0)
    }

    #[allow(dead_code)] // library surface: pending the settlement-chain decision
    pub fn stake_of(&self, node: &str) -> u64 {
        self.node_stake.get(node).copied().unwrap_or(0)
    }
}

#[derive(Debug, Default)]
pub struct ReputationBook {
    node_trust: HashMap<String, f64>,
    outcomes: Vec<Outcome>,
    reviews: Vec<Review>,
    pub collusion: CollusionConfig,
}

#[derive(Serialize, Deserialize, Default)]
struct BookFile {
    node_trust: HashMap<String, f64>,
    outcomes: Vec<Outcome>,
    reviews: Vec<Review>,
}

impl ReputationBook {
    /// Persist to a JSON file (atomically).
    pub fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        let f = BookFile { node_trust: self.node_trust.clone(), outcomes: self.outcomes.clone(), reviews: self.reviews.clone() };
        if let Some(d) = path.parent() {
            std::fs::create_dir_all(d)?;
        }
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, serde_json::to_vec(&f).expect("book serializes"))?;
        std::fs::rename(tmp, path)
    }

    /// Load from a JSON file; a missing file is an empty book.
    pub fn load(path: &std::path::Path) -> std::io::Result<Self> {
        match std::fs::read(path) {
            Ok(b) => {
                let f: BookFile = serde_json::from_slice(&b).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                Ok(ReputationBook { node_trust: f.node_trust, outcomes: f.outcomes, reviews: f.reviews, ..Default::default() })
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(ReputationBook::default()),
            Err(e) => Err(e),
        }
    }

    pub fn set_node_trust(&mut self, node: &str, trust: f64) {
        self.node_trust.insert(node.to_string(), trust.clamp(0.0, 1.0));
    }

    pub fn record_outcome(&mut self, o: Outcome) {
        self.outcomes.push(o);
    }

    #[allow(dead_code)] // library surface: used by tests and integrators, or pending the settlement-chain decision
    pub fn record_review(&mut self, r: Review) {
        self.reviews.push(r);
    }

    fn prior(&self, a: &AgentKey) -> f64 {
        self.node_trust.get(&a.0).copied().unwrap_or(0.5)
    }

    pub fn flags(&self) -> CollusionFlags {
        detect_collusion(&self.reviews, &self.collusion)
    }

    /// Reputation of `agent` in `role`, in [0,1].
    pub fn reputation(&self, agent: &AgentKey, role: Role) -> f64 {
        let flags = self.flags();
        self.reputation_with(agent, role, &flags, 1)
    }

    /// Weight an agent's reviews carry: its (executor-or-reviewer) reputation,
    /// cut by the collusion penalty if flagged.
    #[allow(dead_code)] // library surface: used by tests and integrators, or pending the settlement-chain decision
    pub fn review_weight(&self, agent: &AgentKey) -> f64 {
        let flags = self.flags();
        self.review_weight_with(agent, &flags, 1)
    }

    fn review_weight_with(&self, agent: &AgentKey, flags: &CollusionFlags, depth: u8) -> f64 {
        let base = self.reputation_with(agent, Role::Reviewer, flags, depth);
        if flags.flagged(agent) { base * COLLUSION_PENALTY } else { base }
    }

    fn reputation_with(&self, agent: &AgentKey, role: Role, flags: &CollusionFlags, depth: u8) -> f64 {
        let p = self.prior(agent);
        let mut alpha = p * PRIOR_STRENGTH;
        let mut beta = (1.0 - p) * PRIOR_STRENGTH;
        for o in self.outcomes.iter().filter(|o| &o.agent == agent && o.role == role) {
            if o.success { alpha += 1.0 } else { beta += 1.0 }
        }
        for r in self.reviews.iter().filter(|r| &r.target == agent && r.role == role) {
            // Reviewer weight: its own reputation as reviewer (one level deep to stay finite),
            // times the collusion penalty. Unknown reviewers weigh by their node prior.
            let w = if depth == 0 {
                self.prior(&r.reviewer)
            } else {
                let base = self.reputation_with(&r.reviewer, Role::Reviewer, flags, depth - 1);
                if flags.flagged(&r.reviewer) { base * COLLUSION_PENALTY } else { base }
            };
            alpha += w * r.score;
            beta += w * (1.0 - r.score);
        }
        alpha / (alpha + beta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn a(n: &str) -> AgentKey { (format!("node-{n}"), format!("key-{n}")) }

    #[test]
    fn reputation_survives_a_save_and_load() {
        let path = std::env::temp_dir().join(format!("slk-rep-{}/book.json", rand::random::<u64>()));
        let mut b = ReputationBook::default();
        b.set_node_trust("n", 0.7);
        for _ in 0..5 { b.record_outcome(Outcome { agent: a("x"), role: Role::Executor, success: false }); }
        b.record_review(Review { task: "t".into(), reviewer: a("y"), target: a("x"), role: Role::Executor, score: 0.2 });
        b.save(&path).unwrap();
        let r = ReputationBook::load(&path).unwrap();
        assert_eq!(r.reputation(&a("x"), Role::Executor), b.reputation(&a("x"), Role::Executor));
        assert!(ReputationBook::load(&path.with_file_name("absent.json")).unwrap().flags().ring_members.is_empty());
        std::fs::write(&path, "{").unwrap();
        assert!(ReputationBook::load(&path).is_err());
    }

    #[test]
    fn node_trust_is_the_prior_and_evidence_takes_over() {
        let mut b = ReputationBook::default();
        b.set_node_trust("node-good", 0.9);
        b.set_node_trust("node-bad", 0.1);
        let (g, x) = (("node-good".to_string(), "k".to_string()), ("node-bad".to_string(), "k".to_string()));
        assert!((b.reputation(&g, Role::Executor) - 0.9).abs() < 1e-9);
        assert!((b.reputation(&x, Role::Executor) - 0.1).abs() < 1e-9);
        for _ in 0..40 {
            b.record_outcome(Outcome { agent: x.clone(), role: Role::Executor, success: true });
            b.record_outcome(Outcome { agent: g.clone(), role: Role::Executor, success: false });
        }
        assert!(b.reputation(&x, Role::Executor) > 0.85, "evidence beats a low prior");
        assert!(b.reputation(&g, Role::Executor) < 0.15, "failures beat a high prior");
        // roles are independent
        assert!((b.reputation(&g, Role::Planner) - 0.9).abs() < 1e-9);
    }

    #[test]
    fn creating_agents_costs_stake_and_is_quota_limited() {
        let mut r = AgentRegistry::new(10, 2);
        assert_eq!(r.register("n", "k1"), Err(RegistryError::Insufficient { have: 0, need: 10 }));
        r.deposit("n", 25);
        r.register("n", "k1").unwrap();
        r.register("n", "k2").unwrap();
        assert_eq!(r.stake_of("n"), 5);
        assert_eq!(r.register("n", "k3"), Err(RegistryError::QuotaReached(2)));
        assert_eq!(r.register("n", "k1"), Err(RegistryError::Duplicate));
        // slashing burns the locked stake: it is not refunded to the node
        assert_eq!(r.slash("n", "k1"), 10);
        assert_eq!(r.stake_of("n"), 5);
        assert!(!r.is_registered("n", "k1"));
        // a fresh key does not inherit anything: it must pay again
        r.deposit("n", 5);
        r.register("n", "k3").unwrap();
    }

    /// Simulate tasks. Honest agents h0..h5 do good work (quality 0.85) and review truthfully.
    /// Colluders c0..c2 do bad work (0.15), rate each other 1.0 and every honest agent 0.0.
    fn simulate(tasks: usize, defended: bool) -> (ReputationBook, Vec<AgentKey>, Vec<AgentKey>) {
        let honest: Vec<AgentKey> = (0..6).map(|i| a(&format!("h{i}"))).collect();
        let cols: Vec<AgentKey> = (0..3).map(|i| a(&format!("c{i}"))).collect();
        let mut b = ReputationBook::default();
        for k in honest.iter().chain(&cols) {
            b.set_node_trust(&k.0, 0.5);
        }
        let all: Vec<&AgentKey> = honest.iter().chain(&cols).collect();
        for t in 0..tasks {
            let worker = all[t % all.len()];
            let is_col = cols.contains(worker);
            let quality = if is_col { 0.15 } else { 0.85 };
            for reviewer in &all {
                if *reviewer == worker { continue; }
                let score = if cols.contains(reviewer) {
                    if is_col { 1.0 } else { 0.0 }
                } else {
                    // honest reviewer: truthful with a little deterministic noise
                    (quality + if (t + reviewer.1.len()) % 2 == 0 { 0.03f64 } else { -0.03f64 }).clamp(0.0, 1.0)
                };
                b.record_review(Review { task: format!("t{t}"), reviewer: (*reviewer).clone(), target: worker.clone(), role: Role::Executor, score });
            }
        }
        if !defended {
            // disable detection by making thresholds unreachable
            b.collusion = CollusionConfig { min_mutual: usize::MAX, min_co_outlier_tasks: usize::MAX, ..Default::default() };
        }
        (b, honest, cols)
    }

    #[test]
    fn colluding_ring_is_detected_and_downweighted_within_a_bounded_number_of_tasks() {
        let (b, honest, cols) = simulate(36, true);
        let flags = b.flags();
        for c in &cols {
            assert!(flags.flagged(c), "colluder {c:?} not flagged");
        }
        for h in &honest {
            assert!(!flags.flagged(h), "honest {h:?} wrongly flagged");
        }
        let w_col = b.review_weight(&cols[0]);
        let w_hon = b.review_weight(&honest[0]);
        assert!(w_col < 0.1 * w_hon + 1e-9, "colluder review weight {w_col} vs honest {w_hon}");
    }

    #[test]
    fn defence_is_what_removes_the_colluders_advantage() {
        let (undefended, honest, cols) = simulate(36, false);
        let (defended, _, _) = simulate(36, true);
        let gap = |b: &ReputationBook| b.reputation(&honest[0], Role::Executor) - b.reputation(&cols[0], Role::Executor);
        // Without detection, colluders' mutual 1.0s keep their reputation close to honest agents'
        // despite worst-quality work; with detection the gap is large.
        assert!(gap(&defended) > gap(&undefended) + 0.1, "defended gap {} vs undefended {}", gap(&defended), gap(&undefended));
        assert!(defended.reputation(&cols[0], Role::Executor) < 0.4);
        assert!(defended.reputation(&honest[0], Role::Executor) > 0.6);
    }

    #[test]
    fn early_in_the_run_there_is_not_enough_evidence_to_flag_anyone() {
        let (b, _, cols) = simulate(3, true);
        assert!(!b.flags().flagged(&cols[0]));
    }

    #[test]
    fn honest_mutual_praise_between_good_agents_is_not_a_ring() {
        // Everyone is good and everyone rates everyone high: no gap between inside and outside views.
        let ids: Vec<AgentKey> = (0..5).map(|i| a(&format!("g{i}"))).collect();
        let mut reviews = vec![];
        for t in 0..10 {
            for r in &ids {
                for w in &ids {
                    if r != w {
                        reviews.push(Review { task: format!("t{t}"), reviewer: r.clone(), target: w.clone(), role: Role::Executor, score: 0.9 });
                    }
                }
            }
        }
        let f = detect_collusion(&reviews, &CollusionConfig::default());
        assert!(f.ring_members.is_empty() && f.correlated_outliers.is_empty());
    }

    #[test]
    fn correlated_outlier_voters_are_flagged() {
        // Three honest reviewers say ~0.5; two agents always vote 1.0 together on every task.
        let mk = |n: &str| a(n);
        let mut reviews = vec![];
        for t in 0..6 {
            let target = mk(&format!("t{}", t % 2));
            for (r, s) in [("h1", 0.5), ("h2", 0.52), ("h3", 0.48), ("x1", 1.0), ("x2", 1.0)] {
                reviews.push(Review { task: format!("task{t}"), reviewer: mk(r), target: target.clone(), role: Role::Executor, score: s });
            }
        }
        let f = detect_collusion(&reviews, &CollusionConfig::default());
        assert!(f.correlated_outliers.contains(&mk("x1")) && f.correlated_outliers.contains(&mk("x2")));
        assert!(!f.correlated_outliers.contains(&mk("h1")));
    }
}
