//! Lineage credit and payouts (A6).
//!
//! Who deserves what for a result that was built on earlier candidates?
//! The credit rule is a Shapley value over the candidate lineage:
//!
//! * Players: every candidate in the ancestry (inclusive) of each target
//!   candidate (the Gems).
//! * Value of a coalition S: the best score among target candidates whose
//!   *entire* ancestry lies inside S (zero if none). A result is worth
//!   something only if everything it was built on is present, so credit flows
//!   down the lineage to the ancestors it depended on.
//! * Within a candidate's credit, `executor_share` goes to its author and the
//!   remainder is split equally among the data owners that evaluated it.
//!
//! Exact Shapley is used up to `EXACT_MAX_PLAYERS`; beyond that a permutation
//! sampling estimate is used. **The approximation is part of the published rule**:
//! `CreditRule::VERSION`, the sample count and the seed (derived from the
//! program's evidence root) fully determine the output, so any node can
//! recompute the table and check its payout. Results are bit-identical across
//! nodes running the same build because the permutation order and the
//! summation order are fixed.

use crate::agents::settlement::{SettlementChain, SettlementError};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub const EXACT_MAX_PLAYERS: usize = 10;

#[derive(Debug, Clone)]
pub struct CreditRule {
    pub samples: u32,
    pub executor_share: f64,
}

impl CreditRule {
    pub const VERSION: &'static str = "slakshna/credit/shapley-lineage/v1";

    pub fn standard() -> Self {
        CreditRule { samples: 2000, executor_share: 0.7 }
    }
}

#[derive(Debug, Clone)]
pub struct LineageCandidate {
    pub id: String,
    pub author: String,
    pub parents: Vec<String>,
    /// Data owners (node ids) that produced findings for it.
    pub evaluators: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct Target {
    pub candidate: String,
    pub score: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CreditTable {
    /// Per candidate Shapley credit; sums to the value of the grand coalition.
    pub per_candidate: BTreeMap<String, f64>,
    /// Per node credit after the executor/evaluator split.
    pub per_node: BTreeMap<String, f64>,
    pub exact: bool,
}

fn ancestry<'a>(cands: &'a BTreeMap<&str, &LineageCandidate>, id: &str) -> BTreeSet<&'a str> {
    let mut out = BTreeSet::new();
    let mut stack = vec![id];
    while let Some(x) = stack.pop() {
        let Some(c) = cands.get(x) else { continue };
        if out.insert(c.id.as_str()) {
            stack.extend(c.parents.iter().map(String::as_str));
        }
    }
    out
}

/// Seed derived from the evidence root: published, not chosen by any payee.
pub fn credit_seed(evidence_root: &str) -> u64 {
    let h = Sha256::digest([CreditRule::VERSION.as_bytes(), b"\0", evidence_root.as_bytes()].concat());
    u64::from_le_bytes(h[..8].try_into().unwrap())
}

pub fn compute(candidates: &[LineageCandidate], targets: &[Target], rule: &CreditRule, evidence_root: &str) -> CreditTable {
    let by_id: BTreeMap<&str, &LineageCandidate> = candidates.iter().map(|c| (c.id.as_str(), c)).collect();
    // Targets and their ancestries (only candidates we actually know).
    let tgt: Vec<(&str, f64, BTreeSet<&str>)> = {
        let mut v: Vec<_> = targets
            .iter()
            .filter(|t| by_id.contains_key(t.candidate.as_str()) && t.score.is_finite() && t.score > 0.0)
            .map(|t| (t.candidate.as_str(), t.score, ancestry(&by_id, &t.candidate)))
            .collect();
        v.sort_by(|a, b| a.0.cmp(b.0));
        v
    };
    let players: Vec<&str> = tgt.iter().flat_map(|t| t.2.iter().copied()).collect::<BTreeSet<_>>().into_iter().collect();
    let idx: BTreeMap<&str, usize> = players.iter().enumerate().map(|(i, p)| (*p, i)).collect();
    let n = players.len();

    // value(S) over a bitset of players.
    let masks: Vec<(f64, Vec<usize>)> = tgt.iter().map(|(_, s, anc)| (*s, anc.iter().map(|a| idx[a]).collect())).collect();
    let value = |in_set: &dyn Fn(usize) -> bool| -> f64 {
        masks.iter().filter(|(_, anc)| anc.iter().all(|&i| in_set(i))).map(|(s, _)| *s).fold(0.0, f64::max)
    };

    let mut phi = vec![0.0f64; n];
    let exact = n <= EXACT_MAX_PLAYERS;
    if n > 0 && exact {
        // phi_i = sum over S not containing i of |S|!(n-|S|-1)!/n! (v(S+i) - v(S))
        let fact: Vec<f64> = (0..=n).scan(1.0, |a, k| { if k > 0 { *a *= k as f64; } Some(*a) }).collect();
        for i in 0..n {
            for mask in 0u32..(1 << n) {
                if mask & (1 << i) != 0 { continue; }
                let s = mask.count_ones() as usize;
                let w = fact[s] * fact[n - s - 1] / fact[n];
                let with = value(&|j| mask & (1 << j) != 0 || j == i);
                let without = value(&|j| mask & (1 << j) != 0);
                phi[i] += w * (with - without);
            }
        }
    } else if n > 0 {
        let mut rng = ChaCha20Rng::seed_from_u64(credit_seed(evidence_root));
        let mut order: Vec<usize> = (0..n).collect();
        for _ in 0..rule.samples {
            order.shuffle(&mut rng);
            let mut present = vec![false; n];
            let mut prev = 0.0;
            for &i in &order {
                present[i] = true;
                let v = value(&|j| present[j]);
                phi[i] += v - prev;
                prev = v;
            }
        }
        phi.iter_mut().for_each(|x| *x /= rule.samples as f64);
    }

    let per_candidate: BTreeMap<String, f64> = players.iter().zip(&phi).map(|(p, v)| (p.to_string(), *v)).collect();
    let mut per_node: BTreeMap<String, f64> = BTreeMap::new();
    for (id, credit) in &per_candidate {
        let c = by_id[id.as_str()];
        let evals: BTreeSet<&String> = c.evaluators.iter().collect();
        let exec = if evals.is_empty() { *credit } else { credit * rule.executor_share };
        *per_node.entry(c.author.clone()).or_default() += exec;
        for e in &evals {
            *per_node.entry((*e).clone()).or_default() += (credit - exec) / evals.len() as f64;
        }
    }
    CreditTable { per_candidate, per_node, exact }
}

/// Convert credits to integer payouts of exactly `pool` units (largest
/// remainder; ties broken by node id so the table is reproducible).
pub fn payouts(per_node: &BTreeMap<String, f64>, pool: u64) -> BTreeMap<String, u64> {
    let total: f64 = per_node.values().filter(|v| **v > 0.0).sum();
    let mut out: BTreeMap<String, u64> = BTreeMap::new();
    if total <= 0.0 || pool == 0 {
        return out;
    }
    let mut rem: Vec<(f64, &String)> = vec![];
    let mut given = 0u64;
    for (k, v) in per_node.iter().filter(|(_, v)| **v > 0.0) {
        let exact = v / total * pool as f64;
        let base = exact.floor() as u64;
        given += base;
        out.insert(k.clone(), base);
        rem.push((exact - base as f64, k));
    }
    rem.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then_with(|| a.1.cmp(b.1)));
    for (_, k) in rem.into_iter().take((pool - given) as usize) {
        *out.get_mut(k).unwrap() += 1;
    }
    out
}

/// Escrow `pool` from `funder` and release it by the payout table (milestone payout).
#[allow(dead_code)] // library surface: used by tests and integrators, or pending the settlement-chain decision
pub fn settle_milestone(
    ledger: &dyn SettlementChain,
    milestone_id: &str,
    funder: &str,
    pool: u64,
    table: &BTreeMap<String, u64>,
    evidence: &[String],
) -> Result<(), SettlementError> {
    ledger.escrow(milestone_id, funder, pool, evidence)?;
    let splits: Vec<(String, u64)> = table.iter().map(|(k, v)| (k.clone(), *v)).collect();
    ledger.release(milestone_id, &splits, evidence)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agents::settlement::MemoryLedger;

    fn c(id: &str, author: &str, parents: &[&str], evals: &[&str]) -> LineageCandidate {
        LineageCandidate { id: id.into(), author: author.into(), parents: parents.iter().map(|s| s.to_string()).collect(), evaluators: evals.iter().map(|s| s.to_string()).collect() }
    }

    fn sum(m: &BTreeMap<String, f64>) -> f64 { m.values().sum() }

    #[test]
    fn single_candidate_gets_all_credit() {
        let t = compute(&[c("a", "alice", &[], &[])], &[Target { candidate: "a".into(), score: 0.8 }], &CreditRule::standard(), "r");
        assert!((t.per_node["alice"] - 0.8).abs() < 1e-12);
        assert!(t.exact);
    }

    #[test]
    fn chain_splits_credit_equally_between_dependent_links() {
        // a <- b <- c: the result needs all three, so symmetric players share equally.
        let cands = [c("a", "x", &[], &[]), c("b", "y", &["a"], &[]), c("c", "z", &["b"], &[])];
        let t = compute(&cands, &[Target { candidate: "c".into(), score: 0.9 }], &CreditRule::standard(), "r");
        for n in ["x", "y", "z"] {
            assert!((t.per_node[n] - 0.3).abs() < 1e-12, "{n}: {}", t.per_node[n]);
        }
    }

    #[test]
    fn efficiency_axiom_credit_sums_to_best_target_value() {
        let cands = [c("a", "x", &[], &[]), c("b", "y", &["a"], &[]), c("d", "w", &["a"], &[]), c("e", "v", &[], &[])];
        let targets = [Target { candidate: "b".into(), score: 0.6 }, Target { candidate: "d".into(), score: 0.9 }, Target { candidate: "e".into(), score: 0.3 }];
        let t = compute(&cands, &targets, &CreditRule::standard(), "r");
        // grand coalition value = max score = 0.9
        assert!((sum(&t.per_candidate) - 0.9).abs() < 1e-12);
        assert!((sum(&t.per_node) - 0.9).abs() < 1e-12);
        // 'b' is a worse substitute for 'd' (same parent): it earns credit only in
        // coalitions without 'd', so strictly less than 'd', but not nothing.
        assert!(t.per_candidate["d"] > t.per_candidate["b"]);
        assert!(t.per_candidate["b"] > 0.0);
    }

    #[test]
    fn evaluators_share_the_non_executor_part() {
        let t = compute(&[c("a", "alice", &[], &["hospA", "hospB"])], &[Target { candidate: "a".into(), score: 1.0 }], &CreditRule::standard(), "r");
        assert!((t.per_node["alice"] - 0.7).abs() < 1e-12);
        assert!((t.per_node["hospA"] - 0.15).abs() < 1e-12);
        assert!((t.per_node["hospB"] - 0.15).abs() < 1e-12);
    }

    #[test]
    fn sampled_estimate_is_deterministic_and_close_to_exact() {
        // 12 players forces sampling. Chain of 12: exact Shapley is value/12 each.
        let mut cands = vec![c("n0", "p0", &[], &[])];
        for i in 1..12 {
            cands.push(c(&format!("n{i}"), &format!("p{i}"), &[&format!("n{}", i - 1)], &[]));
        }
        let tg = [Target { candidate: "n11".into(), score: 1.2 }];
        let a = compute(&cands, &tg, &CreditRule::standard(), "root");
        let b = compute(&cands, &tg, &CreditRule::standard(), "root");
        assert!(!a.exact);
        assert_eq!(a, b, "same inputs must give bit-identical tables");
        // Only the grand coalition has value, so each player's exact share is 1.2/12 = 0.1;
        // the sampled estimate must be near that.
        for v in a.per_candidate.values() {
            assert!((v - 0.1).abs() < 0.03, "sampled share {v} too far from exact 0.1");
        }
        assert!((sum(&a.per_candidate) - 1.2).abs() < 1e-9, "efficiency holds exactly even when sampled");
    }

    #[test]
    fn sampled_estimate_tracks_exact_on_a_branchy_lineage() {
        // 6 parents each alone enough for target t_i (value i+1); exact path (n=12) vs forced sampling (same game, n>10 requires > 10 players).
        // Build 11 players: root r, children k0..k9 each a target with score 1..10.
        let mut cands = vec![c("r", "R", &[], &[])];
        let mut tg = vec![];
        for i in 0..10 {
            cands.push(c(&format!("k{i}"), &format!("K{i}"), &["r"], &[]));
            tg.push(Target { candidate: format!("k{i}"), score: (i + 1) as f64 });
        }
        let sampled = compute(&cands, &tg, &CreditRule { samples: 20000, executor_share: 1.0 }, "root-x");
        assert!(!sampled.exact);
        // Analytic Shapley for this "max" game: root is needed by everyone. Efficiency: total = 10.
        assert!((sum(&sampled.per_candidate) - 10.0).abs() < 1e-9);
        // The best child (k9, value 10) must earn more than the worst (k0, value 1), and the root a positive share.
        assert!(sampled.per_candidate["k9"] > sampled.per_candidate["k0"]);
        assert!(sampled.per_candidate["r"] > 0.0);
        // Different evidence roots draw different permutations, so estimates differ slightly but stay close.
        let alt = compute(&cands, &tg, &CreditRule { samples: 20000, executor_share: 1.0 }, "root-y");
        assert!((alt.per_candidate["r"] - sampled.per_candidate["r"]).abs() < 0.15);
    }

    #[test]
    fn payouts_sum_exactly_to_pool_and_are_reproducible() {
        let mut m = BTreeMap::new();
        m.insert("a".to_string(), 1.0);
        m.insert("b".to_string(), 1.0);
        m.insert("c".to_string(), 1.0);
        let p = payouts(&m, 100);
        assert_eq!(p.values().sum::<u64>(), 100);
        assert_eq!(p, payouts(&m, 100));
        assert_eq!(p["a"], 34, "ties on remainder go to the lowest id");
        assert!(payouts(&BTreeMap::new(), 10).is_empty());
    }

    #[test]
    fn milestone_settlement_moves_the_pool_through_escrow() {
        let ledger = MemoryLedger::default();
        let ev = vec!["a".repeat(64)];
        ledger.credit("funder", 100, &ev).unwrap();
        let mut per = BTreeMap::new();
        per.insert("x".to_string(), 3.0);
        per.insert("y".to_string(), 1.0);
        let table = payouts(&per, 80);
        settle_milestone(&ledger, "m1", "funder", 80, &table, &ev).unwrap();
        assert_eq!((ledger.balance("x"), ledger.balance("y"), ledger.balance("funder")), (60, 20, 20));
    }

    #[test]
    fn unknown_or_nonpositive_targets_are_ignored() {
        let t = compute(&[c("a", "x", &[], &[])], &[Target { candidate: "ghost".into(), score: 1.0 }, Target { candidate: "a".into(), score: -1.0 }], &CreditRule::standard(), "r");
        assert!(t.per_node.is_empty());
    }
}
