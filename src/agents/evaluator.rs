//! Federated evaluation (A4).
//!
//! The data owner runs an `EvaluatorService` next to its private dataset. It
//! scores candidates and returns *typed findings only*: a metric value, the
//! evaluator code hash and a Merkle commitment to the evaluation data. Raw rows
//! never leave the site, except to an explicitly authorized auditor for a
//! sampled spot check (rows come with Merkle proofs to the committed root).
//!
//! Leakage controls: a per-requester query budget, an optional differential
//! privacy budget with Laplace noise calibrated to the metric's sensitivity,
//! and coarse rounding of reported values.
//!
//! Verification of results that others cannot see:
//!   * shared benchmark  -> redundant evaluation, results must agree (exact class)
//!   * private data      -> auditor spot check on a random sample (probabilistic class)
//!   * suspect results   -> triage before the costly checks

use crate::agents::merkle::{self, MerkleTree, Proof};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::sync::Mutex;

/// One labelled example of the evaluation set.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Row {
    pub input: Vec<u8>,
    pub label: Vec<u8>,
}

impl Row {
    pub fn leaf_bytes(&self) -> Vec<u8> {
        let mut v = (self.input.len() as u64).to_le_bytes().to_vec();
        v.extend_from_slice(&self.input);
        v.extend_from_slice(&self.label);
        v
    }
}

/// The candidate under evaluation: maps an input to a prediction. In a
/// deployment this is backed by a sandboxed run of the candidate.
pub trait Predictor {
    fn predict(&self, input: &[u8]) -> Vec<u8>;

    /// Batch prediction; containerized candidates override this to run once per batch.
    fn predict_all(&self, inputs: &[&[u8]]) -> Vec<Vec<u8>> {
        inputs.iter().map(|i| self.predict(i)).collect()
    }
}

impl<F: Fn(&[u8]) -> Vec<u8>> Predictor for F {
    fn predict(&self, input: &[u8]) -> Vec<u8> {
        self(input)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Metric {
    /// Fraction of exact label matches, in [0,1]; sensitivity 1/n.
    Accuracy,
    /// 1 - mean absolute error between UTF-8 decimal prediction/label, clipped
    /// per example to [0,1]; sensitivity 1/n.
    ClippedMaeScore,
}

impl Metric {
    pub fn name(self) -> &'static str {
        match self {
            Metric::Accuracy => "accuracy",
            Metric::ClippedMaeScore => "clipped_mae_score",
        }
    }

    /// Hash identifying the exact scoring code (name + version tag). Bump the
    /// version string when the implementation changes.
    pub fn code_hash(self) -> String {
        let id = match self {
            Metric::Accuracy => "slakshna/metric/accuracy/v1",
            Metric::ClippedMaeScore => "slakshna/metric/clipped_mae_score/v1",
        };
        hex::encode(Sha256::digest(id.as_bytes()))
    }

    /// Per-example score in [0,1].
    pub fn example_score(self, pred: &[u8], label: &[u8]) -> f64 {
        match self {
            Metric::Accuracy => (pred == label) as u8 as f64,
            Metric::ClippedMaeScore => {
                let p = std::str::from_utf8(pred).ok().and_then(|s| s.trim().parse::<f64>().ok());
                let l = std::str::from_utf8(label).ok().and_then(|s| s.trim().parse::<f64>().ok());
                match (p, l) {
                    (Some(p), Some(l)) if p.is_finite() && l.is_finite() => 1.0 - (p - l).abs().min(1.0),
                    _ => 0.0,
                }
            }
        }
    }

    pub fn score(self, rows: &[Row], pred: &dyn Predictor) -> f64 {
        let inputs: Vec<&[u8]> = rows.iter().map(|r| r.input.as_slice()).collect();
        let preds = pred.predict_all(&inputs);
        rows.iter().enumerate().map(|(i, r)| self.example_score(preds.get(i).map(Vec::as_slice).unwrap_or(&[]), &r.label)).sum::<f64>() / rows.len() as f64
    }
}

/// What the evaluator publishes for a candidate. Maps onto `research.finding`
/// (value, evaluator_code_hash, data_merkle_root) plus the size of the data.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Attestation {
    pub metric: String,
    pub value: f64,
    pub evaluator_code_hash: String,
    pub data_merkle_root: String,
    pub n_examples: u64,
    /// Whether DP noise was added to `value`.
    pub noised: bool,
}

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum EvalError {
    #[error("query budget exhausted for requester {0}")]
    QueryBudget(String),
    #[error("privacy budget exhausted")]
    PrivacyBudget,
    #[error("requester {0} is not an authorized auditor")]
    NotAuditor(String),
    #[error("sample too large (max {0} rows)")]
    SampleTooLarge(usize),
    #[error("empty dataset")]
    Empty,
    #[error("bad sample index")]
    BadIndex,
}

#[derive(Debug, Clone)]
pub struct EvaluatorConfig {
    pub metric: Metric,
    /// Max evaluations per requester for this dataset.
    pub queries_per_requester: u32,
    /// Total epsilon the dataset may spend; None disables DP noise.
    pub epsilon_total: Option<f64>,
    /// Epsilon spent per query when DP is on.
    pub epsilon_per_query: f64,
    /// Round reported values to this many decimals (coarse reporting).
    pub decimals: u32,
    pub auditors: HashSet<String>,
    /// Max rows an auditor may pull in one spot check.
    pub max_audit_rows: usize,
}

impl EvaluatorConfig {
    pub fn new(metric: Metric) -> Self {
        EvaluatorConfig {
            metric,
            queries_per_requester: 20,
            epsilon_total: None,
            epsilon_per_query: 0.5,
            decimals: 4,
            auditors: HashSet::new(),
            max_audit_rows: 64,
        }
    }
}

pub struct EvaluatorService {
    rows: Vec<Row>,
    tree: MerkleTree,
    cfg: EvaluatorConfig,
    queries: Mutex<HashMap<String, u32>>,
    epsilon_spent: Mutex<f64>,
    rng: Mutex<ChaCha20Rng>,
}

impl EvaluatorService {
    pub fn new(rows: Vec<Row>, cfg: EvaluatorConfig) -> Result<Self, EvalError> {
        let leaves: Vec<Vec<u8>> = rows.iter().map(Row::leaf_bytes).collect();
        let tree = MerkleTree::new(&leaves).ok_or(EvalError::Empty)?;
        Ok(EvaluatorService {
            rows,
            tree,
            cfg,
            queries: Mutex::new(HashMap::new()),
            epsilon_spent: Mutex::new(0.0),
            rng: Mutex::new(ChaCha20Rng::from_entropy()),
        })
    }

    /// Deterministic noise source, for reproducible tests.
    #[allow(dead_code)] // library surface: used by tests and integrators, or pending the settlement-chain decision
    pub fn with_rng_seed(self, seed: u64) -> Self {
        *self.rng.lock().unwrap() = ChaCha20Rng::seed_from_u64(seed);
        self
    }

    pub fn len(&self) -> u64 {
        self.rows.len() as u64
    }

    pub fn data_root(&self) -> String {
        self.tree.root()
    }

    pub fn evaluate(&self, requester: &str, candidate: &dyn Predictor) -> Result<Attestation, EvalError> {
        {
            let mut q = self.queries.lock().unwrap();
            let used = q.entry(requester.to_string()).or_default();
            if *used >= self.cfg.queries_per_requester {
                return Err(EvalError::QueryBudget(requester.into()));
            }
            *used += 1;
        }
        let n = self.rows.len() as f64;
        let mut value = self.cfg.metric.score(&self.rows, candidate);
        let noised = if let Some(total) = self.cfg.epsilon_total {
            let mut spent = self.epsilon_spent.lock().unwrap();
            if *spent + self.cfg.epsilon_per_query > total + 1e-12 {
                // Refund the query: nothing was released.
                *self.queries.lock().unwrap().get_mut(requester).unwrap() -= 1;
                return Err(EvalError::PrivacyBudget);
            }
            *spent += self.cfg.epsilon_per_query;
            // Laplace mechanism: sensitivity of a mean of [0,1] scores is 1/n.
            let scale = (1.0 / n) / self.cfg.epsilon_per_query;
            value += laplace(&mut *self.rng.lock().unwrap(), scale);
            value = value.clamp(0.0, 1.0);
            true
        } else {
            false
        };
        let p = 10f64.powi(self.cfg.decimals as i32);
        Ok(Attestation {
            metric: self.cfg.metric.name().into(),
            value: (value * p).round() / p,
            evaluator_code_hash: self.cfg.metric.code_hash(),
            data_merkle_root: self.tree.root(),
            n_examples: self.rows.len() as u64,
            noised,
        })
    }

    #[allow(dead_code)] // library surface: used by tests and integrators, or pending the settlement-chain decision
    pub fn epsilon_spent(&self) -> f64 {
        *self.epsilon_spent.lock().unwrap()
    }

    /// Release sampled rows with Merkle proofs to an authorized auditor only.
    pub fn audit_sample(&self, auditor: &str, indices: &[u64]) -> Result<Vec<(Row, Proof)>, EvalError> {
        if !self.cfg.auditors.contains(auditor) {
            return Err(EvalError::NotAuditor(auditor.into()));
        }
        if indices.len() > self.cfg.max_audit_rows {
            return Err(EvalError::SampleTooLarge(self.cfg.max_audit_rows));
        }
        indices
            .iter()
            .map(|&i| {
                let row = self.rows.get(i as usize).ok_or(EvalError::BadIndex)?.clone();
                let proof = self.tree.prove(i).ok_or(EvalError::BadIndex)?;
                Ok((row, proof))
            })
            .collect()
    }
}

fn laplace(rng: &mut ChaCha20Rng, scale: f64) -> f64 {
    // Inverse CDF; u in (-0.5, 0.5) excluding the endpoints.
    let u: f64 = loop {
        let u = rng.gen::<f64>() - 0.5;
        if u.abs() < 0.5 && u != -0.5 {
            break u;
        }
    };
    -scale * u.signum() * (1.0 - 2.0 * u.abs()).ln()
}

// ---------------------------------------------------------------------------
// Verification
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum SpotCheck {
    /// Sample is consistent with the claimed score.
    Consistent { sample_score: f64, allowed_gap: f64 },
    Inconsistent { sample_score: f64, allowed_gap: f64 },
    /// A revealed row does not belong to the committed data.
    BadProof { index: u64 },
}

/// Choose `k` distinct indices the evaluator cannot predict: derived from the
/// auditor's secret seed, never from anything the evaluator controls.
pub fn sample_indices(auditor_seed: u64, n: u64, k: usize) -> Vec<u64> {
    let k = k.min(n as usize);
    let mut rng = ChaCha20Rng::seed_from_u64(auditor_seed);
    let mut chosen = HashSet::new();
    while chosen.len() < k {
        chosen.insert(rng.gen_range(0..n));
    }
    let mut v: Vec<u64> = chosen.into_iter().collect();
    v.sort_unstable();
    v
}

/// Auditor side: re-score the sampled rows with the same predictor and check
/// (1) every row is in the committed Merkle root, (2) the sample score is
/// within `tolerance` plus a 3-sigma sampling band of the claimed value.
/// `claimed_noise_scale` widens the band when the claim carries DP noise.
pub fn spot_check(
    claim: &Attestation,
    sample: &[(Row, Proof)],
    candidate: &dyn Predictor,
    tolerance: f64,
    claimed_noise_scale: f64,
) -> SpotCheck {
    let metric = match claim.metric.as_str() {
        "accuracy" => Metric::Accuracy,
        _ => Metric::ClippedMaeScore,
    };
    for (row, proof) in sample {
        if proof.leaf_count != claim.n_examples || !merkle::verify(&claim.data_merkle_root, &row.leaf_bytes(), proof) {
            return SpotCheck::BadProof { index: proof.index };
        }
    }
    let k = sample.len().max(1) as f64;
    let sample_score = sample
        .iter()
        .map(|(r, _)| metric.example_score(&candidate.predict(&r.input), &r.label))
        .sum::<f64>()
        / k;
    let p = claim.value.clamp(0.0, 1.0);
    // Finite-population-agnostic binomial standard error, conservative.
    let sigma = (p * (1.0 - p) / k).sqrt().max(0.5 / k);
    let allowed_gap = tolerance + 3.0 * sigma + 3.0 * claimed_noise_scale;
    if (sample_score - claim.value).abs() <= allowed_gap {
        SpotCheck::Consistent { sample_score, allowed_gap }
    } else {
        SpotCheck::Inconsistent { sample_score, allowed_gap }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Consensus {
    pub value: f64,
    /// Indices of attestations that disagree with the consensus.
    pub outliers: Vec<usize>,
}

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum ConsensusError {
    #[error("not enough evaluators ({0} < {1})")]
    TooFew(usize, usize),
    #[error("attestations are not over the same benchmark and evaluator code")]
    Mismatch,
}

/// Redundant evaluation on a shared benchmark: every evaluator must have used
/// the same data root and scoring code; the consensus is the median, and
/// attestations further than `tolerance` from it are flagged.
pub fn redundant_consensus(atts: &[Attestation], min_evaluators: usize, tolerance: f64) -> Result<Consensus, ConsensusError> {
    if atts.len() < min_evaluators {
        return Err(ConsensusError::TooFew(atts.len(), min_evaluators));
    }
    let first = &atts[0];
    if atts.iter().any(|a| a.data_merkle_root != first.data_merkle_root || a.evaluator_code_hash != first.evaluator_code_hash) {
        return Err(ConsensusError::Mismatch);
    }
    let mut v: Vec<f64> = atts.iter().map(|a| a.value).collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    let med = if n % 2 == 1 { v[n / 2] } else { (v[n / 2 - 1] + v[n / 2]) / 2.0 };
    let outliers = atts.iter().enumerate().filter(|(_, a)| (a.value - med).abs() > tolerance).map(|(i, _)| i).collect();
    Ok(Consensus { value: med, outliers })
}

/// Cheap suspicion test run before costly checks (the role `decide()` plays in
/// the plan). Flags values far from the peer median in robust (MAD) terms, or
/// suspiciously perfect scores.
pub trait Triage {
    fn is_suspect(&self, value: f64, peer_values: &[f64]) -> bool;
}

pub struct OutlierTriage {
    pub mad_threshold: f64,
    pub perfect_is_suspect: bool,
}

impl Default for OutlierTriage {
    fn default() -> Self {
        OutlierTriage { mad_threshold: 4.0, perfect_is_suspect: true }
    }
}

impl Triage for OutlierTriage {
    fn is_suspect(&self, value: f64, peers: &[f64]) -> bool {
        if !value.is_finite() {
            return true;
        }
        if self.perfect_is_suspect && value >= 1.0 && peers.iter().any(|p| *p < 1.0) {
            return true;
        }
        if peers.len() < 3 {
            return false;
        }
        let med = median(peers);
        let mad = median(&peers.iter().map(|p| (p - med).abs()).collect::<Vec<_>>()).max(1e-3);
        (value - med).abs() / (1.4826 * mad) > self.mad_threshold
    }
}

fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = s.len();
    if n % 2 == 1 { s[n / 2] } else { (s[n / 2 - 1] + s[n / 2]) / 2.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dataset(n: usize) -> Vec<Row> {
        (0..n).map(|i| Row { input: format!("{i}").into_bytes(), label: format!("{}", i % 3).into_bytes() }).collect()
    }

    /// Predictor right on a fixed fraction of rows.
    fn model(right_mod: usize) -> impl Fn(&[u8]) -> Vec<u8> {
        move |inp: &[u8]| {
            let i: usize = std::str::from_utf8(inp).unwrap().parse().unwrap();
            if i % right_mod == 0 { format!("{}", i % 3).into_bytes() } else { b"x".to_vec() }
        }
    }

    fn svc(n: usize) -> EvaluatorService {
        let mut cfg = EvaluatorConfig::new(Metric::Accuracy);
        cfg.auditors.insert("auditor".into());
        EvaluatorService::new(dataset(n), cfg).unwrap()
    }

    #[test]
    fn finding_carries_metric_code_hash_and_commitment_but_no_rows() {
        let s = svc(100);
        let a = s.evaluate("alice", &model(2)).unwrap();
        assert_eq!(a.value, 0.5);
        assert_eq!(a.n_examples, 100);
        assert_eq!(a.data_merkle_root, s.data_root());
        assert_eq!(a.evaluator_code_hash, Metric::Accuracy.code_hash());
        let json = serde_json::to_string(&a).unwrap();
        assert!(!json.contains("input") && !json.contains("label"));
    }

    #[test]
    fn query_budget_is_per_requester() {
        let mut cfg = EvaluatorConfig::new(Metric::Accuracy);
        cfg.queries_per_requester = 2;
        let s = EvaluatorService::new(dataset(10), cfg).unwrap();
        assert!(s.evaluate("a", &model(1)).is_ok());
        assert!(s.evaluate("a", &model(1)).is_ok());
        assert_eq!(s.evaluate("a", &model(1)), Err(EvalError::QueryBudget("a".into())));
        assert!(s.evaluate("b", &model(1)).is_ok());
    }

    #[test]
    fn dp_noise_is_unbiased_bounded_and_budgeted() {
        let mut cfg = EvaluatorConfig::new(Metric::Accuracy);
        cfg.epsilon_total = Some(2.0);
        cfg.epsilon_per_query = 0.5;
        cfg.queries_per_requester = 100;
        cfg.decimals = 8;
        let s = EvaluatorService::new(dataset(200), cfg).unwrap().with_rng_seed(7);
        let mut vals = vec![];
        for _ in 0..4 {
            let a = s.evaluate("a", &model(2)).unwrap();
            assert!(a.noised && (0.0..=1.0).contains(&a.value));
            vals.push(a.value);
        }
        assert!(vals.windows(2).any(|w| w[0] != w[1]), "noise must vary between queries");
        assert!((s.epsilon_spent() - 2.0).abs() < 1e-9);
        assert_eq!(s.evaluate("a", &model(2)), Err(EvalError::PrivacyBudget));
        // refused query must not consume the requester's query allowance
        assert_eq!(*s.queries.lock().unwrap().get("a").unwrap(), 4);
    }

    #[test]
    fn laplace_noise_has_expected_scale() {
        let mut rng = ChaCha20Rng::seed_from_u64(1);
        let scale = 0.1;
        let xs: Vec<f64> = (0..20000).map(|_| laplace(&mut rng, scale)).collect();
        let mean = xs.iter().sum::<f64>() / xs.len() as f64;
        let mean_abs = xs.iter().map(|x| x.abs()).sum::<f64>() / xs.len() as f64;
        assert!(mean.abs() < 0.01, "mean {mean}");
        assert!((mean_abs - scale).abs() < 0.01, "E|X| = scale, got {mean_abs}");
    }

    #[test]
    fn only_authorized_auditors_get_rows_and_proofs_verify() {
        let s = svc(50);
        assert_eq!(s.audit_sample("mallory", &[1]).unwrap_err(), EvalError::NotAuditor("mallory".into()));
        let rows = s.audit_sample("auditor", &[0, 7, 49]).unwrap();
        for (r, p) in &rows {
            assert!(merkle::verify(&s.data_root(), &r.leaf_bytes(), p));
        }
        assert_eq!(s.audit_sample("auditor", &[50]).unwrap_err(), EvalError::BadIndex);
        let big: Vec<u64> = (0..65).collect();
        assert!(matches!(s.audit_sample("auditor", &big), Err(EvalError::SampleTooLarge(_))));
    }

    #[test]
    fn spot_check_reproduces_honest_score_within_tolerance() {
        let s = svc(1000);
        let m = model(2);
        let claim = s.evaluate("alice", &m).unwrap();
        let idx = sample_indices(42, 1000, 60);
        let sample = s.audit_sample("auditor", &idx).unwrap();
        assert!(matches!(spot_check(&claim, &sample, &m, 0.02, 0.0), SpotCheck::Consistent { .. }));
    }

    #[test]
    fn spot_check_catches_inflated_score_and_forged_rows() {
        let s = svc(1000);
        let m = model(2);
        let mut claim = s.evaluate("alice", &m).unwrap();
        claim.value = 0.95; // data owner inflates 0.5 -> 0.95
        let idx = sample_indices(9, 1000, 60);
        let sample = s.audit_sample("auditor", &idx).unwrap();
        assert!(matches!(spot_check(&claim, &sample, &m, 0.02, 0.0), SpotCheck::Inconsistent { .. }));

        // Rows swapped for easier ones fail the Merkle proof.
        let honest = s.evaluate("alice", &m).unwrap();
        let mut forged = s.audit_sample("auditor", &idx).unwrap();
        forged[0].0.label = b"changed".to_vec();
        assert!(matches!(spot_check(&honest, &forged, &m, 0.02, 0.0), SpotCheck::BadProof { .. }));
    }

    #[test]
    fn sample_indices_are_distinct_sorted_and_seed_dependent() {
        let a = sample_indices(1, 100, 10);
        assert_eq!(a.len(), 10);
        assert!(a.windows(2).all(|w| w[0] < w[1]));
        assert_ne!(a, sample_indices(2, 100, 10));
        assert_eq!(sample_indices(1, 5, 10).len(), 5);
    }

    fn att(v: f64, root: &str) -> Attestation {
        Attestation { metric: "accuracy".into(), value: v, evaluator_code_hash: Metric::Accuracy.code_hash(), data_merkle_root: root.into(), n_examples: 10, noised: false }
    }

    #[test]
    fn redundant_evaluation_flags_the_liar() {
        let atts = vec![att(0.80, "r"), att(0.80, "r"), att(0.81, "r"), att(0.99, "r")];
        let c = redundant_consensus(&atts, 3, 0.02).unwrap();
        assert!((c.value - 0.805).abs() < 1e-9);
        assert_eq!(c.outliers, vec![3]);
        assert_eq!(redundant_consensus(&atts[..2], 3, 0.02), Err(ConsensusError::TooFew(2, 3)));
        let mixed = vec![att(0.8, "r"), att(0.8, "other"), att(0.8, "r")];
        assert_eq!(redundant_consensus(&mixed, 3, 0.02), Err(ConsensusError::Mismatch));
    }

    #[test]
    fn triage_flags_outliers_perfect_scores_and_nan() {
        let t = OutlierTriage::default();
        let peers = [0.70, 0.72, 0.71, 0.69, 0.73];
        assert!(!t.is_suspect(0.71, &peers));
        assert!(t.is_suspect(0.99, &peers));
        assert!(t.is_suspect(f64::NAN, &peers));
        assert!(t.is_suspect(1.0, &[0.9]));
        assert!(!t.is_suspect(0.5, &[0.9])); // too few peers to call it
    }

    #[test]
    fn clipped_mae_metric() {
        let rows = vec![Row { input: b"a".to_vec(), label: b"1.0".to_vec() }, Row { input: b"b".to_vec(), label: b"0.0".to_vec() }];
        let p = |i: &[u8]| if i == b"a" { b"0.9".to_vec() } else { b"garbage".to_vec() };
        let v = Metric::ClippedMaeScore.score(&rows, &p);
        assert!((v - 0.45).abs() < 1e-9);
    }
}
