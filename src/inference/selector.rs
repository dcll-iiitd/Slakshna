//! Query-time adapter selection (federated inference, step 3).
//!
//! Pure function over the locally maintained registry: no network access.

use crate::registry::AdapterMeta;
use std::collections::HashMap;
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Strategy {
    /// Highest locally-computed trust; newest round breaks ties.
    Trust,
    /// This node's own personalised adapter.
    SelfOnly,
    /// Newest round; locally-computed trust breaks ties.
    Latest,
}

impl FromStr for Strategy {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "trust" => Ok(Strategy::Trust),
            "self" => Ok(Strategy::SelfOnly),
            "latest" => Ok(Strategy::Latest),
            other => Err(format!("unknown selection strategy '{other}'")),
        }
    }
}

pub struct SelectionContext<'a> {
    pub own_node_id: &'a str,
    /// This node's own view of every peer's trust (from `trust_rankings`).
    pub local_trust: &'a HashMap<String, f64>,
    pub min_trust: f64,
    /// Adapters for any other base model are skipped. Empty disables the check.
    pub base_model: &'a str,
}

/// Pick the adapter to serve a query with. Falls back to this node's own
/// adapter when the strategy finds nothing; `None` means "use the base model".
pub fn select<'a>(
    candidates: &[&'a AdapterMeta],
    strategy: Strategy,
    ctx: &SelectionContext,
) -> Option<&'a AdapterMeta> {
    let trust_of = |m: &AdapterMeta| ctx.local_trust.get(&m.node_id).copied().unwrap_or(0.0);
    let usable: Vec<&'a AdapterMeta> = candidates
        .iter()
        .copied()
        .filter(|m| ctx.base_model.is_empty() || m.base_model == ctx.base_model)
        // A node's own adapter is always acceptable; peers must clear min_trust.
        .filter(|m| m.node_id == ctx.own_node_id || trust_of(*m) >= ctx.min_trust)
        .collect();

    let newest_own = || {
        usable
            .iter()
            .filter(|m| m.node_id == ctx.own_node_id)
            .max_by_key(|m| m.round)
            .copied()
    };

    let preferred = match strategy {
        Strategy::SelfOnly => newest_own(),
        Strategy::Trust => usable.iter().copied().max_by(|a, b| {
            trust_of(*a)
                .partial_cmp(&trust_of(*b))
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.round.cmp(&b.round))
                .then_with(|| b.node_id.cmp(&a.node_id))
        }),
        Strategy::Latest => usable.iter().copied().max_by(|a, b| {
            a.round
                .cmp(&b.round)
                .then(
                    trust_of(*a)
                        .partial_cmp(&trust_of(*b))
                        .unwrap_or(std::cmp::Ordering::Equal),
                )
                .then_with(|| b.node_id.cmp(&a.node_id))
        }),
    };

    preferred.or_else(newest_own)
}

/// Mixing weights for "System One": pick up to `k` distinct adapters and give
/// each a weight from a softmax over this node's own trust in its author.
///
/// * only adapters for our base model, above `min_trust` (our own is always eligible);
/// * the node's own adapter is always included when it has one (personalisation);
/// * identical weights (same sha256) from different nodes are counted once, keeping
///   the more trusted advertiser;
/// * weights are positive and sum to 1.
pub fn select_top_k<'a>(
    candidates: &[&'a AdapterMeta],
    ctx: &SelectionContext,
    k: usize,
    weight_temperature: f64,
) -> Vec<(&'a AdapterMeta, f64)> {
    if k == 0 {
        return Vec::new();
    }
    let trust_of = |m: &AdapterMeta| ctx.local_trust.get(&m.node_id).copied().unwrap_or(0.0);

    let mut usable: Vec<&'a AdapterMeta> = candidates
        .iter()
        .copied()
        .filter(|m| ctx.base_model.is_empty() || m.base_model == ctx.base_model)
        .filter(|m| m.node_id == ctx.own_node_id || trust_of(m) >= ctx.min_trust)
        .collect();

    // Most trusted first; newer round, then node id, make the order deterministic.
    usable.sort_by(|a, b| {
        trust_of(b)
            .partial_cmp(&trust_of(a))
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(b.round.cmp(&a.round))
            .then_with(|| a.node_id.cmp(&b.node_id))
    });

    let mut chosen: Vec<&'a AdapterMeta> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let own = usable.iter().copied().filter(|m| m.node_id == ctx.own_node_id).max_by_key(|m| m.round);
    if let Some(m) = own {
        seen.insert(m.sha256.clone());
        chosen.push(m);
    }
    for m in usable {
        if chosen.len() >= k {
            break;
        }
        if seen.insert(m.sha256.clone()) {
            chosen.push(m);
        }
    }

    // Numerically stable softmax over trust / temperature.
    let t = weight_temperature.max(1e-6);
    let logits: Vec<f64> = chosen.iter().map(|m| trust_of(m) / t).collect();
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|l| (l - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    chosen.into_iter().zip(exps).map(|(m, e)| (m, e / sum)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta(node: &str, round: u64, base: &str) -> AdapterMeta {
        AdapterMeta {
            node_id: node.into(),
            endpoint_id: String::new(),
            pubkey: String::new(),
            round,
            sha256: format!("{node}{round}"),
            blake3: String::new(),
            size: 1,
            base_model: base.into(),
            rank: 8,
            target_modules: vec![],
            adapter_config: String::new(),
            trust_score: 0.0,
            timestamp: 0,
            signature: String::new(),
        }
    }

    fn ctx<'a>(trust: &'a HashMap<String, f64>, min: f64) -> SelectionContext<'a> {
        SelectionContext { own_node_id: "me", local_trust: trust, min_trust: min, base_model: "base" }
    }

    #[test]
    fn trust_picks_most_trusted_peer() {
        let (me, a, b) = (meta("me", 3, "base"), meta("a", 3, "base"), meta("b", 3, "base"));
        let trust = HashMap::from([("a".to_string(), 0.2), ("b".to_string(), 0.9), ("me".to_string(), 0.1)]);
        let got = select(&[&me, &a, &b], Strategy::Trust, &ctx(&trust, 0.0)).unwrap();
        assert_eq!(got.node_id, "b");
    }

    #[test]
    fn self_strategy_returns_own_adapter() {
        let (me, a) = (meta("me", 2, "base"), meta("a", 9, "base"));
        let trust = HashMap::from([("a".to_string(), 5.0)]);
        let got = select(&[&me, &a], Strategy::SelfOnly, &ctx(&trust, 0.0)).unwrap();
        assert_eq!(got.node_id, "me");
    }

    #[test]
    fn min_trust_filters_peers_but_falls_back_to_own() {
        let (me, a) = (meta("me", 1, "base"), meta("a", 5, "base"));
        let trust = HashMap::from([("a".to_string(), 0.1)]);
        let got = select(&[&me, &a], Strategy::Trust, &ctx(&trust, 0.5)).unwrap();
        assert_eq!(got.node_id, "me");
    }

    #[test]
    fn wrong_base_model_is_never_selected() {
        let a = meta("a", 5, "other-base");
        let trust = HashMap::from([("a".to_string(), 9.0)]);
        assert!(select(&[&a], Strategy::Trust, &ctx(&trust, 0.0)).is_none());
    }

    #[test]
    fn latest_picks_newest_round() {
        let (a, b) = (meta("a", 2, "base"), meta("b", 7, "base"));
        let trust = HashMap::new();
        let got = select(&[&a, &b], Strategy::Latest, &ctx(&trust, 0.0)).unwrap();
        assert_eq!(got.node_id, "b");
    }

    fn same_hash(node: &str, round: u64, sha: &str) -> AdapterMeta {
        let mut m = meta(node, round, "base");
        m.sha256 = sha.into();
        m
    }

    #[test]
    fn top_k_weights_sum_to_one_and_follow_trust() {
        let (a, b, c) = (meta("a", 1, "base"), meta("b", 1, "base"), meta("c", 1, "base"));
        let trust = HashMap::from([("a".to_string(), 3.0), ("b".to_string(), 1.0), ("c".to_string(), 0.0)]);
        let got = select_top_k(&[&a, &b, &c], &ctx(&trust, 0.0), 3, 1.0);
        assert_eq!(got.len(), 3);
        assert!((got.iter().map(|(_, w)| w).sum::<f64>() - 1.0).abs() < 1e-9);
        assert_eq!(got[0].0.node_id, "a");
        assert!(got[0].1 > got[1].1 && got[1].1 > got[2].1);
    }

    #[test]
    fn top_k_caps_k_and_always_includes_own() {
        let me = meta("me", 1, "base");
        let peers: Vec<AdapterMeta> = (0..5).map(|i| meta(&format!("p{i}"), 1, "base")).collect();
        let mut refs: Vec<&AdapterMeta> = peers.iter().collect();
        refs.push(&me);
        // Own trust is the lowest, yet it must be a member.
        let mut trust: HashMap<String, f64> = (0..5).map(|i| (format!("p{i}"), 10.0 + i as f64)).collect();
        trust.insert("me".into(), 0.0);
        let got = select_top_k(&refs, &ctx(&trust, 0.0), 3, 1.0);
        assert_eq!(got.len(), 3);
        assert!(got.iter().any(|(m, _)| m.node_id == "me"));
    }

    #[test]
    fn top_k_dedupes_identical_weights() {
        let (a, b, c) = (same_hash("a", 1, "X"), same_hash("b", 1, "X"), same_hash("c", 1, "Y"));
        let trust = HashMap::from([("a".to_string(), 2.0), ("b".to_string(), 1.0), ("c".to_string(), 0.5)]);
        let got = select_top_k(&[&a, &b, &c], &ctx(&trust, 0.0), 3, 1.0);
        let hashes: Vec<&str> = got.iter().map(|(m, _)| m.sha256.as_str()).collect();
        assert_eq!(hashes, vec!["X", "Y"]);
        assert_eq!(got[0].0.node_id, "a");
    }

    #[test]
    fn top_k_respects_min_trust_and_base_model() {
        let (a, b, c) = (meta("a", 1, "base"), meta("b", 1, "base"), meta("c", 1, "other"));
        let trust = HashMap::from([("a".to_string(), 1.0), ("b".to_string(), 0.1), ("c".to_string(), 9.0)]);
        let got = select_top_k(&[&a, &b, &c], &ctx(&trust, 0.5), 3, 1.0);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].0.node_id, "a");
        assert!((got[0].1 - 1.0).abs() < 1e-9);
    }

    #[test]
    fn top_k_high_temperature_flattens_weights() {
        let (a, b) = (meta("a", 1, "base"), meta("b", 1, "base"));
        let trust = HashMap::from([("a".to_string(), 4.0), ("b".to_string(), 0.0)]);
        let sharp = select_top_k(&[&a, &b], &ctx(&trust, 0.0), 2, 0.5);
        let flat = select_top_k(&[&a, &b], &ctx(&trust, 0.0), 2, 100.0);
        assert!(sharp[0].1 > flat[0].1);
        assert!((flat[0].1 - 0.5).abs() < 0.05);
    }

    #[test]
    fn top_k_empty_registry_is_empty() {
        let trust = HashMap::new();
        assert!(select_top_k(&[], &ctx(&trust, 0.0), 3, 1.0).is_empty());
    }

    #[test]
    fn empty_registry_means_base_model() {
        let trust = HashMap::new();
        assert!(select(&[], Strategy::Trust, &ctx(&trust, 0.0)).is_none());
    }
}
