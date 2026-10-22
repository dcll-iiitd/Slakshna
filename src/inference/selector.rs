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

/// Candidates in the order a query should try them, best first. The caller walks the
/// list, so one unreachable or corrupt adapter falls through to the next best one
/// instead of straight to the base model. Falls back to this node's own adapter
/// (always in the list unless `SelfOnly` has none). Empty means "use the base model".
///
/// Ties are broken towards this node's own adapter (when nothing separates two
/// adapters, the personalised one is the safe choice), then by node id so the order
/// is deterministic rather than depending on hash-map iteration.
pub fn rank<'a>(
    candidates: &[&'a AdapterMeta],
    strategy: Strategy,
    ctx: &SelectionContext,
) -> Vec<&'a AdapterMeta> {
    use std::cmp::Ordering;

    let trust_of = |m: &AdapterMeta| ctx.local_trust.get(&m.node_id).copied().unwrap_or(0.0);
    let is_own = |m: &AdapterMeta| m.node_id == ctx.own_node_id;
    let mut usable: Vec<&'a AdapterMeta> = candidates
        .iter()
        .copied()
        .filter(|m| ctx.base_model.is_empty() || m.base_model == ctx.base_model)
        // A node's own adapter is always acceptable; peers must clear min_trust.
        .filter(|m| is_own(m) || trust_of(m) >= ctx.min_trust)
        .collect();

    let by_trust = |a: &AdapterMeta, b: &AdapterMeta| {
        trust_of(b).partial_cmp(&trust_of(a)).unwrap_or(Ordering::Equal)
    };
    // Descending in everything that is "better", then own first, then node id.
    let tail = |a: &AdapterMeta, b: &AdapterMeta| is_own(b).cmp(&is_own(a)).then_with(|| a.node_id.cmp(&b.node_id));

    match strategy {
        Strategy::SelfOnly => {
            usable.retain(|m| is_own(m));
            usable.sort_by(|a, b| b.round.cmp(&a.round));
        }
        Strategy::Trust => usable.sort_by(|a, b| {
            by_trust(a, b).then(b.round.cmp(&a.round)).then_with(|| tail(a, b))
        }),
        Strategy::Latest => usable.sort_by(|a, b| {
            b.round.cmp(&a.round).then_with(|| by_trust(a, b)).then_with(|| tail(a, b))
        }),
    }
    usable
}

/// The single best candidate, see [`rank`].
pub fn select<'a>(
    candidates: &[&'a AdapterMeta],
    strategy: Strategy,
    ctx: &SelectionContext,
) -> Option<&'a AdapterMeta> {
    rank(candidates, strategy, ctx).into_iter().next()
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

    #[test]
    fn ties_go_to_the_own_adapter_not_to_whoever_sorts_first() {
        // No trust information at all (e.g. a freshly restarted serve-only node), same round.
        let (me, a) = (meta("zzz-me", 3, "base"), meta("aaa", 3, "base"));
        let trust = HashMap::new();
        let c = SelectionContext { own_node_id: "zzz-me", local_trust: &trust, min_trust: 0.0, base_model: "base" };
        assert_eq!(select(&[&a, &me], Strategy::Trust, &c).unwrap().node_id, "zzz-me");
        assert_eq!(select(&[&a, &me], Strategy::Latest, &c).unwrap().node_id, "zzz-me");
    }

    #[test]
    fn rank_is_a_full_fallback_order() {
        let (me, a, b) = (meta("me", 1, "base"), meta("a", 2, "base"), meta("b", 2, "base"));
        let trust = HashMap::from([("a".to_string(), 0.3), ("b".to_string(), 0.6), ("me".to_string(), 0.1)]);
        let order: Vec<&str> = rank(&[&me, &a, &b], Strategy::Trust, &ctx(&trust, 0.0))
            .iter()
            .map(|m| m.node_id.as_str())
            .collect();
        assert_eq!(order, vec!["b", "a", "me"]);
    }

    #[test]
    fn rank_self_only_has_just_own_adapters_newest_first() {
        let (old, new, a) = (meta("me", 1, "base"), meta("me", 4, "base"), meta("a", 9, "base"));
        let trust = HashMap::from([("a".to_string(), 9.0)]);
        let order: Vec<u64> = rank(&[&old, &a, &new], Strategy::SelfOnly, &ctx(&trust, 0.0))
            .iter()
            .map(|m| m.round)
            .collect();
        assert_eq!(order, vec![4, 1]);
    }

    #[test]
    fn empty_registry_means_base_model() {
        let trust = HashMap::new();
        assert!(select(&[], Strategy::Trust, &ctx(&trust, 0.0)).is_none());
    }
}
