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
    fn empty_registry_means_base_model() {
        let trust = HashMap::new();
        assert!(select(&[], Strategy::Trust, &ctx(&trust, 0.0)).is_none());
    }
}
