//! Decentralized planning (A5).
//!
//! Each generation a planner committee is drawn by VRF sortition over node
//! keys, weighted by planning trust. Every committee member submits a proposal
//! citing the same evidence root. The agenda is a pure function of
//! (evidence root, committee, proposals), so any node can recompute it:
//!
//! * proposals citing another evidence root are discarded;
//! * a planner's weight is capped so no single planner can carry an item;
//! * an item is adopted only with support from at least two distinct planners
//!   and at least `min_support` of the (capped) committee weight;
//! * a per-tag diversity quota limits how many adopted items share a tag;
//! * everything proposed but not adopted is kept as recorded dissent.

use crate::agents::records::AgendaItem;
use crate::agents::vrf::{self, VrfProof};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub fn sortition_alpha(program: &str, generation: u64, evidence_root: &str) -> Vec<u8> {
    let mut a = b"slakshna/planner-committee/v1\0".to_vec();
    a.extend_from_slice(program.as_bytes());
    a.extend_from_slice(&generation.to_le_bytes());
    a.extend_from_slice(evidence_root.as_bytes());
    a
}

/// A node's claim to a committee ticket for one (program, generation, evidence).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Ticket {
    pub node_id: String,
    pub public_key: String,
    pub proof: String,
}

impl Ticket {
    pub fn create(node_id: &str, seed: &[u8; 32], program: &str, generation: u64, evidence_root: &str) -> Ticket {
        let (_, proof) = vrf::prove(seed, &sortition_alpha(program, generation, evidence_root));
        Ticket { node_id: node_id.into(), public_key: hex::encode(vrf::public_key(seed)), proof: proof.to_hex() }
    }

    /// Verified VRF output, bound to the node id (the id must derive from the key).
    fn verified_output(&self, alpha: &[u8]) -> Option<[u8; 64]> {
        if !crate::identity::node_id_matches_key(&self.node_id, &self.public_key) {
            return None;
        }
        let pk: [u8; 32] = hex::decode(&self.public_key).ok()?.try_into().ok()?;
        vrf::verify(&pk, alpha, &VrfProof::from_hex(&self.proof)?)
    }
}

/// Weighted sortition without replacement (Efraimidis–Spirakis): each ticket
/// gets key = -ln(u)/weight with u the VRF draw; the `k` smallest keys win.
/// Nodes with non-positive weight, invalid tickets and duplicates are skipped.
pub fn select_committee(
    tickets: &[Ticket],
    weights: &BTreeMap<String, f64>,
    program: &str,
    generation: u64,
    evidence_root: &str,
    k: usize,
) -> Vec<String> {
    let alpha = sortition_alpha(program, generation, evidence_root);
    let mut seen = BTreeSet::new();
    let mut keyed: Vec<(f64, String)> = vec![];
    for t in tickets {
        let w = weights.get(&t.node_id).copied().unwrap_or(0.0);
        if !(w.is_finite() && w > 0.0) {
            continue;
        }
        // Verify before de-duplicating: an invalid ticket carrying someone
        // else's id must not displace that node's real ticket.
        if let Some(out) = t.verified_output(&alpha) {
            if seen.insert(t.node_id.clone()) {
                keyed.push((-vrf::unit_interval(&out).ln() / w, t.node_id.clone()));
            }
        }
    }
    keyed.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then_with(|| a.1.cmp(&b.1)));
    keyed.into_iter().take(k).map(|(_, id)| id).collect()
}

pub use crate::agents::records::ProposedItem;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Proposal {
    pub planner: String,
    pub evidence_root: String,
    pub items: Vec<ProposedItem>,
}

#[derive(Debug, Clone)]
pub struct MergeConfig {
    /// Max share of total committee weight any one planner may carry.
    pub weight_cap: f64,
    /// Share of capped committee weight an item needs.
    pub min_support: f64,
    pub max_items: usize,
    pub max_per_tag: usize,
    /// Distinct planners that must propose an item. 2 (default) means nothing is adopted without
    /// agreement; 1 lets any planner's idea in, bounded by `solo_per_planner`.
    pub min_supporters: usize,
    /// With `min_supporters == 1`: most items one planner may get adopted on its own proposal alone, so a
    /// single (possibly malicious) planner cannot flood the agenda.
    pub solo_per_planner: usize,
}

impl Default for MergeConfig {
    fn default() -> Self {
        MergeConfig { weight_cap: 0.3, min_support: 0.34, max_items: 8, max_per_tag: 3, min_supporters: 2, solo_per_planner: 2 }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Dissent {
    pub item_id: String,
    pub proposers: Vec<String>,
    pub support: f64,
    pub reason: &'static str,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MergedAgenda {
    pub items: Vec<AgendaItem>,
    pub dissent: Vec<Dissent>,
    /// Planners whose proposals were discarded (wrong evidence root / not on committee).
    pub rejected_planners: Vec<String>,
}

/// Deterministic trust-weighted merge. `committee` maps planner -> planning
/// trust weight for exactly the members drawn by `select_committee`.
pub fn merge(
    evidence_root: &str,
    committee: &BTreeMap<String, f64>,
    proposals: &[Proposal],
    cfg: &MergeConfig,
) -> MergedAgenda {
    // One proposal per committee planner, citing the right evidence.
    let mut valid: BTreeMap<&str, &Proposal> = BTreeMap::new();
    let mut rejected = BTreeSet::new();
    let mut sorted: Vec<&Proposal> = proposals.iter().collect();
    sorted.sort_by(|a, b| a.planner.cmp(&b.planner).then_with(|| serde_json::to_string(a).unwrap().cmp(&serde_json::to_string(b).unwrap())));
    for p in sorted {
        if !committee.contains_key(&p.planner) || p.evidence_root != evidence_root || valid.contains_key(p.planner.as_str()) {
            rejected.insert(p.planner.clone());
            continue;
        }
        valid.insert(&p.planner, p);
    }

    // Cap each planner's weight at `weight_cap` of the *capped* total, found
    // by iterating to the fixed point (capping against the raw total would let
    // a very heavy planner dominate the denominator).
    let mut capped: BTreeMap<&str, f64> = committee.iter().map(|(k, w)| (k.as_str(), w.max(0.0))).collect();
    for _ in 0..64 {
        let t: f64 = capped.values().sum();
        let limit = cfg.weight_cap * t;
        let mut changed = false;
        for w in capped.values_mut() {
            if *w > limit + 1e-12 {
                *w = limit;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    let capped_total: f64 = capped.values().sum();

    struct Cand<'a> {
        item: &'a ProposedItem,
        supporters: BTreeSet<&'a str>,
        support: f64,
    }
    let mut cands: BTreeMap<&str, Cand> = BTreeMap::new();
    for (planner, p) in &valid {
        let mut once = BTreeSet::new();
        for it in &p.items {
            if !once.insert(it.id.as_str()) {
                continue; // a planner counts once per item
            }
            let c = cands.entry(it.id.as_str()).or_insert_with(|| Cand { item: it, supporters: BTreeSet::new(), support: 0.0 });
            // On conflicting definitions of one id, the definition of the lowest planner id wins (iteration order).
            c.supporters.insert(planner);
            c.support += capped[planner];
        }
    }

    let mut ranked: Vec<&Cand> = cands.values().collect();
    ranked.sort_by(|a, b| b.support.partial_cmp(&a.support).unwrap().then_with(|| a.item.id.cmp(&b.item.id)));

    let mut items = vec![];
    let mut dissent = vec![];
    let mut per_tag: BTreeMap<&str, usize> = BTreeMap::new();
    let mut solo_used: BTreeMap<&str, usize> = BTreeMap::new();
    for c in ranked {
        let share = if capped_total > 0.0 { c.support / capped_total } else { 0.0 };
        let solo = c.supporters.len() == 1;
        let solo_planner = c.supporters.iter().next().copied();
        let reason = if c.supporters.len() < cfg.min_supporters && valid.len() >= 2 {
            Some("single supporter")
        } else if solo && cfg.min_supporters <= 1 && solo_used.get(solo_planner.unwrap_or("")).copied().unwrap_or(0) >= cfg.solo_per_planner {
            Some("solo quota")
        } else if cfg.min_supporters >= 2 && share + 1e-12 < cfg.min_support {
            Some("insufficient support")
        } else if items.len() >= cfg.max_items {
            Some("agenda full")
        } else if per_tag.get(c.item.tag.as_str()).copied().unwrap_or(0) >= cfg.max_per_tag {
            Some("diversity quota")
        } else {
            None
        };
        match reason {
            None => {
                if solo {
                    *solo_used.entry(solo_planner.unwrap_or("")).or_default() += 1;
                }
                *per_tag.entry(c.item.tag.as_str()).or_default() += 1;
                items.push(AgendaItem { id: c.item.id.clone(), description: c.item.description.clone(), requirements: c.item.requirements.clone() });
            }
            Some(reason) => dissent.push(Dissent {
                item_id: c.item.id.clone(),
                proposers: c.supporters.iter().map(|s| s.to_string()).collect(),
                support: share,
                reason,
            }),
        }
    }
    MergedAgenda { items, dissent, rejected_planners: rejected.into_iter().collect() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agents::records::Requirements;
    use crate::identity::Keypair;

    fn nodes(n: usize) -> Vec<(Keypair, String)> {
        (0..n).map(|_| { let k = Keypair::generate(); let id = k.node_id().to_string(); (k, id) }).collect()
    }

    fn tickets(ns: &[(Keypair, String)], prog: &str, gen: u64, root: &str) -> Vec<Ticket> {
        ns.iter().map(|(k, id)| Ticket::create(id, &k.to_bytes(), prog, gen, root)).collect()
    }

    fn weights(ns: &[(Keypair, String)], w: f64) -> BTreeMap<String, f64> {
        ns.iter().map(|(_, id)| (id.clone(), w)).collect()
    }

    #[test]
    fn committee_is_verifiable_deterministic_and_order_independent() {
        let ns = nodes(10);
        let t = tickets(&ns, "p", 1, "r");
        let w = weights(&ns, 1.0);
        let a = select_committee(&t, &w, "p", 1, "r", 4);
        assert_eq!(a.len(), 4);
        let mut rev = t.clone();
        rev.reverse();
        assert_eq!(a, select_committee(&rev, &w, "p", 1, "r", 4));
        // different evidence/generation -> different draw (overwhelmingly likely)
        let t2 = tickets(&ns, "p", 2, "r");
        assert_ne!(a, select_committee(&t2, &w, "p", 2, "r", 4));
    }

    #[test]
    fn invalid_forged_or_zero_weight_tickets_never_win() {
        let ns = nodes(6);
        let mut t = tickets(&ns, "p", 1, "r");
        let mut w = weights(&ns, 1.0);
        // ticket proving for a different generation is invalid for this one
        t[0] = Ticket::create(&ns[0].1, &ns[0].0.to_bytes(), "p", 99, "r");
        // node 1 claims node 2's id with its own key
        t[1].node_id = ns[2].1.clone();
        w.insert(ns[3].1.clone(), 0.0);
        let c = select_committee(&t, &w, "p", 1, "r", 6);
        assert!(!c.contains(&ns[0].1));
        assert!(!c.contains(&ns[3].1));
        // the impersonated id is only selectable via its real ticket
        assert!(c.iter().filter(|x| **x == ns[2].1).count() <= 1);
        // valid: nodes 2, 4, 5 only. node0 has a proof for another generation, node1's
        // ticket claims node2's id with its own key, node3 has zero weight.
        assert_eq!(c.len(), 3);
        assert!(c.contains(&ns[2].1), "stolen-id ticket must not displace the real one");
    }

    #[test]
    fn higher_trust_is_selected_more_often() {
        let ns = nodes(2);
        let mut w = weights(&ns, 1.0);
        w.insert(ns[0].1.clone(), 5.0);
        let mut wins = 0;
        let trials = 60;
        for g in 0..trials {
            let t = tickets(&ns, "p", g, "r");
            if select_committee(&t, &w, "p", g, "r", 1)[0] == ns[0].1 { wins += 1; }
        }
        // expected 5/6 ≈ 50 of 60
        assert!(wins > 40, "heavy node won only {wins}/{trials}");
    }

    fn item(id: &str, tag: &str) -> ProposedItem {
        ProposedItem { id: id.into(), description: format!("do {id}"), tag: tag.into(), requirements: Requirements::default() }
    }

    fn prop(p: &str, root: &str, items: &[(&str, &str)]) -> Proposal {
        Proposal { planner: p.into(), evidence_root: root.into(), items: items.iter().map(|(i, t)| item(i, t)).collect() }
    }

    fn committee(ids: &[&str]) -> BTreeMap<String, f64> {
        ids.iter().map(|i| (i.to_string(), 1.0)).collect()
    }

    #[test]
    fn agenda_is_recomputable_from_any_order() {
        let c = committee(&["a", "b", "c", "d"]);
        let ps = vec![
            prop("a", "r", &[("x", "t1"), ("y", "t2")]),
            prop("b", "r", &[("x", "t1"), ("z", "t3")]),
            prop("c", "r", &[("x", "t1"), ("y", "t2")]),
            prop("d", "r", &[("y", "t2")]),
        ];
        let m1 = merge("r", &c, &ps, &MergeConfig::default());
        let mut rev = ps.clone();
        rev.reverse();
        assert_eq!(m1, merge("r", &c, &rev, &MergeConfig::default()));
        let ids: Vec<_> = m1.items.iter().map(|i| i.id.as_str()).collect();
        assert_eq!(ids, ["x", "y"]); // x: 3 supporters, y: 3 supporters; z has one
        assert!(m1.dissent.iter().any(|d| d.item_id == "z" && d.reason == "single supporter"));
    }

    #[test]
    fn one_malicious_planner_cannot_steer_the_generation() {
        let c = committee(&["a", "b", "c", "m"]);
        // m floods junk items, even 100 of them, and omits the honest one
        let junk: Vec<(String, String)> = (0..100).map(|i| (format!("junk{i}"), format!("tag{i}"))).collect();
        let junk_refs: Vec<(&str, &str)> = junk.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
        let ps = vec![
            prop("a", "r", &[("good", "t")]),
            prop("b", "r", &[("good", "t")]),
            prop("c", "r", &[("good", "t")]),
            prop("m", "r", &junk_refs),
        ];
        let m = merge("r", &c, &ps, &MergeConfig::default());
        assert_eq!(m.items.iter().map(|i| i.id.as_str()).collect::<Vec<_>>(), vec!["good"]);
        assert_eq!(m.dissent.len(), 100);
    }

    #[test]
    fn solo_proposals_can_be_allowed_but_one_planner_cannot_flood() {
        let c = committee(&["a", "b", "c"]);
        // three planners with entirely different ideas (what independent LLM planners produce)
        let ps = vec![prop("a", "r", &[("a1", "t1"), ("a2", "t2")]), prop("b", "r", &[("b1", "t3")]), prop("c", "r", &[("c1", "t4"), ("c2", "t5"), ("c3", "t6")])];
        // default: nothing is adopted without agreement
        assert!(merge("r", &c, &ps, &MergeConfig::default()).items.is_empty());
        let open = MergeConfig { min_supporters: 1, solo_per_planner: 2, ..Default::default() };
        let m = merge("r", &c, &ps, &open);
        let ids: Vec<_> = m.items.iter().map(|i| i.id.as_str()).collect();
        assert_eq!(ids.len(), 5, "2 from a, 1 from b, 2 of c's 3: {ids:?}");
        assert!(m.dissent.iter().any(|d| d.reason == "solo quota" && d.proposers == vec!["c".to_string()]));
        // a flooding planner still gets at most its quota
        let junk: Vec<(String, String)> = (0..50).map(|i| (format!("j{i}"), format!("tag{i}"))).collect();
        let junk_refs: Vec<(&str, &str)> = junk.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
        let flood = vec![prop("a", "r", &[("ok", "t")]), prop("b", "r", &[("ok", "t")]), prop("c", "r", &junk_refs)];
        let m = merge("r", &c, &flood, &open);
        assert_eq!(m.items.iter().filter(|i| i.id.starts_with('j')).count(), 2);
        assert!(m.items.iter().any(|i| i.id == "ok"));
        // deterministic regardless of proposal order
        let mut rev = ps.clone();
        rev.reverse();
        assert_eq!(merge("r", &c, &rev, &open), merge("r", &c, &ps, &open));
    }

    #[test]
    fn heavy_planner_weight_is_capped() {
        let mut c = committee(&["a", "b", "c"]);
        c.insert("whale".into(), 1000.0);
        let ps = vec![prop("whale", "r", &[("w", "t")]), prop("a", "r", &[("ok", "t")]), prop("b", "r", &[("ok", "t")]), prop("c", "r", &[("ok", "t")])];
        let m = merge("r", &c, &ps, &MergeConfig::default());
        assert_eq!(m.items.iter().map(|i| i.id.as_str()).collect::<Vec<_>>(), vec!["ok"]);
    }

    #[test]
    fn wrong_evidence_duplicates_and_non_members_are_discarded() {
        let c = committee(&["a", "b", "c"]);
        let ps = vec![
            prop("a", "r", &[("x", "t")]),
            prop("b", "r", &[("x", "t")]),
            prop("c", "STALE", &[("x", "t"), ("y", "t")]),
            prop("outsider", "r", &[("y", "t")]),
            prop("a", "r", &[("y", "t")]), // second proposal from a: ignored
        ];
        let m = merge("r", &c, &ps, &MergeConfig::default());
        assert!(m.rejected_planners.contains(&"c".to_string()));
        assert!(m.rejected_planners.contains(&"outsider".to_string()));
        assert!(m.items.iter().all(|i| i.id == "x"));
    }

    #[test]
    fn diversity_quota_and_agenda_cap() {
        let c = committee(&["a", "b", "c"]);
        let items: Vec<(&str, &str)> = vec![("i1", "same"), ("i2", "same"), ("i3", "same"), ("i4", "same"), ("i5", "other")];
        let ps: Vec<_> = ["a", "b", "c"].iter().map(|p| prop(p, "r", &items)).collect();
        let cfg = MergeConfig { max_per_tag: 2, max_items: 3, ..Default::default() };
        let m = merge("r", &c, &ps, &cfg);
        let ids: Vec<_> = m.items.iter().map(|i| i.id.as_str()).collect();
        assert_eq!(ids, ["i1", "i2", "i5"]);
        assert!(m.dissent.iter().any(|d| d.reason == "diversity quota"));
    }
}
