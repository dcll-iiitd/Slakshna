//! Distributed task market (A3).
//!
//! Agenda items become claimable tasks. A node claims an item with a lease,
//! keeps it with heartbeats, and finishes it with a completion that commits to
//! the result hash. When the lease lapses (crash, partition) the item becomes
//! claimable again, so a generation completes even if a node dies mid-way.
//!
//! The market is a deterministic function of the signed claim records plus
//! the current time, so every node derives the same owner for an item:
//! among live claims the owner is the one with the lowest `(at, claim hash)`.
//!
//! Limit: `at` is self-reported, so a malicious claimant can backdate a claim
//! to preempt a live holder. The market does not prevent that (there is no
//! trusted clock); the damage is bounded because the claim must stay alive
//! with signed heartbeats, and the behaviour is attributable and feeds
//! reputation (M4) and slashing. A holder that is preempted simply sees
//! `status != Claimed{by: me}` and stops.

use crate::agents::records::*;
use crate::history::{RecordKind, UpdateRecord};
use std::collections::{BTreeMap, HashMap};

/// What a node offers (extends ServiceAnnounce): used to check requirements.
#[derive(Debug, Clone, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub struct CapabilityAd {
    pub node_id: String,
    pub gpu_class: Option<String>,
    pub data_access: Vec<String>,
    /// GPU-seconds the node is willing to spend on one item.
    pub gpu_seconds: u64,
}

impl CapabilityAd {
    pub fn satisfies(&self, req: &Requirements) -> Result<(), String> {
        if let Some(g) = &req.gpu_class {
            if self.gpu_class.as_ref() != Some(g) {
                return Err(format!("needs GPU class {g}"));
            }
        }
        if let Some(d) = &req.data_access {
            if !self.data_access.contains(d) {
                return Err(format!("needs access to data '{d}'"));
            }
        }
        if req.gpu_seconds > self.gpu_seconds {
            return Err("item needs more GPU-seconds than the node offers".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct ClaimState {
    hash: String,
    claimant: String,
    at: i64,
    lease: i64,
    last_beat: i64,
    released: bool,
}

impl ClaimState {
    fn expires_at(&self) -> i64 {
        self.last_beat + self.lease
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ItemStatus {
    Open,
    Claimed { by: String, claim: String, expires_at: i64 },
    Done { by: String, result: String },
}

#[derive(Debug, Default)]
pub struct Market {
    /// (agenda hash, item id) -> requirements
    items: BTreeMap<(String, String), Requirements>,
    claims: HashMap<(String, String), Vec<ClaimState>>,
    /// (agenda, item) -> completions (claim hash, completer, at, result)
    completions: HashMap<(String, String), Vec<(String, String, i64, String)>>,
    pending_events: Vec<(String, String, ClaimEvent)>, // author, record hash, event
}

#[derive(Debug, thiserror::Error, PartialEq)]
#[allow(dead_code)] // library surface: used by tests and integrators, or pending the settlement-chain decision
pub enum MarketError {
    #[error("unknown item")]
    UnknownItem,
    #[error("item already claimed by {0}")]
    Taken(String),
    #[error("item already completed")]
    Done,
    #[error("requirements not met: {0}")]
    Requirements(String),
}

impl Market {
    pub fn from_records<'a>(records: impl IntoIterator<Item = &'a UpdateRecord>) -> Self {
        let mut m = Market::default();
        for r in records {
            m.apply(r);
        }
        m
    }

    /// Ingest a verified record: an Agenda posts items, claim events drive leases.
    pub fn apply(&mut self, r: &UpdateRecord) {
        let RecordKind::Provenance { kind, body } = &r.kind else { return };
        if r.verify().is_err() {
            return;
        }
        match kind.as_str() {
            KIND_AGENDA => {
                if let Ok(a) = serde_json::from_value::<Agenda>(body.clone()) {
                    // Only committee members may post an agenda's items.
                    if !a.committee.contains(&r.node_id) {
                        return;
                    }
                    for it in a.items {
                        self.items.entry((r.hash.clone(), it.id)).or_insert(it.requirements);
                    }
                    self.flush_pending();
                }
            }
            KIND_CLAIM => {
                if let Ok(e) = serde_json::from_value::<ClaimEvent>(body.clone()) {
                    self.pending_events.push((r.node_id.clone(), r.hash.clone(), e));
                    self.flush_pending();
                }
            }
            _ => {}
        }
    }

    /// Events can arrive before the agenda or before their claim (gossip
    /// order); keep retrying until they apply. Applying is idempotent.
    fn flush_pending(&mut self) {
        loop {
            let before = self.pending_events.len();
            let mut left = vec![];
            for (author, hash, e) in std::mem::take(&mut self.pending_events) {
                if !self.try_apply_event(&author, &hash, &e) {
                    left.push((author, hash, e));
                }
            }
            self.pending_events = left;
            if self.pending_events.len() == before {
                break;
            }
        }
    }

    fn try_apply_event(&mut self, author: &str, hash: &str, e: &ClaimEvent) -> bool {
        let key = (e.agenda.clone(), e.item_id.clone());
        if !self.items.contains_key(&key) {
            return false;
        }
        match e.action {
            ClaimAction::Claim => {
                let list = self.claims.entry(key).or_default();
                if !list.iter().any(|c| c.hash == hash) {
                    list.push(ClaimState { hash: hash.into(), claimant: author.into(), at: e.at, lease: e.lease_secs as i64, last_beat: e.at, released: false });
                }
                true
            }
            ClaimAction::Heartbeat | ClaimAction::Release | ClaimAction::Complete => {
                let want = e.claim.as_deref().unwrap_or("");
                let Some(c) = self.claims.get_mut(&key).and_then(|l| l.iter_mut().find(|c| c.hash == want)) else {
                    return false; // claim not seen yet
                };
                if c.claimant != author {
                    return true; // someone else's claim: permanently ignored
                }
                match e.action {
                    ClaimAction::Heartbeat => c.last_beat = c.last_beat.max(e.at),
                    ClaimAction::Release => c.released = true,
                    ClaimAction::Complete => {
                        // Only valid if the lease was live when completing.
                        if e.at < c.expires_at() && !c.released {
                            let v = self.completions.entry(key).or_default();
                            if !v.iter().any(|x| x.0 == hash) {
                                v.push((hash.into(), author.into(), e.at, e.result.clone().unwrap_or_default()));
                            }
                        }
                    }
                    ClaimAction::Claim => unreachable!(),
                }
                true
            }
        }
    }

    #[allow(dead_code)] // library surface: used by tests and integrators, or pending the settlement-chain decision
    pub fn requirements(&self, agenda: &str, item: &str) -> Option<&Requirements> {
        self.items.get(&(agenda.to_string(), item.to_string()))
    }

    /// Status of an item as seen by an observer whose clock reads `now`.
    pub fn status(&self, agenda: &str, item: &str, now: i64) -> Option<ItemStatus> {
        let key = (agenda.to_string(), item.to_string());
        self.items.get(&key)?;
        if let Some(done) = self.completions.get(&key).and_then(|v| v.iter().min_by(|a, b| (a.2, &a.0).cmp(&(b.2, &b.0)))) {
            return Some(ItemStatus::Done { by: done.1.clone(), result: done.3.clone() });
        }
        let owner = self
            .claims
            .get(&key)
            .into_iter()
            .flatten()
            .filter(|c| !c.released && c.expires_at() > now)
            .map(|c| (c.at, &c.hash, c))
            .min_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
        Some(match owner {
            Some((_, _, c)) => ItemStatus::Claimed { by: c.claimant.clone(), claim: c.hash.clone(), expires_at: c.expires_at() },
            None => ItemStatus::Open,
        })
    }

    /// Items a node with `ad` could claim right now.
    pub fn claimable(&self, ad: &CapabilityAd, now: i64) -> Vec<(String, String)> {
        self.items
            .iter()
            .filter(|((a, i), req)| ad.satisfies(req).is_ok() && self.status(a, i, now) == Some(ItemStatus::Open))
            .map(|(k, _)| k.clone())
            .collect()
    }

    /// Check that `ad` may claim the item now (local pre-flight before emitting a claim record).
    #[allow(dead_code)] // library surface: used by tests and integrators, or pending the settlement-chain decision
    pub fn can_claim(&self, ad: &CapabilityAd, agenda: &str, item: &str, now: i64) -> Result<(), MarketError> {
        let req = self.requirements(agenda, item).ok_or(MarketError::UnknownItem)?;
        match self.status(agenda, item, now).ok_or(MarketError::UnknownItem)? {
            ItemStatus::Done { .. } => return Err(MarketError::Done),
            ItemStatus::Claimed { by, .. } => return Err(MarketError::Taken(by)),
            ItemStatus::Open => {}
        }
        ad.satisfies(req).map_err(MarketError::Requirements)
    }

    /// Distinct nodes that claimed the item and then released it (a failed attempt).
    pub fn failures(&self, agenda: &str, item: &str) -> usize {
        let key = (agenda.to_string(), item.to_string());
        let claimants: std::collections::BTreeSet<&str> = self.claims.get(&key).into_iter().flatten().filter(|c| c.released).map(|c| c.claimant.as_str()).collect();
        claimants.len()
    }

    /// Every item of the agenda is either done or has been failed by at least `max_failures` distinct nodes
    /// (so the generation can move on instead of waiting on an item nobody can build).
    pub fn all_settled(&self, agenda: &str, now: i64, max_failures: usize) -> bool {
        let mut any = false;
        for (a, i) in self.items.keys().filter(|(a, _)| a == agenda) {
            any = true;
            let done = matches!(self.status(a, i, now), Some(ItemStatus::Done { .. }));
            if !done && self.failures(a, i) < max_failures {
                return false;
            }
        }
        any
    }

    pub fn all_done(&self, agenda: &str, now: i64) -> bool {
        let mut any = false;
        for (a, i) in self.items.keys().filter(|(a, _)| a == agenda) {
            any = true;
            if !matches!(self.status(a, i, now), Some(ItemStatus::Done { .. })) {
                return false;
            }
        }
        any
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agents::graph::make_record;
    use crate::identity::Keypair;

    struct Site { kp: Keypair, id: String }
    fn site() -> Site { let kp = Keypair::generate(); let id = kp.node_id().to_string(); Site { kp, id } }

    struct Log { recs: Vec<UpdateRecord> }
    impl Log {
        fn emit<T: serde::Serialize>(&mut self, s: &Site, kind: &str, body: &T) -> String {
            let prev = self.recs.iter().rev().find(|r| r.node_id == s.id).map(|r| r.hash.clone()).unwrap_or("0".repeat(64));
            let mut r = make_record(&s.id, &prev, kind, body);
            r.sign(&s.kp);
            let h = r.hash.clone();
            self.recs.push(r);
            h
        }
    }

    fn agenda(log: &mut Log, planner: &Site, n: usize, req: Requirements) -> String {
        log.emit(planner, KIND_AGENDA, &Agenda {
            program: "a".repeat(64), generation: 1, evidence_root: "b".repeat(64),
            committee: vec![planner.id.clone()],
            items: (0..n).map(|i| AgendaItem { id: format!("i{i}"), description: "d".into(), requirements: req.clone() }).collect(),
        })
    }

    fn ev(agenda: &str, item: &str, action: ClaimAction, at: i64, lease: u64, claim: Option<&str>, result: Option<&str>) -> ClaimEvent {
        ClaimEvent { agenda: agenda.into(), item_id: item.into(), action, at, lease_secs: lease, claim: claim.map(String::from), result: result.map(String::from) }
    }

    #[test]
    fn claim_heartbeat_complete_lifecycle() {
        let (p, a) = (site(), site());
        let mut log = Log { recs: vec![] };
        let ag = agenda(&mut log, &p, 1, Requirements::default());
        let c = log.emit(&a, KIND_CLAIM, &ev(&ag, "i0", ClaimAction::Claim, 100, 60, None, None));
        let m = Market::from_records(&log.recs);
        assert!(matches!(m.status(&ag, "i0", 120), Some(ItemStatus::Claimed { ref by, .. }) if *by == a.id));
        log.emit(&a, KIND_CLAIM, &ev(&ag, "i0", ClaimAction::Heartbeat, 150, 0, Some(&c), None));
        let m = Market::from_records(&log.recs);
        assert!(matches!(m.status(&ag, "i0", 200), Some(ItemStatus::Claimed { expires_at: 210, .. })));
        log.emit(&a, KIND_CLAIM, &ev(&ag, "i0", ClaimAction::Complete, 160, 0, Some(&c), Some(&"f".repeat(64))));
        let m = Market::from_records(&log.recs);
        assert_eq!(m.status(&ag, "i0", 500), Some(ItemStatus::Done { by: a.id.clone(), result: "f".repeat(64) }));
        assert!(m.all_done(&ag, 500));
    }

    #[test]
    fn lapsed_lease_reopens_item_and_late_completion_is_void() {
        let (p, a, b) = (site(), site(), site());
        let mut log = Log { recs: vec![] };
        let ag = agenda(&mut log, &p, 1, Requirements::default());
        let ca = log.emit(&a, KIND_CLAIM, &ev(&ag, "i0", ClaimAction::Claim, 100, 30, None, None));
        let m = Market::from_records(&log.recs);
        assert_eq!(m.status(&ag, "i0", 131), Some(ItemStatus::Open), "no heartbeat -> reassignable");
        // b takes over; a's zombie completion after expiry must not count
        log.emit(&b, KIND_CLAIM, &ev(&ag, "i0", ClaimAction::Claim, 140, 30, None, None));
        log.emit(&a, KIND_CLAIM, &ev(&ag, "i0", ClaimAction::Complete, 145, 0, Some(&ca), Some(&"9".repeat(64))));
        let m = Market::from_records(&log.recs);
        assert!(matches!(m.status(&ag, "i0", 150), Some(ItemStatus::Claimed { ref by, .. }) if *by == b.id));
    }

    #[test]
    fn earliest_live_claim_owns_the_item_deterministically() {
        let (p, a, b) = (site(), site(), site());
        let mut log = Log { recs: vec![] };
        let ag = agenda(&mut log, &p, 1, Requirements::default());
        log.emit(&b, KIND_CLAIM, &ev(&ag, "i0", ClaimAction::Claim, 1001, 100, None, None));
        log.emit(&a, KIND_CLAIM, &ev(&ag, "i0", ClaimAction::Claim, 1000, 100, None, None));
        let want = |m: &Market| matches!(m.status(&ag, "i0", 1010), Some(ItemStatus::Claimed { ref by, .. }) if *by == a.id);
        assert!(want(&Market::from_records(&log.recs)));
        let mut rev = log.recs.clone();
        rev.reverse();
        assert!(want(&Market::from_records(&rev)), "owner must not depend on arrival order");
    }

    #[test]
    fn backdated_claim_can_preempt_but_dies_without_heartbeats() {
        // Documents the limit stated in the module docs.
        let (p, a, b) = (site(), site(), site());
        let mut log = Log { recs: vec![] };
        let ag = agenda(&mut log, &p, 1, Requirements::default());
        log.emit(&a, KIND_CLAIM, &ev(&ag, "i0", ClaimAction::Claim, 1000, 100, None, None));
        log.emit(&b, KIND_CLAIM, &ev(&ag, "i0", ClaimAction::Claim, 900, 150, None, None));
        let m = Market::from_records(&log.recs);
        assert!(matches!(m.status(&ag, "i0", 1010), Some(ItemStatus::Claimed { ref by, .. }) if *by == b.id));
        // b never heartbeats: its claim lapses at 1050 and a's still-live claim takes over.
        assert!(matches!(m.status(&ag, "i0", 1051), Some(ItemStatus::Claimed { ref by, .. }) if *by == a.id));
        assert_eq!(m.status(&ag, "i0", 1101), Some(ItemStatus::Open));
    }

    #[test]
    fn items_failed_by_enough_distinct_nodes_are_settled_without_being_done() {
        let (p, a, b) = (site(), site(), site());
        let mut log = Log { recs: vec![] };
        let ag = agenda(&mut log, &p, 2, Requirements::default());
        let c0 = log.emit(&a, KIND_CLAIM, &ev(&ag, "i0", ClaimAction::Claim, 100, 60, None, None));
        log.emit(&a, KIND_CLAIM, &ev(&ag, "i0", ClaimAction::Release, 101, 0, Some(&c0), None));
        let c0b = log.emit(&a, KIND_CLAIM, &ev(&ag, "i0", ClaimAction::Claim, 102, 60, None, None));
        log.emit(&a, KIND_CLAIM, &ev(&ag, "i0", ClaimAction::Release, 103, 0, Some(&c0b), None));
        let m = Market::from_records(&log.recs);
        assert_eq!(m.failures(&ag, "i0"), 1, "the same node failing twice is one failure");
        let c1 = log.emit(&b, KIND_CLAIM, &ev(&ag, "i0", ClaimAction::Claim, 110, 60, None, None));
        log.emit(&b, KIND_CLAIM, &ev(&ag, "i0", ClaimAction::Release, 111, 0, Some(&c1), None));
        let m = Market::from_records(&log.recs);
        assert_eq!(m.failures(&ag, "i0"), 2);
        // i1 is untouched, so the agenda is not settled yet
        assert!(!m.all_settled(&ag, 200, 2));
        let c = log.emit(&a, KIND_CLAIM, &ev(&ag, "i1", ClaimAction::Claim, 120, 60, None, None));
        log.emit(&a, KIND_CLAIM, &ev(&ag, "i1", ClaimAction::Complete, 121, 0, Some(&c), Some(&"f".repeat(64))));
        let m = Market::from_records(&log.recs);
        assert!(m.all_settled(&ag, 200, 2), "i0 given up on, i1 done");
        assert!(!m.all_done(&ag, 200), "but i0 was never done");
        assert!(!m.all_settled(&ag, 200, 3), "with a higher threshold i0 is still being retried");
    }

    #[test]
    fn events_may_arrive_before_their_agenda_or_claim() {
        let (p, a) = (site(), site());
        let mut log = Log { recs: vec![] };
        let ag = agenda(&mut log, &p, 1, Requirements::default());
        let c = log.emit(&a, KIND_CLAIM, &ev(&ag, "i0", ClaimAction::Claim, 100, 60, None, None));
        log.emit(&a, KIND_CLAIM, &ev(&ag, "i0", ClaimAction::Complete, 110, 0, Some(&c), Some(&"f".repeat(64))));
        let mut rev = log.recs.clone();
        rev.reverse(); // complete, claim, agenda
        let m = Market::from_records(&rev);
        assert!(m.all_done(&ag, 200));
    }

    #[test]
    fn only_the_claimant_can_heartbeat_complete_or_release() {
        let (p, a, evil) = (site(), site(), site());
        let mut log = Log { recs: vec![] };
        let ag = agenda(&mut log, &p, 1, Requirements::default());
        let c = log.emit(&a, KIND_CLAIM, &ev(&ag, "i0", ClaimAction::Claim, 100, 30, None, None));
        log.emit(&evil, KIND_CLAIM, &ev(&ag, "i0", ClaimAction::Release, 105, 0, Some(&c), None));
        log.emit(&evil, KIND_CLAIM, &ev(&ag, "i0", ClaimAction::Complete, 106, 0, Some(&c), Some(&"1".repeat(64))));
        let m = Market::from_records(&log.recs);
        assert!(matches!(m.status(&ag, "i0", 110), Some(ItemStatus::Claimed { ref by, .. }) if *by == a.id));
    }

    #[test]
    fn agenda_from_non_committee_member_posts_nothing() {
        let (p, outsider) = (site(), site());
        let mut log = Log { recs: vec![] };
        log.emit(&outsider, KIND_AGENDA, &Agenda {
            program: "a".repeat(64), generation: 1, evidence_root: "b".repeat(64),
            committee: vec![p.id.clone()], items: vec![AgendaItem { id: "x".into(), description: "d".into(), requirements: Requirements::default() }],
        });
        assert!(Market::from_records(&log.recs).claimable(&CapabilityAd::default(), 0).is_empty());
    }

    #[test]
    fn requirements_gate_claims() {
        let (p, cpu, gpu) = (site(), site(), site());
        let mut log = Log { recs: vec![] };
        let req = Requirements { gpu_class: Some("A100".into()), data_access: Some("hospital".into()), gpu_seconds: 100 };
        let ag = agenda(&mut log, &p, 1, req);
        let m = Market::from_records(&log.recs);
        let cpu_ad = CapabilityAd { node_id: cpu.id.clone(), ..Default::default() };
        let gpu_ad = CapabilityAd { node_id: gpu.id.clone(), gpu_class: Some("A100".into()), data_access: vec!["hospital".into()], gpu_seconds: 500 };
        assert!(matches!(m.can_claim(&cpu_ad, &ag, "i0", 0), Err(MarketError::Requirements(_))));
        assert_eq!(m.can_claim(&gpu_ad, &ag, "i0", 0), Ok(()));
        assert_eq!(m.claimable(&gpu_ad, 0).len(), 1);
        assert!(m.claimable(&cpu_ad, 0).is_empty());
        assert_eq!(m.can_claim(&gpu_ad, &ag, "nope", 0), Err(MarketError::UnknownItem));
    }

    /// A generation over three sites completes although one node dies mid-generation.
    #[test]
    fn generation_completes_when_one_node_is_killed_mid_generation() {
        let p = site();
        let sites = [site(), site(), site()];
        let mut log = Log { recs: vec![] };
        let ag = agenda(&mut log, &p, 6, Requirements::default());
        let ad = |s: &Site| CapabilityAd { node_id: s.id.clone(), gpu_seconds: 1000, ..Default::default() };
        let lease = 30i64;
        let mut now = 1000i64;
        // live claim of each site: item -> claim hash
        let mut held: [Option<(String, String, i64)>; 3] = [None, None, None]; // (item, claim, started)
        let dead_after = 1010i64; // site 1 is killed at t=1010
        let mut guard = 0;
        while !Market::from_records(&log.recs).all_done(&ag, now) {
            guard += 1;
            assert!(guard < 500, "generation did not complete");
            for (k, s) in sites.iter().enumerate() {
                if k == 1 && now >= dead_after { continue; } // dead: no claims, no heartbeats
                let m = Market::from_records(&log.recs);
                match held[k].clone() {
                    None => {
                        if let Some((a, i)) = m.claimable(&ad(s), now).into_iter().next() {
                            let c = log.emit(s, KIND_CLAIM, &ev(&a, &i, ClaimAction::Claim, now, lease as u64, None, None));
                            held[k] = Some((i, c, now));
                        }
                    }
                    Some((i, c, started)) => {
                        if now - started >= 12 {
                            log.emit(s, KIND_CLAIM, &ev(&ag, &i, ClaimAction::Complete, now, 0, Some(&c), Some(&format!("{:064x}", now))));
                            held[k] = None;
                        } else {
                            log.emit(s, KIND_CLAIM, &ev(&ag, &i, ClaimAction::Heartbeat, now, 0, Some(&c), None));
                        }
                    }
                }
            }
            now += 5;
        }
        let m = Market::from_records(&log.recs);
        for i in 0..6 {
            assert!(matches!(m.status(&ag, &format!("i{i}"), now), Some(ItemStatus::Done { .. })), "item {i}");
        }
        // The killed site held a claim when it died and never finished it; every
        // completion after its death came from the survivors.
        let dead = &sites[1].id;
        for i in 0..6 {
            if let Some(ItemStatus::Done { by, .. }) = m.status(&ag, &format!("i{i}"), now) {
                let done_at = log.recs.iter().filter_map(|r| match &r.kind {
                    RecordKind::Provenance { body, .. } => serde_json::from_value::<ClaimEvent>(body.clone()).ok()
                        .filter(|e| e.action == ClaimAction::Complete && e.item_id == format!("i{i}")).map(|e| (r.node_id.clone(), e.at)),
                    _ => None,
                }).filter(|(n, _)| *n == by).map(|(_, at)| at).next().unwrap();
                assert!(by != *dead || done_at < dead_after, "dead node completed item {i} after dying");
            }
        }
        assert!(now > dead_after + lease, "the test must actually run past the dead node's lease expiry");
    }
}
