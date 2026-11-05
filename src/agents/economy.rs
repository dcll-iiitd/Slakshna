//! Agent economy and settlement (M5).
//!
//! * Prices come from service announcements.
//! * Principals have spending budgets; escrow reserves from them.
//! * A task's escrow is released on an accepted result, or by an arbiter's
//!   ruling. Sub-contracting splits the escrow along the delegation tree: each
//!   agent keeps its budget minus what it passed down.
//! * Every settlement sums exactly to the escrow: retained shares go to
//!   accepted agents and everything else returns to the payer.

use crate::agents::settlement::{SettlementChain, SettlementError};
use std::collections::{BTreeMap, HashMap};
use std::sync::Mutex;

#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)] // library surface: used by tests and integrators, or pending the settlement-chain decision
pub struct ServiceAnnouncement {
    pub node: String,
    pub skill: String,
    pub price: u64,
    pub announced_at: i64,
    pub ttl_secs: i64,
}

/// Latest unexpired announcement per (node, skill).
#[derive(Debug, Default)]
#[allow(dead_code)] // library surface: used by tests and integrators, or pending the settlement-chain decision
pub struct PriceBook {
    latest: HashMap<(String, String), ServiceAnnouncement>,
}

impl PriceBook {
    #[allow(dead_code)] // library surface: used by tests and integrators, or pending the settlement-chain decision
    pub fn announce(&mut self, a: ServiceAnnouncement) {
        let k = (a.node.clone(), a.skill.clone());
        if self.latest.get(&k).map_or(true, |old| a.announced_at >= old.announced_at) {
            self.latest.insert(k, a);
        }
    }

    #[allow(dead_code)] // library surface: used by tests and integrators, or pending the settlement-chain decision
    pub fn quote(&self, node: &str, skill: &str, now: i64) -> Option<u64> {
        let a = self.latest.get(&(node.to_string(), skill.to_string()))?;
        (now < a.announced_at + a.ttl_secs).then_some(a.price)
    }

    /// Cheapest live offer for a skill; ties broken by node id.
    #[allow(dead_code)] // library surface: used by tests and integrators, or pending the settlement-chain decision
    pub fn cheapest(&self, skill: &str, now: i64) -> Option<(String, u64)> {
        self.latest
            .values()
            .filter(|a| a.skill == skill && now < a.announced_at + a.ttl_secs)
            .map(|a| (a.price, a.node.clone()))
            .min()
            .map(|(p, n)| (n, p))
    }
}

#[derive(Debug, thiserror::Error, PartialEq)]
#[allow(dead_code)] // library surface: used by tests and integrators, or pending the settlement-chain decision
pub enum EconomyError {
    #[error("principal {principal} budget exceeded: {remaining} left, {needed} needed")]
    Budget { principal: String, remaining: u64, needed: u64 },
    #[error("invalid delegation tree: {0}")]
    BadTree(String),
    #[error(transparent)]
    Settlement(#[from] SettlementError),
}

#[derive(Debug, Clone)]
#[allow(dead_code)] // library surface: used by tests and integrators, or pending the settlement-chain decision
pub struct DelegationNode {
    pub agent: String,
    pub parent: Option<usize>,
    /// Escrow amount assigned to this node's whole subtree (its own work and its sub-contracts).
    pub budget: u64,
    /// Did the delegator accept this node's result (a2a.acceptance)?
    pub accepted: bool,
}

/// Validate the tree and compute what each node keeps: budget - sum(children budgets).
#[allow(dead_code)] // library surface: used by tests and integrators, or pending the settlement-chain decision
pub fn retained(tree: &[DelegationNode]) -> Result<Vec<u64>, EconomyError> {
    if tree.is_empty() {
        return Err(EconomyError::BadTree("empty".into()));
    }
    if tree[0].parent.is_some() {
        return Err(EconomyError::BadTree("node 0 must be the root".into()));
    }
    let mut child_sum = vec![0u64; tree.len()];
    for (i, n) in tree.iter().enumerate().skip(1) {
        match n.parent {
            Some(p) if p < i => child_sum[p] += n.budget, // parents precede children: no cycles
            _ => return Err(EconomyError::BadTree(format!("node {i} must have an earlier parent"))),
        }
    }
    tree.iter()
        .enumerate()
        .map(|(i, n)| n.budget.checked_sub(child_sum[i]).ok_or_else(|| EconomyError::BadTree(format!("node {i} passes down more than its budget"))))
        .collect()
}

#[allow(dead_code)] // library surface: used by tests and integrators, or pending the settlement-chain decision
pub struct Economy<'a> {
    ledger: &'a dyn SettlementChain,
    budgets: Mutex<HashMap<String, u64>>,
}

impl<'a> Economy<'a> {
    #[allow(dead_code)] // library surface: used by tests and integrators, or pending the settlement-chain decision
    pub fn new(ledger: &'a dyn SettlementChain) -> Self {
        Economy { ledger, budgets: Mutex::new(HashMap::new()) }
    }

    #[allow(dead_code)] // library surface: used by tests and integrators, or pending the settlement-chain decision
    pub fn set_budget(&self, principal: &str, amount: u64) {
        self.budgets.lock().unwrap().insert(principal.into(), amount);
    }

    #[allow(dead_code)] // library surface: used by tests and integrators, or pending the settlement-chain decision
    pub fn budget(&self, principal: &str) -> u64 {
        self.budgets.lock().unwrap().get(principal).copied().unwrap_or(0)
    }

    /// Reserve from the principal's budget and lock the amount in escrow.
    #[allow(dead_code)] // library surface: used by tests and integrators, or pending the settlement-chain decision
    pub fn open_task(&self, task_id: &str, payer: &str, amount: u64, evidence: &[String]) -> Result<(), EconomyError> {
        {
            let mut b = self.budgets.lock().unwrap();
            let rem = b.get(payer).copied().unwrap_or(0);
            if rem < amount {
                return Err(EconomyError::Budget { principal: payer.into(), remaining: rem, needed: amount });
            }
            b.insert(payer.into(), rem - amount);
        }
        if let Err(e) = self.ledger.escrow(task_id, payer, amount, evidence) {
            *self.budgets.lock().unwrap().entry(payer.into()).or_default() += amount; // roll back the reservation
            return Err(e.into());
        }
        Ok(())
    }

    /// Release along the delegation tree. `ruling_bps` (basis points of each retained
    /// share actually paid) lets an arbiter split a disputed escrow; 10_000 pays in full.
    #[allow(dead_code)] // library surface: used by tests and integrators, or pending the settlement-chain decision
    pub fn settle(&self, task_id: &str, payer: &str, tree: &[DelegationNode], ruling_bps: u64, evidence: &[String]) -> Result<BTreeMap<String, u64>, EconomyError> {
        if ruling_bps > 10_000 {
            return Err(EconomyError::BadTree("ruling above 100%".into()));
        }
        let keep = retained(tree)?;
        let total = tree[0].budget;
        let mut pay: BTreeMap<String, u64> = BTreeMap::new();
        let mut paid = 0u64;
        for (n, k) in tree.iter().zip(&keep) {
            if n.accepted {
                let amt = (*k as u128 * ruling_bps as u128 / 10_000) as u64;
                if amt > 0 {
                    *pay.entry(n.agent.clone()).or_default() += amt;
                    paid += amt;
                }
            }
        }
        let refund = total - paid;
        if refund > 0 {
            *pay.entry(payer.to_string()).or_default() += refund;
            // refunded budget is available to the principal again
            *self.budgets.lock().unwrap().entry(payer.into()).or_default() += refund;
        }
        let splits: Vec<(String, u64)> = pay.iter().map(|(k, v)| (k.clone(), *v)).collect();
        self.ledger.release(task_id, &splits, evidence)?;
        Ok(pay)
    }

    /// Return the whole escrow to the payer (task rejected outright or cancelled).
    #[allow(dead_code)] // library surface: pending the settlement-chain decision
    pub fn cancel(&self, task_id: &str, payer: &str, amount: u64, evidence: &[String]) -> Result<(), EconomyError> {
        self.ledger.refund(task_id, evidence)?;
        *self.budgets.lock().unwrap().entry(payer.into()).or_default() += amount;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agents::settlement::MemoryLedger;

    fn ev() -> Vec<String> { vec!["e".repeat(64)] }

    fn three_level(accepted: [bool; 3]) -> Vec<DelegationNode> {
        vec![
            DelegationNode { agent: "A".into(), parent: None, budget: 100, accepted: accepted[0] },
            DelegationNode { agent: "B".into(), parent: Some(0), budget: 60, accepted: accepted[1] },
            DelegationNode { agent: "C".into(), parent: Some(1), budget: 25, accepted: accepted[2] },
        ]
    }

    fn setup() -> MemoryLedger {
        let l = MemoryLedger::default();
        l.credit("payer", 1000, &ev()).unwrap();
        l
    }

    #[test]
    fn three_level_delegation_settles_exactly() {
        let l = setup();
        let e = Economy::new(&l);
        e.set_budget("payer", 500);
        e.open_task("t1", "payer", 100, &ev()).unwrap();
        assert_eq!(e.budget("payer"), 400);
        let pay = e.settle("t1", "payer", &three_level([true; 3]), 10_000, &ev()).unwrap();
        assert_eq!(pay["A"], 40); // 100 - 60
        assert_eq!(pay["B"], 35); // 60 - 25
        assert_eq!(pay["C"], 25);
        assert!(!pay.contains_key("payer"));
        assert_eq!((l.balance("A"), l.balance("B"), l.balance("C"), l.balance("payer")), (40, 35, 25, 900));
        // value is conserved
        assert_eq!(l.balance("A") + l.balance("B") + l.balance("C") + l.balance("payer"), 1000);
    }

    #[test]
    fn rejected_work_is_refunded_and_others_are_still_paid() {
        let l = setup();
        let e = Economy::new(&l);
        e.set_budget("payer", 500);
        e.open_task("t1", "payer", 100, &ev()).unwrap();
        // B's result was rejected by A; C's (deeper) work was fine; A's own deliverable accepted by payer.
        let pay = e.settle("t1", "payer", &three_level([true, false, true]), 10_000, &ev()).unwrap();
        assert_eq!((pay["A"], pay["C"], pay["payer"]), (40, 25, 35));
        assert!(!pay.contains_key("B"));
        assert_eq!(e.budget("payer"), 400 + 35, "refunded amount is spendable again");
    }

    #[test]
    fn arbitration_ruling_splits_disputed_escrow() {
        let l = setup();
        let e = Economy::new(&l);
        e.set_budget("payer", 500);
        e.open_task("t1", "payer", 100, &ev()).unwrap();
        let pay = e.settle("t1", "payer", &three_level([true; 3]), 5_000, &ev()).unwrap();
        assert_eq!((pay["A"], pay["B"], pay["C"]), (20, 17, 12));
        assert_eq!(pay["payer"], 100 - 20 - 17 - 12);
        assert_eq!(pay.values().sum::<u64>(), 100);
    }

    #[test]
    fn invalid_trees_are_refused_before_any_money_moves() {
        let l = setup();
        let e = Economy::new(&l);
        e.set_budget("payer", 500);
        e.open_task("t1", "payer", 100, &ev()).unwrap();
        let mut t = three_level([true; 3]);
        t[1].budget = 120; // B is passed more than A holds... and A passes down more than it has
        assert!(matches!(e.settle("t1", "payer", &t, 10_000, &ev()), Err(EconomyError::BadTree(_))));
        let mut t = three_level([true; 3]);
        t[2].parent = Some(2); // cycle
        assert!(matches!(e.settle("t1", "payer", &t, 10_000, &ev()), Err(EconomyError::BadTree(_))));
        assert!(matches!(e.settle("t1", "payer", &three_level([true; 3]), 10_001, &ev()), Err(EconomyError::BadTree(_))));
        // escrow untouched: still settleable
        assert!(e.settle("t1", "payer", &three_level([true; 3]), 10_000, &ev()).is_ok());
        // and not settleable twice
        assert!(matches!(e.settle("t1", "payer", &three_level([true; 3]), 10_000, &ev()), Err(EconomyError::Settlement(SettlementError::Settled(_)))));
    }

    #[test]
    fn principal_budgets_are_enforced_and_rolled_back_on_ledger_failure() {
        let l = setup();
        let e = Economy::new(&l);
        e.set_budget("payer", 50);
        assert_eq!(e.open_task("t1", "payer", 60, &ev()), Err(EconomyError::Budget { principal: "payer".into(), remaining: 50, needed: 60 }));
        // ledger refuses (duplicate id): the reservation must be returned
        e.open_task("t2", "payer", 20, &ev()).unwrap();
        assert!(e.open_task("t2", "payer", 20, &ev()).is_err());
        assert_eq!(e.budget("payer"), 30);
    }

    #[test]
    fn cancel_returns_escrow_and_budget() {
        let l = setup();
        let e = Economy::new(&l);
        e.set_budget("payer", 100);
        e.open_task("t", "payer", 70, &ev()).unwrap();
        e.cancel("t", "payer", 70, &ev()).unwrap();
        assert_eq!((l.balance("payer"), e.budget("payer")), (1000, 100));
    }

    #[test]
    fn prices_come_from_live_announcements() {
        let mut p = PriceBook::default();
        let ann = |n: &str, price, at, ttl| ServiceAnnouncement { node: n.into(), skill: "evaluate".into(), price, announced_at: at, ttl_secs: ttl };
        p.announce(ann("a", 50, 100, 60));
        p.announce(ann("b", 30, 100, 60));
        p.announce(ann("c", 10, 100, 5));
        assert_eq!(p.cheapest("evaluate", 103), Some(("c".into(), 10)));
        assert_eq!(p.cheapest("evaluate", 110), Some(("b".into(), 30)), "expired offer is ignored");
        p.announce(ann("b", 80, 120, 60)); // newer announcement replaces
        p.announce(ann("b", 1, 90, 60)); // stale one does not
        assert_eq!(p.quote("b", "evaluate", 130), Some(80));
        assert_eq!(p.cheapest("evaluate", 130), Some(("a".into(), 50)));
        assert_eq!(p.cheapest("nothing", 130), None);
        assert_eq!(p.quote("a", "evaluate", 1000), None);
    }
}
