//! Settlement-chain interface (A6, M5).
//!
//! The handoff leaves the real ledger undecided, so everything the agent
//! layers need is behind this trait: credit, escrow, release and slash. Each
//! call carries the provenance record hashes that justify it. `MemoryLedger`
//! is a complete single-process reference implementation used for tests and
//! single-node runs; it is not a consensus ledger.

use std::collections::HashMap;
use std::sync::Mutex;

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
#[allow(dead_code)] // library surface: used by tests and integrators, or pending the settlement-chain decision
pub enum SettlementError {
    #[error("insufficient balance: {account} has {have}, needs {need}")]
    Insufficient { account: String, have: u64, need: u64 },
    #[error("unknown escrow {0}")]
    UnknownEscrow(String),
    #[error("escrow {0} already exists")]
    DuplicateEscrow(String),
    #[error("escrow {0} already settled")]
    Settled(String),
    #[error("split does not match escrow amount ({got} != {want})")]
    BadSplit { got: u64, want: u64 },
    #[error("evidence required")]
    NoEvidence,
}

#[allow(dead_code)] // library surface: used by tests and integrators, or pending the settlement-chain decision
pub trait SettlementChain: Send + Sync {
    fn credit(&self, account: &str, amount: u64, evidence: &[String]) -> Result<(), SettlementError>;
    /// Lock `amount` from `payer` under `escrow_id`.
    fn escrow(&self, escrow_id: &str, payer: &str, amount: u64, evidence: &[String]) -> Result<(), SettlementError>;
    /// Pay out an escrow along `splits` (must sum exactly to the escrowed amount).
    fn release(&self, escrow_id: &str, splits: &[(String, u64)], evidence: &[String]) -> Result<(), SettlementError>;
    /// Return an unreleased escrow to the payer (rejected or arbitrated task).
    fn refund(&self, escrow_id: &str, evidence: &[String]) -> Result<(), SettlementError>;
    /// Burn up to `amount` of `account`'s balance; returns the amount burned.
    fn slash(&self, account: &str, amount: u64, evidence: &[String]) -> Result<u64, SettlementError>;
    fn balance(&self, account: &str) -> u64;
}

#[derive(Default)]
#[allow(dead_code)] // library surface: used by tests and integrators, or pending the settlement-chain decision
struct Ledger {
    balances: HashMap<String, u64>,
    escrows: HashMap<String, (String, u64, bool)>, // payer, amount, settled
}

#[derive(Default)]
#[allow(dead_code)] // library surface: used by tests and integrators, or pending the settlement-chain decision
pub struct MemoryLedger {
    inner: Mutex<Ledger>,
}

#[allow(dead_code)] // library surface: used by tests and integrators, or pending the settlement-chain decision
fn need_evidence(e: &[String]) -> Result<(), SettlementError> {
    if e.is_empty() { Err(SettlementError::NoEvidence) } else { Ok(()) }
}

impl SettlementChain for MemoryLedger {
    fn credit(&self, account: &str, amount: u64, evidence: &[String]) -> Result<(), SettlementError> {
        need_evidence(evidence)?;
        let mut l = self.inner.lock().unwrap();
        *l.balances.entry(account.to_string()).or_default() += amount;
        Ok(())
    }

    fn escrow(&self, id: &str, payer: &str, amount: u64, evidence: &[String]) -> Result<(), SettlementError> {
        need_evidence(evidence)?;
        let mut l = self.inner.lock().unwrap();
        if l.escrows.contains_key(id) {
            return Err(SettlementError::DuplicateEscrow(id.into()));
        }
        let have = l.balances.get(payer).copied().unwrap_or(0);
        if have < amount {
            return Err(SettlementError::Insufficient { account: payer.into(), have, need: amount });
        }
        *l.balances.get_mut(payer).unwrap() -= amount;
        l.escrows.insert(id.into(), (payer.into(), amount, false));
        Ok(())
    }

    fn release(&self, id: &str, splits: &[(String, u64)], evidence: &[String]) -> Result<(), SettlementError> {
        need_evidence(evidence)?;
        let mut l = self.inner.lock().unwrap();
        let (_, amount, settled) = *l.escrows.get(id).ok_or_else(|| SettlementError::UnknownEscrow(id.into()))?;
        if settled {
            return Err(SettlementError::Settled(id.into()));
        }
        let got: u64 = splits.iter().map(|(_, a)| a).sum();
        if got != amount {
            return Err(SettlementError::BadSplit { got, want: amount });
        }
        for (acct, a) in splits {
            *l.balances.entry(acct.clone()).or_default() += a;
        }
        l.escrows.get_mut(id).unwrap().2 = true;
        Ok(())
    }

    fn refund(&self, id: &str, evidence: &[String]) -> Result<(), SettlementError> {
        need_evidence(evidence)?;
        let mut l = self.inner.lock().unwrap();
        let (payer, amount, settled) = l.escrows.get(id).cloned().ok_or_else(|| SettlementError::UnknownEscrow(id.into()))?;
        if settled {
            return Err(SettlementError::Settled(id.into()));
        }
        *l.balances.entry(payer).or_default() += amount;
        l.escrows.get_mut(id).unwrap().2 = true;
        Ok(())
    }

    fn slash(&self, account: &str, amount: u64, evidence: &[String]) -> Result<u64, SettlementError> {
        need_evidence(evidence)?;
        let mut l = self.inner.lock().unwrap();
        let b = l.balances.entry(account.to_string()).or_default();
        let burned = amount.min(*b);
        *b -= burned;
        Ok(burned)
    }

    fn balance(&self, account: &str) -> u64 {
        self.inner.lock().unwrap().balances.get(account).copied().unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ev() -> Vec<String> { vec!["a".repeat(64)] }

    #[test]
    fn escrow_release_conserves_value() {
        let l = MemoryLedger::default();
        l.credit("alice", 100, &ev()).unwrap();
        l.escrow("t1", "alice", 60, &ev()).unwrap();
        assert_eq!(l.balance("alice"), 40);
        l.release("t1", &[("bob".into(), 45), ("carol".into(), 15)], &ev()).unwrap();
        assert_eq!((l.balance("bob"), l.balance("carol"), l.balance("alice")), (45, 15, 40));
        assert_eq!(l.release("t1", &[("bob".into(), 60)], &ev()), Err(SettlementError::Settled("t1".into())));
    }

    #[test]
    fn bad_split_and_overdraft_rejected() {
        let l = MemoryLedger::default();
        l.credit("a", 10, &ev()).unwrap();
        assert!(matches!(l.escrow("t", "a", 11, &ev()), Err(SettlementError::Insufficient { .. })));
        l.escrow("t", "a", 10, &ev()).unwrap();
        assert_eq!(l.release("t", &[("b".into(), 9)], &ev()), Err(SettlementError::BadSplit { got: 9, want: 10 }));
        assert_eq!(l.balance("b"), 0);
    }

    #[test]
    fn refund_and_slash() {
        let l = MemoryLedger::default();
        l.credit("a", 10, &ev()).unwrap();
        l.escrow("t", "a", 10, &ev()).unwrap();
        l.refund("t", &ev()).unwrap();
        assert_eq!(l.balance("a"), 10);
        assert_eq!(l.slash("a", 25, &ev()).unwrap(), 10);
        assert_eq!(l.balance("a"), 0);
    }

    #[test]
    fn every_call_needs_evidence() {
        let l = MemoryLedger::default();
        assert_eq!(l.credit("a", 1, &[]), Err(SettlementError::NoEvidence));
    }
}
