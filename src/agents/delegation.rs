//! Agent key delegation (A1).
//!
//! A node signs an `agent.key_delegation` provenance record naming an agent
//! public key, a scope and an expiry. The agent then signs its own outputs
//! with that key; anyone holding the node's provenance log can check that the
//! key was delegated, is in scope, has not expired and has not been revoked.

use crate::agents::records::{KeyDelegation, KIND_KEY_DELEGATION};
use crate::history::{RecordKind, UpdateRecord};
use crate::identity::{self, Keypair};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

#[allow(dead_code)] // library surface: used by tests and integrators, or pending the settlement-chain decision
const AGENT_MSG_DOMAIN: &[u8] = b"slakshna/agent-msg/v1\0";

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum DelegationError {
    #[error("no delegation from node {0} for this agent key")]
    Unknown(String),
    #[error("agent key delegation was revoked")]
    Revoked,
    #[error("agent key delegation is not valid at this time")]
    Expired,
    #[error("scope '{0}' not granted to this agent key")]
    OutOfScope(String),
    #[error("agent signature invalid")]
    #[allow(dead_code)] // library surface: used by tests and integrators, or pending the settlement-chain decision
    BadSignature,
}

#[derive(Debug, Clone)]
struct Grant {
    scope: HashSet<String>,
    issued_at: i64,
    expires_at: i64,
}

/// Delegations derived from a set of provenance records. Order independent:
/// revocation is sticky, and multiple grants for one key are merged by taking
/// the union of scopes and the widest validity window.
#[derive(Debug, Default, Clone)]
pub struct DelegationRegistry {
    grants: HashMap<(String, String), Grant>,
    revoked: HashSet<(String, String)>,
}

impl DelegationRegistry {
    pub fn from_records<'a>(records: impl IntoIterator<Item = &'a UpdateRecord>) -> Self {
        let mut reg = DelegationRegistry::default();
        for r in records {
            reg.add(r);
        }
        reg
    }

    /// Add one record (ignored unless it is a verified key delegation).
    pub fn add(&mut self, record: &UpdateRecord) {
        let RecordKind::Provenance { kind, body } = &record.kind else { return };
        if kind != KIND_KEY_DELEGATION || record.verify().is_err() {
            return;
        }
        let Ok(d) = serde_json::from_value::<KeyDelegation>(body.clone()) else { return };
        let key = (record.node_id.clone(), d.agent_public_key.clone());
        if d.revoked {
            self.revoked.insert(key);
            return;
        }
        let g = self.grants.entry(key).or_insert(Grant {
            scope: HashSet::new(),
            issued_at: d.issued_at,
            expires_at: d.expires_at,
        });
        g.scope.extend(d.scope);
        g.issued_at = g.issued_at.min(d.issued_at);
        g.expires_at = g.expires_at.max(d.expires_at);
    }

    pub fn check(&self, node_id: &str, agent_pk: &str, scope: &str, now: i64) -> Result<(), DelegationError> {
        let key = (node_id.to_string(), agent_pk.to_string());
        if self.revoked.contains(&key) {
            return Err(DelegationError::Revoked);
        }
        let g = self.grants.get(&key).ok_or_else(|| DelegationError::Unknown(node_id.to_string()))?;
        if now < g.issued_at || now >= g.expires_at {
            return Err(DelegationError::Expired);
        }
        if !g.scope.contains(scope) {
            return Err(DelegationError::OutOfScope(scope.to_string()));
        }
        Ok(())
    }

    /// Verify a message signed by an agent key against the registry.
    #[allow(dead_code)] // library surface: used by tests and integrators, or pending the settlement-chain decision
    pub fn verify_envelope(&self, env: &AgentEnvelope, now: i64) -> Result<(), DelegationError> {
        self.check(&env.node_id, &env.agent_public_key, &env.scope, now)?;
        if !identity::verify_hex(&env.agent_public_key, &env.signing_bytes(), &env.signature) {
            return Err(DelegationError::BadSignature);
        }
        Ok(())
    }
}

/// An agent's signed statement about a payload (e.g. the hash of a run result).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[allow(dead_code)] // library surface: used by tests and integrators, or pending the settlement-chain decision
pub struct AgentEnvelope {
    pub node_id: String,
    pub agent_public_key: String,
    pub scope: String,
    pub signed_at: i64,
    /// Hex SHA-256 of the payload being attested.
    pub payload_hash: String,
    pub signature: String,
}

impl AgentEnvelope {
    #[allow(dead_code)] // library surface: used by tests and integrators, or pending the settlement-chain decision
    fn signing_bytes(&self) -> Vec<u8> {
        let mut m = AGENT_MSG_DOMAIN.to_vec();
        for part in [
            self.node_id.as_bytes(),
            self.agent_public_key.as_bytes(),
            self.scope.as_bytes(),
            self.signed_at.to_string().as_bytes(),
            self.payload_hash.as_bytes(),
        ] {
            m.extend_from_slice(&(part.len() as u64).to_le_bytes());
            m.extend_from_slice(part);
        }
        m
    }

    #[allow(dead_code)] // library surface: used by tests and integrators, or pending the settlement-chain decision
    pub fn sign(agent: &Keypair, node_id: &str, scope: &str, signed_at: i64, payload_hash: &str) -> Self {
        let mut env = AgentEnvelope {
            node_id: node_id.to_string(),
            agent_public_key: agent.public_key_hex(),
            scope: scope.to_string(),
            signed_at,
            payload_hash: payload_hash.to_string(),
            signature: String::new(),
        };
        env.signature = agent.sign_hex(&env.signing_bytes());
        env
    }
}

/// Build an unsigned `agent.key_delegation` record for the node to sign.
pub fn delegation_record(
    node_id: &str,
    prev_hash: &str,
    agent_pk: &str,
    scope: &[&str],
    issued_at: i64,
    expires_at: i64,
    revoked: bool,
) -> UpdateRecord {
    UpdateRecord {
        node_id: node_id.to_string(),
        prev_hash: prev_hash.to_string(),
        kind: crate::agents::records::provenance(
            KIND_KEY_DELEGATION,
            &KeyDelegation {
                agent_public_key: agent_pk.to_string(),
                scope: scope.iter().map(|s| s.to_string()).collect(),
                issued_at,
                expires_at,
                revoked,
            },
        ),
        signature: String::new(),
        hash: String::new(),
        public_key: String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn signed(node: &Keypair, agent_pk: &str, scope: &[&str], t0: i64, t1: i64, revoked: bool) -> UpdateRecord {
        let mut r = delegation_record(node.node_id().as_str(), &"0".repeat(64), agent_pk, scope, t0, t1, revoked);
        r.sign(node);
        r
    }

    #[test]
    fn delegated_agent_signature_verifies_within_scope_and_window() {
        let node = Keypair::generate();
        let agent = Keypair::generate();
        let rec = signed(&node, &agent.public_key_hex(), &["research.run"], 100, 200, false);
        let reg = DelegationRegistry::from_records([&rec]);
        let nid = node.node_id().to_string();
        let env = AgentEnvelope::sign(&agent, &nid, "research.run", 150, &"ab".repeat(32));
        assert_eq!(reg.verify_envelope(&env, 150), Ok(()));
        // expiry and not-yet-valid
        assert_eq!(reg.verify_envelope(&env, 200), Err(DelegationError::Expired));
        assert_eq!(reg.verify_envelope(&env, 99), Err(DelegationError::Expired));
        // out of scope
        let env2 = AgentEnvelope::sign(&agent, &nid, "a2a.task", 150, &"ab".repeat(32));
        assert_eq!(reg.verify_envelope(&env2, 150), Err(DelegationError::OutOfScope("a2a.task".into())));
    }

    #[test]
    fn tampered_envelope_and_foreign_key_fail() {
        let node = Keypair::generate();
        let agent = Keypair::generate();
        let rec = signed(&node, &agent.public_key_hex(), &["s"], 0, 1000, false);
        let reg = DelegationRegistry::from_records([&rec]);
        let nid = node.node_id().to_string();
        let mut env = AgentEnvelope::sign(&agent, &nid, "s", 5, &"00".repeat(32));
        env.payload_hash = "11".repeat(32);
        assert_eq!(reg.verify_envelope(&env, 5), Err(DelegationError::BadSignature));
        // Another key was never delegated.
        let rogue = Keypair::generate();
        let env = AgentEnvelope::sign(&rogue, &nid, "s", 5, &"00".repeat(32));
        assert!(matches!(reg.verify_envelope(&env, 5), Err(DelegationError::Unknown(_))));
    }

    #[test]
    fn revocation_is_sticky_regardless_of_order() {
        let node = Keypair::generate();
        let agent = Keypair::generate();
        let grant = signed(&node, &agent.public_key_hex(), &["s"], 0, 1000, false);
        let revoke = signed(&node, &agent.public_key_hex(), &["s"], 0, 1000, true);
        let nid = node.node_id().to_string();
        let env = AgentEnvelope::sign(&agent, &nid, "s", 5, &"00".repeat(32));
        for recs in [[&grant, &revoke], [&revoke, &grant]] {
            let reg = DelegationRegistry::from_records(recs);
            assert_eq!(reg.verify_envelope(&env, 5), Err(DelegationError::Revoked));
        }
    }

    #[test]
    fn delegation_signed_by_someone_else_is_ignored() {
        let node = Keypair::generate();
        let agent = Keypair::generate();
        let mut rec = signed(&node, &agent.public_key_hex(), &["s"], 0, 1000, false);
        rec.signature = "00".repeat(64); // forged
        let reg = DelegationRegistry::from_records([&rec]);
        let env = AgentEnvelope::sign(&agent, node.node_id().as_str(), "s", 5, &"00".repeat(32));
        assert!(reg.verify_envelope(&env, 5).is_err());
    }
}
