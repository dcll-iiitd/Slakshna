//! Federated adapter registry.
//!
//! Every node gossips a small, signed `AdapterMeta` after each aggregation
//! round and keeps the metadata of the whole federation here. No weights travel
//! over gossip: the registry only says *who* holds *which* adapter (by hash).

use crate::identity::{verify_with_public_key, Keypair, NodeId};
use crate::state::State;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, warn};

/// Rounds of metadata kept per node. Older entries are dropped from memory and RocksDB.
const ROUNDS_KEPT_PER_NODE: usize = 3;
const MAX_CONFIG_BYTES: usize = 16 * 1024;
const MAX_TARGET_MODULES: usize = 64;

/// Lightweight, signed advertisement of one node's adapter for one round.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AdapterMeta {
    /// Federation NodeId (bech32) of the node that trained the adapter.
    pub node_id: String,
    /// Iroh EndpointId peers dial to fetch the weights.
    pub endpoint_id: String,
    /// Hex Ed25519 public key; must hash to `node_id` and verify `signature`.
    pub pubkey: String,
    pub round: u64,
    /// SHA-256 (hex) of `adapter_model.safetensors`. The identity of the adapter.
    pub sha256: String,
    /// BLAKE3 (hex) of the same file, the iroh-blobs address used for transfer.
    pub blake3: String,
    pub size: u64,
    pub base_model: String,
    pub rank: u32,
    pub target_modules: Vec<String>,
    /// PEFT `adapter_config.json` contents, so a receiver can rebuild the
    /// adapter directory from the single weights blob.
    pub adapter_config: String,
    /// The announcer's own aggregate trust at announcement time. Informational:
    /// receivers rank with their locally computed trust, not this claim.
    pub trust_score: f64,
    pub timestamp: i64,
    /// Hex Ed25519 signature over `signing_bytes()`.
    #[serde(default)]
    pub signature: String,
}

impl AdapterMeta {
    /// Canonical bytes covered by the signature: this struct with `signature` empty.
    pub fn signing_bytes(&self) -> Vec<u8> {
        let mut unsigned = self.clone();
        unsigned.signature = String::new();
        serde_json::to_vec(&unsigned).unwrap_or_default()
    }

    pub fn sign(&mut self, keypair: &Keypair) {
        self.pubkey = keypair.public_key_hex();
        self.signature = keypair.sign_hex(&self.signing_bytes());
    }

    /// Structural checks plus signature and identity binding.
    pub fn verify(&self) -> Result<(), &'static str> {
        if !is_hex(&self.sha256, 64) {
            return Err("sha256 is not 64 hex chars");
        }
        if !is_hex(&self.blake3, 64) {
            return Err("blake3 is not 64 hex chars");
        }
        if self.size == 0 {
            return Err("empty adapter");
        }
        if self.adapter_config.len() > MAX_CONFIG_BYTES || self.target_modules.len() > MAX_TARGET_MODULES {
            return Err("oversized metadata");
        }
        let pk = hex::decode(&self.pubkey).map_err(|_| "pubkey is not hex")?;
        if NodeId::from_public_key(&pk).0 != self.node_id {
            return Err("pubkey does not match node_id");
        }
        let sig = hex::decode(&self.signature).map_err(|_| "signature is not hex")?;
        if !verify_with_public_key(&pk, &self.signing_bytes(), &sig) {
            return Err("bad signature");
        }
        Ok(())
    }
}

fn is_hex(s: &str, len: usize) -> bool {
    s.len() == len && s.bytes().all(|b| b.is_ascii_hexdigit())
}

#[derive(Default)]
pub struct AdapterRegistry {
    by_node: HashMap<String, BTreeMap<u64, AdapterMeta>>,
    /// (node_id, sha256) pairs whose weights failed hash verification.
    rejected: HashSet<(String, String)>,
}

impl AdapterRegistry {
    /// Rebuild the registry from RocksDB, dropping anything that no longer verifies.
    pub async fn load(state: &Arc<RwLock<State>>) -> Self {
        let raw = state.read().await.load_adapter_metas();
        let mut reg = AdapterRegistry::default();
        for json in raw {
            match serde_json::from_str::<AdapterMeta>(&json) {
                Ok(meta) if meta.verify().is_ok() => {
                    reg.by_node.entry(meta.node_id.clone()).or_default().insert(meta.round, meta);
                }
                _ => debug!("Dropping unreadable persisted adapter metadata"),
            }
        }
        reg
    }

    /// Verify and store an advertisement. Returns true if it was new.
    pub async fn insert(&mut self, meta: AdapterMeta, state: &Arc<RwLock<State>>) -> bool {
        if let Err(why) = meta.verify() {
            warn!("🚫 Rejected adapter metadata from {}: {}", meta.node_id, why);
            return false;
        }
        let rounds = self.by_node.entry(meta.node_id.clone()).or_default();
        if rounds.get(&meta.round) == Some(&meta) {
            return false;
        }
        let json = serde_json::to_string(&meta).unwrap_or_default();
        {
            let mut st = state.write().await;
            if let Err(e) = st.save_adapter_meta(&meta.node_id, meta.round, &json) {
                debug!("Could not persist adapter metadata: {}", e);
            }
        }
        let node_id = meta.node_id.clone();
        rounds.insert(meta.round, meta);
        while rounds.len() > ROUNDS_KEPT_PER_NODE {
            let oldest = *rounds.keys().next().expect("non-empty");
            rounds.remove(&oldest);
            let mut st = state.write().await;
            let _ = st.delete_adapter_meta(&node_id, oldest);
        }
        true
    }

    /// Newest usable advertisement of every node.
    pub fn latest_per_node(&self) -> Vec<&AdapterMeta> {
        self.by_node
            .values()
            .filter_map(|rounds| {
                rounds
                    .values()
                    .rev()
                    .find(|m| !self.rejected.contains(&(m.node_id.clone(), m.sha256.clone())))
            })
            .collect()
    }

    /// All usable advertisements for a given content hash (other holders of the
    /// same weights are valid alternative sources).
    pub fn holders_of(&self, sha256: &str) -> Vec<&AdapterMeta> {
        self.by_node
            .values()
            .flat_map(|r| r.values())
            .filter(|m| m.sha256 == sha256 && !self.rejected.contains(&(m.node_id.clone(), m.sha256.clone())))
            .collect()
    }

    /// Remember that `node_id` served bytes that did not match `sha256`.
    pub fn mark_rejected(&mut self, node_id: &str, sha256: &str) {
        self.rejected.insert((node_id.to_string(), sha256.to_string()));
    }

    pub fn len(&self) -> usize {
        self.by_node.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(crate) fn signed_meta(kp: &Keypair, round: u64) -> AdapterMeta {
        let mut m = AdapterMeta {
            node_id: kp.node_id().0,
            endpoint_id: "ep".into(),
            pubkey: String::new(),
            round,
            sha256: "a".repeat(64),
            blake3: "b".repeat(64),
            size: 10,
            base_model: "base".into(),
            rank: 8,
            target_modules: vec!["q_proj".into()],
            adapter_config: "{}".into(),
            trust_score: 0.5,
            timestamp: 0,
            signature: String::new(),
        };
        m.sign(kp);
        m
    }

    #[test]
    fn valid_meta_verifies() {
        let kp = Keypair::generate();
        assert!(signed_meta(&kp, 1).verify().is_ok());
    }

    #[test]
    fn tampered_meta_is_rejected() {
        let kp = Keypair::generate();
        let mut m = signed_meta(&kp, 1);
        m.sha256 = "c".repeat(64);
        assert_eq!(m.verify(), Err("bad signature"));
    }

    #[test]
    fn impersonation_is_rejected() {
        let victim = Keypair::generate();
        let attacker = Keypair::generate();
        let mut m = signed_meta(&attacker, 1);
        m.node_id = victim.node_id().0;
        m.sign(&attacker); // valid signature, but the key does not own node_id
        assert_eq!(m.verify(), Err("pubkey does not match node_id"));
    }
}
