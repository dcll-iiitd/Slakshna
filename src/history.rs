use crate::config::Config;
use crate::state::State;
use crate::identity::{ self, Keypair, NodeId };
use serde::{ Deserialize, Serialize };
use sha2::{ Sha256, Digest };
use std::sync::Arc;
use tokio::sync::RwLock;


pub type BoxError = Box<dyn std::error::Error + Send + Sync>;

// How many of the most-trusted peers make up the aggregation cohort for an epoch.
pub const COHORT_SIZE: usize = 4;

// --- LOCAL UPDATE HISTORY: ONE APPEND-ONLY LOG PER PEER ---

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecordKind {
    /// A sparsified, compressed model delta produced by one local training round.
    ModelUpdate {
        delta_hash: String,
        compressed_delta: String,
    },
    /// One node's trust assessment of another node's model update.
    PeerReview {
        target_node: String,
        update_hash: String,
        loss_drop: f64,
        trust_score: f64,
    },
    /// A typed provenance record for the agent layers (Steps 4-5): research
    /// loop objects, agent key delegations, A2A cards/tasks, policy audits.
    /// `kind` is the dotted kind name (e.g. `research.finding`); `body` is
    /// validated against its schema in `crate::agents::records`.
    Provenance {
        kind: String,
        body: serde_json::Value,
    },
}

fn verified_cache() -> &'static std::sync::Mutex<std::collections::HashSet<[u8; 32]>> {
    static C: std::sync::OnceLock<std::sync::Mutex<std::collections::HashSet<[u8; 32]>>> = std::sync::OnceLock::new();
    C.get_or_init(Default::default)
}

/// Domain separation prefix for record signatures, so a signature over a
/// record hash can never be replayed as a signature over anything else.
const RECORD_SIGNING_DOMAIN: &[u8] = b"slakshna/record/v1\0";

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum RecordError {
    #[error("record hash does not match its contents")]
    BadHash,
    #[error("record is unsigned (no public key or signature)")]
    Unsigned,
    #[error("node id does not match the signing public key")]
    NodeIdMismatch,
    #[error("signature verification failed")]
    BadSignature,
    #[error("invalid provenance body: {0}")]
    BadBody(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateRecord {
    pub node_id: String,
    pub prev_hash: String,
    pub kind: RecordKind,
    pub signature: String,
    pub hash: String,
    /// Hex Ed25519 public key of the author; `node_id` must derive from it.
    #[serde(default)]
    pub public_key: String,
}

impl UpdateRecord {
    pub fn calculate_hash(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(&self.node_id);
        hasher.update(&self.prev_hash);
        match &self.kind {
            RecordKind::ModelUpdate { delta_hash, compressed_delta } => {
                hasher.update(b"model_update");
                hasher.update(delta_hash);
                hasher.update(compressed_delta);
            }
            RecordKind::PeerReview { target_node, update_hash, loss_drop, trust_score } => {
                hasher.update(b"peer_review");
                hasher.update(target_node);
                hasher.update(update_hash);
                hasher.update(loss_drop.to_le_bytes());
                hasher.update(trust_score.to_le_bytes());
            }
            RecordKind::Provenance { kind, body } => {
                hasher.update(b"provenance");
                hasher.update((kind.len() as u64).to_le_bytes());
                hasher.update(kind);
                // serde_json maps are ordered (BTreeMap), so this is canonical.
                hasher.update(serde_json::to_vec(body).unwrap_or_default());
            }
        }
        hex::encode(hasher.finalize())
    }

    fn signing_message(&self) -> Vec<u8> {
        let mut m = RECORD_SIGNING_DOMAIN.to_vec();
        m.extend_from_slice(self.hash.as_bytes());
        m
    }

    /// Fill in the author key, hash and Ed25519 signature. `node_id` must be
    /// the keypair's own id.
    pub fn sign(&mut self, keypair: &Keypair) {
        self.public_key = keypair.public_key_hex();
        self.hash = self.calculate_hash();
        self.signature = keypair.sign_hex(&self.signing_message());
    }

    /// Full validity check: contents hash, author binding, signature, and
    /// (for provenance records) the typed body schema.
    pub fn verify(&self) -> Result<(), RecordError> {
        if self.hash != self.calculate_hash() {
            return Err(RecordError::BadHash);
        }
        if self.public_key.is_empty() || self.signature.is_empty() {
            return Err(RecordError::Unsigned);
        }
        // The signature/body checks depend only on the fields below, so a record that
        // passed once passes again: remember successes (never failures).
        let cache_key: [u8; 32] = {
            let mut h = Sha256::new();
            for part in [&self.hash, &self.signature, &self.public_key, &self.node_id] {
                h.update((part.len() as u64).to_le_bytes());
                h.update(part.as_bytes());
            }
            h.finalize().into()
        };
        if verified_cache().lock().map(|c| c.contains(&cache_key)).unwrap_or(false) {
            return Ok(());
        }
        self.verify_uncached()?;
        if let Ok(mut c) = verified_cache().lock() {
            if c.len() >= 500_000 {
                c.clear();
            }
            c.insert(cache_key);
        }
        Ok(())
    }

    fn verify_uncached(&self) -> Result<(), RecordError> {
        if !identity::node_id_matches_key(&self.node_id, &self.public_key) {
            return Err(RecordError::NodeIdMismatch);
        }
        if !identity::verify_hex(&self.public_key, &self.signing_message(), &self.signature) {
            return Err(RecordError::BadSignature);
        }
        if let RecordKind::Provenance { kind, body } = &self.kind {
            crate::agents::records::validate_body(kind, body).map_err(RecordError::BadBody)?;
        }
        Ok(())
    }
}

/// The node's local store of federated learning work.
///
/// Every node keeps one append-only log per participant: its own model updates
/// and peer reviews, plus everything received over the gossip mesh. The
/// compressed deltas staged here are what get shared with peers and what the
/// ML engine reads back when it aggregates.
pub struct UpdateHistory {
    pub config: Config,
    pub state: Arc<RwLock<State>>,
    pub node_identity: NodeId,
    pub peer_updates: std::collections::HashMap<String, Vec<UpdateRecord>>, // Per-node update log
}

impl UpdateHistory {
    pub async fn new(
        config: Config,
        state: Arc<RwLock<State>>,
        node_identity: NodeId
    ) -> Result<Self, BoxError> {
        Ok(UpdateHistory {
            config,
            state,
            node_identity,
            peer_updates: std::collections::HashMap::new(),
        })
    }

    // ========================================================================
    // FEDERATED LEARNING WORK
    // ========================================================================


    /// Append a record to its author's log. Unsigned, forged or malformed
    /// records are rejected; re-delivered records (same hash) are ignored.
    pub fn record_update(&mut self, record: UpdateRecord) -> Result<(), RecordError> {
        // Re-delivery of a record we already hold, byte for byte (it was verified when stored),
        // is a no-op. A record with the same hash but a different signature is not a
        // re-delivery: it falls through to full verification and is rejected if forged.
        if self.peer_updates.get(&record.node_id).map_or(false, |l| {
            l.iter().any(|r| r.hash == record.hash && r.signature == record.signature && r.public_key == record.public_key)
        }) {
            return Ok(());
        }
        if let Err(e) = record.verify() {
            tracing::warn!("Rejected record from {}: {}", record.node_id, e);
            return Err(e);
        }

        let log = self.peer_updates.entry(record.node_id.clone()).or_insert_with(Vec::new);
        if log.iter().any(|r| r.hash == record.hash) {
            return Ok(()); // valid, but we already hold this record under another valid signature
        }
        log.push(record);
        Ok(())
    }

    /// Author, sign and append a provenance record to this node's own log,
    /// chained to the log's current tail. Returns the record so the caller can
    /// broadcast it. Taking `&mut self` makes tail-read + append atomic.
    pub fn author_provenance(
        &mut self,
        keypair: &Keypair,
        kind: &str,
        body: serde_json::Value,
    ) -> Result<UpdateRecord, RecordError> {
        let node_id = self.node_identity.to_string();
        let prev_hash = self
            .peer_updates
            .get(&node_id)
            .and_then(|l| l.last())
            .map(|r| r.hash.clone())
            .unwrap_or_else(|| "0".repeat(64));
        let mut r = UpdateRecord {
            node_id,
            prev_hash,
            kind: RecordKind::Provenance { kind: kind.to_string(), body },
            signature: String::new(),
            hash: String::new(),
            public_key: String::new(),
        };
        r.sign(keypair);
        self.record_update(r.clone())?;
        Ok(r)
    }

    /// All provenance records of a given kind, across every node's log, in
    /// a deterministic order (author, then position in that author's log).
    pub fn provenance(&self, kind: &str) -> Vec<&UpdateRecord> {
        let mut nodes: Vec<&String> = self.peer_updates.keys().collect();
        nodes.sort();
        let mut out = Vec::new();
        for n in nodes {
            for r in &self.peer_updates[n] {
                if let RecordKind::Provenance { kind: k, .. } = &r.kind {
                    if k == kind {
                        out.push(r);
                    }
                }
            }
        }
        out
    }

    pub fn top_trusted_peers(&self, k: usize) -> Vec<String> {
        self.trust_rankings(k)
            .into_iter()
            .map(|(id, _)| id)
            .collect()
    }

    pub fn trust_rankings(&self, k: usize) -> Vec<(String, f64)> {
        let mut global_trust: std::collections::HashMap<String, f64> =
            std::collections::HashMap::new();

        // Accumulate trust weights from every peer review on record
        for (_, log) in &self.peer_updates {
            for record in log {
                if let RecordKind::PeerReview { target_node, trust_score, .. } = &record.kind {
                    let trust = global_trust.entry(target_node.clone()).or_insert(0.0);
                    *trust += trust_score;
                }
            }
        }

        let mut ranked: Vec<_> = global_trust.into_iter().collect();

        // Sort descending by trust
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then_with(|| a.0.cmp(&b.0)));

        ranked.into_iter().take(k).collect()
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    fn review(kp: &Keypair) -> UpdateRecord {
        let mut r = UpdateRecord {
            node_id: kp.node_id().to_string(),
            prev_hash: "0".repeat(64),
            kind: RecordKind::PeerReview {
                target_node: "peer".into(),
                update_hash: "u".into(),
                loss_drop: 0.1,
                trust_score: 0.7,
            },
            signature: String::new(),
            hash: String::new(),
            public_key: String::new(),
        };
        r.sign(kp);
        r
    }

    async fn history(kp: &Keypair) -> UpdateHistory {
        let dir = std::env::temp_dir().join(format!("slk-hist-{}", rand::random::<u64>()));
        let cfg: Config = toml::from_str(&format!(
            "[federation]\nid=\"t\"\nname=\"t\"\n[node]\nid=\"x\"\ndata_dir=\"{}\"\n[network]\nhost=\"127.0.0.1\"\napi_port=0\nws_port=0\np2p_port=0\n[logging]\nlevel=\"info\"\n",
            dir.display()
        ))
        .unwrap();
        let state = Arc::new(RwLock::new(State::new(&cfg.node.data_dir).unwrap()));
        UpdateHistory::new(cfg, state, kp.node_id()).await.unwrap()
    }

    #[test]
    fn signed_record_verifies() {
        assert_eq!(review(&Keypair::generate()).verify(), Ok(()));
    }

    #[test]
    fn placeholder_signature_strings_are_rejected() {
        // What the node used to emit before real signing.
        let kp = Keypair::generate();
        let mut r = review(&kp);
        r.signature = "node_signature_123".into();
        assert_eq!(r.verify(), Err(RecordError::BadSignature));
        r.public_key.clear();
        r.signature.clear();
        assert_eq!(r.verify(), Err(RecordError::Unsigned));
    }

    #[test]
    fn tampering_with_content_or_author_is_detected() {
        let kp = Keypair::generate();
        let mut r = review(&kp);
        if let RecordKind::PeerReview { trust_score, .. } = &mut r.kind {
            *trust_score = 99.0;
        }
        assert_eq!(r.verify(), Err(RecordError::BadHash));
        // Re-hashing after tampering does not help without the key.
        r.hash = r.calculate_hash();
        assert_eq!(r.verify(), Err(RecordError::BadSignature));

        // Impersonation: someone signs with their own key but claims another id.
        let victim = Keypair::generate();
        let mut forged = review(&Keypair::generate());
        forged.node_id = victim.node_id().to_string();
        forged.sign(&Keypair::generate());
        assert_eq!(forged.verify(), Err(RecordError::NodeIdMismatch));
    }

    #[test]
    fn signature_does_not_transfer_between_records() {
        let kp = Keypair::generate();
        let a = review(&kp);
        let mut b = review(&kp);
        b.prev_hash = "1".repeat(64);
        b.hash = b.calculate_hash();
        b.signature = a.signature.clone();
        assert_eq!(b.verify(), Err(RecordError::BadSignature));
    }

    #[tokio::test]
    async fn history_rejects_invalid_and_dedupes_redelivery() {
        let kp = Keypair::generate();
        let mut h = history(&kp).await;
        let good = review(&kp);
        assert!(h.record_update(good.clone()).is_ok());
        assert!(h.record_update(good.clone()).is_ok());
        assert_eq!(h.peer_updates[&kp.node_id().to_string()].len(), 1);

        let mut bad = review(&kp);
        bad.signature = "00".repeat(64);
        assert_eq!(h.record_update(bad), Err(RecordError::BadSignature));
        assert_eq!(h.peer_updates[&kp.node_id().to_string()].len(), 1);
    }

    #[test]
    fn provenance_body_must_match_schema() {
        let kp = Keypair::generate();
        let mut r = UpdateRecord {
            node_id: kp.node_id().to_string(),
            prev_hash: "0".repeat(64),
            kind: RecordKind::Provenance {
                kind: "research.candidate".into(),
                body: serde_json::json!({"program": "nope"}),
            },
            signature: String::new(),
            hash: String::new(),
            public_key: String::new(),
        };
        r.sign(&kp);
        assert!(matches!(r.verify(), Err(RecordError::BadBody(_))));
    }

    #[test]
    fn model_update_hash_is_unchanged_by_signing_fields() {
        // The wire hash of pre-existing kinds must not change, so old logs stay valid.
        let r = UpdateRecord {
            node_id: "n".into(),
            prev_hash: "p".into(),
            kind: RecordKind::ModelUpdate { delta_hash: "d".into(), compressed_delta: "c".into() },
            signature: String::new(),
            hash: String::new(),
            public_key: "anything".into(),
        };
        let mut hasher = Sha256::new();
        hasher.update("n");
        hasher.update("p");
        hasher.update(b"model_update");
        hasher.update("d");
        hasher.update("c");
        assert_eq!(r.calculate_hash(), hex::encode(hasher.finalize()));
    }
}
