use crate::config::Config;
use crate::state::State;
use crate::identity::{verify_with_public_key, Keypair, NodeId};
use serde::{ Deserialize, Serialize };
use sha2::{ Sha256, Digest };
use std::sync::Arc;
use tokio::sync::RwLock;


pub type BoxError = Box<dyn std::error::Error + Send + Sync>;

// How many of the most-trusted peers make up the aggregation cohort for an epoch.
pub const COHORT_SIZE: usize = 4;

/// Per-round decay of a review's influence in `rank_trust`: a review from `n`
/// rounds before the newest counts `TRUST_DECAY^n` as much.
pub const TRUST_DECAY: f64 = 0.9;

/// Domain separator so a record signature can never be replayed as any other
/// kind of signature made with the same key.
const RECORD_SIG_DOMAIN: &[u8] = b"slakshna/record/v1";

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
        /// Federated round this review was made in. Covered by the record hash
        /// (and so by the signature) when non-zero; 0 marks legacy records.
        #[serde(default)]
        round: u64,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateRecord {
    pub node_id: String,
    pub prev_hash: String,
    pub kind: RecordKind,
    /// Hex Ed25519 signature over `RECORD_SIG_DOMAIN || hash`.
    pub signature: String,
    pub hash: String,
    /// Hex Ed25519 public key of the author; must hash to `node_id`. Empty on
    /// legacy (unsigned) records.
    #[serde(default)]
    pub pubkey: String,
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
            RecordKind::PeerReview { target_node, update_hash, loss_drop, trust_score, round } => {
                hasher.update(b"peer_review");
                hasher.update(target_node);
                hasher.update(update_hash);
                hasher.update(loss_drop.to_le_bytes());
                hasher.update(trust_score.to_le_bytes());
                // Only when set, so legacy records (round 0) keep their old hash.
                if *round != 0 {
                    hasher.update(round.to_le_bytes());
                }
            }
        }
        hex::encode(hasher.finalize())
    }

    fn signing_message(&self) -> Vec<u8> {
        let mut msg = RECORD_SIG_DOMAIN.to_vec();
        msg.extend_from_slice(self.hash.as_bytes());
        msg
    }

    /// Recompute the hash, then sign it with this node's key. The hash covers
    /// `node_id`, `prev_hash` and the whole `kind`, so the signature does too.
    pub fn sign(&mut self, keypair: &Keypair) {
        self.hash = self.calculate_hash();
        self.pubkey = keypair.public_key_hex();
        self.signature = keypair.sign_hex(&self.signing_message());
    }

    /// The signature is valid and the key really belongs to `node_id`.
    pub fn verify_signature(&self) -> Result<(), &'static str> {
        let pk = hex::decode(&self.pubkey).map_err(|_| "pubkey is not hex")?;
        if NodeId::from_public_key(&pk).0 != self.node_id {
            return Err("pubkey does not match node_id");
        }
        let sig = hex::decode(&self.signature).map_err(|_| "signature is not hex")?;
        if !verify_with_public_key(&pk, &self.signing_message(), &sig) {
            return Err("bad signature");
        }
        Ok(())
    }
}

/// Decide whether to accept a record received from the network.
///
/// * the hash must match the content;
/// * a record that carries a key must verify, always;
/// * a record with no key is accepted only when `require_signed` is false
///   (mixed federations during a migration).
pub fn check_record(record: &UpdateRecord, require_signed: bool) -> Result<(), &'static str> {
    if record.hash != record.calculate_hash() {
        return Err("hash mismatch");
    }
    if record.pubkey.is_empty() {
        return if require_signed { Err("unsigned record") } else { Ok(()) };
    }
    record.verify_signature()
}

/// Federation-wide trust in each node, from the peer reviews on record.
///
/// * a node's review of itself is ignored (no self-promotion);
/// * each reviewer's reviews of a target are averaged with weight
///   `decay^(rounds old)`, so recent rounds dominate and the score stays on the
///   scale of a single review instead of growing with the number of rounds;
/// * each reviewer then gets one vote, weighted by how recently they reviewed,
///   so a reviewer who stopped reporting fades out and a chatty one cannot
///   outvote the rest.
///
/// Scores are therefore comparable across rounds, on the same scale as the
/// per-round weights nodes report (roughly 0..1).
pub fn rank_trust(
    peer_updates: &std::collections::HashMap<String, Vec<UpdateRecord>>,
    decay: f64,
) -> Vec<(String, f64)> {
    rank_trust_with(peer_updates, &[], decay)
}

/// `rank_trust` plus reviews restored from disk (`extra`), which are kept out of
/// `peer_updates` so a restarted node does not look like it has already heard from everyone.
pub fn rank_trust_with(
    peer_updates: &std::collections::HashMap<String, Vec<UpdateRecord>>,
    extra: &[UpdateRecord],
    decay: f64,
) -> Vec<(String, f64)> {
    use std::collections::HashMap;

    // (reviewer, target) -> [(round, score)]
    let mut pairs: HashMap<(&str, &str), Vec<(u64, f64)>> = HashMap::new();
    let mut newest = 0u64;
    let live = peer_updates.iter().flat_map(|(reviewer, log)| log.iter().map(move |r| (reviewer.as_str(), r)));
    let restored = extra.iter().map(|r| (r.node_id.as_str(), r));
    for (reviewer, record) in live.chain(restored) {
        if let RecordKind::PeerReview { target_node, trust_score, round, .. } = &record.kind {
            if target_node == reviewer || !trust_score.is_finite() {
                continue;
            }
            newest = newest.max(*round);
            pairs.entry((reviewer, target_node.as_str())).or_default().push((*round, *trust_score));
        }
    }

    let age_weight = |round: u64| decay.powi(newest.saturating_sub(round).min(i32::MAX as u64) as i32);

    // target -> [(vote weight, reviewer's decayed average)]
    let mut votes: HashMap<&str, Vec<(f64, f64)>> = HashMap::new();
    for ((_, target), reviews) in &pairs {
        let (num, den) = reviews.iter().fold((0.0, 0.0), |(n, d), (r, s)| {
            let w = age_weight(*r);
            (n + w * s, d + w)
        });
        let avg = if den > 0.0 {
            num / den
        } else {
            reviews.iter().map(|(_, s)| s).sum::<f64>() / reviews.len() as f64
        };
        let latest = reviews.iter().map(|(r, _)| *r).max().unwrap_or(0);
        votes.entry(*target).or_default().push((age_weight(latest), avg));
    }

    let mut ranked: Vec<(String, f64)> = votes
        .into_iter()
        .map(|(target, vs)| {
            let den: f64 = vs.iter().map(|(w, _)| w).sum();
            let score = if den > 0.0 {
                vs.iter().map(|(w, a)| w * a).sum::<f64>() / den
            } else {
                vs.iter().map(|(_, a)| a).sum::<f64>() / vs.len() as f64
            };
            (target.to_string(), score)
        })
        .collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then_with(|| a.0.cmp(&b.0)));
    ranked
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
    /// Peer reviews loaded from disk at startup; used for trust only.
    restored_reviews: Vec<UpdateRecord>,
}

impl UpdateHistory {
    pub async fn new(
        config: Config,
        state: Arc<RwLock<State>>,
        node_identity: NodeId
    ) -> Result<Self, BoxError> {
        // Trust must survive a restart: reload persisted reviews, dropping any that no
        // longer verify (e.g. unsigned ones after `require_signed_records` was turned on).
        let require_signed = config.network.require_signed_records;
        let restored_reviews: Vec<UpdateRecord> = state
            .read()
            .await
            .load_reviews()
            .into_iter()
            .filter_map(|json| serde_json::from_str::<UpdateRecord>(&json).ok())
            .filter(|r| matches!(r.kind, RecordKind::PeerReview { .. }) && check_record(r, require_signed).is_ok())
            .collect();
        if !restored_reviews.is_empty() {
            tracing::info!("Restored {} peer review(s) from disk", restored_reviews.len());
        }
        Ok(UpdateHistory {
            config,
            state,
            node_identity,
            peer_updates: std::collections::HashMap::new(),
            restored_reviews,
        })
    }

    // ========================================================================
    // FEDERATED LEARNING WORK
    // ========================================================================


    pub async fn record_update(&mut self, record: UpdateRecord) {
        if let Err(why) = check_record(&record, self.config.network.require_signed_records) {
            tracing::warn!("Rejected update record from {}: {}", record.node_id, why);
            return;
        }

        // Keep reviews on disk so trust outlives a restart.
        if let RecordKind::PeerReview { target_node, round, .. } = &record.kind {
            if let Ok(json) = serde_json::to_string(&record) {
                let mut st = self.state.write().await;
                if let Err(e) = st.save_review(&record.node_id, *round, target_node, &json) {
                    tracing::debug!("Could not persist peer review: {}", e);
                }
            }
        }

        let log = self.peer_updates.entry(record.node_id.clone()).or_insert_with(Vec::new);

        // Simple append for now
        log.push(record);
    }

    pub fn top_trusted_peers(&self, k: usize) -> Vec<String> {
        self.trust_rankings(k)
            .into_iter()
            .map(|(id, _)| id)
            .collect()
    }

    pub fn trust_rankings(&self, k: usize) -> Vec<(String, f64)> {
        rank_trust_with(&self.peer_updates, &self.restored_reviews, TRUST_DECAY).into_iter().take(k).collect()
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn zero() -> String {
        "0".repeat(64)
    }

    fn review(kp: &Keypair, target: &str, round: u64, score: f64) -> UpdateRecord {
        let mut r = UpdateRecord {
            node_id: kp.node_id().0,
            prev_hash: zero(),
            kind: RecordKind::PeerReview {
                target_node: target.to_string(),
                update_hash: zero(),
                loss_drop: 0.0,
                trust_score: score,
                round,
            },
            signature: String::new(),
            hash: String::new(),
            pubkey: String::new(),
        };
        r.sign(kp);
        r
    }

    fn log(records: Vec<UpdateRecord>) -> HashMap<String, Vec<UpdateRecord>> {
        let mut m: HashMap<String, Vec<UpdateRecord>> = HashMap::new();
        for r in records {
            m.entry(r.node_id.clone()).or_default().push(r);
        }
        m
    }

    #[test]
    fn signed_record_is_accepted() {
        let kp = Keypair::generate();
        let r = review(&kp, "t", 1, 0.5);
        assert!(check_record(&r, true).is_ok());
    }

    #[test]
    fn tampering_after_signing_is_rejected() {
        let kp = Keypair::generate();
        let mut r = review(&kp, "t", 1, 0.5);
        if let RecordKind::PeerReview { trust_score, .. } = &mut r.kind {
            *trust_score = 0.99;
        }
        assert_eq!(check_record(&r, true), Err("hash mismatch"));
        // Recomputing the hash does not help: the signature no longer matches.
        r.hash = r.calculate_hash();
        assert_eq!(check_record(&r, true), Err("bad signature"));
    }

    #[test]
    fn round_is_covered_by_the_signature() {
        let kp = Keypair::generate();
        let mut r = review(&kp, "t", 3, 0.5);
        if let RecordKind::PeerReview { round, .. } = &mut r.kind {
            *round = 99; // a relay trying to make an old review look fresh
        }
        assert_eq!(check_record(&r, true), Err("hash mismatch"));
    }

    #[test]
    fn unsigned_records_are_rejected_only_when_required() {
        let kp = Keypair::generate();
        let mut r = review(&kp, "t", 1, 0.5);
        r.pubkey = String::new();
        r.signature = "review_sig_1".into(); // the old placeholder
        assert_eq!(check_record(&r, true), Err("unsigned record"));
        assert!(check_record(&r, false).is_ok());
    }

    #[test]
    fn a_bad_signature_is_rejected_even_when_unsigned_records_are_allowed() {
        let kp = Keypair::generate();
        let mut r = review(&kp, "t", 1, 0.5);
        r.signature = "00".repeat(64);
        assert_eq!(check_record(&r, false), Err("bad signature"));
    }

    #[test]
    fn forging_another_nodes_review_is_rejected() {
        let victim = Keypair::generate();
        let attacker = Keypair::generate();
        let mut r = review(&attacker, "t", 1, 0.9);
        r.node_id = victim.node_id().0; // claim to be the victim
        r.sign(&attacker); // valid signature, but not the victim's key
        assert_eq!(check_record(&r, true), Err("pubkey does not match node_id"));
    }

    #[test]
    fn legacy_round_zero_keeps_its_old_hash() {
        let kp = Keypair::generate();
        let with_zero = review(&kp, "t", 0, 0.5);
        let mut legacy_input = Sha256::new();
        legacy_input.update(&with_zero.node_id);
        legacy_input.update(&with_zero.prev_hash);
        legacy_input.update(b"peer_review");
        legacy_input.update("t");
        legacy_input.update(zero());
        legacy_input.update(0f64.to_le_bytes());
        legacy_input.update(0.5f64.to_le_bytes());
        assert_eq!(with_zero.hash, hex::encode(legacy_input.finalize()));
    }

    #[test]
    fn self_reviews_do_not_count() {
        let a = Keypair::generate();
        let b = Keypair::generate();
        let (an, bn) = (a.node_id().0, b.node_id().0);
        let m = log(vec![review(&a, &an, 1, 1.0), review(&a, &bn, 1, 0.2)]);
        let ranked = rank_trust(&m, 0.9);
        assert_eq!(ranked.len(), 1); // `a` was only ever reviewed by itself
        assert_eq!(ranked[0].0, bn);
    }

    #[test]
    fn score_is_an_average_not_a_sum_that_grows_with_rounds() {
        let a = Keypair::generate();
        let target = Keypair::generate().node_id().0;
        let one = rank_trust(&log(vec![review(&a, &target, 1, 0.4)]), 0.9);
        let many = rank_trust(&log((1..=50).map(|r| review(&a, &target, r, 0.4)).collect()), 0.9);
        assert!((one[0].1 - 0.4).abs() < 1e-9);
        assert!((many[0].1 - 0.4).abs() < 1e-9);
    }

    #[test]
    fn recent_reviews_outweigh_old_ones() {
        let a = Keypair::generate();
        let target = Keypair::generate().node_id().0;
        // Great long ago, poor lately.
        let mut recs: Vec<UpdateRecord> = (1..=10).map(|r| review(&a, &target, r, 0.9)).collect();
        recs.extend((11..=20).map(|r| review(&a, &target, r, 0.1)));
        let score = rank_trust(&log(recs), 0.9)[0].1;
        assert!(score < 0.5, "recent poor reviews should dominate, got {score}");
    }

    #[test]
    fn reviewers_get_one_vote_each() {
        let (a, b, c) = (Keypair::generate(), Keypair::generate(), Keypair::generate());
        let target = Keypair::generate().node_id().0;
        // `a` reviews 30 times with 0.9; `b` and `c` once each with 0.1.
        let mut recs: Vec<UpdateRecord> = (1..=30).map(|r| review(&a, &target, r, 0.9)).collect();
        recs.push(review(&b, &target, 30, 0.1));
        recs.push(review(&c, &target, 30, 0.1));
        let score = rank_trust(&log(recs), 0.9)[0].1;
        assert!((score - (0.9 + 0.1 + 0.1) / 3.0).abs() < 1e-9, "got {score}");
    }

    #[test]
    fn a_reviewer_who_went_quiet_fades_out() {
        let (old, fresh) = (Keypair::generate(), Keypair::generate());
        let target = Keypair::generate().node_id().0;
        let recs = vec![review(&old, &target, 1, 0.0), review(&fresh, &target, 60, 1.0)];
        let score = rank_trust(&log(recs), 0.9)[0].1;
        assert!(score > 0.99, "got {score}");
    }

    #[test]
    fn legacy_unrounded_reviews_do_not_produce_nan() {
        let a = Keypair::generate();
        let target = Keypair::generate().node_id().0;
        let recs = vec![review(&a, &target, 0, 0.3), review(&a, &target, 5000, 0.5)];
        let s = rank_trust(&log(recs), 0.9)[0].1;
        assert!(s.is_finite() && (0.3..=0.5).contains(&s), "got {s}");
    }

    #[test]
    fn restored_reviews_count_without_being_live_records() {
        let a = Keypair::generate();
        let target = Keypair::generate().node_id().0;
        let restored = vec![review(&a, &target, 4, 0.7)];
        let r = rank_trust_with(&HashMap::new(), &restored, 0.9);
        assert_eq!(r.len(), 1);
        assert!((r[0].1 - 0.7).abs() < 1e-9);
        // and a live review of the same round does not double-count
        let both = rank_trust_with(&log(vec![review(&a, &target, 4, 0.7)]), &restored, 0.9);
        assert!((both[0].1 - 0.7).abs() < 1e-9);
    }

    #[test]
    fn ranking_is_sorted_and_deterministic() {
        let a = Keypair::generate();
        let (x, y) = (Keypair::generate().node_id().0, Keypair::generate().node_id().0);
        let m = log(vec![review(&a, &x, 1, 0.2), review(&a, &y, 1, 0.8)]);
        let ranked = rank_trust(&m, 0.9);
        assert_eq!(ranked[0].0, y);
        assert_eq!(ranked[1].0, x);
    }
}
