use crate::config::Config;
use crate::state::State;
use crate::identity::NodeId;
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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateRecord {
    pub node_id: String,
    pub epoch: u64,
    pub prev_hash: String,
    pub kind: RecordKind,
    pub signature: String,
    pub hash: String,
}

impl UpdateRecord {
    pub fn calculate_hash(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(&self.node_id);
        hasher.update(self.epoch.to_le_bytes());
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
        }
        hex::encode(hasher.finalize())
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


    pub fn record_update(&mut self, record: UpdateRecord) {
        if record.hash != record.calculate_hash() {
            tracing::warn!("Invalid update record hash from {}", record.node_id);
            return;
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
