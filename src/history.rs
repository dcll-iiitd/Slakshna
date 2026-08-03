use crate::config::Config;
use crate::state::State;
use crate::address::Address;
use serde::{ Deserialize, Serialize };
use sha2::{ Sha256, Digest };
use std::sync::Arc;
use tokio::sync::RwLock;


pub type BoxError = Box<dyn std::error::Error + Send + Sync>;

// Global constant for committee size
pub const COMMITTEE_SIZE: usize = 4;

// --- LAYER 1: ASYNCHRONOUS MODEL-LATTICE ---

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LatticeBlockType {
    Proposal {
        payload_hash: String,
        compressed_delta: String, 
    },
    Evaluation {
        target_node: String,
        proposal_hash: String,
        loss_drop: f64,
        trust_score: f64,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatticeBlock {
    pub node_id: String,
    pub prev_hash: String,
    pub block_type: LatticeBlockType,
    pub signature: String,
    pub hash: String,
}

impl LatticeBlock {
    pub fn calculate_hash(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(&self.node_id);
        hasher.update(&self.prev_hash);
        match &self.block_type {
            LatticeBlockType::Proposal { payload_hash, compressed_delta } => {
                hasher.update(b"proposal");
                hasher.update(payload_hash);
                hasher.update(compressed_delta);
            }
            LatticeBlockType::Evaluation { target_node, proposal_hash, loss_drop, trust_score } => {
                hasher.update(b"evaluation");
                hasher.update(target_node);
                hasher.update(proposal_hash);
                hasher.update(loss_drop.to_le_bytes());
                hasher.update(trust_score.to_le_bytes());
            }
        }
        hex::encode(hasher.finalize())
    }
}

pub struct Blockchain {
    pub config: Config,
    pub state: Arc<RwLock<State>>,
    pub master_address: Address,
    pub lattice_chains: std::collections::HashMap<String, Vec<LatticeBlock>>, // Personal chains per node
}

impl Blockchain {
    pub async fn new(
        config: Config,
        state: Arc<RwLock<State>>,
        master_address: Address
    ) -> Result<Self, BoxError> {
        Ok(Blockchain {
            config,
            state,
            master_address,
            lattice_chains: std::collections::HashMap::new(),
        })
    }

    // ========================================================================
    // LAYER 1: FEDERATED LEARNING WORK
    // ========================================================================

    
    pub fn add_lattice_block(&mut self, block: LatticeBlock) {
        if block.hash != block.calculate_hash() {
            tracing::warn!("Invalid lattice block hash from {}", block.node_id);
            return;
        }
        
        let chain = self.lattice_chains.entry(block.node_id.clone()).or_insert_with(Vec::new);
        
        // Simple append for now
        chain.push(block);
    }

    pub fn get_elected_committee(&self, k: usize) -> Vec<String> {
        let mut global_reputation: std::collections::HashMap<String, f64> = std::collections::HashMap::new();

        // Accumulate trust weights from all Evaluation blocks
        for (_, chain) in &self.lattice_chains {
            for block in chain {
                if let LatticeBlockType::Evaluation { target_node, trust_score, .. } = &block.block_type {
                    let rep = global_reputation.entry(target_node.clone()).or_insert(0.0);
                    *rep += trust_score;
                }
            }
        }

        let mut eligible_nodes: Vec<_> = global_reputation.into_iter().collect();

        // Sort descending by score
        eligible_nodes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then_with(|| a.0.cmp(&b.0)));

        eligible_nodes.into_iter().take(k).map(|(id, _)| id).collect()
    }

    pub fn get_committee_with_reputation(&self, k: usize) -> Vec<(String, f64)> {
        let mut global_reputation: std::collections::HashMap<String, f64> = std::collections::HashMap::new();

        for (_, chain) in &self.lattice_chains {
            for block in chain {
                if let LatticeBlockType::Evaluation { target_node, trust_score, .. } = &block.block_type {
                    let rep = global_reputation.entry(target_node.clone()).or_insert(0.0);
                    *rep += trust_score;
                }
            }
        }

        let mut eligible_nodes: Vec<_> = global_reputation.into_iter().collect();

        eligible_nodes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then_with(|| a.0.cmp(&b.0)));

        eligible_nodes.into_iter().take(k).collect()
    }

}
