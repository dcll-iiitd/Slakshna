pub mod mesh;

use crate::history::{UpdateRecord, BoxError};
use crate::state::HistorySnapshot;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// The wire format exchanged between peers over the gossip topic.
///
/// Every participant speaks the same protocol in both directions — there is no
/// client/server or bootstrap/peer split. Renaming a variant or field is a
/// wire-format break: all nodes in a federation must run the same build.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum P2PMessage {
    Hello { node_id: String },
    Welcome { node_id: String, round: u64, peers: Vec<String> },
    GetState,
    HistorySnapshot(HistorySnapshot),
    NewUpdate(UpdateRecord),
    GetUpdate { index: u64 },
    Ping,
    Pong,
}

#[async_trait]
pub trait Network: Send + Sync {
    async fn start(&mut self) -> Result<(), BoxError>;
    async fn broadcast_update(&self, record: &UpdateRecord) -> Result<(), BoxError>;

    async fn get_active_peer_ids(&self) -> Vec<String>;

    fn peer_count(&self) -> usize;
    fn browser_count(&self) -> usize;
}
