use crate::identity::{ NodeId, Keypair };
use rocksdb::{ DB, Options };
use serde::{ Deserialize, Serialize };
use std::path::Path;

type BoxError = Box<dyn std::error::Error + Send + Sync>;

pub struct State {
    db: DB,
    keypair: Option<Keypair>,
}

impl State {
    pub fn new(data_dir: &str) -> Result<Self, BoxError> {
        let path = Path::new(data_dir).join("rocksdb");
        std::fs::create_dir_all(&path)?;

        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.set_max_open_files(100);

        let db = DB::open(&opts, path)?;

        Ok(State { db, keypair: None })
    }

    /// Load this node's federation identity, generating one on first start.
    /// The keypair is persisted, so a node keeps the same id across restarts
    /// as long as its data_dir survives.
    pub fn get_or_create_node_identity(&mut self) -> Result<NodeId, BoxError> {
        if let Some(bytes) = self.db.get(b"meta:keypair")? {
            let key_bytes: [u8; 32] = bytes
                .as_slice()
                .try_into()
                .map_err(|_| BoxError::from("Invalid keypair bytes"))?;
            let keypair = Keypair::from_bytes(&key_bytes)?;
            let node_id = keypair.node_id();
            self.keypair = Some(keypair);
            return Ok(node_id);
        }

        let keypair = Keypair::generate();
        let node_id = keypair.node_id();

        self.db.put(b"meta:keypair", keypair.to_bytes())?;
        self.keypair = Some(keypair);

        Ok(node_id)
    }

    pub fn get_keypair(&self) -> Option<&Keypair> {
        self.keypair.as_ref()
    }

    // Round operations (how many federated epochs this node has completed)
    pub fn set_round(&mut self, round: u64) -> Result<(), BoxError> {
        self.db.put(b"meta:round", round.to_le_bytes())?;
        Ok(())
    }

    pub fn get_round(&self) -> Result<u64, BoxError> {
        if let Some(bytes) = self.db.get(b"meta:round")? {
            Ok(
                u64::from_le_bytes(
                    bytes
                        .as_slice()
                        .try_into()
                        .map_err(|_| BoxError::from("Invalid round bytes"))?
                )
            )
        } else {
            Ok(0)
        }
    }

    // History snapshot for sync
    pub fn get_history_snapshot(&self) -> Result<HistorySnapshot, BoxError> {
        let round = self.get_round()?;
        let recent_updates = Vec::new();

        Ok(HistorySnapshot {
            round,
            recent_updates,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistorySnapshot {
    pub round: u64,
    pub recent_updates: Vec<crate::history::UpdateRecord>,
}
