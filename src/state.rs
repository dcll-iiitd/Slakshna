// use crate::chain::Block;
use crate::address::{ Address, Keypair };
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

    pub fn get_or_create_master_address(&mut self) -> Result<Address, BoxError> {
        if let Some(bytes) = self.db.get(b"meta:keypair")? {
            let key_bytes: [u8; 32] = bytes
                .as_slice()
                .try_into()
                .map_err(|_| BoxError::from("Invalid keypair bytes"))?;
            let keypair = Keypair::from_bytes(&key_bytes)?;
            let address = keypair.address();
            self.keypair = Some(keypair);
            return Ok(address);
        }

        let keypair = Keypair::generate();
        let address = keypair.address();

        self.db.put(b"meta:keypair", keypair.to_bytes())?;
        self.keypair = Some(keypair);

        Ok(address)
    }

    pub fn get_keypair(&self) -> Option<&Keypair> {
        self.keypair.as_ref()
    }

    // ADD TO: src/state.rs inside impl State

    pub fn set_state_root(&mut self, root: &str) -> Result<(), BoxError> {
        self.db.put(b"meta:state_root", root.as_bytes())?;
        Ok(())
    }

    pub fn get_state_root(&self) -> Result<String, BoxError> {
        if let Some(bytes) = self.db.get(b"meta:state_root")? {
            Ok(String::from_utf8(bytes.to_vec())?)
        } else {
            Ok("0".repeat(64)) // Genesis root
        }
    }

    // ==================== LAYER 1 BLOCK OPERATIONS ====================
    

    

    // Height operations (tracking Layer 2 height primarily)
    pub fn set_height(&mut self, height: u64) -> Result<(), BoxError> {
        self.db.put(b"meta:height", height.to_le_bytes())?;
        Ok(())
    }

    pub fn get_height(&self) -> Result<u64, BoxError> {
        if let Some(bytes) = self.db.get(b"meta:height")? {
            Ok(
                u64::from_le_bytes(
                    bytes
                        .as_slice()
                        .try_into()
                        .map_err(|_| BoxError::from("Invalid height bytes"))?
                )
            )
        } else {
            Ok(0)
        }
    }

    // State snapshot for sync
    pub fn get_state_snapshot(&self) -> Result<StateSnapshot, BoxError> {
        let height = self.get_height()?;
        let recent_blocks = Vec::new();

        Ok(StateSnapshot {
            height,
            recent_blocks,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateSnapshot {
    pub height: u64,
    pub recent_blocks: Vec<crate::chain::LatticeBlock>,
}
