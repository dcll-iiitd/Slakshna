//! Artifact transfer over iroh-blobs (A2): candidate diffs and other artifacts
//! move by BLAKE3 hash, the same hash the provenance records carry, so a node
//! that sees `research.candidate{diff_hash}` can fetch exactly those bytes from
//! any peer that has them, and the transfer is verified end to end by the
//! protocol (a peer cannot serve different bytes under that hash).

use crate::agents::research::{blake3_hex, ArtifactStore};
use async_trait::async_trait;
use iroh::{Endpoint, EndpointId};
use iroh_blobs::api::downloader::Downloader;
use iroh_blobs::api::Store;
use iroh_blobs::Hash;
use std::sync::Arc;
use std::time::Duration;

/// Largest artifact a node will fetch on behalf of the research loop.
pub const MAX_ARTIFACT_BYTES: u64 = 256 * 1024 * 1024;

pub struct IrohArtifacts {
    store: Store,
    downloader: Downloader,
    providers: Arc<dyn Fn() -> Vec<EndpointId> + Send + Sync>,
    timeout: Duration,
}

impl IrohArtifacts {
    /// `providers` is asked for candidate sources at fetch time (e.g. the gossip neighbours).
    pub fn new(store: Store, endpoint: &Endpoint, providers: Arc<dyn Fn() -> Vec<EndpointId> + Send + Sync>) -> Self {
        let downloader = store.downloader(endpoint);
        IrohArtifacts { store, downloader, providers, timeout: Duration::from_secs(60) }
    }

    #[allow(dead_code)] // library surface: used by tests and integrators, or pending the settlement-chain decision
    pub fn with_timeout(mut self, t: Duration) -> Self {
        self.timeout = t;
        self
    }

    #[allow(dead_code)] // library surface: used by tests and integrators, or pending the settlement-chain decision
    pub fn store(&self) -> &Store {
        &self.store
    }

    /// Parse a 64-char hex BLAKE3 hash. Validated by hand first: `Hash::from_str`
    /// panics on strings of the wrong length, and these come from remote records.
    fn parse(hash: &str) -> Option<Hash> {
        if !crate::agents::records::is_hash(hash) {
            return None;
        }
        let b: [u8; 32] = hex::decode(hash).ok()?.try_into().ok()?;
        Some(Hash::from_bytes(b))
    }
}

#[async_trait]
impl ArtifactStore for IrohArtifacts {
    async fn put(&self, bytes: &[u8]) -> Result<String, String> {
        let tag = self.store.blobs().add_bytes(bytes.to_vec()).await.map_err(|e| e.to_string())?;
        let h = tag.hash.to_hex();
        debug_assert_eq!(h, blake3_hex(bytes));
        Ok(h)
    }

    async fn get(&self, hash: &str) -> Option<Vec<u8>> {
        let h = Self::parse(hash)?;
        if !self.store.blobs().has(h).await.unwrap_or(false) {
            let providers = (self.providers)();
            if providers.is_empty() {
                return None;
            }
            let dl = self.downloader.download(h, providers);
            tokio::time::timeout(self.timeout, dl).await.ok()?.ok()?;
        }
        let bytes = self.store.blobs().get_bytes(h).await.ok()?;
        (bytes.len() as u64 <= MAX_ARTIFACT_BYTES).then(|| bytes.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use iroh::endpoint::presets;
    use iroh::protocol::Router;
    use iroh::RelayMode;
    use iroh_blobs::store::mem::MemStore;
    use iroh_blobs::BlobsProtocol;

    async fn node() -> (Router, Endpoint, MemStore) {
        let ep = Endpoint::builder(presets::Minimal).relay_mode(RelayMode::Disabled).bind().await.unwrap();
        let store = MemStore::new();
        let router = Router::builder(ep.clone()).accept(iroh_blobs::ALPN, BlobsProtocol::new(&store, None)).spawn();
        (router, ep, store)
    }

    #[tokio::test]
    async fn artifact_put_on_one_node_is_fetched_by_hash_on_another() {
        use iroh::address_lookup::MemoryLookup;
        let (ra, ea, sa) = node().await;
        let lookup = MemoryLookup::new();
        lookup.add_endpoint_info(ea.addr());
        let eb = Endpoint::builder(presets::Minimal).relay_mode(RelayMode::Disabled).address_lookup(lookup).bind().await.unwrap();
        let sb = MemStore::new();

        let a = IrohArtifacts::new((*sa).clone(), &ea, Arc::new(Vec::new));
        let peer = ea.id();
        let b = IrohArtifacts::new((*sb).clone(), &eb, Arc::new(move || vec![peer])).with_timeout(Duration::from_secs(20));

        let payload = b"candidate diff bytes".repeat(1000);
        let hash = a.put(&payload).await.unwrap();
        assert_eq!(hash, blake3_hex(&payload), "record hash == blob hash");
        assert!(!sb.blobs().has(IrohArtifacts::parse(&hash).unwrap()).await.unwrap(), "B starts without it");
        assert_eq!(b.get(&hash).await.unwrap(), payload);
        assert!(sb.blobs().has(IrohArtifacts::parse(&hash).unwrap()).await.unwrap(), "B now holds and can serve it");
        // unknown hash / garbage / nobody to ask
        assert!(b.get("not-a-hash").await.is_none());
        let lonely = IrohArtifacts::new((*MemStore::new()).clone(), &eb, Arc::new(Vec::new));
        assert!(lonely.get(&hash).await.is_none());
        ra.shutdown().await.unwrap();
    }
}
