//! Content-addressed adapter transfer over iroh-blobs.
//!
//! All iroh-blobs API usage is confined to this file. Adapter weights are single
//! `adapter_model.safetensors` blobs addressed by BLAKE3; the caller layers the
//! advertised SHA-256 check on top (see `inference::fetch_adapter`).

use crate::history::BoxError;
use iroh::{Endpoint, EndpointId};
use iroh_blobs::{store::fs::FsStore, BlobsProtocol, Hash};
use std::path::Path;
use std::str::FromStr;

pub struct BlobService {
    store: FsStore,
    endpoint: Endpoint,
}

impl BlobService {
    /// Open the on-disk store at `{data_dir}/blobs` and build the protocol
    /// handler to register on the Router under `iroh_blobs::ALPN`.
    pub async fn new(data_dir: &str, endpoint: Endpoint) -> Result<(Self, BlobsProtocol), BoxError> {
        let path = Path::new(data_dir).join("blobs");
        std::fs::create_dir_all(&path)?;
        let store = FsStore::load(&path).await.map_err(|e| format!("blob store: {e}"))?;
        let protocol = BlobsProtocol::new(&store, None);
        Ok((BlobService { store, endpoint }, protocol))
    }

    /// Import a local file and return its BLAKE3 hash (hex). Serving starts
    /// immediately because the store backs the registered protocol handler.
    pub async fn add_file(&self, path: &Path) -> Result<String, BoxError> {
        let tag = self
            .store
            .blobs()
            .add_path(path)
            .await
            .map_err(|e| format!("add blob {}: {e}", path.display()))?;
        Ok(tag.hash.to_string())
    }

    /// Download the blob `blake3_hex` from `peer` (an EndpointId string) and
    /// export it to `dest`. Integrity of the BLAKE3 stream is checked by
    /// iroh-blobs; the caller still verifies the SHA-256.
    pub async fn fetch(&self, blake3_hex: &str, peer: &str, dest: &Path) -> Result<(), BoxError> {
        let hash = Hash::from_str(blake3_hex).map_err(|e| format!("bad blake3 hash: {e}"))?;
        let peer: EndpointId = peer.parse().map_err(|e| format!("bad endpoint id: {e}"))?;

        let downloader = self.store.downloader(&self.endpoint);
        downloader
            .download(hash, vec![peer])
            .await
            .map_err(|e| format!("download from {}: {e}", peer.fmt_short()))?;

        self.store
            .blobs()
            .export(hash, dest)
            .await
            .map_err(|e| format!("export blob: {e}"))?;
        Ok(())
    }
}
