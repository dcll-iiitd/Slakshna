use std::sync::Arc;
use iroh::{Endpoint, RelayMode, SecretKey};
use iroh::endpoint::presets;
use iroh::protocol::Router;
use iroh::address_lookup::MemoryLookup;
use iroh_blobs::store::fs::FsStore;
use iroh_blobs::BlobsProtocol;
use iroh_blobs::ticket::BlobTicket;
use iroh_blobs::BlobFormat;
use iroh_blobs::ALPN as BLOBS_ALPN;

#[tokio::test]
async fn test_iroh_blobs_p2p_transfer() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let test_dir = std::env::temp_dir().join(format!("slakshna_test_{}", uuid::Uuid::new_v4()));
    let node1_dir = test_dir.join("node1");
    let node2_dir = test_dir.join("node2");
    tokio::fs::create_dir_all(&node1_dir).await?;
    tokio::fs::create_dir_all(&node2_dir).await?;

    // --- NODE 1 SETUP ---
    let node1_store = Arc::new(FsStore::load(node1_dir.join("blobs")).await?);
    let node1_blobs = BlobsProtocol::new(node1_store.as_ref(), None);

    let node1_direct = MemoryLookup::new();
    let node1_secret = SecretKey::generate();
    let node1_endpoint = Endpoint::builder(presets::N0)
        .secret_key(node1_secret)
        .ca_tls_config(iroh::tls::CaTlsConfig::insecure_skip_verify())
        .bind_addr("127.0.0.1:0".parse::<std::net::SocketAddr>()?)?
        .alpns(vec![BLOBS_ALPN.to_vec()])
        .address_lookup(node1_direct.clone())
        .relay_mode(RelayMode::Disabled)
        .bind()
        .await?;

    let _node1_router = Router::builder(node1_endpoint.clone())
        .accept(BLOBS_ALPN, node1_blobs)
        .spawn();

    // --- NODE 2 SETUP ---
    let node2_store = Arc::new(FsStore::load(node2_dir.join("blobs")).await?);
    let node2_blobs = BlobsProtocol::new(node2_store.as_ref(), None);

    let node2_direct = MemoryLookup::new();
    let node2_secret = SecretKey::generate();
    let node2_endpoint = Endpoint::builder(presets::N0)
        .secret_key(node2_secret)
        .ca_tls_config(iroh::tls::CaTlsConfig::insecure_skip_verify())
        .bind_addr("127.0.0.1:0".parse::<std::net::SocketAddr>()?)?
        .alpns(vec![BLOBS_ALPN.to_vec()])
        .address_lookup(node2_direct.clone())
        .relay_mode(RelayMode::Disabled)
        .bind()
        .await?;

    let node2_downloader = node2_store.downloader(&node2_endpoint);

    let _node2_router = Router::builder(node2_endpoint.clone())
        .accept(BLOBS_ALPN, node2_blobs)
        .spawn();

    // --- CREATE SYNTHETIC MODEL DELTA ON NODE 1 (10 MB) ---
    let payload_size = 10 * 1024 * 1024; // 10 MB payload
    let synthetic_data: Vec<u8> = (0..payload_size).map(|i| (i % 256) as u8).collect();
    let source_file = node1_dir.join("model_delta_epoch_1.pt");
    tokio::fs::write(&source_file, &synthetic_data).await?;

    // Node 1 stages the blob
    let outcome = node1_store.add_path(&source_file).await?;
    let hash = outcome.hash;

    // Node 1 generates a ticket
    let ticket = BlobTicket::new(node1_endpoint.addr(), hash, BlobFormat::Raw);
    let ticket_str = ticket.to_string();

    // --- NODE 2 RECEIVES TICKET AND DOWNLOADS OVER QUIC ---
    let received_ticket: BlobTicket = ticket_str.parse()?;
    let (addr, blob_hash, _format) = received_ticket.into_parts();
    assert_eq!(hash, blob_hash);

    // Register sender address in node 2's direct lookup
    node2_direct.add_endpoint_info(addr.clone());

    // Download blob directly from Node 1 via Iroh Blobs
    node2_downloader.download(blob_hash, vec![addr.id]).await?;

    // Node 2 exports the downloaded blob to disk
    let destination_file = node2_dir.join("exported_delta.pt");
    node2_store.export(blob_hash, &destination_file).await?;

    // Verify downloaded file integrity
    let exported_data = tokio::fs::read(&destination_file).await?;
    assert_eq!(exported_data.len(), payload_size);
    assert_eq!(exported_data, synthetic_data);

    println!("✅ Iroh Blobs 10MB P2P QUIC transfer verified successfully!");
    Ok(())
}
