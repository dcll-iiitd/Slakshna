use std::sync::Arc;
use std::time::Instant;
use iroh::{Endpoint, RelayMode, SecretKey};
use iroh::endpoint::presets;
use iroh::protocol::Router;
use iroh::address_lookup::MemoryLookup;
use iroh_blobs::store::fs::FsStore;
use iroh_blobs::BlobsProtocol;
use iroh_blobs::ticket::BlobTicket;
use iroh_blobs::BlobFormat;
use iroh_blobs::ALPN as BLOBS_ALPN;

struct EdgeTestNode {
    store: Arc<FsStore>,
    endpoint: Endpoint,
    direct_lookup: MemoryLookup,
    _router: Router,
    work_dir: std::path::PathBuf,
}

impl EdgeTestNode {
    async fn create(name: &str, base_dir: &std::path::Path) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let work_dir = base_dir.join(name);
        tokio::fs::create_dir_all(&work_dir).await?;

        let blobs_dir = work_dir.join("blobs");
        let store = Arc::new(FsStore::load(blobs_dir).await?);
        let blobs = BlobsProtocol::new(store.as_ref(), None);

        let direct_lookup = MemoryLookup::new();
        let secret = SecretKey::generate();
        let endpoint = Endpoint::builder(presets::N0)
            .secret_key(secret)
            .ca_tls_config(iroh::tls::CaTlsConfig::insecure_skip_verify())
            .bind_addr("127.0.0.1:0".parse::<std::net::SocketAddr>()?)?
            .alpns(vec![BLOBS_ALPN.to_vec()])
            .address_lookup(direct_lookup.clone())
            .relay_mode(RelayMode::Disabled)
            .bind()
            .await?;

        let router = Router::builder(endpoint.clone())
            .accept(BLOBS_ALPN, blobs)
            .spawn();

        Ok(Self {
            store,
            endpoint,
            direct_lookup,
            _router: router,
            work_dir,
        })
    }

    async fn stage_file(&self, filename: &str, data: &[u8]) -> Result<(BlobTicket, u64), Box<dyn std::error::Error + Send + Sync>> {
        let file_path = self.work_dir.join(filename);
        tokio::fs::write(&file_path, data).await?;
        let outcome = self.store.add_path(&file_path).await?;
        let ticket = BlobTicket::new(self.endpoint.addr(), outcome.hash, BlobFormat::Raw);
        Ok((ticket, data.len() as u64))
    }
}

#[tokio::test]
async fn test_edgecase_zero_byte_and_boundary_sizes() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let test_dir = std::env::temp_dir().join(format!("slakshna_edge_boundaries_{}", uuid::Uuid::new_v4()));
    tokio::fs::create_dir_all(&test_dir).await?;

    let sender = EdgeTestNode::create("sender", &test_dir).await?;
    let receiver = EdgeTestNode::create("receiver", &test_dir).await?;
    let downloader = receiver.store.downloader(&receiver.endpoint);

    // Test extreme boundary sizes: 0 bytes, 1 byte, 1023 (chunk - 1), 1024 (exact chunk), 1025 (chunk + 1)
    let boundary_sizes = vec![0, 1, 17, 1023, 1024, 1025, 65536, 1048576];

    for &size in &boundary_sizes {
        let data: Vec<u8> = (0..size).map(|i| (i % 251) as u8).collect();
        let fname = format!("boundary_{}_bytes.dat", size);
        let (ticket, _) = sender.stage_file(&fname, &data).await?;

        let (addr, hash, _) = ticket.into_parts();
        receiver.direct_lookup.add_endpoint_info(addr.clone());

        downloader.download(hash, vec![addr.id]).await?;

        let export_path = receiver.work_dir.join(format!("received_{}_bytes.dat", size));
        receiver.store.export(hash, &export_path).await?;

        let downloaded = tokio::fs::read(&export_path).await?;
        assert_eq!(downloaded.len(), size, "Size mismatch for boundary size {}", size);
        assert_eq!(downloaded, data, "Data mismatch for boundary size {}", size);
        println!("✅ Verified boundary payload size: {} bytes", size);
    }

    let _ = tokio::fs::remove_dir_all(&test_dir).await;
    Ok(())
}

#[tokio::test]
async fn test_edgecase_malformed_and_invalid_tickets() {
    let malformed_inputs = vec![
        "",
        "not_a_ticket",
        "blob1234567890",
        "blob:invalidbase32",
        "blob00000000000000000000000000000000",
    ];

    for input in malformed_inputs {
        let parsed = input.parse::<BlobTicket>();
        assert!(parsed.is_err(), "Expected parsing error for invalid ticket: '{}'", input);
    }
    println!("✅ All malformed ticket variations safely rejected without panics");
}

#[tokio::test]
async fn test_edgecase_unreachable_peer_fails_gracefully() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let test_dir = std::env::temp_dir().join(format!("slakshna_edge_unreachable_{}", uuid::Uuid::new_v4()));
    tokio::fs::create_dir_all(&test_dir).await?;

    let receiver = EdgeTestNode::create("receiver", &test_dir).await?;
    let downloader = receiver.store.downloader(&receiver.endpoint);

    // Create a fictitious hash and fictitious endpoint ID that doesn't exist
    let fake_hash = iroh_blobs::Hash::from([42u8; 32]);
    let fake_secret = SecretKey::generate();
    let fake_peer_id = fake_secret.public();

    println!("Attempting download from unreachable peer (verifying graceful timeout/error)...");
    let start = Instant::now();

    // Using a timeout so the test verifies it fails within reasonable time
    let download_result = tokio::time::timeout(
        std::time::Duration::from_secs(3),
        downloader.download(fake_hash, vec![fake_peer_id])
    ).await;

    // It should either return Err or timeout cleanly without panicking
    match download_result {
        Ok(Err(e)) => println!("✅ Download from unreachable peer failed cleanly with error: {:?}", e),
        Err(_) => println!("✅ Download from unreachable peer timed out as expected"),
        Ok(Ok(_)) => panic!("Unreachable peer download should never succeed!"),
    }
    println!("Total duration for unreachable attempt: {:?}", start.elapsed());

    let _ = tokio::fs::remove_dir_all(&test_dir).await;
    Ok(())
}

#[tokio::test]
async fn test_edgecase_concurrent_duplicate_downloads() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let test_dir = std::env::temp_dir().join(format!("slakshna_edge_duplicate_{}", uuid::Uuid::new_v4()));
    tokio::fs::create_dir_all(&test_dir).await?;

    let sender = EdgeTestNode::create("sender", &test_dir).await?;
    let receiver = EdgeTestNode::create("receiver", &test_dir).await?;
    let downloader = receiver.store.downloader(&receiver.endpoint);

    // Stage 15 MB payload
    let payload_size = 15 * 1024 * 1024;
    let data: Vec<u8> = (0..payload_size).map(|i| (i % 256) as u8).collect();
    let (ticket, _) = sender.stage_file("concurrent_dup.pt", &data).await?;

    let (addr, hash, _) = ticket.into_parts();
    receiver.direct_lookup.add_endpoint_info(addr.clone());

    println!("Launching 4 concurrent download requests for the EXACT SAME hash...");
    let mut handles = Vec::new();
    for _i in 0..4 {
        let dl = downloader.clone();
        let peer_id = addr.id;
        handles.push(tokio::spawn(async move {
            dl.download(hash, vec![peer_id]).await
        }));
    }

    for (i, h) in handles.into_iter().enumerate() {
        let res = h.await?;
        assert!(res.is_ok(), "Concurrent duplicate download #{} failed: {:?}", i, res);
    }
    println!("✅ All 4 concurrent duplicate downloads succeeded without contention!");

    // Export and verify data
    let export_path = receiver.work_dir.join("verified_dup.pt");
    receiver.store.export(hash, &export_path).await?;
    let downloaded = tokio::fs::read(&export_path).await?;
    assert_eq!(downloaded, data);
    println!("✅ Downloaded payload integrity 100% verified after concurrent downloads!");

    let _ = tokio::fs::remove_dir_all(&test_dir).await;
    Ok(())
}

#[tokio::test]
async fn test_edgecase_atomic_export_pattern() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let test_dir = std::env::temp_dir().join(format!("slakshna_edge_atomic_{}", uuid::Uuid::new_v4()));
    tokio::fs::create_dir_all(&test_dir).await?;

    let node = EdgeTestNode::create("node", &test_dir).await?;
    let data = b"CRITICAL_MODEL_DELTA_WEIGHTS_PRECISION_TEST";
    let (ticket, _) = node.stage_file("source.pt", data).await?;

    let (_, hash, _) = ticket.into_parts();
    let final_dest = node.work_dir.join("model_delta.pt");
    let temp_dest = node.work_dir.join(format!("model_delta.tmp.{}", uuid::Uuid::new_v4()));

    // Perform atomic export: write to temp file then rename
    node.store.export(hash, &temp_dest).await?;
    assert!(temp_dest.exists());
    assert!(!final_dest.exists());

    tokio::fs::rename(&temp_dest, &final_dest).await?;
    assert!(!temp_dest.exists());
    assert!(final_dest.exists());

    let content = tokio::fs::read(&final_dest).await?;
    assert_eq!(content, data);
    println!("✅ Atomic export pattern verified cleanly!");

    let _ = tokio::fs::remove_dir_all(&test_dir).await;
    Ok(())
}
