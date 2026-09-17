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
use iroh_blobs::api::downloader::Downloader;
use iroh_blobs::ALPN as BLOBS_ALPN;

struct TestNode {
    #[allow(dead_code)]
    id: usize,
    store: Arc<FsStore>,
    endpoint: Endpoint,
    direct_lookup: MemoryLookup,
    downloader: Downloader,
    _router: Router,
    work_dir: std::path::PathBuf,
}

impl TestNode {
    async fn create(id: usize, base_dir: &std::path::Path) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let work_dir = base_dir.join(format!("node_{}", id));
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

        let downloader = store.downloader(&endpoint);
        let router = Router::builder(endpoint.clone())
            .accept(BLOBS_ALPN, blobs)
            .spawn();

        Ok(Self {
            id,
            store,
            endpoint,
            direct_lookup,
            downloader,
            _router: router,
            work_dir,
        })
    }

    async fn stage_payload(&self, filename: &str, data: &[u8]) -> Result<(BlobTicket, u64), Box<dyn std::error::Error + Send + Sync>> {
        let file_path = self.work_dir.join(filename);
        tokio::fs::write(&file_path, data).await?;
        let outcome = self.store.add_path(&file_path).await?;
        let ticket = BlobTicket::new(self.endpoint.addr(), outcome.hash, BlobFormat::Raw);
        Ok((ticket, data.len() as u64))
    }

    async fn download_and_verify(
        &self,
        ticket: &BlobTicket,
        export_name: &str,
        expected_len: usize,
    ) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
        let (addr, hash, _format) = ticket.clone().into_parts();
        self.direct_lookup.add_endpoint_info(addr.clone());

        self.downloader.download(hash, vec![addr.id]).await?;

        let export_path = self.work_dir.join(export_name);
        self.store.export(hash, &export_path).await?;

        let downloaded_data = tokio::fs::read(&export_path).await?;
        assert_eq!(downloaded_data.len(), expected_len, "Payload size mismatch!");
        Ok(downloaded_data)
    }
}

#[tokio::test]
async fn test_multi_round_large_pretraining_payloads() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let test_dir = std::env::temp_dir().join(format!("slakshna_multiround_{}", uuid::Uuid::new_v4()));
    tokio::fs::create_dir_all(&test_dir).await?;

    let node_a = TestNode::create(1, &test_dir).await?;
    let node_b = TestNode::create(2, &test_dir).await?;

    println!("\n=======================================================");
    println!("🚀 STARTING MULTI-ROUND LARGE PAYLOAD PRETRAINING TEST");
    println!("=======================================================");

    // Test 3 rounds with varying large sizes (20MB, 40MB, 60MB)
    let payload_sizes = vec![20 * 1024 * 1024, 40 * 1024 * 1024, 60 * 1024 * 1024];

    for (round_idx, &size) in payload_sizes.iter().enumerate() {
        let round = round_idx + 1;
        println!("\n--- Round {} (Payload Size: {:.2} MB) ---", round, size as f64 / (1024.0 * 1024.0));

        // Generate synthetic weights pattern
        let seed = (round * 37) as u8;
        let synthetic_weights: Vec<u8> = (0..size).map(|i| (i as u8) ^ seed).collect();

        // Node A stages delta
        let start_stage = Instant::now();
        let (ticket, staged_size) = node_a.stage_payload(&format!("delta_round_{}.pt", round), &synthetic_weights).await?;
        let stage_dur = start_stage.elapsed();
        println!("Node A staged {:.2} MB in {:?}", staged_size as f64 / (1024.0 * 1024.0), stage_dur);

        // Node B downloads delta over P2P QUIC via Iroh Blobs
        let start_transfer = Instant::now();
        let downloaded = node_b.download_and_verify(&ticket, &format!("node_b_received_round_{}.pt", round), size).await?;
        let transfer_dur = start_transfer.elapsed();

        let throughput_mbps = (size as f64 / (1024.0 * 1024.0)) / transfer_dur.as_secs_f64();
        println!(
            "Node B downloaded & verified round {} delta in {:?} (Throughput: {:.2} MB/s)",
            round, transfer_dur, throughput_mbps
        );

        assert_eq!(downloaded, synthetic_weights, "Round {} content verification failed!", round);
        println!("✅ Round {} bit-for-bit integrity verified!", round);
    }

    println!("\n✅ Multi-round large payload pretraining test PASSED successfully!\n");
    let _ = tokio::fs::remove_dir_all(&test_dir).await;
    Ok(())
}

#[tokio::test]
async fn test_mesh_concurrent_large_payload_exchange() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let test_dir = std::env::temp_dir().join(format!("slakshna_mesh_scalability_{}", uuid::Uuid::new_v4()));
    tokio::fs::create_dir_all(&test_dir).await?;

    println!("\n=======================================================");
    println!("🚀 STARTING 3-NODE MESH CONCURRENT PRETRAINING EXCHANGE");
    println!("=======================================================");

    let num_nodes = 3;
    let payload_size = 25 * 1024 * 1024; // 25 MB per node = 75 MB concurrent swarm data
    println!("Nodes in mesh: {}, Payload per node: {:.2} MB (Total: {:.2} MB)",
             num_nodes, payload_size as f64 / (1024.0 * 1024.0), (num_nodes * payload_size) as f64 / (1024.0 * 1024.0));

    let mut nodes = Vec::new();
    for i in 1..=num_nodes {
        nodes.push(Arc::new(TestNode::create(i, &test_dir).await?));
    }

    // Each node generates and stages a distinct large payload
    let mut tickets = Vec::new();
    let mut original_data = Vec::new();

    for (i, node) in nodes.iter().enumerate() {
        let pattern = ((i + 1) * 73) as u8;
        let data: Vec<u8> = (0..payload_size).map(|k| (k as u8).wrapping_add(pattern)).collect();
        let (ticket, _) = node.stage_payload(&format!("model_update_node_{}.pt", i + 1), &data).await?;
        tickets.push(ticket);
        original_data.push(data);
    }

    let start_mesh_sync = Instant::now();

    // Spawn concurrent download tasks: every node downloads deltas from all OTHER nodes simultaneously
    let mut join_handles = Vec::new();

    for (receiver_idx, receiver_node) in nodes.iter().enumerate() {
        for (sender_idx, sender_ticket) in tickets.iter().enumerate() {
            if receiver_idx == sender_idx {
                continue; // Skip self
            }
            let node_clone = Arc::clone(receiver_node);
            let ticket_clone = sender_ticket.clone();
            let expected_bytes = original_data[sender_idx].clone();

            let handle = tokio::spawn(async move {
                let export_name = format!("from_node_{}_by_node_{}.pt", sender_idx + 1, receiver_idx + 1);
                let received = node_clone.download_and_verify(&ticket_clone, &export_name, payload_size).await?;
                assert_eq!(received, expected_bytes);
                Result::<(), Box<dyn std::error::Error + Send + Sync>>::Ok(())
            });
            join_handles.push(handle);
        }
    }

    // Await all concurrent transfers
    for handle in join_handles {
        handle.await??;
    }

    let mesh_dur = start_mesh_sync.elapsed();
    let total_transferred_mb = (num_nodes * (num_nodes - 1) * payload_size) as f64 / (1024.0 * 1024.0);
    let aggregate_throughput = total_transferred_mb / mesh_dur.as_secs_f64();

    println!("\n✅ All {} concurrent P2P transfers completed in {:?}", num_nodes * (num_nodes - 1), mesh_dur);
    println!("Total data transferred across mesh: {:.2} MB", total_transferred_mb);
    println!("Aggregate Mesh Throughput: {:.2} MB/s", aggregate_throughput);
    println!("✅ 3-Node Mesh Concurrent Pretraining Exchange PASSED successfully!\n");

    let _ = tokio::fs::remove_dir_all(&test_dir).await;
    Ok(())
}

#[tokio::test]
async fn test_large_100mb_pretraining_delta_transfer() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let test_dir = std::env::temp_dir().join(format!("slakshna_100mb_{}", uuid::Uuid::new_v4()));
    tokio::fs::create_dir_all(&test_dir).await?;

    let node_a = TestNode::create(1, &test_dir).await?;
    let node_b = TestNode::create(2, &test_dir).await?;

    println!("\n=======================================================");
    println!("🚀 STARTING 100 MB SINGLE DELTA PRETRAINING BENCHMARK");
    println!("=======================================================");

    let payload_size = 100 * 1024 * 1024; // 100 MB
    let mut synthetic_payload = Vec::with_capacity(payload_size);
    for i in 0..payload_size {
        synthetic_payload.push((i % 251) as u8);
    }

    let start_stage = Instant::now();
    let (ticket, staged_len) = node_a.stage_payload("pretraining_delta_100mb.pt", &synthetic_payload).await?;
    let stage_dur = start_stage.elapsed();
    println!("Staged 100 MB delta in {:?} ({:.2} MB/s staging speed)", stage_dur, 100.0 / stage_dur.as_secs_f64());
    assert_eq!(staged_len, payload_size as u64);

    let start_transfer = Instant::now();
    let downloaded = node_b.download_and_verify(&ticket, "received_100mb.pt", payload_size).await?;
    let transfer_dur = start_transfer.elapsed();
    let throughput_mbps = 100.0 / transfer_dur.as_secs_f64();
    println!("Downloaded & verified 100 MB delta in {:?} (Throughput: {:.2} MB/s)", transfer_dur, throughput_mbps);

    assert_eq!(downloaded, synthetic_payload, "100 MB payload corruption detected!");
    println!("✅ 100 MB pretraining delta transfer verified with 100% bit-level integrity!\n");

    let _ = tokio::fs::remove_dir_all(&test_dir).await;
    Ok(())
}

#[tokio::test]
async fn test_massive_4node_swarm_large_payload_exchange() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let test_dir = std::env::temp_dir().join(format!("slakshna_4node_swarm_{}", uuid::Uuid::new_v4()));
    tokio::fs::create_dir_all(&test_dir).await?;

    println!("\n=======================================================");
    println!("🚀 STARTING 4-NODE SWARM LARGE PRETRAINING EXCHANGE");
    println!("=======================================================");

    let num_nodes = 4;
    let payload_size = 50 * 1024 * 1024; // 50 MB per node = 200 MB concurrent swarm data
    println!("Nodes in swarm: {}, Payload per node: {:.2} MB (Total swarm data: {:.2} MB)",
             num_nodes, payload_size as f64 / (1024.0 * 1024.0), (num_nodes * payload_size) as f64 / (1024.0 * 1024.0));

    let mut nodes = Vec::new();
    for i in 1..=num_nodes {
        nodes.push(Arc::new(TestNode::create(i, &test_dir).await?));
    }

    let mut tickets = Vec::new();
    let mut original_data = Vec::new();

    for (i, node) in nodes.iter().enumerate() {
        let pattern = ((i + 1) * 97) as u8;
        let data: Vec<u8> = (0..payload_size).map(|k| (k as u8).wrapping_add(pattern)).collect();
        let (ticket, _) = node.stage_payload(&format!("model_update_node_{}.pt", i + 1), &data).await?;
        tickets.push(ticket);
        original_data.push(data);
    }

    let start_mesh_sync = Instant::now();
    let mut join_handles = Vec::new();

    for (receiver_idx, receiver_node) in nodes.iter().enumerate() {
        for (sender_idx, sender_ticket) in tickets.iter().enumerate() {
            if receiver_idx == sender_idx {
                continue;
            }
            let node_clone = Arc::clone(receiver_node);
            let ticket_clone = sender_ticket.clone();
            let expected_bytes = original_data[sender_idx].clone();

            let handle = tokio::spawn(async move {
                let export_name = format!("from_node_{}_by_node_{}.pt", sender_idx + 1, receiver_idx + 1);
                let received = node_clone.download_and_verify(&ticket_clone, &export_name, payload_size).await?;
                assert_eq!(received, expected_bytes);
                Result::<(), Box<dyn std::error::Error + Send + Sync>>::Ok(())
            });
            join_handles.push(handle);
        }
    }

    for handle in join_handles {
        handle.await??;
    }

    let mesh_dur = start_mesh_sync.elapsed();
    let total_transferred_mb = (num_nodes * (num_nodes - 1) * payload_size) as f64 / (1024.0 * 1024.0);
    let aggregate_throughput = total_transferred_mb / mesh_dur.as_secs_f64();

    println!("\n✅ All {} concurrent P2P transfers completed in {:?}", num_nodes * (num_nodes - 1), mesh_dur);
    println!("Total data transferred across 4-node swarm: {:.2} MB", total_transferred_mb);
    println!("Aggregate Swarm Throughput: {:.2} MB/s", aggregate_throughput);
    println!("✅ 4-Node Swarm Large Pretraining Exchange PASSED successfully!\n");

    let _ = tokio::fs::remove_dir_all(&test_dir).await;
    Ok(())
}
