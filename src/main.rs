mod config;
mod history;
mod identity;
mod state;
mod network;
mod api;

use crate::config::Config;
use crate::history::UpdateHistory;
use crate::state::State;
use crate::network::Network;
use crate::api::start_api_server;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{ info, Level };
use tracing_subscriber::FmtSubscriber;

type BoxError = Box<dyn std::error::Error + Send + Sync>;

use serde::Deserialize;

/// The JSON contract with `ml_engine.py`: the engine's final stdout line.
#[derive(Deserialize)]
struct MLEngineOutput {
    weights: std::collections::HashMap<String, f64>,
    model_hash: String,
    validation_score: f64,
    metadata: String,
    compressed_delta: String, // The sparsified delta, base64 encoded
}

#[tokio::main]
async fn main() -> Result<(), BoxError> {
    // Parse command line args
    let args: Vec<String> = std::env::args().collect();

    let config_path = if args.len() > 2 && args[1] == "--config" {
        args[2].clone()
    } else {
        "config.toml".to_string()
    };

    // Load config
    let mut config = Config::load(&config_path)?;

    // Setup logging
    let _subscriber = FmtSubscriber::builder()
        .with_max_level(match config.logging.level.as_str() {
            "debug" => Level::DEBUG,
            "info" => Level::INFO,
            "warn" => Level::WARN,
            "error" => Level::ERROR,
            _ => Level::INFO,
        })
        .with_target(false)
        .pretty()
        .init();

    print_banner();

    info!("Loading config from: {}", config_path);
    info!("Federation: {}", config.federation.id);
    info!("Node ID: {}", config.node.id);
    info!("Node Type: {}", config.node.node_type);

    // Initialize state (RocksDB)
    let state = Arc::new(RwLock::new(State::new(&config.node.data_dir)?));

    // Load (or generate) this node's federation identity
    let node_identity = {
        let mut state_guard = state.write().await;
        let id = state_guard.get_or_create_node_identity()?;
        info!("Node Identity: {}", id);
        id
    };

    config.node.id = node_identity.to_string();

    // Initialize the local update history
    let history = Arc::new(
        RwLock::new(UpdateHistory::new(config.clone(), state.clone(), node_identity.clone()).await?)
    );

    // Initialize network
    let network = Arc::new(
        RwLock::new(crate::network::mesh::MeshNetwork::new(config.clone(), history.clone(), state.clone()))
    );

    // Start network
    {
        let mut net = network.write().await;
        net.start().await?;
    }

    // Start API server
    let api_handle = tokio::spawn(
        start_api_server(config.clone(), history.clone(), state.clone(), network.clone())
    );

    let hist = history.clone();
    let net = network.clone();
    let node_id = config.node.id.clone();


    // ==========================================
    // FEDERATED TRAINING LOOP (one pass per epoch)
    // ==========================================
    let hist_loop = hist.clone();
    let net_loop = net.clone();
    let node_id_loop = node_id.clone();
    let gpu_id_loop = config.node.gpu_id;
    let config_loop = config.clone();

    tokio::spawn(async move {
        let epoch_duration = config_loop.training.epoch_duration_secs;
        let sync_deadline = config_loop.training.sync_deadline_secs;
        let expected_peers = config_loop.training.expected_peers;

        loop {
            // 1. EXACT CLOCK SYNCHRONIZATION
            let now = chrono::Utc::now().timestamp() as u64;
            let next_boundary = now + (epoch_duration - (now % epoch_duration));
            let wait_time = next_boundary - now;

            tracing::info!("⏳ Syncing clock... waiting {}s for exact epoch boundary", wait_time);
            tokio::time::sleep(tokio::time::Duration::from_secs(wait_time)).await;

            let epoch_start = next_boundary;
            tracing::info!("🏁 NEW EPOCH STARTED (Global Time: {})", epoch_start);

            // --- PHASE A: LOCAL TRAINING & PEER DELTA EXCHANGE ---
            tracing::info!("🧠 Phase A: Real Local Training & Peer Exchange executing...");

            let peer_deltas_dir = format!("{}/network_deltas", config_loop.node.data_dir);
            std::fs::create_dir_all(&peer_deltas_dir).unwrap_or_default();

            // Stage every peer's most recent model update on disk for the ML engine
            {
                let history = hist_loop.read().await;
                let mut extracted_count = 0u32;
                let mut skipped_count = 0u32;

                // Iterate over every peer's update log so we don't depend on live mesh membership
                for (peer_id, records) in &history.peer_updates {
                    if peer_id == &node_id_loop { continue; }

                    // Find their latest model update
                    let mut found_update = false;
                    for record in records.iter().rev() {
                        if
                            let crate::history::RecordKind::ModelUpdate {
                                compressed_delta,
                                ..
                            } = &record.kind
                        {
                            let path = format!("{}/{}_delta.b64", peer_deltas_dir, peer_id);
                            let _ = std::fs::write(&path, compressed_delta);
                            tracing::info!("📥 Network Delta Extracted: Saved {} bytes from peer {} to {}", compressed_delta.len(), peer_id, path);
                            extracted_count += 1;
                            found_update = true;
                            break;
                        }
                    }
                    if !found_update {
                        tracing::debug!("⏭️ Peer {} has {} records but no model update", peer_id, records.len());
                        skipped_count += 1;
                    }
                }
                tracing::info!("📊 Delta extraction summary: {} extracted, {} peers without updates, data_dir={}", extracted_count, skipped_count, config_loop.node.data_dir);
            }

            let mut python_args = vec!["ml_engine.py".to_string(), node_id_loop.clone()];
            {
                let history = hist_loop.read().await;
                for peer_id in history.peer_updates.keys() {
                    if peer_id != &node_id_loop {
                        python_args.push(peer_id.clone());
                    }
                }
            }

            let mut cmd = tokio::process::Command::new("python");
            cmd.args(&python_args).current_dir(".");

            // CRITICAL: Pass the data_dir to Python so it reads delta files from the correct path
            cmd.env("IIITD_DATA_DIR", &config_loop.node.data_dir);

            // Pin this ML process to the GPU assigned in the node config
            if let Some(gid) = gpu_id_loop {
                cmd.env("CUDA_VISIBLE_DEVICES", gid.to_string());
                tracing::info!("🔥 ML Engine pinned to GPU {}", gid);
            }

            let output = cmd.output().await;
            let my_prev_hash = {
                let history = hist_loop.read().await;
                if let Some(records) = history.peer_updates.get(&node_id_loop) {
                    if let Some(last_record) = records.last() {
                        last_record.hash.clone()
                    } else {
                        "0".repeat(64)
                    }
                } else {
                    "0".repeat(64)
                }
            };

            let mut my_update = crate::history::UpdateRecord {
                node_id: node_id_loop.clone(),
                prev_hash: my_prev_hash,
                kind: crate::history::RecordKind::ModelUpdate {
                    delta_hash: "error_hash".to_string(),
                    compressed_delta: String::new(),
                },
                signature: format!("node_signature_{}", epoch_start),
                hash: String::new(),
            };

            let mut reviews = Vec::new();

            match output {
                Ok(out) if out.status.success() => {
                    let stdout_str = String::from_utf8_lossy(&out.stdout);
                    let json_str = stdout_str.lines().last().unwrap_or("");

                    if let Ok(ml_data) = serde_json::from_str::<MLEngineOutput>(json_str) {
                        tracing::info!(
                            "✅ Local Training Complete! Score: {:.4}",
                            ml_data.validation_score
                        );

                        // Record our own model update
                        my_update.kind = crate::history::RecordKind::ModelUpdate {
                            delta_hash: ml_data.model_hash.clone(),
                            compressed_delta: ml_data.compressed_delta.clone(),
                        };
                        my_update.hash = my_update.calculate_hash();

                        let mut current_tail_hash = my_update.hash.clone();

                        let history = hist_loop.read().await;

                        // Record one peer review per evaluated peer
                        for (peer_id, weight) in ml_data.weights {
                            let mut reviewed_update_hash = None;
                            if let Some(peer_records) = history.peer_updates.get(&peer_id) {
                                for record in peer_records.iter().rev() {
                                    if
                                        let crate::history::RecordKind::ModelUpdate { .. } =
                                            &record.kind
                                    {
                                        reviewed_update_hash = Some(record.hash.clone());
                                        break;
                                    }
                                }
                            }

                            let target_update_hash = match reviewed_update_hash {
                                Some(h) => h,
                                None => {
                                    tracing::warn!(
                                        "Skipping review for {}, no model update on record.",
                                        peer_id
                                    );
                                    continue;
                                }
                            };

                            let mut review = crate::history::UpdateRecord {
                                node_id: node_id_loop.clone(),
                                prev_hash: current_tail_hash.clone(),
                                kind: crate::history::RecordKind::PeerReview {
                                    target_node: peer_id,
                                    update_hash: target_update_hash,
                                    loss_drop: 0.0, // Simplified for now
                                    trust_score: weight,
                                },
                                signature: format!("review_sig_{}", epoch_start),
                                hash: String::new(),
                            };
                            review.hash = review.calculate_hash();
                            current_tail_hash = review.hash.clone();
                            reviews.push(review);
                        }
                        drop(history);
                    }
                }
                Ok(out) => {
                    let stderr_str = String::from_utf8_lossy(&out.stderr);
                    tracing::error!(
                        "❌ Python ML Engine failed (Exit Code {}): {}",
                        out.status.code().unwrap_or(-1),
                        stderr_str
                    );
                }
                Err(e) => tracing::error!("❌ Failed to start Python process: {}", e),
            }

            {
                let mut history = hist_loop.write().await;
                history.record_update(my_update.clone());
                for review in &reviews {
                    history.record_update(review.clone());
                }
            }

            {
                let network_guard = net_loop.read().await;
                let _ = network_guard.broadcast_update(&my_update).await;
                for review in &reviews {
                    let _ = network_guard.broadcast_update(review).await;
                }
            }

            // 2. MID-EPOCH SYNC: Intelligent Sync Barrier
            let sync_target_time = epoch_start + sync_deadline;

            tracing::info!("⏳ Waiting for peer updates to propagate across the federation...");

            loop {
                let current_time = chrono::Utc::now().timestamp() as u64;

                let unique_count = {
                    let history = hist_loop.read().await;
                    history.peer_updates.len()
                };

                if unique_count >= expected_peers {
                    tracing::info!(
                        "✅ Collected updates from all {} participants early!",
                        unique_count
                    );
                    break;
                }

                if current_time >= sync_target_time {
                    tracing::warn!(
                        "⚠️ Hit sync deadline with only {} participants. Proceeding anyway...",
                        unique_count
                    );
                    break;
                }

                tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
            }

            let now_after_wait = chrono::Utc::now().timestamp() as u64;
            if sync_target_time > now_after_wait {
                tokio::time::sleep(
                    tokio::time::Duration::from_secs(sync_target_time - now_after_wait)
                ).await;
            }

            // ==========================================
            // PHASE B: TRUST RANKING FOR THIS EPOCH
            // ==========================================
            let cohort = {
                let history = hist_loop.read().await;
                history.top_trusted_peers(crate::history::COHORT_SIZE) // 👈 Using global constant
            };

            tracing::info!("🏅 Most-trusted cohort this epoch: {:?}", cohort);

            if cohort.contains(&node_id_loop) {
                tracing::info!("⚙️ This node ranks among the most-trusted contributors!");
            } else {
                tracing::info!("💤 This node is outside the top cohort. Waiting for next epoch...");
            }
        }
    });

    // Print status
    info!("==================================================");
    info!("🟢 Node is LIVE (Peer-to-Peer Federated Learning Active)");
    info!("==================================================");
    info!("Transport: Iroh QUIC (with automatic STUN + DERP relay)");
    info!("WS:   ws://{}:{}", config.network.host, config.network.ws_port);
    info!("API:  http://{}:{}", config.network.host, config.network.api_port);
    info!("==================================================");

    // Wait for API server
    api_handle.await??;

    Ok(())
}

fn print_banner() {
    println!(
        r#"
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║                        S L A K S H N A                        ║
║                                                               ║
║         Decentralized Peer-to-Peer Federated Learning         ║
║      Asynchronous local training over an Iroh QUIC mesh       ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
"#
    );
}
