mod config;
mod history;
mod identity;
mod agents;
mod state;
mod network;
mod api;
mod api_agents;

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
    compressed_delta: String, // Versioned sparse-quantized payload, base64 encoded
    #[serde(default)]
    round: Option<u64>,
    #[serde(default)]
    total_epochs: Option<u64>,
    #[serde(default)]
    is_finished: bool,
}

#[tokio::main]
async fn main() -> Result<(), BoxError> {
    // Parse command line args
    let args: Vec<String> = std::env::args().collect();

    // `iiitd llm-bench --servers name=url|model,... --test gsm8k_test.json --train gsm8k_train.json [options]`:
    // real LLM agents on the GPUs (OpenAI-compatible servers), results with bootstrap confidence intervals.
    if args.get(1).map(String::as_str) == Some("llm-bench") {
        use crate::agents::llm_bench as lb;
        use crate::agents::testbed::Behavior;
        let opt = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned();
        let servers: Vec<lb::ModelServer> = opt("--servers")
            .ok_or("--servers name=url|model,... required")?
            .split(',')
            .map(|s| {
                let (name, rest) = s.split_once('=').ok_or("server must be name=url|model")?;
                let (url, model) = rest.split_once('|').ok_or("server must be name=url|model")?;
                Ok::<_, BoxError>(lb::ModelServer { name: name.into(), url: url.into(), model: model.into() })
            })
            .collect::<Result<_, _>>()?;
        let test = lb::load_gsm8k(&opt("--test").ok_or("--test required")?)?;
        let train = lb::load_gsm8k(&opt("--train").ok_or("--train required")?)?;
        let num = |n: &str, d: usize| opt(n).and_then(|v| v.parse().ok()).unwrap_or(d);
        let cfg = lb::BenchConfig {
            servers,
            workers: num("--workers", 9),
            ks: opt("--ks").map(|v| v.split(',').filter_map(|x| x.parse().ok()).collect()).unwrap_or_else(|| vec![0, 2, 3]),
            adversaries: opt("--adversaries").map(|v| v.split(',').filter_map(|x| match x { "colluding" => Some(Behavior::Colluding), "lying" => Some(Behavior::Lying), "silent" => Some(Behavior::Silent), _ => None }).collect()).unwrap_or_else(|| vec![Behavior::Colluding, Behavior::Lying]),
            problems: num("--problems", 60),
            seeds: (0..num("--seeds", 3) as u64).collect(),
            redundancy: num("--redundancy", 3),
            temperature: opt("--temperature").and_then(|v| v.parse().ok()).unwrap_or(0.7),
            concurrency: num("--concurrency", 6),
            quorum: opt("--quorum").and_then(|v| v.parse().ok()).unwrap_or(0.0),
            calibration: num("--calibration", 8),
        };
        let cells = lb::run(&cfg, &test, &train).await?;
        println!("{}", lb::report(&cells));
        if let Some(out) = opt("--out") {
            std::fs::write(&out, serde_json::to_vec_pretty(&lb::to_json(&cells))?)?;
            eprintln!("raw results written to {out}");
        }
        return Ok(());
    }

    // `iiitd testbed [tasks]`: M7 evaluation matrix over real QUIC, printed as a Markdown table.
    if args.get(1).map(String::as_str) == Some("testbed") {
        let tasks: usize = args.get(2).and_then(|t| t.parse().ok()).unwrap_or(40);
        use crate::agents::testbed::{run_matrix, to_markdown, Behavior};
        let rows = run_matrix(9, &[0, 1, 2, 3], &[Behavior::Colluding, Behavior::Lying, Behavior::Silent], tasks, 3).await?;
        println!("{}", to_markdown(&rows, 9));
        return Ok(());
    }

    let config_path = if args.len() > 2 && args[1] == "--config" {
        args[2].clone()
    } else {
        "config.toml".to_string()
    };

    // Load config
    let mut config = Config::load(&config_path)?;

    use tracing_subscriber::EnvFilter;

    let filter = EnvFilter::new(format!("{},iroh=error", config.logging.level.as_str()));

    // Setup logging
    let _subscriber = FmtSubscriber::builder()
        .with_env_filter(filter)
        .with_target(false)
        .pretty()
        .init();

    print_banner();

    info!("Loading config from: {}", config_path);
    info!("Federation: {}", config.federation.id);
    info!("Node ID: {}", config.node.id);

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

    // Steps 4-5: federated agents and A2A (off unless [agents] enabled = true)
    let agent_service = if config.agents.enabled {
        let keypair = state.read().await.get_keypair().cloned().ok_or("node keypair missing")?;
        match crate::agents::runtime::start(&config, keypair, history.clone(), network.clone()).await {
            Ok(rt) => Some(rt.service),
            Err(e) => {
                tracing::error!("❌ agents failed to start: {}", e);
                return Err(e);
            }
        }
    } else {
        None
    };

    // Start API server
    let api_handle = tokio::spawn(
        start_api_server(config.clone(), history.clone(), state.clone(), network.clone(), agent_service)
    );

    let hist = history.clone();
    let net = network.clone();
    let state_loop = state.clone();
    let node_id = config.node.id.clone();


    // ==========================================
    // FEDERATED TRAINING LOOP (one pass per epoch)
    // ==========================================
    let hist_loop = hist.clone();
    let net_loop = net.clone();
    let node_id_loop = node_id.clone();
    let keypair_loop = state.read().await.get_keypair().cloned().ok_or("node keypair missing")?;
    let gpu_id_loop = config.node.gpu_id;
    let config_loop = config.clone();

    tokio::spawn(async move {
        let epoch_duration = config_loop.training.epoch_duration_secs;
        let sync_deadline = config_loop.training.sync_deadline_secs;
        let expected_peers = config_loop.training.expected_peers;
        let mut completed_epochs = 0u64;

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
            cmd.env("SLAKSHNA_COMPRESSION_ENABLED", config_loop.compression.enabled.to_string());
            cmd.env("SLAKSHNA_DELTA_SPARSITY", config_loop.compression.sparsity.to_string());
            cmd.env("SLAKSHNA_DELTA_QUANTIZATION", &config_loop.compression.quantization);
            cmd.env("SLAKSHNA_ALLOW_LEGACY_DELTA_FORMAT", config_loop.compression.allow_legacy_delta_format.to_string());
            cmd.env("SLAKSHNA_MAX_DELTA_PAYLOAD_BYTES", config_loop.compression.max_payload_bytes.to_string());
            cmd.env("SLAKSHNA_MAX_DELTA_TENSOR_ELEMENTS", config_loop.compression.max_tensor_elements.to_string());

            // Multi-GPU Support
            let num_gpus = config_loop.node.num_gpus.unwrap_or(1);
            cmd.env("SLAKSHNA_NUM_GPUS", num_gpus.to_string());
            
            if let Some(gid) = gpu_id_loop {
                if num_gpus > 1 {
                    let devices: Vec<String> = (gid..gid+num_gpus).map(|id| id.to_string()).collect();
                    let devices_str = devices.join(",");
                    cmd.env("CUDA_VISIBLE_DEVICES", &devices_str);
                    tracing::info!("🔥 Multi-GPU Enabled: ML Engine pinned to GPUs {}", devices_str);
                } else {
                    cmd.env("CUDA_VISIBLE_DEVICES", gid.to_string());
                    tracing::info!("🔥 ML Engine pinned to GPU {}", gid);
                }
            } else if num_gpus > 1 {
                tracing::info!("🔥 Multi-GPU Enabled: Using {} GPUs (CUDA_VISIBLE_DEVICES not constrained)", num_gpus);
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
                signature: String::new(),
                hash: String::new(),
                public_key: String::new(),
            };

            let mut reviews = Vec::new();
            let mut epoch_is_final = false;
            let mut current_round_num = completed_epochs + 1;

            match output {
                Ok(out) if out.status.success() => {
                    let stdout_str = String::from_utf8_lossy(&out.stdout);
                    let json_str = stdout_str.lines().last().unwrap_or("");

                    if let Ok(ml_data) = serde_json::from_str::<MLEngineOutput>(json_str) {
                        tracing::info!(
                            "✅ Local Training Complete! Score: {:.4}",
                            ml_data.validation_score
                        );

                        if let Some(r) = ml_data.round {
                            current_round_num = r;
                        }
                        if ml_data.is_finished {
                            epoch_is_final = true;
                        }

                        // Record our own model update
                        my_update.kind = crate::history::RecordKind::ModelUpdate {
                            delta_hash: ml_data.model_hash.clone(),
                            compressed_delta: ml_data.compressed_delta.clone(),
                        };
                        my_update.sign(&keypair_loop);

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
                                signature: String::new(),
                                hash: String::new(),
                                public_key: String::new(),
                            };
                            review.sign(&keypair_loop);
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

            // A failed epoch leaves `my_update` unsigned; it is neither
            // recorded nor broadcast.
            if my_update.verify().is_ok() {
                {
                    let mut history = hist_loop.write().await;
                    let _ = history.record_update(my_update.clone());
                    for review in &reviews {
                        let _ = history.record_update(review.clone());
                    }
                }

                let network_guard = net_loop.read().await;
                let _ = network_guard.broadcast_update(&my_update).await;
                for review in &reviews {
                    let _ = network_guard.broadcast_update(review).await;
                }
            }

            completed_epochs = current_round_num;
            {
                let mut state_guard = state_loop.write().await;
                let _ = state_guard.set_round(completed_epochs);
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

            if epoch_is_final {
                tracing::info!(
                    "🏁 TARGET FEDERATED EPOCHS REACHED (Round {}). Federated training completed successfully! Node shutting down gracefully.",
                    completed_epochs
                );
                tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
                std::process::exit(0);
            }
        }
    });

    // Print status
    info!("==================================================");
    info!("🟢 Node is LIVE (Peer-to-Peer Federated Learning Active)");
    info!("==================================================");
    info!("Transport: Iroh QUIC (with automatic STUN + DERP relay)");
    info!(
        "Discovery: mdns={} dht={} dns={} relay={}",
        config.discovery.mdns,
        config.discovery.dht,
        config.discovery.dns,
        config.discovery.relay
    );
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