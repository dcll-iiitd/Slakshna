mod config;
mod chain;
mod address;
mod state;
mod network;
mod api;

use crate::config::Config;
use crate::chain::Blockchain;
use crate::state::State;
use crate::network::Network;
use crate::api::start_api_server;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{ info, Level };
use tracing_subscriber::FmtSubscriber;

type BoxError = Box<dyn std::error::Error + Send + Sync>;

use serde::Deserialize;

// #[derive(Deserialize)]
// struct MLEngineOutput {
//     weights: std::collections::HashMap<String, f64>,
//     model_hash: String,
//     validation_score: f64,
//     metadata: String,
// }

#[derive(Deserialize)]
struct MLEngineOutput {
    weights: std::collections::HashMap<String, f64>,
    model_hash: String,
    validation_score: f64,
    metadata: String,
    compressed_delta: String, // NEW: Catching the weights from Python
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
    info!("Chain ID: {}", config.chain.chain_id);
    info!("Node ID: {}", config.node.id);
    info!("Node Type: {}", config.node.node_type);

    // Initialize state (RocksDB)
    let state = Arc::new(RwLock::new(State::new(&config.node.data_dir)?));

    // Generate or load master address
    let master_address = {
        let mut state_guard = state.write().await;
        let addr = state_guard.get_or_create_master_address()?;
        info!("Master Address: {}", addr);
        addr
    };

    config.node.id = master_address.to_string();

    // Initialize blockchain
    let blockchain = Arc::new(
        RwLock::new(Blockchain::new(config.clone(), state.clone(), master_address.clone()).await?)
    );

    // Initialize network
    let network = Arc::new(
        RwLock::new(crate::network::mesh::MeshNetwork::new(config.clone(), blockchain.clone(), state.clone()))
    );

    // Start network
    {
        let mut net = network.write().await;
        net.start().await?;
    }

    // Start API server
    let api_handle = tokio::spawn(
        start_api_server(config.clone(), blockchain.clone(), state.clone(), network.clone())
    );

    let bc = blockchain.clone();
    let net = network.clone();
    let node_id = config.node.id.clone();


    // ==========================================
    // THREAD 2: LAYER 1 / ML EPOCHS (Every 60 seconds)
    // ==========================================
    let bc_l1 = bc.clone();
    let net_l1 = net.clone();
    let node_id_l1 = node_id.clone();
    let gpu_id_l1 = config.node.gpu_id;
    let config_l1 = config.clone();

    tokio::spawn(async move {
        let epoch_duration = 600_u64; // Increased to 600s for LLM fine-tuning

        loop {
            // 1. EXACT CLOCK SYNCHRONIZATION
            let now = chrono::Utc::now().timestamp() as u64;
            let next_boundary = now + (epoch_duration - (now % epoch_duration));
            let wait_time = next_boundary - now;

            tracing::info!("⏳ Syncing clock... waiting {}s for exact epoch boundary", wait_time);
            tokio::time::sleep(tokio::time::Duration::from_secs(wait_time)).await;

            let epoch_start = next_boundary;
            tracing::info!("🏁 NEW EPOCH STARTED (Global Time: {})", epoch_start);

            // --- PHASE A: OFF-CHAIN L2C LEARNING ---
            tracing::info!("🧠 Phase A: Real Local Training & Peer Exchange executing...");

            let peer_deltas_dir = format!("{}/network_deltas", config_l1.node.data_dir);
            std::fs::create_dir_all(&peer_deltas_dir).unwrap_or_default();

            // NEW: Extract received peer proposals from the blockchain and stage them for Python
            {
                let blockchain = bc_l1.read().await;
                let mut extracted_count = 0u32;
                let mut skipped_count = 0u32;

                // Iterate over all chains in the lattice, instead of active_peers, to avoid libp2p ID mismatch
                for (peer_id, chain) in &blockchain.lattice_chains {
                    if peer_id == &node_id_l1 { continue; }
                    
                    // Find their latest proposal
                    let mut found_proposal = false;
                    for block in chain.iter().rev() {
                        if
                            let crate::chain::LatticeBlockType::Proposal {
                                compressed_delta,
                                ..
                            } = &block.block_type
                        {
                            let path = format!("{}/{}_delta.b64", peer_deltas_dir, peer_id);
                            let _ = std::fs::write(&path, compressed_delta);
                            tracing::info!("📥 Network Delta Extracted: Saved {} bytes from peer {} to {}", compressed_delta.len(), peer_id, path);
                            extracted_count += 1;
                            found_proposal = true;
                            break;
                        }
                    }
                    if !found_proposal {
                        tracing::debug!("⏭️ Peer {} has {} blocks in lattice but no Proposal block", peer_id, chain.len());
                        skipped_count += 1;
                    }
                }
                tracing::info!("📊 Delta extraction summary: {} extracted, {} peers without proposals, data_dir={}", extracted_count, skipped_count, config_l1.node.data_dir);
            }

            let mut python_args = vec!["ml_engine.py".to_string(), node_id_l1.clone()];
            {
                let blockchain = bc_l1.read().await;
                for peer_id in blockchain.lattice_chains.keys() {
                    if peer_id != &node_id_l1 {
                        python_args.push(peer_id.clone());
                    }
                }
            }

            // let output = tokio::process::Command
            //     ::new("python")
            //     .args(&python_args)
            //     .current_dir(".")
            //     .output().await;
            let mut cmd = tokio::process::Command::new("python");
            cmd.args(&python_args).current_dir(".");

            // CRITICAL: Pass the data_dir to Python so it reads delta files from the correct path
            cmd.env("IIITD_DATA_DIR", &config_l1.node.data_dir);

            // Pin this ML process to the GPU assigned in the node config
            if let Some(gid) = gpu_id_l1 {
                cmd.env("CUDA_VISIBLE_DEVICES", gid.to_string());
                tracing::info!("🔥 ML Engine pinned to GPU {}", gid);
            }

            let output = cmd.output().await;
            let my_prev_hash = {
                let blockchain = bc_l1.read().await;
                if let Some(chain) = blockchain.lattice_chains.get(&node_id_l1) {
                    if let Some(last_block) = chain.last() {
                        last_block.hash.clone()
                    } else {
                        "0".repeat(64)
                    }
                } else {
                    "0".repeat(64)
                }
            };

            // let mut my_proposal = crate::chain::LatticeBlock {
            //     node_id: node_id_l1.clone(),
            //     prev_hash: my_prev_hash,
            //     block_type: crate::chain::LatticeBlockType::Proposal {
            //         payload_hash: "error_hash".to_string(),
            //         storage_uri: "local".to_string(),
            //     },
            //     signature: format!("node_signature_{}", epoch_start),
            //     hash: String::new(),
            // };
            let mut my_proposal = crate::chain::LatticeBlock {
                node_id: node_id_l1.clone(),
                prev_hash: my_prev_hash,
                block_type: crate::chain::LatticeBlockType::Proposal {
                    payload_hash: "error_hash".to_string(),
                    // Remove storage_uri: "local".to_string(),
                    compressed_delta: String::new(), // <--- USE THIS INSTEAD
                },
                signature: format!("node_signature_{}", epoch_start),
                hash: String::new(),
            };

            let mut evaluations = Vec::new();

            match output {
                Ok(out) if out.status.success() => {
                    let stdout_str = String::from_utf8_lossy(&out.stdout);
                    let json_str = stdout_str.lines().last().unwrap_or("");

                    if let Ok(ml_data) = serde_json::from_str::<MLEngineOutput>(json_str) {
                        tracing::info!(
                            "✅ Local Training Complete! Score: {:.4}",
                            ml_data.validation_score
                        );

                        // Create Proposal Block
                        my_proposal.block_type = crate::chain::LatticeBlockType::Proposal {
                            payload_hash: ml_data.model_hash.clone(),
                            compressed_delta: ml_data.compressed_delta.clone(),
                        };
                        my_proposal.hash = my_proposal.calculate_hash();

                        let mut current_tail_hash = my_proposal.hash.clone();

                        let blockchain = bc_l1.read().await;

                        // Create Evaluation Blocks
                        for (peer_id, weight) in ml_data.weights {
                            let mut actual_proposal_hash = None;
                            if let Some(peer_chain) = blockchain.lattice_chains.get(&peer_id) {
                                for block in peer_chain.iter().rev() {
                                    if
                                        let crate::chain::LatticeBlockType::Proposal { .. } =
                                            &block.block_type
                                    {
                                        actual_proposal_hash = Some(block.hash.clone());
                                        break;
                                    }
                                }
                            }

                            let target_proposal_hash = match actual_proposal_hash {
                                Some(h) => h,
                                None => {
                                    tracing::warn!(
                                        "Skipping evaluation for {}, no proposal found in lattice.",
                                        peer_id
                                    );
                                    continue;
                                }
                            };

                            let mut eval_block = crate::chain::LatticeBlock {
                                node_id: node_id_l1.clone(),
                                prev_hash: current_tail_hash.clone(),
                                block_type: crate::chain::LatticeBlockType::Evaluation {
                                    target_node: peer_id,
                                    proposal_hash: target_proposal_hash,
                                    loss_drop: 0.0, // Simplified for now
                                    trust_score: weight,
                                },
                                signature: format!("eval_sig_{}", epoch_start),
                                hash: String::new(),
                            };
                            eval_block.hash = eval_block.calculate_hash();
                            current_tail_hash = eval_block.hash.clone();
                            evaluations.push(eval_block);
                        }
                        drop(blockchain);
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
                let mut blockchain = bc_l1.write().await;
                blockchain.add_lattice_block(my_proposal.clone());
                for eval in &evaluations {
                    blockchain.add_lattice_block(eval.clone());
                }
            }

            {
                let network_guard = net_l1.read().await;
                let _ = network_guard.broadcast_lattice_block(&my_proposal).await;
                for eval in &evaluations {
                    let _ = network_guard.broadcast_lattice_block(eval).await;
                }
            }

            // 2. MID-EPOCH SYNC: Intelligent Sync Barrier
            let target_l1_time = epoch_start + 300;

            tracing::info!("⏳ Waiting for ML submissions to propagate across network...");

            loop {
                let current_time = chrono::Utc::now().timestamp() as u64;

                let unique_count = {
                    let blockchain = bc_l1.read().await;
                    blockchain.lattice_chains.len()
                };

                if unique_count >= 6 {
                    tracing::info!(
                        "✅ Collected all {} unique ML submissions early!",
                        unique_count
                    );
                    break;
                }

                if current_time >= target_l1_time {
                    tracing::warn!(
                        "⚠️ Hit L1 deadline with only {} submissions. Proceeding anyway...",
                        unique_count
                    );
                    break;
                }

                tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
            }

            let now_after_wait = chrono::Utc::now().timestamp() as u64;
            if target_l1_time > now_after_wait {
                tokio::time::sleep(
                    tokio::time::Duration::from_secs(target_l1_time - now_after_wait)
                ).await;
            }

            // ==========================================
            // PHASE B: L1 CONSENSUS & L2 FINALIZATION
            // ==========================================
            let _current_l1_hash = "lattice_consensus".to_string();

            let committee = {
                let blockchain = bc_l1.read().await;
                blockchain.get_elected_committee(crate::chain::COMMITTEE_SIZE) // 👈 Using global constant
            };

            tracing::info!("🏛️ Committee Elected: {:?}", committee);

            if committee.contains(&node_id_l1) {
                tracing::info!("⚙️ Node elected to committee for this epoch!");
            } else {
                tracing::info!("💤 Node not in committee. Waiting for next epoch...");
            }
        }
    });

    // Print status
    info!("==================================================");
    info!("🟢 Node is LIVE (Federated Learning Consensus Active)");
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
║        ██╗██╗██╗████████╗██████╗                              ║
║        ██║██║██║╚══██╔══╝██╔══██╗                             ║
║        ██║██║██║   ██║   ██║  ██║                             ║
║        ██║██║██║   ██║   ██║  ██║                             ║
║        ██║██║██║   ██║   ██████╔╝                             ║
║        ╚═╝╚═╝╚═╝   ╚═╝   ╚═════╝                              ║
║                                                               ║
║   IIITD VIRTUAL MACHINE - FEDERATED LEARNING CONSENSUS        ║
║   A Dual-Layer Blockchain for AI & Value Transfer             ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
"#
    );
}
