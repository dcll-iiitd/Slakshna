use crate::chain::{ Blockchain, BoxError };
use crate::config::Config;
use crate::state::State;
use crate::network::Network;
use axum::{
    extract::{ Path, State as AxumState, WebSocketUpgrade, ws::{ WebSocket, Message } },
    http::StatusCode,
    response::{ IntoResponse, Json },
    routing::get,
    Router,
};
use futures::{ SinkExt, StreamExt };
use serde::Serialize;
use std::sync::Arc;
use tokio::sync::RwLock;
use tower_http::cors::CorsLayer;
use tracing::info;

type SharedState = Arc<AppState>;

struct AppState {
    config: Config,
    blockchain: Arc<RwLock<Blockchain>>,
    state: Arc<RwLock<State>>,
    network: Arc<RwLock<crate::network::mesh::MeshNetwork>>,
}

pub async fn start_api_server(
    config: Config,
    blockchain: Arc<RwLock<Blockchain>>,
    state: Arc<RwLock<State>>,
    // network: Arc<RwLock<StarNetwork>>,
    network: Arc<RwLock<crate::network::mesh::MeshNetwork>>
) -> Result<(), BoxError> {
    let app_state = Arc::new(AppState {
        config: config.clone(),
        blockchain,
        state,
        network,
    });

    let app = Router::new()
        .route("/", get(index))
        .route("/status", get(get_status))
        .route("/block/:height", get(get_block))
        .route("/block/latest", get(get_latest_block))
        .route("/blocks", get(get_blocks))
        .route("/leaderboard", get(get_leaderboard))
        .route("/ws", get(ws_handler))
        .layer(CorsLayer::permissive())
        .with_state(app_state);

    let addr = format!("{}:{}", config.network.host, config.network.api_port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;

    axum::serve(listener, app).await?;

    Ok(())
}

async fn index() -> impl IntoResponse {
    Json(
        serde_json::json!({
        "name": "SLAKSHNA Federated Learning System",
        "version": "1.0.0",
        "architecture": "Asynchronous Model-Lattice",
        "endpoints": {
            "chain": {
                "status": "GET /status",
                "blocks": "GET /blocks",
                "block": "GET /block/:height",
                "latest": "GET /block/latest"
            },
            "network": {
                "leaderboard": "GET /leaderboard",
                "ws": "GET /ws"
            }
        }
    })
    )
}

#[derive(Serialize)]
struct StatusResponse {
    chain_id: String,
    chain_name: String,
    height: u64,
    peers: usize,
    browsers: usize,
    node_type: String,
}

async fn get_status(AxumState(state): AxumState<SharedState>) -> impl IntoResponse {
    let state_guard = state.state.read().await;
    let height = state_guard.get_height().unwrap_or(0);
    drop(state_guard);

    let network = state.network.read().await;
    let peers = network.peer_count();
    let browsers = network.browser_count();
    drop(network);

    Json(StatusResponse {
        chain_id: state.config.chain.chain_id.clone(),
        chain_name: state.config.chain.chain_name.clone(),
        height,
        peers,
        browsers,
        node_type: state.config.node.node_type.clone(),
    })
}

async fn get_block(
    Path(height): Path<u64>,
    AxumState(state): AxumState<SharedState>
) -> impl IntoResponse {
    let chain_guard = state.blockchain.read().await;
    let index = height as usize;
    let mut matching_blocks = Vec::new();
    for (_node, blocks) in chain_guard.lattice_chains.iter() {
        if let Some(block) = blocks.get(index) {
            matching_blocks.push(block.clone());
        }
    }
    if !matching_blocks.is_empty() {
        Json(serde_json::json!({
            "success": true,
            "blocks": matching_blocks
        })).into_response()
    } else {
        (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "success": false,
                "error": "not_found",
                "message": format!("No lattice blocks found at index/height {}", height)
            }))
        ).into_response()
    }
}

async fn get_latest_block(AxumState(state): AxumState<SharedState>) -> impl IntoResponse {
    let chain_guard = state.blockchain.read().await;
    let mut latest_block = None;
    let mut max_len = 0;
    for (_node, blocks) in chain_guard.lattice_chains.iter() {
        if blocks.len() >= max_len {
            if let Some(block) = blocks.last() {
                max_len = blocks.len();
                latest_block = Some(block.clone());
            }
        }
    }
    if let Some(block) = latest_block {
        Json(serde_json::json!({
            "success": true,
            "block": block
        })).into_response()
    } else {
        (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "success": false,
                "error": "not_found",
                "message": "No lattice blocks available yet"
            }))
        ).into_response()
    }
}

async fn get_blocks(AxumState(state): AxumState<SharedState>) -> impl IntoResponse {
    let chain_guard = state.blockchain.read().await;
    let mut all_blocks = Vec::new();
    for (_node, blocks) in chain_guard.lattice_chains.iter() {
        all_blocks.extend(blocks.clone());
    }
    Json(serde_json::json!({
        "success": true,
        "blocks": all_blocks
    })).into_response()
}

async fn get_leaderboard(AxumState(state): AxumState<SharedState>) -> impl IntoResponse {
    let chain_guard = state.blockchain.read().await;
    let committee = chain_guard.get_committee_with_reputation(100);
    let rankings: Vec<_> = committee.into_iter().map(|(node, score)| {
        serde_json::json!({
            "node": node,
            "reputation_score": score
        })
    }).collect();
    Json(serde_json::json!({
        "success": true,
        "leaderboard": rankings
    })).into_response()
}

async fn ws_handler(
    ws: WebSocketUpgrade,
    AxumState(state): AxumState<SharedState>
) -> impl IntoResponse {
    let config = state.config.clone();
    let db_state = state.state.clone();
    let network = state.network.clone();

    ws.on_upgrade(move |socket| handle_browser_socket(socket, config, db_state, network))
}

async fn handle_browser_socket(
    socket: WebSocket,
    config: Config,
    state: Arc<RwLock<State>>,
    _network: Arc<RwLock<crate::network::mesh::MeshNetwork>>
) {
    let (mut sender, mut receiver) = socket.split();

    let browser_id = uuid::Uuid::new_v4().to_string();
    info!("🌐 Browser connected: {}", &browser_id[..8]);

    let status = {
        let state_guard = state.read().await;
        let height = state_guard.get_height().unwrap_or(0);
        serde_json::json!({
            "type": "welcome",
            "height": height,
            "chain_id": config.chain.chain_id
        })
    };
    let _ = sender.send(Message::Text(status.to_string())).await;

    while let Some(Ok(msg)) = receiver.next().await {
        if let Message::Text(_text) = msg {
            // TODO: Handle browser queries
        }
    }

    info!("🌐 Browser disconnected: {}", &browser_id[..8]);
}
