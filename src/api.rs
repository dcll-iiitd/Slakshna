use crate::history::{ UpdateHistory, BoxError };
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
    history: Arc<RwLock<UpdateHistory>>,
    state: Arc<RwLock<State>>,
    network: Arc<RwLock<crate::network::mesh::MeshNetwork>>,
}

pub async fn start_api_server(
    config: Config,
    history: Arc<RwLock<UpdateHistory>>,
    state: Arc<RwLock<State>>,
    network: Arc<RwLock<crate::network::mesh::MeshNetwork>>
) -> Result<(), BoxError> {
    let app_state = Arc::new(AppState {
        config: config.clone(),
        history,
        state,
        network,
    });

    let app = Router::new()
        .route("/", get(index))
        .route("/status", get(get_status))
        .route("/updates/:index", get(get_update))
        .route("/updates/latest", get(get_latest_update))
        .route("/updates", get(get_updates))
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
        "architecture": "Asynchronous Peer-to-Peer Federated Learning",
        "endpoints": {
            "history": {
                "status": "GET /status",
                "updates": "GET /updates",
                "update": "GET /updates/:index",
                "latest": "GET /updates/latest"
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
    federation_id: String,
    federation_name: String,
    round: u64,
    peers: usize,
    browsers: usize,
    node_type: String,
}

async fn get_status(AxumState(state): AxumState<SharedState>) -> impl IntoResponse {
    let state_guard = state.state.read().await;
    let round = state_guard.get_round().unwrap_or(0);
    drop(state_guard);

    let network = state.network.read().await;
    let peers = network.peer_count();
    let browsers = network.browser_count();
    drop(network);

    Json(StatusResponse {
        federation_id: state.config.federation.id.clone(),
        federation_name: state.config.federation.name.clone(),
        round,
        peers,
        browsers,
        node_type: state.config.node.node_type.clone(),
    })
}

/// Returns the record at `index` in every peer's update log — i.e. what each
/// participant contributed in that federated round.
async fn get_update(
    Path(index): Path<u64>,
    AxumState(state): AxumState<SharedState>
) -> impl IntoResponse {
    let history_guard = state.history.read().await;
    let idx = index as usize;
    let mut matching_records = Vec::new();
    for (_node, records) in history_guard.peer_updates.iter() {
        if let Some(record) = records.get(idx) {
            matching_records.push(record.clone());
        }
    }
    if !matching_records.is_empty() {
        Json(serde_json::json!({
            "success": true,
            "updates": matching_records
        })).into_response()
    } else {
        (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "success": false,
                "error": "not_found",
                "message": format!("No update records found at index {}", index)
            }))
        ).into_response()
    }
}

async fn get_latest_update(AxumState(state): AxumState<SharedState>) -> impl IntoResponse {
    let history_guard = state.history.read().await;
    let mut latest_record = None;
    let mut max_len = 0;
    for (_node, records) in history_guard.peer_updates.iter() {
        if records.len() >= max_len {
            if let Some(record) = records.last() {
                max_len = records.len();
                latest_record = Some(record.clone());
            }
        }
    }
    if let Some(record) = latest_record {
        Json(serde_json::json!({
            "success": true,
            "update": record
        })).into_response()
    } else {
        (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "success": false,
                "error": "not_found",
                "message": "No update records available yet"
            }))
        ).into_response()
    }
}

async fn get_updates(AxumState(state): AxumState<SharedState>) -> impl IntoResponse {
    let history_guard = state.history.read().await;
    let mut all_records = Vec::new();
    for (_node, records) in history_guard.peer_updates.iter() {
        all_records.extend(records.clone());
    }
    Json(serde_json::json!({
        "success": true,
        "updates": all_records
    })).into_response()
}

async fn get_leaderboard(AxumState(state): AxumState<SharedState>) -> impl IntoResponse {
    let history_guard = state.history.read().await;
    let rankings: Vec<_> = history_guard
        .trust_rankings(100)
        .into_iter()
        .map(|(node, score)| {
            serde_json::json!({
                "node": node,
                "trust_score": score
            })
        })
        .collect();
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
        let round = state_guard.get_round().unwrap_or(0);
        serde_json::json!({
            "type": "welcome",
            "round": round,
            "federation_id": config.federation.id
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
