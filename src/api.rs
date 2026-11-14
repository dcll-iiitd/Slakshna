use crate::history::{ UpdateHistory, BoxError };
use crate::config::Config;
use crate::state::State;
use crate::network::Network;
use crate::rag::{ QueryRequest, RagService };
use axum::{
    extract::{ Path, State as AxumState, WebSocketUpgrade, ws::{ WebSocket, Message } },
    http::StatusCode,
    response::{ IntoResponse, Json },
    routing::{ get, post },
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
    /// Present only when `[rag] enabled = true`.
    rag: Option<Arc<RagService>>,
}

pub async fn start_api_server(
    config: Config,
    history: Arc<RwLock<UpdateHistory>>,
    state: Arc<RwLock<State>>,
    network: Arc<RwLock<crate::network::mesh::MeshNetwork>>,
    rag: Option<Arc<RagService>>
) -> Result<(), BoxError> {
    let app_state = Arc::new(AppState {
        config: config.clone(),
        history,
        state,
        network,
        rag,
    });

    let app = Router::new()
        .route("/", get(index))
        .route("/status", get(get_status))
        .route("/updates/:index", get(get_update))
        .route("/updates/latest", get(get_latest_update))
        .route("/updates", get(get_updates))
        .route("/leaderboard", get(get_leaderboard))
        .route("/peers", get(get_peers))
        .route("/ws", get(ws_handler))
        .route("/rag/query", post(rag_query))
        .route("/rag/ingest", post(rag_ingest))
        .route("/rag/status", get(rag_status))
        .route("/rag/debug/hello", get(rag_debug_hello))
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
                "peers": "GET /peers",
                "ws": "GET /ws"
            },
            "rag": {
                "query": "POST /rag/query",
                "ingest": "POST /rag/ingest",
                "status": "GET /rag/status"
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
    /// This node's Iroh EndpointId — hand it to any other node to have it join.
    endpoint_id: Option<String>,
}

async fn get_status(AxumState(state): AxumState<SharedState>) -> impl IntoResponse {
    let state_guard = state.state.read().await;
    let round = state_guard.get_round().unwrap_or(0);
    drop(state_guard);

    let network = state.network.read().await;
    let peers = network.peer_count();
    let browsers = network.browser_count();
    let endpoint_id = network.endpoint_id().await;
    drop(network);

    Json(StatusResponse {
        federation_id: state.config.federation.id.clone(),
        federation_name: state.config.federation.name.clone(),
        round,
        peers,
        browsers,
        endpoint_id,
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

/// Live gossip neighbours plus every peer this node has ever reached.
/// The `known` set is what the node redials on its next start — it is the
/// federation's own memory of its membership, held by no central party.
async fn get_peers(AxumState(state): AxumState<SharedState>) -> impl IntoResponse {
    let network = state.network.read().await;
    let connected = network.get_active_peer_ids().await;
    let endpoint_id = network.endpoint_id().await;
    drop(network);

    let known = {
        let state_guard = state.state.read().await;
        state_guard.known_peers()
    };

    Json(serde_json::json!({
        "success": true,
        "endpoint_id": endpoint_id,
        "connected": connected,
        "known": known
    }))
}

fn rag_error(status: StatusCode, error: &str, message: String) -> axum::response::Response {
    (
        status,
        Json(serde_json::json!({
            "success": false,
            "error": error,
            "message": message
        })),
    ).into_response()
}

fn rag_disabled() -> axum::response::Response {
    rag_error(
        StatusCode::SERVICE_UNAVAILABLE,
        "rag_disabled",
        "RAG is not enabled on this node ([rag] enabled = true)".to_string(),
    )
}

/// Answer a question from retrieved documents.
async fn rag_query(
    AxumState(state): AxumState<SharedState>,
    Json(request): Json<QueryRequest>
) -> impl IntoResponse {
    let Some(rag) = &state.rag else { return rag_disabled() };
    if request.query.trim().is_empty() {
        return rag_error(StatusCode::BAD_REQUEST, "empty_query", "query must not be empty".to_string());
    }
    if request.retrieval == Some(crate::config::Retrieval::Directory) && !rag.config.directory {
        return rag_error(
            StatusCode::BAD_REQUEST,
            "directory_disabled",
            "retrieval \"directory\" needs [rag] directory = true on this node".to_string(),
        );
    }
    let link = state.network.read().await.rag_link().await;
    match rag.query(request, link).await {
        Ok(response) => Json(response).into_response(),
        Err(e) => rag_error(StatusCode::BAD_GATEWAY, "rag_failed", e.to_string()),
    }
}

/// Add documents to this node's index. The body is passed to the sidecar as is:
/// `{"path": "..."}` or `{"documents": [{"id": "...", "text": "..."}]}`.
async fn rag_ingest(
    AxumState(state): AxumState<SharedState>,
    Json(body): Json<serde_json::Value>
) -> impl IntoResponse {
    let Some(rag) = &state.rag else { return rag_disabled() };
    match rag.sidecar.ingest(&body).await {
        Ok(result) => {
            // New documents mean a new Merkle root; sign it before peers ask.
            if let Err(e) = rag.refresh_own_root().await {
                tracing::warn!("⚠️ Ingest succeeded but the new RAG root could not be signed: {}", e);
            }
            Json(result).into_response()
        }
        Err(e) => rag_error(StatusCode::BAD_GATEWAY, "rag_failed", e.to_string()),
    }
}

async fn rag_status(AxumState(state): AxumState<SharedState>) -> impl IntoResponse {
    let Some(rag) = &state.rag else { return rag_disabled() };
    let target_ids: Vec<iroh::EndpointId> = match state.network.read().await.rag_link().await {
        Some(link) => link.targets,
        None => Vec::new(),
    };
    let targets: Vec<String> = target_ids.iter().map(|id| id.to_string()).collect();
    match rag.sidecar.health().await {
        Ok(sidecar) => Json(serde_json::json!({
            "success": true,
            "sidecar": sidecar,
            "peer_targets": targets,
            "own_root": rag.own_root().await,
            "retrieval": rag.config.retrieval,
            "registry": rag.registry_status(&target_ids),
            "directory": rag.directory_status(),
            "top_k": rag.config.top_k,
            "final_k": rag.config.final_k
        })).into_response(),
        Err(e) => rag_error(StatusCode::BAD_GATEWAY, "rag_failed", e.to_string()),
    }
}

/// Debugging aid: fetch every peer target's signed root and check each
/// signature against that peer's EndpointId.
async fn rag_debug_hello(AxumState(state): AxumState<SharedState>) -> impl IntoResponse {
    let Some(rag) = &state.rag else { return rag_disabled() };
    let Some(link) = state.network.read().await.rag_link().await else {
        return Json(serde_json::json!({ "success": true, "peers": [] })).into_response();
    };
    let peers = rag.refresh_peer_roots(&link).await;
    Json(serde_json::json!({ "success": true, "peers": peers })).into_response()
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
