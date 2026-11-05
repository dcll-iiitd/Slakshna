//! A2A gateway (M1): JSON-RPC 2.0 over HTTP(S) for agents outside the federation.
//!
//! * `GET /.well-known/agent-card.json` - the signed Agent Card (public).
//! * `POST /a2a` - JSON-RPC methods `SendMessage`, `SendStreamingMessage`,
//!   `SubscribeToTask`, `GetTask`, `ListTasks`, `CancelTask`, `GetExtendedAgentCard`.
//!   Requires `Authorization: Bearer <key>`; keys are configured as SHA-256 hashes.
//!
//! Outside calls land in the same task store as calls over the QUIC ALPN, so every
//! lifecycle step becomes a signed provenance record exactly as inside the federation;
//! the caller appears as the principal `api:<name>`.
//!
//! Method names and message shapes follow A2A v1.0 as published at
//! a2a-protocol.org (PascalCase JSON-RPC methods; Task/Message/Part/Artifact in
//! camelCase). Streaming events use the `StreamResponse` shape
//! (`task`, `statusUpdate`, `artifactUpdate`).

use crate::agents::a2a::{Message, TaskError, TaskEvent};
use crate::agents::service::AgentService;
use axum::body::Body;
use axum::extract::{DefaultBodyLimit, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::convert::Infallible;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const MAX_BODY: usize = 1024 * 1024;
const RATE_LIMIT_PER_MIN: u32 = 120;

pub struct GatewayState {
    pub svc: Arc<AgentService>,
    /// sha256(key) hex -> principal
    keys: HashMap<String, String>,
    buckets: Mutex<HashMap<String, (Instant, u32)>>,
}

impl GatewayState {
    pub fn new(svc: Arc<AgentService>) -> Arc<Self> {
        let keys = svc.cfg.api_keys.iter().map(|k| (k.key_sha256.to_lowercase(), k.principal.clone())).collect();
        Arc::new(GatewayState { svc, keys, buckets: Mutex::new(HashMap::new()) })
    }

    fn authenticate(&self, headers: &HeaderMap) -> Result<String, StatusCode> {
        let Some(v) = headers.get(header::AUTHORIZATION).and_then(|v| v.to_str().ok()) else { return Err(StatusCode::UNAUTHORIZED) };
        let Some(token) = v.strip_prefix("Bearer ") else { return Err(StatusCode::UNAUTHORIZED) };
        let digest = hex::encode(Sha256::digest(token.trim().as_bytes()));
        // Lookup is by hash of the presented key, so no secret-dependent comparison happens here.
        self.keys.get(&digest).cloned().ok_or(StatusCode::UNAUTHORIZED)
    }

    fn rate_ok(&self, principal: &str) -> bool {
        let mut b = self.buckets.lock().unwrap();
        let e = b.entry(principal.to_string()).or_insert((Instant::now(), 0));
        if e.0.elapsed() >= Duration::from_secs(60) {
            *e = (Instant::now(), 0);
        }
        e.1 += 1;
        e.1 <= RATE_LIMIT_PER_MIN
    }
}

pub fn router(state: Arc<GatewayState>) -> Router {
    Router::new()
        .route("/.well-known/agent-card.json", get(card))
        .route("/a2a", post(rpc))
        .layer(DefaultBodyLimit::max(MAX_BODY))
        .with_state(state)
}

async fn card(State(st): State<Arc<GatewayState>>) -> Json<Value> {
    Json(serde_json::to_value(st.svc.card()).expect("card serializes"))
}

fn rpc_ok(id: &Value, result: Value) -> Value {
    json!({"jsonrpc": "2.0", "id": id, "result": result})
}

fn rpc_err(id: &Value, code: i64, message: impl Into<String>) -> Value {
    json!({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message.into()}})
}

fn task_error(id: &Value, e: TaskError) -> Value {
    match e {
        TaskError::NotFound(_) => rpc_err(id, -32001, e.to_string()), // TaskNotFoundError
        TaskError::BadTransition(..) => rpc_err(id, -32002, e.to_string()), // TaskNotCancelableError
        TaskError::InvalidMessage(_) => rpc_err(id, -32602, e.to_string()),
        TaskError::Io(_) => rpc_err(id, -32603, e.to_string()),
    }
}

async fn rpc(State(st): State<Arc<GatewayState>>, headers: HeaderMap, body: axum::body::Bytes) -> Response {
    let principal = match st.authenticate(&headers) {
        Ok(p) => p,
        Err(code) => return (code, [(header::WWW_AUTHENTICATE, "Bearer")], "unauthorized").into_response(),
    };
    if !st.rate_ok(&principal) {
        return (StatusCode::TOO_MANY_REQUESTS, "rate limit exceeded").into_response();
    }
    let req: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => return Json(rpc_err(&Value::Null, -32700, format!("parse error: {e}"))).into_response(),
    };
    let id = req.get("id").cloned().unwrap_or(Value::Null);
    if req["jsonrpc"] != "2.0" || !req["method"].is_string() {
        return Json(rpc_err(&id, -32600, "invalid request")).into_response();
    }
    let method = req["method"].as_str().unwrap();
    let params = req.get("params").cloned().unwrap_or(Value::Null);
    let sender = format!("api:{principal}");
    let svc = &st.svc;

    let parse_msg = |p: &Value| -> Result<Message, String> {
        serde_json::from_value(p.get("message").cloned().unwrap_or(Value::Null)).map_err(|e| format!("invalid message: {e}"))
    };

    match method {
        "SendMessage" => match parse_msg(&params) {
            Ok(m) => match svc.send_message(&sender, m) {
                Ok(t) => Json(rpc_ok(&id, json!({"task": t}))).into_response(),
                Err(e) => Json(task_error(&id, e)).into_response(),
            },
            Err(e) => Json(rpc_err(&id, -32602, e)).into_response(),
        },
        "GetTask" => match params["id"].as_str() {
            Some(tid) => match owned(svc, tid, &sender).and_then(|_| svc.tasks.get(tid)) {
                Ok(t) => Json(rpc_ok(&id, serde_json::to_value(t).unwrap())).into_response(),
                Err(e) => Json(task_error(&id, e)).into_response(),
            },
            None => Json(rpc_err(&id, -32602, "id required")).into_response(),
        },
        "ListTasks" => {
            // Callers see only the tasks they submitted.
            let mine: Vec<_> = svc.tasks.list().into_iter().filter(|t| t.metadata.as_ref().and_then(|m| m["sender"].as_str()) == Some(sender.as_str())).collect();
            Json(rpc_ok(&id, json!({"tasks": mine}))).into_response()
        }
        "CancelTask" => match params["id"].as_str() {
            Some(tid) => match owned(svc, tid, &sender).and_then(|_| svc.cancel_task(tid)) {
                Ok(t) => Json(rpc_ok(&id, serde_json::to_value(t).unwrap())).into_response(),
                Err(e) => Json(task_error(&id, e)).into_response(),
            },
            None => Json(rpc_err(&id, -32602, "id required")).into_response(),
        },
        "GetExtendedAgentCard" => Json(rpc_ok(&id, serde_json::to_value(svc.card()).unwrap())).into_response(),
        "SendStreamingMessage" => match parse_msg(&params) {
            Ok(m) => match svc.send_message(&sender, m) {
                Ok(t) => sse(st.clone(), id, t.id).into_response(),
                Err(e) => Json(task_error(&id, e)).into_response(),
            },
            Err(e) => Json(rpc_err(&id, -32602, e)).into_response(),
        },
        "SubscribeToTask" => match params["id"].as_str() {
            Some(tid) => match owned(svc, tid, &sender) {
                Ok(_) => sse(st.clone(), id, tid.to_string()).into_response(),
                Err(e) => Json(task_error(&id, e)).into_response(),
            },
            None => Json(rpc_err(&id, -32602, "id required")).into_response(),
        },
        other => Json(rpc_err(&id, -32601, format!("method '{other}' not found"))).into_response(),
    }
}

/// A caller may only touch tasks it created.
fn owned(svc: &AgentService, tid: &str, sender: &str) -> Result<(), TaskError> {
    let t = svc.tasks.get(tid)?;
    if t.metadata.as_ref().and_then(|m| m["sender"].as_str()) == Some(sender) {
        Ok(())
    } else {
        Err(TaskError::NotFound(tid.into())) // do not reveal that it exists
    }
}

fn sse(st: Arc<GatewayState>, rpc_id: Value, task_id: String) -> Sse<impl futures::Stream<Item = Result<Event, Infallible>>> {
    let svc = st.svc.clone();
    let (tx, rx) = tokio::sync::mpsc::channel::<Event>(64);
    tokio::spawn(async move {
        let send = |v: Value| Event::default().data(serde_json::to_string(&rpc_ok(&rpc_id, v)).unwrap());
        // Subscribe first, then read the state, so no transition is lost.
        let mut events = svc.tasks.subscribe();
        let Ok(task) = svc.tasks.get(&task_id) else { return };
        let done = task.status.state.is_terminal();
        if tx.send(send(json!({"task": task}))).await.is_err() || done {
            return;
        }
        let deadline = tokio::time::Instant::now() + Duration::from_secs(svc.cfg.task_timeout_secs + 60);
        loop {
            match tokio::time::timeout_at(deadline, events.recv()).await {
                Ok(Ok(TaskEvent::Status { task_id: t, status, is_final })) if t == task_id => {
                    let ctx = svc.tasks.get(&t).map(|x| x.context_id).unwrap_or_default();
                    let ev = send(json!({"statusUpdate": {"taskId": t, "contextId": ctx, "status": status, "final": is_final}}));
                    if tx.send(ev).await.is_err() || is_final {
                        return;
                    }
                }
                Ok(Ok(TaskEvent::Artifact { task_id: t, artifact })) if t == task_id => {
                    let ctx = svc.tasks.get(&t).map(|x| x.context_id).unwrap_or_default();
                    if tx.send(send(json!({"artifactUpdate": {"taskId": t, "contextId": ctx, "artifact": artifact}}))).await.is_err() {
                        return;
                    }
                }
                Ok(Ok(_)) => {}
                Ok(Err(tokio::sync::broadcast::error::RecvError::Lagged(_))) => {
                    if let Ok(t) = svc.tasks.get(&task_id) {
                        let fin = t.status.state.is_terminal();
                        if tx.send(send(json!({"task": t}))).await.is_err() || fin {
                            return;
                        }
                    }
                }
                _ => return,
            }
        }
    });
    Sse::new(tokio_stream_from(rx)).keep_alive(KeepAlive::new().interval(Duration::from_secs(15)))
}

fn tokio_stream_from(mut rx: tokio::sync::mpsc::Receiver<Event>) -> impl futures::Stream<Item = Result<Event, Infallible>> {
    futures::stream::poll_fn(move |cx| rx.poll_recv(cx).map(|o| o.map(Ok)))
}

/// Serve the gateway until the returned future is dropped. HTTPS when both
/// `tls_cert` and `tls_key` are configured, plain HTTP otherwise.
pub async fn serve(state: Arc<GatewayState>, listener: tokio::net::TcpListener) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let app = router(state.clone());
    match (&state.svc.cfg.tls_cert, &state.svc.cfg.tls_key) {
        (Some(cert), Some(key)) => serve_tls(app, listener, cert, key).await,
        (None, None) => {
            axum::serve(listener, app).await?;
            Ok(())
        }
        _ => Err("agents.tls_cert and agents.tls_key must be set together".into()),
    }
}

async fn serve_tls(app: Router, listener: tokio::net::TcpListener, cert: &str, key: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    use hyper_util::rt::{TokioExecutor, TokioIo};
    use hyper_util::server::conn::auto::Builder;
    use hyper_util::service::TowerToHyperService;
    use rustls_pki_types::pem::PemObject;
    use rustls_pki_types::{CertificateDer, PrivateKeyDer};

    let certs: Vec<CertificateDer<'static>> = CertificateDer::pem_file_iter(cert)?.collect::<Result<_, _>>()?;
    let key = PrivateKeyDer::from_pem_file(key)?;
    let mut cfg = rustls::ServerConfig::builder_with_provider(Arc::new(rustls::crypto::ring::default_provider()))
        .with_safe_default_protocol_versions()?
        .with_no_client_auth()
        .with_single_cert(certs, key)?;
    cfg.alpn_protocols = vec![b"h2".to_vec(), b"http/1.1".to_vec()];
    let acceptor = tokio_rustls::TlsAcceptor::from(Arc::new(cfg));
    loop {
        let (tcp, _) = listener.accept().await?;
        let acceptor = acceptor.clone();
        let app = app.clone();
        tokio::spawn(async move {
            let Ok(tls) = acceptor.accept(tcp).await else { return };
            let _ = Builder::new(TokioExecutor::new())
                .serve_connection(TokioIo::new(tls), TowerToHyperService::new(app.map_request(|r: axum::http::Request<hyper::body::Incoming>| r.map(Body::new))))
                .await;
        });
    }
}

use tower::ServiceExt as _;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agents::a2a::{Part, Role, TaskState};
    use crate::agents::a2a_client::{A2aClient, ClientError, RemoteAgentSkill};
    use crate::agents::records::KIND_A2A_TASK;
    use crate::agents::service::tests::{make_service, tmpdir, TestSkill};
    use crate::config::ApiKeyConfig;
    use crate::history::RecordKind;
    use crate::identity::Keypair;

    fn key(principal: &str, secret: &str) -> ApiKeyConfig {
        ApiKeyConfig { principal: principal.into(), key_sha256: hex::encode(Sha256::digest(secret.as_bytes())) }
    }

    fn msg(id: &str, skill: &str, text: &str) -> Message {
        Message { message_id: id.into(), role: Role::User, parts: vec![Part::text(text)], context_id: None, task_id: None, metadata: Some(json!({"skill": skill})), extensions: vec![], reference_task_ids: vec![] }
    }

    async fn start(dir: &str, delay_ms: u64, tls: Option<(String, String)>) -> (Arc<AgentService>, String) {
        let svc = make_service(dir, Keypair::generate(), |c| {
            c.api_keys = vec![key("alice", "alice-secret"), key("bob", "bob-secret")];
            if let Some((cert, k)) = &tls {
                c.tls_cert = Some(cert.clone());
                c.tls_key = Some(k.clone());
            }
        })
        .await;
        let mut s = TestSkill::new("shout");
        s.delay_ms = delay_ms;
        svc.register_skill(Arc::new(s));
        svc.ensure_delegation().await;
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let scheme = if tls.is_some() { "https" } else { "http" };
        let url = format!("{scheme}://127.0.0.1:{port}");
        svc.set_public_url(&url);
        let state = GatewayState::new(svc.clone());
        tokio::spawn(async move { let _ = serve(state, listener).await; });
        (svc, url)
    }

    fn alice(url: &str) -> A2aClient { A2aClient::new(url, Some("alice-secret".into())).unwrap() }

    #[tokio::test]
    async fn outside_client_discovers_verifies_and_calls_a_slakshna_agent() {
        let (svc, url) = start(&tmpdir("g1"), 0, None).await;
        let c = alice(&url);
        let (card, signed) = c.fetch_card().await.unwrap();
        assert!(signed && card.verified_key().as_deref() == Some(svc.agent_public_key().as_str()));
        assert!(card.supported_interfaces.iter().any(|i| i.protocol_binding == "JSONRPC" && i.url.ends_with("/a2a")));
        assert_eq!(card.skills[0].id, "shout");
        let t = c.send_message(&msg("m1", "shout", "from outside")).await.unwrap();
        let done = c.wait(&t.id, Duration::from_secs(5)).await.unwrap();
        assert_eq!(done.status.state, TaskState::Completed);
        assert_eq!(done.artifacts[0].parts[0].text.as_deref(), Some("FROM OUTSIDE"));
        // Outside work is as attributable as inside work: signed task records name the API principal.
        tokio::time::sleep(Duration::from_millis(150)).await;
        let h = svc.history.read().await;
        let recs = h.provenance(KIND_A2A_TASK);
        assert_eq!(recs.len(), 3);
        assert!(recs.iter().all(|r| r.verify().is_ok()));
        assert_eq!(done.metadata.unwrap()["sender"], "api:alice");
        drop(h);
        let _ = RecordKind::ModelUpdate { delta_hash: String::new(), compressed_delta: String::new() };
    }

    #[tokio::test]
    async fn authentication_is_required_and_callers_are_isolated() {
        let (_svc, url) = start(&tmpdir("g2"), 200, None).await;
        let anon = A2aClient::new(&url, None).unwrap();
        assert!(matches!(anon.send_message(&msg("x", "shout", "x")).await, Err(ClientError::Http(e)) if e.contains("401")));
        let wrong = A2aClient::new(&url, Some("nope".into())).unwrap();
        assert!(wrong.send_message(&msg("x", "shout", "x")).await.is_err());
        // the card stays public
        anon.fetch_card().await.unwrap();

        let a = alice(&url);
        let b = A2aClient::new(&url, Some("bob-secret".into())).unwrap();
        let t = a.send_message(&msg("m", "shout", "mine")).await.unwrap();
        // bob can neither see nor cancel alice's task, and learns nothing about it
        assert!(matches!(b.get_task(&t.id).await, Err(ClientError::Rpc { code: -32001, .. })));
        assert!(matches!(b.cancel_task(&t.id).await, Err(ClientError::Rpc { code: -32001, .. })));
        let lst = b.call("ListTasks", json!({})).await.unwrap();
        assert_eq!(lst["tasks"].as_array().unwrap().len(), 0);
        let lst = a.call("ListTasks", json!({})).await.unwrap();
        assert_eq!(lst["tasks"].as_array().unwrap().len(), 1);
        assert_eq!(a.wait(&t.id, Duration::from_secs(5)).await.unwrap().status.state, TaskState::Completed);
    }

    #[tokio::test]
    async fn json_rpc_errors_use_the_standard_and_a2a_codes() {
        let (_svc, url) = start(&tmpdir("g3"), 0, None).await;
        let c = alice(&url);
        let code = |r: Result<Value, ClientError>| match r { Err(ClientError::Rpc { code, .. }) => code, o => panic!("expected rpc error, got {o:?}") };
        assert_eq!(code(c.call("Nope", json!({})).await), -32601);
        assert_eq!(code(c.call("SendMessage", json!({"message": {"bad": 1}})).await), -32602);
        assert_eq!(code(c.call("GetTask", json!({"id": "missing"})).await), -32001);
        assert_eq!(code(c.call("GetTask", json!({})).await), -32602);
        // an empty message is rejected by validation
        let mut m = msg("e", "shout", "x");
        m.parts.clear();
        assert_eq!(code(c.call("SendMessage", json!({"message": m})).await), -32602);
        // malformed JSON and a non-2.0 envelope
        let raw = reqwest::Client::new();
        let r: Value = raw.post(format!("{url}/a2a")).bearer_auth("alice-secret").body("{oops").send().await.unwrap().json().await.unwrap();
        assert_eq!(r["error"]["code"], -32700);
        let r: Value = raw.post(format!("{url}/a2a")).bearer_auth("alice-secret").json(&json!({"id": 1, "method": "GetTask"})).send().await.unwrap().json().await.unwrap();
        assert_eq!(r["error"]["code"], -32600);
        // oversized bodies are refused
        let big = "x".repeat(2 * 1024 * 1024);
        let r = raw.post(format!("{url}/a2a")).bearer_auth("alice-secret").body(big).send().await.unwrap();
        assert_eq!(r.status(), 413);
    }

    #[tokio::test]
    async fn streaming_delivers_task_status_and_artifact_events() {
        let (_svc, url) = start(&tmpdir("g4"), 150, None).await;
        let events = alice(&url).send_streaming(&msg("s", "shout", "stream"), Duration::from_secs(10)).await.unwrap();
        assert!(events[0].get("task").is_some(), "first event is the task snapshot");
        assert!(events.iter().any(|e| e.get("artifactUpdate").is_some()));
        let last = events.last().unwrap();
        assert_eq!(last["statusUpdate"]["final"], true);
        assert_eq!(last["statusUpdate"]["status"]["state"], "TASK_STATE_COMPLETED");
    }

    /// M1 done-when, both directions: an outside client calls a Slakshna agent, and a
    /// Slakshna agent calls an outside one (here: another gateway, speaking only the wire protocol).
    #[tokio::test]
    async fn slakshna_agent_calls_an_outside_agent_through_a_remote_skill() {
        let (_outside, outside_url) = start(&tmpdir("g5a"), 0, None).await; // plays "someone else's" A2A server
        // The local agent exposes skill "ask-outside" that forwards to the outside agent's "shout" skill.
        let local = make_service(&tmpdir("g5b"), Keypair::generate(), |_| {}).await;
        local.register_skill(Arc::new(RemoteAgentSkill {
            local_id: "ask-outside".into(),
            remote_skill: "shout".into(),
            client: A2aClient::new(&outside_url, Some("alice-secret".into())).unwrap(),
            timeout: Duration::from_secs(10),
        }));
        let t = local.send_message("endpoint:requester", msg("r1", "ask-outside", "relay this")).unwrap();
        let done = crate::agents::service::tests::wait_state(&local, &t.id, TaskState::Completed).await;
        assert_eq!(done.artifacts[0].parts[0].text.as_deref(), Some("RELAY THIS"));
        // a remote that is down fails the task cleanly instead of hanging
        let dead = make_service(&tmpdir("g5c"), Keypair::generate(), |_| {}).await;
        dead.register_skill(Arc::new(RemoteAgentSkill { local_id: "ask".into(), remote_skill: "shout".into(), client: A2aClient::new("http://127.0.0.1:1", None).unwrap(), timeout: Duration::from_secs(2) }));
        let t = dead.send_message("endpoint:r", msg("r2", "ask", "x")).unwrap();
        crate::agents::service::tests::wait_state(&dead, &t.id, TaskState::Failed).await;
    }

    #[tokio::test]
    async fn gateway_serves_https_when_a_certificate_is_configured() {
        let dir = tmpdir("g6");
        let ck = rcgen::generate_simple_self_signed(vec!["127.0.0.1".to_string(), "localhost".to_string()]).unwrap();
        let cert_path = format!("{dir}/cert.pem");
        let key_path = format!("{dir}/key.pem");
        std::fs::write(&cert_path, ck.cert.pem()).unwrap();
        std::fs::write(&key_path, ck.signing_key.serialize_pem()).unwrap();
        let (_svc, url) = start(&dir, 0, Some((cert_path.clone(), key_path))).await;
        assert!(url.starts_with("https://"));
        let ca = reqwest::Certificate::from_pem(std::fs::read(&cert_path).unwrap().as_slice()).unwrap();
        let http = reqwest::Client::builder().add_root_certificate(ca).build().unwrap();
        let c = A2aClient::with_http(http, &url, Some("alice-secret".into())).unwrap();
        c.fetch_card().await.unwrap();
        let t = c.send_message(&msg("t", "shout", "secure")).await.unwrap();
        assert_eq!(c.wait(&t.id, Duration::from_secs(5)).await.unwrap().artifacts[0].parts[0].text.as_deref(), Some("SECURE"));
        // a client that does not trust the certificate cannot talk to it
        let strict = A2aClient::new(&url, Some("alice-secret".into())).unwrap();
        assert!(strict.fetch_card().await.is_err());
        // plain HTTP against the TLS port does not work either
        let plain = A2aClient::new(&url.replace("https", "http"), None).unwrap();
        assert!(plain.fetch_card().await.is_err());
    }

    #[tokio::test]
    async fn half_configured_tls_is_an_error() {
        let svc = make_service(&tmpdir("g7"), Keypair::generate(), |c| c.tls_cert = Some("x".into())).await;
        let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        assert!(serve(GatewayState::new(svc), l).await.is_err());
    }
}
