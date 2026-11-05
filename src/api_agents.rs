//! HTTP management API for the agent layer. Everything except the public card
//! and the signed-halt endpoint requires the admin bearer key
//! (`agents.admin_key_sha256`); with no key configured those endpoints are off.

use crate::agents::graph::EvidenceGraph;
use crate::agents::safety::HaltMsg;
use crate::api::SharedState;
use axum::extract::{Path, Query, State as AxumState};
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{IntoResponse, Json, Response};
use axum::routing::{get, post};
use axum::Router;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::HashMap;

pub fn routes() -> Router<SharedState> {
    Router::new()
        .route("/agents/card", get(card))
        .route("/agents/tasks", get(tasks))
        .route("/agents/tasks/:id", get(task))
        .route("/agents/directory", get(directory))
        .route("/agents/approvals", get(approvals))
        .route("/agents/approvals/:id/approve", post(approve))
        .route("/agents/programs", get(programs).post(start_program))
        .route("/agents/programs/:id", get(program))
        .route("/agents/programs/:id/gate", post(gate))
        .route("/agents/artifacts/:hash", get(artifact))
        .route("/agents/programs/:id/debug", get(program_debug))
        .route("/agents/halt", post(halt))
        .route("/agents/delegate", post(delegate))
        .route("/agents/team", get(team))
        .route("/agents/send", post(send_mail))
        .route("/agents/mailbox", get(mailbox))
        .route("/agents/blackboard/:collab", get(board).post(board_post))
        .route("/agents/chain/:task", get(chain))
        .route("/agents/status", get(status))
}

fn disabled() -> Response {
    (StatusCode::NOT_FOUND, Json(json!({"error": "agents are not enabled on this node"}))).into_response()
}

#[allow(clippy::result_large_err)]
fn admin(st: &SharedState, headers: &HeaderMap) -> Result<(), Response> {
    let Some(want) = st.config.agents.admin_key_sha256.as_ref() else {
        return Err((StatusCode::FORBIDDEN, Json(json!({"error": "admin API disabled: set agents.admin_key_sha256"}))).into_response());
    };
    let got = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .map(|t| hex::encode(Sha256::digest(t.trim().as_bytes())));
    match got {
        // Compare digests of equal length; the presented secret is hashed first.
        Some(g) if g.eq_ignore_ascii_case(want) => Ok(()),
        _ => Err((StatusCode::UNAUTHORIZED, [(header::WWW_AUTHENTICATE, "Bearer")], Json(json!({"error": "unauthorized"}))).into_response()),
    }
}

async fn card(AxumState(st): AxumState<SharedState>) -> Response {
    match &st.agents {
        Some(a) => Json(serde_json::to_value(a.card()).unwrap()).into_response(),
        None => disabled(),
    }
}

async fn tasks(AxumState(st): AxumState<SharedState>, headers: HeaderMap) -> Response {
    if let Err(r) = admin(&st, &headers) { return r; }
    match &st.agents {
        Some(a) => Json(json!({"tasks": a.tasks.list()})).into_response(),
        None => disabled(),
    }
}

async fn task(AxumState(st): AxumState<SharedState>, headers: HeaderMap, Path(id): Path<String>) -> Response {
    if let Err(r) = admin(&st, &headers) { return r; }
    match &st.agents {
        Some(a) => match a.tasks.get(&id) {
            Ok(t) => Json(serde_json::to_value(t).unwrap()).into_response(),
            Err(e) => (StatusCode::NOT_FOUND, Json(json!({"error": e.to_string()}))).into_response(),
        },
        None => disabled(),
    }
}

async fn directory(AxumState(st): AxumState<SharedState>, headers: HeaderMap, Query(q): Query<HashMap<String, String>>) -> Response {
    if let Err(r) = admin(&st, &headers) { return r; }
    let Some(a) = &st.agents else { return disabled() };
    let rows = match q.get("skill") {
        Some(s) => a.search_agents(s),
        None => a.all_agents(),
    };
    Json(json!({"agents": rows.into_iter().map(|(n, k, c)| json!({"node_id": n, "agent_key": k, "card": c})).collect::<Vec<_>>()})).into_response()
}

async fn approvals(AxumState(st): AxumState<SharedState>, headers: HeaderMap) -> Response {
    if let Err(r) = admin(&st, &headers) { return r; }
    match &st.agents {
        Some(a) => Json(json!({"pending": crate::agents::service::pending_approvals(a).into_iter().map(|(id, task)| json!({"approval": id, "task": task})).collect::<Vec<_>>()})).into_response(),
        None => disabled(),
    }
}

async fn approve(AxumState(st): AxumState<SharedState>, headers: HeaderMap, Path(id): Path<u64>) -> Response {
    if let Err(r) = admin(&st, &headers) { return r; }
    match &st.agents {
        Some(a) => match a.approve(id, "admin") {
            Ok(()) => Json(json!({"approved": id})).into_response(),
            Err(e) => (StatusCode::BAD_REQUEST, Json(json!({"error": e}))).into_response(),
        },
        None => disabled(),
    }
}

/// Lanes, evidence root and credit table of a research program, recomputed from the log.
async fn program(AxumState(st): AxumState<SharedState>, headers: HeaderMap, Path(id): Path<String>, Query(q): Query<HashMap<String, String>>) -> Response {
    if let Err(r) = admin(&st, &headers) { return r; }
    let h = st.history.read().await;
    let recs: Vec<&crate::history::UpdateRecord> = h.peer_updates.values().flatten().collect();
    let g = EvidenceGraph::from_records(recs);
    let Some(lanes) = g.lanes(&id) else {
        return (StatusCode::NOT_FOUND, Json(json!({"error": "unknown program"}))).into_response();
    };
    let entry = |e: &crate::agents::graph::LaneEntry| {
        let c = g.candidate(&e.candidate);
        json!({"candidate": e.candidate, "score": e.score, "evaluators": e.evaluators,
               "diff_hash": c.map(|c| c.body.diff_hash.clone()), "parents": c.map(|c| c.body.parents.clone()), "author": c.map(|c| c.author.clone()),
               "agenda_item": c.and_then(|c| c.body.agenda_item.clone())})
    };
    let credit = crate::agents::credit::compute(&g.lineage(&id), &g.gem_targets(&id), &crate::agents::credit::CreditRule::standard(), &g.evidence_root(&id));
    let candidates = g.candidates_of(&id);
    let n_findings: usize = candidates.iter().map(|c| g.findings_of(&c.id).len()).sum();
    let generation = h
        .provenance(crate::agents::records::KIND_AGENDA)
        .into_iter()
        .filter_map(|r| match &r.kind {
            crate::history::RecordKind::Provenance { body, .. } if body["program"] == id.as_str() => body["generation"].as_u64(),
            _ => None,
        })
        .max()
        .unwrap_or(0);
    Json(json!({
        "program": id,
        "generation": generation,
        "candidates": candidates.len(),
        "findings": n_findings,
        "evidence_root": g.evidence_root(&id),
        "lanes": {
            "incubator": lanes.incubator.iter().map(entry).collect::<Vec<_>>(),
            "frontier": lanes.frontier.iter().map(entry).collect::<Vec<_>>(),
            "gems": lanes.gems.iter().map(entry).collect::<Vec<_>>(),
        },
        "credit": {"rule": crate::agents::credit::CreditRule::VERSION, "exact": credit.exact, "per_node": credit.per_node},
        // Integer payout table for `?pool=N` units: what a milestone settlement would pay each contributor.
        "payouts": q.get("pool").and_then(|p| p.parse::<u64>().ok()).map(|pool| crate::agents::credit::payouts(&credit.per_node, pool)),
    }))
    .into_response()
}

/// Signed halt/resume. No admin key: the signature by a configured halt key is the authorization.
async fn halt(AxumState(st): AxumState<SharedState>, Json(msg): Json<Value>) -> Response {
    let Some(a) = &st.agents else { return disabled() };
    let Ok(msg) = serde_json::from_value::<HaltMsg>(msg) else {
        return (StatusCode::BAD_REQUEST, Json(json!({"error": "malformed halt message"}))).into_response();
    };
    match a.halt(msg).await {
        Ok(changed) => Json(json!({"applied": changed, "halted": a.kill.is_halted()})).into_response(),
        Err(e) => (StatusCode::FORBIDDEN, Json(json!({"error": e}))).into_response(),
    }
}

async fn status(AxumState(st): AxumState<SharedState>, headers: HeaderMap) -> Response {
    if let Err(r) = admin(&st, &headers) { return r; }
    let Some(a) = &st.agents else { return disabled() };
    Json(json!({
        "node_id": a.node_id,
        "agent_key": a.agent_public_key(),
        "skills": a.skill_ids(),
        "halted": a.kill.is_halted(),
        "tasks": a.tasks.list().len(),
    }))
    .into_response()
}

/// Hand a task to agents elsewhere in the federation: contract-net award, redundant
/// execution, reputation-weighted pooling, signed provenance for every hop.
async fn delegate(AxumState(st): AxumState<SharedState>, headers: HeaderMap, Json(body): Json<Value>) -> Response {
    if let Err(r) = admin(&st, &headers) { return r; }
    let Some(a) = &st.agents else { return disabled() };
    let Some(d) = a.delegator() else {
        return (StatusCode::SERVICE_UNAVAILABLE, Json(json!({"error": "delegation is not available yet"}))).into_response();
    };
    let Some(skill) = body["skill"].as_str() else {
        return (StatusCode::BAD_REQUEST, Json(json!({"error": "skill required"}))).into_response();
    };
    let mut req = crate::agents::delegate::DelegateRequest::new(skill, body["input"].clone());
    if let Some(n) = body["redundancy"].as_u64() { req.redundancy = (n as usize).clamp(1, 9); }
    if let Some(n) = body["max_price"].as_u64() { req.max_price = n; }
    if let Some(n) = body["timeout_secs"].as_u64() { req.timeout = std::time::Duration::from_secs(n.clamp(1, 600)); }
    if let Some(q) = body["quorum"].as_f64() { req.quorum = q.clamp(0.01, 1.0); }
    if !body["probe_truth"].is_null() { req.probe_truth = Some(body["probe_truth"].clone()); }
    match d.delegate(req).await {
        Ok(r) => Json(json!({
            "root_task": r.root_task,
            "consensus": r.consensus,
            "share": r.share,
            "cost": r.cost,
            "agents": r.rows.iter().map(|x| json!({
                "node_id": x.node_id, "price": x.price, "reputation_before": x.reputation_before,
                "state": x.state, "remote_task": x.remote_task, "answer": x.answer, "agreed": x.agreed,
            })).collect::<Vec<_>>(),
        }))
        .into_response(),
        Err(e) => (StatusCode::UNPROCESSABLE_ENTITY, Json(json!({"error": e.to_string()}))).into_response(),
    }
}

async fn programs(AxumState(st): AxumState<SharedState>, headers: HeaderMap) -> Response {
    if let Err(r) = admin(&st, &headers) { return r; }
    let h = st.history.read().await;
    let list: Vec<Value> = h
        .provenance(crate::agents::records::KIND_OBJECTIVE)
        .into_iter()
        .map(|r| json!({"program": r.hash, "owner": r.node_id, "objective": match &r.kind { crate::history::RecordKind::Provenance { body, .. } => body.clone(), _ => Value::Null }}))
        .collect();
    Json(json!({"programs": list})).into_response()
}

/// Start a research program: this node becomes its owner.
async fn start_program(AxumState(st): AxumState<SharedState>, headers: HeaderMap, Json(body): Json<Value>) -> Response {
    if let Err(r) = admin(&st, &headers) { return r; }
    let Some(node) = st.agents.as_ref().and_then(|a| a.research()) else {
        return (StatusCode::SERVICE_UNAVAILABLE, Json(json!({"error": "research loop is not enabled on this node"}))).into_response();
    };
    let objective: crate::agents::records::Objective = match serde_json::from_value(body) {
        Ok(o) => o,
        Err(e) => return (StatusCode::BAD_REQUEST, Json(json!({"error": format!("invalid objective: {e}")}))).into_response(),
    };
    match node.start_program(&objective).await {
        Ok(id) => Json(json!({"program": id})).into_response(),
        Err(e) => (StatusCode::BAD_REQUEST, Json(json!({"error": e.to_string()}))).into_response(),
    }
}

/// The owner approves or refuses a human gate (objective, budget, innovation gate).
async fn gate(AxumState(st): AxumState<SharedState>, headers: HeaderMap, Path(id): Path<String>, Json(body): Json<Value>) -> Response {
    if let Err(r) = admin(&st, &headers) { return r; }
    let Some(node) = st.agents.as_ref().and_then(|a| a.research()) else {
        return (StatusCode::SERVICE_UNAVAILABLE, Json(json!({"error": "research loop is not enabled on this node"}))).into_response();
    };
    let (Some(g), Some(ap)) = (body["gate"].as_str(), body["approved"].as_bool()) else {
        return (StatusCode::BAD_REQUEST, Json(json!({"error": "gate and approved required"}))).into_response();
    };
    match node.decide_gate(&id, body["generation"].as_u64().unwrap_or(0), g, ap).await {
        Ok(h) => Json(json!({"decision": h, "gate_open": node.gate_open(&id, g).await})).into_response(),
        Err(e) => (StatusCode::BAD_REQUEST, Json(json!({"error": e.to_string()}))).into_response(),
    }
}

/// Form a research team (researcher, coder, reviewers, evaluator) from the verified directory.
async fn team(AxumState(st): AxumState<SharedState>, headers: HeaderMap) -> Response {
    if let Err(r) = admin(&st, &headers) { return r; }
    let Some(a) = &st.agents else { return disabled() };
    match a.form_team(&crate::agents::coordination::TeamTemplate::research_team()) {
        Ok(t) => Json(json!({"team": t.into_iter().map(|m| json!({"role": m.role, "node_id": m.node_id, "agent_key": m.agent_key})).collect::<Vec<_>>()})).into_response(),
        Err(e) => (StatusCode::UNPROCESSABLE_ENTITY, Json(json!({"error": e}))).into_response(),
    }
}

async fn board(AxumState(st): AxumState<SharedState>, headers: HeaderMap, Path(collab): Path<String>) -> Response {
    if let Err(r) = admin(&st, &headers) { return r; }
    let Some(a) = &st.agents else { return disabled() };
    let b = a.blackboard(&collab).await;
    let proposals: Vec<Value> = b
        .proposals()
        .into_iter()
        .map(|p| json!({
            "id": p.id, "author": p.author, "text": p.post.text,
            "score": b.score(&p.id, &|_| 1.0),
            "replies": b.replies(&p.id).into_iter().map(|r| json!({"id": r.id, "author": r.author, "kind": r.post.kind, "text": r.post.text})).collect::<Vec<_>>(),
        }))
        .collect();
    let results: Vec<Value> = b.results().into_iter().map(|p| json!({"id": p.id, "author": p.author, "text": p.post.text, "score": b.score(&p.id, &|_| 1.0)})).collect();
    Json(json!({"collaboration": collab, "proposals": proposals, "results": results})).into_response()
}

async fn board_post(AxumState(st): AxumState<SharedState>, headers: HeaderMap, Path(collab): Path<String>, Json(body): Json<Value>) -> Response {
    if let Err(r) = admin(&st, &headers) { return r; }
    let Some(a) = &st.agents else { return disabled() };
    let kind: crate::agents::records::PostKind = match serde_json::from_value(body["kind"].clone()) {
        Ok(k) => k,
        Err(_) => return (StatusCode::BAD_REQUEST, Json(json!({"error": "kind must be proposal, critique, support or result"}))).into_response(),
    };
    match a.post(&collab, kind, body["reply_to"].as_str(), body["text"].as_str().unwrap_or("")).await {
        Ok(id) => Json(json!({"post": id})).into_response(),
        Err(e) => (StatusCode::BAD_REQUEST, Json(json!({"error": e}))).into_response(),
    }
}

/// Verify the delegation chain below a task: who asked whom, under what scope.
async fn chain(AxumState(st): AxumState<SharedState>, headers: HeaderMap, Path(task): Path<String>, Query(q): Query<HashMap<String, String>>) -> Response {
    if let Err(r) = admin(&st, &headers) { return r; }
    let Some(a) = &st.agents else { return disabled() };
    let scope: Vec<String> = q.get("scope").map(|s| s.split(',').filter(|x| !x.is_empty()).map(String::from).collect()).unwrap_or_default();
    match a.delegation_chain(&task, &scope).await {
        Ok(hops) => Json(json!({"valid": true, "hops": hops.into_iter().map(|h| json!({"delegator": h.delegator, "parent_task": h.parent_task, "sub_task": h.sub_task, "delegate": h.delegate, "scope": h.scope})).collect::<Vec<_>>()})).into_response(),
        Err(e) => (StatusCode::UNPROCESSABLE_ENTITY, Json(json!({"valid": false, "error": e.to_string()}))).into_response(),
    }
}

/// Send an A2A message to a federation peer (`endpoint:<id>`); if the peer is offline it waits
/// in the durable mailbox and is delivered, once, when the peer returns.
async fn send_mail(AxumState(st): AxumState<SharedState>, headers: HeaderMap, Json(body): Json<Value>) -> Response {
    if let Err(r) = admin(&st, &headers) { return r; }
    let Some(a) = &st.agents else { return disabled() };
    let Some(ep) = st.network.read().await.endpoint() else {
        return (StatusCode::SERVICE_UNAVAILABLE, Json(json!({"error": "network endpoint not ready"}))).into_response();
    };
    let Some(to) = body["to"].as_str() else {
        return (StatusCode::BAD_REQUEST, Json(json!({"error": "'to' (endpoint id) required"}))).into_response();
    };
    let mut msg: crate::agents::a2a::Message = match serde_json::from_value(body["message"].clone()) {
        Ok(m) => m,
        Err(e) => return (StatusCode::BAD_REQUEST, Json(json!({"error": format!("invalid message: {e}")}))).into_response(),
    };
    if let Some(skill) = body["skill"].as_str() {
        msg.metadata = Some(json!({"skill": skill}));
    }
    let id = msg.message_id.clone();
    match a.send_to_peer(&ep, to, msg).await {
        Ok(()) => {
            let delivered = a.mailbox.sent_task(to, &id);
            Json(json!({"queued": delivered.is_none(), "remote_task": delivered})).into_response()
        }
        Err(e) => (StatusCode::BAD_REQUEST, Json(json!({"error": e.to_string()}))).into_response(),
    }
}

async fn mailbox(AxumState(st): AxumState<SharedState>, headers: HeaderMap) -> Response {
    if let Err(r) = admin(&st, &headers) { return r; }
    let Some(a) = &st.agents else { return disabled() };
    Json(json!({"pending": a.mailbox.pending().into_iter().map(|p| json!({"to": p.to, "message_id": p.message.message_id, "attempts": p.attempts, "next_attempt": p.next_attempt})).collect::<Vec<_>>()})).into_response()
}

/// A candidate's patch by its content hash (decoded), from this node's artifact store or its peers.
async fn artifact(AxumState(st): AxumState<SharedState>, headers: HeaderMap, Path(hash): Path<String>) -> Response {
    if let Err(r) = admin(&st, &headers) { return r; }
    let Some(node) = st.agents.as_ref().and_then(|a| a.research()) else {
        return (StatusCode::SERVICE_UNAVAILABLE, Json(json!({"error": "research loop is not enabled on this node"}))).into_response();
    };
    match node.blobs.get(&hash).await {
        Some(b) => match crate::agents::research::Patch::from_bytes(&b) {
            Ok(p) => Json(json!({"hash": hash, "files": p.files.iter().map(|(k, v)| (k.clone(), String::from_utf8_lossy(v).to_string())).collect::<std::collections::BTreeMap<_, _>>()})).into_response(),
            Err(_) => (StatusCode::UNPROCESSABLE_ENTITY, Json(json!({"error": "artifact is not a patch"}))).into_response(),
        },
        None => (StatusCode::NOT_FOUND, Json(json!({"error": "artifact not available"}))).into_response(),
    }
}

async fn program_debug(AxumState(st): AxumState<SharedState>, headers: HeaderMap, Path(id): Path<String>) -> Response {
    if let Err(r) = admin(&st, &headers) { return r; }
    let Some(node) = st.agents.as_ref().and_then(|a| a.research()) else {
        return (StatusCode::SERVICE_UNAVAILABLE, Json(json!({"error": "research loop is not enabled on this node"}))).into_response();
    };
    Json(node.debug_state(&id, chrono::Utc::now().timestamp()).await).into_response()
}
