//! Auditor spot checks (A4): probabilistic verification of scores computed on data
//! nobody else can see.
//!
//! The data owner serves the `audit` skill, but only to principals listed as auditors for
//! that dataset (a trusted-auditor model: sampled rows are released to them, with Merkle
//! proofs to the data commitment in the finding). An auditor picks a random sample with its
//! own secret seed, re-scores it with the same candidate, checks every row against the
//! committed root, and compares the sample score with the claimed score within a
//! statistical tolerance. An inflated claim, or rows that are not in the committed data, is
//! caught with probability that grows with the sample size.

use crate::agents::a2a::{AgentSkill, Artifact, Message, Part, Role, Task, TaskState};
use crate::agents::evaluator::{sample_indices, spot_check, Attestation, Predictor, Row, SpotCheck};
use crate::agents::merkle::Proof;
use crate::agents::protocol::{request, subscribe, AgentRequest, AgentResponse, WireEvent};
use crate::agents::service::{Skill, SkillCtx, SkillError};
use crate::agents::skills::{input_json, EvalRegistry};
use async_trait::async_trait;
use base64::Engine;
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::Duration;

fn b64() -> base64::engine::GeneralPurpose {
    base64::engine::general_purpose::STANDARD
}

pub struct AuditSkill {
    pub registry: Arc<EvalRegistry>,
}

#[async_trait]
impl Skill for AuditSkill {
    fn descriptor(&self) -> AgentSkill {
        AgentSkill {
            id: "audit".into(),
            name: "Dataset audit".into(),
            description: "Releases sampled rows with Merkle proofs to authorized auditors only".into(),
            tags: vec!["audit".into(), "private-data".into()],
            examples: vec![],
            input_modes: vec!["application/json".into()],
            output_modes: vec!["application/json".into()],
        }
    }

    async fn run(&self, ctx: &SkillCtx, _task: &Task, input: &Message) -> Result<Vec<Artifact>, SkillError> {
        let req = input_json(input)?;
        let root = req["root"].as_str().ok_or_else(|| SkillError::BadInput("'root' required".into()))?;
        let svc = self.registry.dataset_by_root(root).ok_or_else(|| SkillError::BadInput("no dataset with that commitment".into()))?;
        let data = if req["info"].as_bool() == Some(true) {
            // Only authorized auditors may even learn the size.
            svc.audit_sample(&ctx.requester, &[]).map_err(|e| SkillError::Failed(e.to_string()))?;
            json!({"n": svc.len()})
        } else {
            let idx: Vec<u64> = req["indices"].as_array().ok_or_else(|| SkillError::BadInput("'indices' required".into()))?.iter().filter_map(|v| v.as_u64()).collect();
            let sample = svc.audit_sample(&ctx.requester, &idx).map_err(|e| SkillError::Failed(e.to_string()))?;
            json!({"rows": sample.iter().map(|(r, p)| json!({"input": b64().encode(&r.input), "label": b64().encode(&r.label), "proof": p})).collect::<Vec<_>>()})
        };
        Ok(vec![Artifact { artifact_id: "audit".into(), name: None, description: None, parts: vec![Part::data(data)], metadata: None }])
    }
}

#[derive(Debug, thiserror::Error)]
pub enum AuditError {
    #[error("could not reach the data owner")]
    Unreachable,
    #[error("data owner refused: {0}")]
    Refused(String),
    #[error("malformed audit response")]
    Malformed,
}

async fn ask(endpoint: &iroh::Endpoint, owner: iroh::EndpointId, data: Value, timeout: Duration) -> Result<Value, AuditError> {
    let msg = Message { message_id: format!("audit-{:032x}", rand::random::<u128>()), role: Role::User, parts: vec![Part::data(data)], context_id: None, task_id: None, metadata: Some(json!({"skill": "audit"})), extensions: vec![], reference_task_ids: vec![] };
    let run = async {
        let task = match request(endpoint, owner, &AgentRequest::SendMessage { message: msg }).await.map_err(|_| AuditError::Unreachable)? {
            AgentResponse::Task { task } => task,
            AgentResponse::Error { message } => return Err(AuditError::Refused(message)),
            _ => return Err(AuditError::Malformed),
        };
        let mut rx = subscribe(endpoint, owner, &task.id).await.map_err(|_| AuditError::Unreachable)?;
        let (mut data, mut fail) = (None, None);
        while let Some(ev) = rx.recv().await {
            match ev {
                WireEvent::Artifact { artifact, .. } => data = artifact.parts.first().and_then(|p| p.data.clone()),
                WireEvent::Status { status, is_final: true, .. } => {
                    if status.state != TaskState::Completed {
                        fail = Some(status.message.and_then(|m| m.parts.first().and_then(|p| p.text.clone())).unwrap_or_else(|| format!("{:?}", status.state)));
                    }
                    break;
                }
                _ => {}
            }
        }
        match (data, fail) {
            (Some(d), None) => Ok(d),
            (_, Some(f)) => Err(AuditError::Refused(f)),
            _ => Err(AuditError::Malformed),
        }
    };
    tokio::time::timeout(timeout, run).await.map_err(|_| AuditError::Unreachable)?
}

/// Audit one claimed score against the data owner. `sample_size` rows are drawn with
/// `seed` (keep it secret from the owner). Returns the spot-check verdict.
pub async fn audit_claim(
    endpoint: &iroh::Endpoint,
    owner: iroh::EndpointId,
    claim: &Attestation,
    candidate: Arc<dyn Predictor + Send + Sync>,
    sample_size: usize,
    seed: u64,
    tolerance: f64,
) -> Result<SpotCheck, AuditError> {
    let timeout = Duration::from_secs(120);
    let info = ask(endpoint, owner, json!({"root": claim.data_merkle_root, "info": true}), timeout).await?;
    let n = info["n"].as_u64().ok_or(AuditError::Malformed)?;
    if n == 0 {
        return Err(AuditError::Malformed);
    }
    let idx = sample_indices(seed, n, sample_size);
    let resp = ask(endpoint, owner, json!({"root": claim.data_merkle_root, "indices": idx}), timeout).await?;
    let rows = resp["rows"].as_array().ok_or(AuditError::Malformed)?;
    if rows.len() != idx.len() {
        return Err(AuditError::Malformed);
    }
    let mut sample: Vec<(Row, Proof)> = vec![];
    for (v, want) in rows.iter().zip(&idx) {
        let proof: Proof = serde_json::from_value(v["proof"].clone()).map_err(|_| AuditError::Malformed)?;
        if proof.index != *want {
            return Ok(SpotCheck::BadProof { index: proof.index }); // owner returned different rows than asked for
        }
        let dec = |k: &str| v[k].as_str().and_then(|s| b64().decode(s).ok()).ok_or(AuditError::Malformed);
        sample.push((Row { input: dec("input")?, label: dec("label")? }, proof));
    }
    // The proofs bind the leaf count to the root, so trust the size only through them.
    let mut claim = claim.clone();
    claim.n_examples = n;
    // Re-scoring may run a sandboxed candidate (blocking): keep it off the async threads.
    let inputs: Vec<Vec<u8>> = sample.iter().map(|(r, _)| r.input.clone()).collect();
    let predictions: Vec<Vec<u8>> = tokio::task::spawn_blocking(move || {
        let refs: Vec<&[u8]> = inputs.iter().map(Vec::as_slice).collect();
        candidate.predict_all(&refs)
    })
    .await
    .map_err(|_| AuditError::Malformed)?;
    let replay = Replay { preds: sample.iter().map(|(r, _)| r.input.clone()).zip(predictions).collect() };
    Ok(spot_check(&claim, &sample, &replay, tolerance, 0.0))
}

/// Predictions already computed for specific inputs.
struct Replay {
    preds: Vec<(Vec<u8>, Vec<u8>)>,
}
impl Predictor for Replay {
    fn predict(&self, input: &[u8]) -> Vec<u8> {
        self.preds.iter().find(|(i, _)| i == input).map(|(_, p)| p.clone()).unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agents::evaluator::{EvaluatorConfig, EvaluatorService, Metric};
    use crate::agents::protocol::{AgentProtocol, AgentSlot, AGENT_ALPN};
    use crate::agents::service::tests::{make_service, tmpdir};
    use crate::agents::skills::{EvalRegistry, EvaluateSkill};
    use crate::identity::Keypair;
    use iroh::address_lookup::MemoryLookup;
    use iroh::endpoint::presets;
    use iroh::protocol::Router;
    use iroh::RelayMode;

    fn rows(n: usize) -> Vec<Row> {
        (0..n).map(|i| Row { input: format!("{i}").into_bytes(), label: format!("{}", i % 3).into_bytes() }).collect()
    }

    fn model(pct: usize) -> impl Fn(&[u8]) -> Vec<u8> + Send + Sync {
        move |inp: &[u8]| {
            let i: usize = std::str::from_utf8(inp).unwrap().parse().unwrap();
            if i % 100 < pct { format!("{}", i % 3).into_bytes() } else { b"x".to_vec() }
        }
    }

    struct Owner { router: Router, ep: iroh::Endpoint, reg: Arc<EvalRegistry>, svc: Arc<crate::agents::service::AgentService> }

    async fn owner(lookup: &MemoryLookup, auditor: &str) -> Owner {
        let kp = Keypair::generate();
        let svc = make_service(&tmpdir("audit"), kp.clone(), |_| {}).await;
        let reg = Arc::new(EvalRegistry::default());
        let mut cfg = EvaluatorConfig::new(Metric::Accuracy);
        cfg.auditors.insert(auditor.to_string());
        cfg.queries_per_requester = 100;
        reg.add_dataset("private", EvaluatorService::new(rows(1000), cfg).unwrap());
        reg.add_predictor("m", Arc::new(model(70)));
        svc.register_skill(Arc::new(AuditSkill { registry: reg.clone() }));
        svc.register_skill(Arc::new(EvaluateSkill { registry: reg.clone(), allowed_images: vec![], runtime: crate::agents::sandbox::Runtime::Podman, work_root: std::env::temp_dir() }));
        let ep = iroh::Endpoint::builder(presets::Minimal).relay_mode(RelayMode::Disabled).alpns(vec![AGENT_ALPN.to_vec()]).address_lookup(lookup.clone()).bind().await.unwrap();
        let slot: AgentSlot = Default::default();
        let _ = slot.set(svc.clone());
        let router = Router::builder(ep.clone()).accept(AGENT_ALPN, AgentProtocol::new(svc.history.clone(), kp.node_id().to_string(), None).with_agent_slot(slot)).spawn();
        lookup.add_endpoint_info(ep.addr());
        Owner { router, ep, reg, svc }
    }

    #[tokio::test]
    async fn an_authorized_auditor_catches_an_inflated_score_and_accepts_an_honest_one() {
        let lookup = MemoryLookup::new();
        let auditor_ep = iroh::Endpoint::builder(presets::Minimal).relay_mode(RelayMode::Disabled).address_lookup(lookup.clone()).bind().await.unwrap();
        let o = owner(&lookup, &format!("endpoint:{}", auditor_ep.id())).await;
        let m: Arc<dyn Predictor + Send + Sync> = Arc::new(model(70));
        let honest = o.reg.dataset("private").unwrap().evaluate("x", m.as_ref()).unwrap();
        assert!((honest.value - 0.7).abs() < 0.01);

        // honest claim: consistent
        let v = audit_claim(&auditor_ep, o.ep.id(), &honest, m.clone(), 60, 1234, 0.02).await.unwrap();
        assert!(matches!(v, SpotCheck::Consistent { .. }), "{v:?}");
        // the owner inflates 0.70 -> 0.95: caught
        let mut inflated = honest.clone();
        inflated.value = 0.95;
        let v = audit_claim(&auditor_ep, o.ep.id(), &inflated, m.clone(), 60, 1234, 0.02).await.unwrap();
        assert!(matches!(v, SpotCheck::Inconsistent { .. }), "{v:?}");
        // a claim over a different data commitment names no dataset
        let mut wrong_root = honest.clone();
        wrong_root.data_merkle_root = "0".repeat(64);
        assert!(matches!(audit_claim(&auditor_ep, o.ep.id(), &wrong_root, m.clone(), 10, 1, 0.02).await, Err(AuditError::Refused(_))));
        let _ = o.router.shutdown().await;
        let _ = o.svc;
    }

    #[tokio::test]
    async fn only_listed_auditors_get_anything_and_rows_are_bounded() {
        let lookup = MemoryLookup::new();
        let stranger = iroh::Endpoint::builder(presets::Minimal).relay_mode(RelayMode::Disabled).address_lookup(lookup.clone()).bind().await.unwrap();
        let o = owner(&lookup, "endpoint:someone-else").await;
        let m: Arc<dyn Predictor + Send + Sync> = Arc::new(model(70));
        let claim = o.reg.dataset("private").unwrap().evaluate("x", m.as_ref()).unwrap();
        let e = audit_claim(&stranger, o.ep.id(), &claim, m.clone(), 10, 1, 0.02).await.unwrap_err();
        assert!(matches!(e, AuditError::Refused(ref msg) if msg.contains("not an authorized auditor")), "{e:?}");
        // a listed auditor still cannot pull more rows than max_audit_rows (64) at once
        let listed = iroh::Endpoint::builder(presets::Minimal).relay_mode(RelayMode::Disabled).address_lookup(lookup.clone()).bind().await.unwrap();
        let o2 = owner(&lookup, &format!("endpoint:{}", listed.id())).await;
        let claim2 = o2.reg.dataset("private").unwrap().evaluate("x", m.as_ref()).unwrap();
        let e = audit_claim(&listed, o2.ep.id(), &claim2, m.clone(), 200, 1, 0.02).await.unwrap_err();
        assert!(matches!(e, AuditError::Refused(ref msg) if msg.contains("sample too large")), "{e:?}");
        let _ = (o.router.shutdown().await, o2.router.shutdown().await);
    }
}

/// Production auditor for the research loop: finds the data owner's endpoint in the verified
/// directory, runs the candidate patch in the sandbox on the sampled rows, and compares.
pub struct NetworkAuditor {
    pub svc: Arc<crate::agents::service::AgentService>,
    pub endpoint: iroh::Endpoint,
    pub profile: crate::agents::sandbox::SandboxProfile,
    pub command: Vec<String>,
    pub base_project: std::path::PathBuf,
    pub work_root: std::path::PathBuf,
    pub sample_size: usize,
    pub tolerance: f64,
}

#[async_trait]
impl crate::agents::research::FindingAuditor for NetworkAuditor {
    async fn audit(&self, f: &crate::agents::records::Finding, author: &str, patch: &crate::agents::research::Patch) -> Option<bool> {
        let owner: iroh::EndpointId = self.svc.endpoint_of(author)?.parse().ok()?;
        let claim = Attestation { metric: f.metric.clone(), value: f.value, evaluator_code_hash: f.evaluator_code_hash.clone(), data_merkle_root: f.data_merkle_root.clone(), n_examples: 0, noised: false };
        let predictor: Arc<dyn Predictor + Send + Sync> = Arc::new(crate::agents::skills::PatchPredictor {
            profile: self.profile.clone(),
            command: self.command.clone(),
            base_project: self.base_project.clone(),
            patch: patch.clone(),
            work_root: self.work_root.clone(),
            wall_clock: Duration::from_secs(600),
            handle: tokio::runtime::Handle::current(),
        });
        // The seed is secret to this auditor: drawn fresh, never shared with the owner.
        let seed = rand::random::<u64>();
        match audit_claim(&self.endpoint, owner, &claim, predictor, self.sample_size, seed, self.tolerance).await {
            Ok(SpotCheck::Consistent { .. }) => Some(true),
            Ok(_) => Some(false),
            Err(e) => {
                tracing::debug!("audit of a finding by {author} not completed: {e}");
                None
            }
        }
    }
}
