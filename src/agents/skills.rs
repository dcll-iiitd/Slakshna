//! Skills a node can offer to other agents.
//!
//! * `chat`     - a chat completion from the federation serving endpoint, through the
//!                persistent Python worker and under the task's token budget.
//! * `evaluate` - federated evaluation (A4): scores a candidate on a private dataset
//!                and returns the typed attestation only. A candidate is either a
//!                predictor registered on this node, or a container image from the
//!                node's allow list run in the sandbox.

use crate::agents::a2a::{AgentSkill, Artifact, Message, Part, Task};
use crate::agents::evaluator::{EvaluatorConfig, EvaluatorService, Metric, Predictor, Row};
use crate::agents::records::KIND_FINDING;
use crate::agents::sandbox::{self, Runtime, SandboxSpec};
use crate::agents::service::{Skill, SkillCtx, SkillError};
use crate::agents::worker::AgentWorker;
use crate::config::DatasetConfig;
use async_trait::async_trait;
use base64::Engine;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use std::time::Duration;

fn b64() -> base64::engine::GeneralPurpose {
    base64::engine::general_purpose::STANDARD
}

/// First `data` part of the message, or the JSON in its first text part.
pub fn input_json(msg: &Message) -> Result<Value, SkillError> {
    for p in &msg.parts {
        if let Some(d) = &p.data {
            return Ok(d.clone());
        }
    }
    for p in &msg.parts {
        if let Some(t) = &p.text {
            return serde_json::from_str(t).map_err(|e| SkillError::BadInput(format!("text part is not JSON: {e}")));
        }
    }
    Err(SkillError::BadInput("message has no data or text part".into()))
}

// ---------------------------------------------------------------------------
// chat
// ---------------------------------------------------------------------------

pub struct ChatSkill {
    pub worker: Arc<AgentWorker>,
    pub max_tokens: u64,
}

#[async_trait]
impl Skill for ChatSkill {
    fn descriptor(&self) -> AgentSkill {
        AgentSkill {
            id: "chat".into(),
            name: "Chat".into(),
            description: "Chat completion from the federation's served model".into(),
            tags: vec!["llm".into(), "chat".into()],
            examples: vec![],
            input_modes: vec!["text/plain".into(), "application/json".into()],
            output_modes: vec!["text/plain".into()],
        }
    }

    async fn run(&self, ctx: &SkillCtx, _task: &Task, input: &Message) -> Result<Vec<Artifact>, SkillError> {
        // Plain text is a single user turn; a data part may carry {"messages": [...]}.
        let messages = match input.parts.iter().find_map(|p| p.data.as_ref()) {
            Some(d) => d.get("messages").cloned().ok_or_else(|| SkillError::BadInput("data part needs a 'messages' array".into()))?,
            None => {
                let text: String = input.parts.iter().filter_map(|p| p.text.as_deref()).collect::<Vec<_>>().join("\n");
                if text.is_empty() {
                    return Err(SkillError::BadInput("empty message".into()));
                }
                json!([{"role": "user", "content": text}])
            }
        };
        let text = self.worker.chat(&ctx.budget, messages, self.max_tokens).await.map_err(|e| SkillError::Failed(e.to_string()))?;
        Ok(vec![Artifact { artifact_id: "reply".into(), name: Some("reply".into()), description: None, parts: vec![Part::text(text)], metadata: None }])
    }
}

// ---------------------------------------------------------------------------
// evaluate
// ---------------------------------------------------------------------------

#[derive(Default)]
pub struct EvalRegistry {
    datasets: RwLock<HashMap<String, Arc<EvaluatorService>>>,
    predictors: RwLock<HashMap<String, Arc<dyn Predictor + Send + Sync>>>,
}

impl EvalRegistry {
    pub fn add_dataset(&self, name: &str, svc: EvaluatorService) {
        self.datasets.write().unwrap().insert(name.into(), Arc::new(svc));
    }
    #[allow(dead_code)] // library surface: used by tests and integrators, or pending the settlement-chain decision
    pub fn add_predictor(&self, name: &str, p: Arc<dyn Predictor + Send + Sync>) {
        self.predictors.write().unwrap().insert(name.into(), p);
    }
    pub fn dataset(&self, name: &str) -> Option<Arc<EvaluatorService>> {
        self.datasets.read().unwrap().get(name).cloned()
    }
    pub fn predictor(&self, name: &str) -> Option<Arc<dyn Predictor + Send + Sync>> {
        self.predictors.read().unwrap().get(name).cloned()
    }
    /// The dataset whose Merkle commitment is `root`.
    pub fn dataset_by_root(&self, root: &str) -> Option<Arc<EvaluatorService>> {
        self.datasets.read().unwrap().values().find(|d| d.data_root() == root).cloned()
    }

    pub fn dataset_names(&self) -> Vec<String> {
        let mut v: Vec<_> = self.datasets.read().unwrap().keys().cloned().collect();
        v.sort();
        v
    }

    /// Load the datasets declared in the node config.
    pub fn load_config(&self, cfgs: &[DatasetConfig]) -> Result<(), String> {
        for c in cfgs {
            let rows = load_rows(&c.path).map_err(|e| format!("dataset '{}': {e}", c.name))?;
            let metric = match c.metric.as_str() {
                "accuracy" => Metric::Accuracy,
                "clipped_mae_score" => Metric::ClippedMaeScore,
                m => return Err(format!("dataset '{}': unknown metric '{m}'", c.name)),
            };
            let mut ec = EvaluatorConfig::new(metric);
            ec.queries_per_requester = c.queries_per_requester;
            ec.epsilon_total = c.epsilon_total;
            ec.auditors = c.auditors.iter().cloned().collect();
            let svc = EvaluatorService::new(rows, ec).map_err(|e| format!("dataset '{}': {e}", c.name))?;
            self.add_dataset(&c.name, svc);
        }
        Ok(())
    }
}

/// JSON-lines rows `{"input": b64, "label": b64}`.
pub fn load_rows(path: &str) -> Result<Vec<Row>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    let mut rows = vec![];
    for (i, line) in text.lines().enumerate().filter(|(_, l)| !l.trim().is_empty()) {
        let v: Value = serde_json::from_str(line).map_err(|e| format!("{path}:{}: {e}", i + 1))?;
        let get = |k: &str| -> Result<Vec<u8>, String> {
            b64().decode(v[k].as_str().ok_or_else(|| format!("{path}:{}: missing '{k}'", i + 1))?).map_err(|e| format!("{path}:{}: {e}", i + 1))
        };
        rows.push(Row { input: get("input")?, label: get("label")? });
    }
    Ok(rows)
}

pub struct EvaluateSkill {
    pub registry: Arc<EvalRegistry>,
    pub allowed_images: Vec<String>,
    pub runtime: Runtime,
    pub work_root: PathBuf,
}

fn image_allowed(allowed: &[String], image: &str) -> bool {
    allowed.iter().any(|a| match a.strip_suffix('*') {
        Some(prefix) => image.starts_with(prefix),
        None => a == image,
    })
}

#[async_trait]
impl Skill for EvaluateSkill {
    fn descriptor(&self) -> AgentSkill {
        AgentSkill {
            id: "evaluate".into(),
            name: "Federated evaluation".into(),
            description: "Scores a candidate on this node's private datasets and returns the metric with a Merkle commitment to the data; rows never leave the node".into(),
            tags: vec!["evaluation".into(), "private-data".into()],
            examples: vec![],
            input_modes: vec!["application/json".into()],
            output_modes: vec!["application/json".into()],
        }
    }

    fn export(&self) -> (crate::agents::safety::ExportKind, &'static str) {
        (crate::agents::safety::ExportKind::Finding, "metrics")
    }

    async fn run(&self, ctx: &SkillCtx, task: &Task, input: &Message) -> Result<Vec<Artifact>, SkillError> {
        let req = input_json(input)?;
        let dataset = req["dataset"].as_str().ok_or_else(|| SkillError::BadInput("'dataset' required".into()))?;
        let svc = self.registry.dataset(dataset).ok_or_else(|| SkillError::BadInput(format!("unknown dataset '{dataset}'")))?;

        let predictor: Arc<dyn Predictor + Send + Sync> = if let Some(name) = req["candidate"].as_str() {
            self.registry.predictor(name).ok_or_else(|| SkillError::BadInput(format!("unknown candidate '{name}'")))?
        } else if let Some(image) = req["image"].as_str() {
            if !image_allowed(&self.allowed_images, image) {
                return Err(SkillError::BadInput(format!("image '{image}' is not on this node's allow list")));
            }
            Arc::new(SandboxPredictor {
                runtime: self.runtime,
                image: image.into(),
                work_root: self.work_root.clone(),
                wall_clock: ctx.budget.wall_remaining().min(Duration::from_secs(600)),
                handle: tokio::runtime::Handle::current(),
            })
        } else {
            return Err(SkillError::BadInput("name a registered 'candidate' or an allowed 'image'".into()));
        };

        let requester = ctx.requester.clone();
        let att = tokio::task::spawn_blocking(move || svc.evaluate(&requester, predictor.as_ref()))
            .await
            .map_err(|e| SkillError::Failed(e.to_string()))?
            .map_err(|e| SkillError::Failed(e.to_string()))?;

        // If the request ties the score to a run, publish it as a research.finding.
        if let (Some(run), Some(cand)) = (req["run"].as_str(), req["candidate_id"].as_str()) {
            if crate::agents::records::is_hash(run) && crate::agents::records::is_hash(cand) {
                ctx.sink.emit(KIND_FINDING, json!({
                    "run": run, "candidate": cand, "metric": att.metric, "value": att.value,
                    "evaluator_code_hash": att.evaluator_code_hash, "data_merkle_root": att.data_merkle_root,
                }));
            }
        }
        let _ = task;
        Ok(vec![Artifact { artifact_id: "attestation".into(), name: Some("attestation".into()), description: None, parts: vec![Part::data(serde_json::to_value(&att).expect("attestation serializes"))], metadata: None }])
    }
}

/// Runs a candidate container once per batch: inputs go in as `/work/in.jsonl`
/// (base64 per line), predictions are read from `/work/out.jsonl` in the same
/// order. The container has no network, a read-only root, and only its own
/// private work directory.
///
/// NOTE: needs Podman (or Apptainer) on the host; not exercised by the unit tests
/// on machines without one.
pub struct SandboxPredictor {
    pub runtime: Runtime,
    pub image: String,
    pub work_root: PathBuf,
    pub wall_clock: Duration,
    pub handle: tokio::runtime::Handle,
}

impl Predictor for SandboxPredictor {
    fn predict(&self, input: &[u8]) -> Vec<u8> {
        self.predict_all(&[input]).pop().unwrap_or_default()
    }

    fn predict_all(&self, inputs: &[&[u8]]) -> Vec<Vec<u8>> {
        let empty = || vec![vec![]; inputs.len()];
        let dir = self.work_root.join(format!("eval-{}", rand::random::<u64>()));
        if std::fs::create_dir_all(&dir).is_err() {
            return empty();
        }
        let cleanup = scopeguard(&dir);
        let body: String = inputs.iter().map(|i| b64().encode(i) + "\n").collect();
        if std::fs::write(dir.join("in.jsonl"), body).is_err() {
            return empty();
        }
        let spec = SandboxSpec::new(self.runtime, &self.image, &dir, vec![], self.wall_clock);
        let name = format!("slk-eval-{}", rand::random::<u32>());
        let res = self.handle.block_on(sandbox::run(&spec, &name));
        let ok = matches!(&res, Ok(r) if r.exit_code == Some(0) && !r.timed_out);
        if !ok {
            return empty();
        }
        let out = std::fs::read_to_string(dir.join("out.jsonl")).unwrap_or_default();
        let mut preds: Vec<Vec<u8>> = out.lines().map(|l| b64().decode(l.trim()).unwrap_or_default()).collect();
        preds.resize(inputs.len(), vec![]);
        preds.truncate(inputs.len());
        drop(cleanup);
        preds
    }
}

struct Cleanup(PathBuf);
impl Drop for Cleanup {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn scopeguard(p: &std::path::Path) -> Cleanup {
    Cleanup(p.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_allow_list_matches_exact_and_prefix() {
        let a = vec!["registry.local/cand:1".to_string(), "registry.local/team/*".to_string()];
        assert!(image_allowed(&a, "registry.local/cand:1"));
        assert!(!image_allowed(&a, "registry.local/cand:2"));
        assert!(image_allowed(&a, "registry.local/team/x:9"));
        assert!(!image_allowed(&a, "evil.example/team/x"));
        assert!(!image_allowed(&[], "anything"));
    }

    #[test]
    fn dataset_files_load_and_reject_garbage() {
        let dir = std::env::temp_dir().join(format!("slk-ds-{}", rand::random::<u64>()));
        std::fs::create_dir_all(&dir).unwrap();
        let good = dir.join("good.jsonl");
        let e = |s: &str| b64().encode(s);
        std::fs::write(&good, format!("{{\"input\":\"{}\",\"label\":\"{}\"}}\n\n{{\"input\":\"{}\",\"label\":\"{}\"}}\n", e("a"), e("1"), e("b"), e("2"))).unwrap();
        let rows = load_rows(good.to_str().unwrap()).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[1].label, b"2");
        let bad = dir.join("bad.jsonl");
        std::fs::write(&bad, "{\"input\": \"!!notb64\", \"label\": \"\"}\n").unwrap();
        assert!(load_rows(bad.to_str().unwrap()).is_err());
        assert!(load_rows("/nonexistent/x.jsonl").is_err());
    }

    #[test]
    fn config_datasets_register_and_unknown_metric_is_an_error() {
        let dir = std::env::temp_dir().join(format!("slk-ds-{}", rand::random::<u64>()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("d.jsonl");
        std::fs::write(&p, format!("{{\"input\":\"{}\",\"label\":\"{}\"}}\n", b64().encode("a"), b64().encode("1"))).unwrap();
        let cfg = |metric: &str| DatasetConfig { name: "d".into(), path: p.to_str().unwrap().into(), metric: metric.into(), queries_per_requester: 5, epsilon_total: None, auditors: vec![] };
        let reg = EvalRegistry::default();
        reg.load_config(&[cfg("accuracy")]).unwrap();
        assert_eq!(reg.dataset_names(), ["d"]);
        assert!(EvalRegistry::default().load_config(&[cfg("nope")]).is_err());
    }
}

/// Predictions from a *candidate patch*: a private worktree is made from the base project, the patch is
/// applied, the inputs are written to `/work/in.jsonl`, and the operator's predict command runs in the
/// sandbox (no network, read-only root); predictions are read from `/work/out.jsonl`.
///
/// NOTE: needs Podman (or Apptainer); not exercised by unit tests on hosts without one.
pub struct PatchPredictor {
    pub profile: sandbox::SandboxProfile,
    pub command: Vec<String>,
    pub base_project: PathBuf,
    pub patch: crate::agents::research::Patch,
    pub work_root: PathBuf,
    pub wall_clock: Duration,
    pub handle: tokio::runtime::Handle,
}

impl Predictor for PatchPredictor {
    fn predict(&self, input: &[u8]) -> Vec<u8> {
        self.predict_all(&[input]).pop().unwrap_or_default()
    }

    fn predict_all(&self, inputs: &[&[u8]]) -> Vec<Vec<u8>> {
        let empty = || vec![vec![]; inputs.len()];
        let id = format!("eval-{:016x}", rand::random::<u64>());
        let Ok(wt) = sandbox::create_worktree(&self.base_project, &self.work_root, &id) else { return empty() };
        let _cleanup = scopeguard(&wt);
        if self.patch.apply(&wt).is_err() {
            return empty();
        }
        let body: String = inputs.iter().map(|i| b64().encode(i) + "\n").collect();
        if std::fs::write(wt.join("in.jsonl"), body).is_err() {
            return empty();
        }
        let spec = self.profile.spec(&wt, self.command.clone(), self.wall_clock);
        let name = format!("slk-patch-eval-{}", rand::random::<u32>());
        match self.handle.block_on(sandbox::run(&spec, &name)) {
            Ok(r) if r.exit_code == Some(0) && !r.timed_out => {}
            _ => return empty(),
        }
        let out = std::fs::read_to_string(wt.join("out.jsonl")).unwrap_or_default();
        let mut preds: Vec<Vec<u8>> = out.lines().map(|l| b64().decode(l.trim()).unwrap_or_default()).collect();
        preds.resize(inputs.len(), vec![]);
        preds
    }
}

/// Evaluator role for the research loop: scores a candidate on this node's private dataset.
pub struct PatchEvaluator {
    pub dataset: Arc<EvaluatorService>,
    pub profile: sandbox::SandboxProfile,
    pub command: Vec<String>,
    pub base_project: PathBuf,
    pub work_root: PathBuf,
    pub wall_clock: Duration,
}

#[async_trait]
impl crate::agents::research::CandidateEvaluator for PatchEvaluator {
    async fn evaluate(&self, patch: &crate::agents::research::Patch, _run: &crate::agents::research::RunReport) -> Option<crate::agents::evaluator::Attestation> {
        let predictor = PatchPredictor {
            profile: self.profile.clone(),
            command: self.command.clone(),
            base_project: self.base_project.clone(),
            patch: patch.clone(),
            work_root: self.work_root.clone(),
            wall_clock: self.wall_clock,
            handle: tokio::runtime::Handle::current(),
        };
        let ds = self.dataset.clone();
        tokio::task::spawn_blocking(move || ds.evaluate("research-loop", &predictor).ok()).await.ok().flatten()
    }
}
