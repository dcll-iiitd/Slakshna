//! Outbound A2A client (M1): lets a Slakshna agent discover and call agents
//! outside the federation over the JSON-RPC binding, and lets outside-facing
//! tools talk to a Slakshna gateway.

use crate::agents::a2a::{AgentCard, Artifact, Message, Task, TaskState};
use crate::agents::service::{Skill, SkillCtx, SkillError};
use async_trait::async_trait;
use serde_json::{json, Value};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

#[derive(Debug, thiserror::Error)]
pub enum ClientError {
    #[error("http: {0}")]
    Http(String),
    #[error("remote error {code}: {message}")]
    Rpc { code: i64, message: String },
    #[error("bad response: {0}")]
    Protocol(String),
    #[error("agent card rejected: {0}")]
    #[allow(dead_code)] // library surface: used by tests and integrators, or pending the settlement-chain decision
    Card(String),
    #[error("timed out waiting for the task")]
    Timeout,
}

impl From<reqwest::Error> for ClientError {
    fn from(e: reqwest::Error) -> Self {
        ClientError::Http(e.to_string())
    }
}

pub struct A2aClient {
    http: reqwest::Client,
    base: String,
    bearer: Option<String>,
    next_id: AtomicU64,
}

impl A2aClient {
    pub fn new(base_url: &str, bearer: Option<String>) -> Result<Self, ClientError> {
        Self::with_http(reqwest::Client::builder().timeout(Duration::from_secs(60)).build()?, base_url, bearer)
    }

    pub fn with_http(http: reqwest::Client, base_url: &str, bearer: Option<String>) -> Result<Self, ClientError> {
        Ok(A2aClient { http, base: base_url.trim_end_matches('/').to_string(), bearer, next_id: AtomicU64::new(1) })
    }

    /// Fetch `/.well-known/agent-card.json`. If the card carries signatures at
    /// least one must verify; returns the card and whether it was signature-verified.
    #[allow(dead_code)] // library surface: used by tests and integrators, or pending the settlement-chain decision
    pub async fn fetch_card(&self) -> Result<(AgentCard, bool), ClientError> {
        let card: AgentCard = self
            .http
            .get(format!("{}/.well-known/agent-card.json", self.base))
            .send()
            .await?
            .error_for_status()?
            .json()
            .await
            .map_err(|e| ClientError::Protocol(e.to_string()))?;
        card.validate().map_err(ClientError::Card)?;
        let signed = !card.signatures.is_empty();
        if signed && card.verified_key().is_none() {
            return Err(ClientError::Card("signature does not verify".into()));
        }
        Ok((card, signed))
    }

    pub async fn call(&self, method: &str, params: Value) -> Result<Value, ClientError> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let mut req = self.http.post(format!("{}/a2a", self.base)).json(&json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}));
        if let Some(b) = &self.bearer {
            req = req.bearer_auth(b);
        }
        let resp = req.send().await?.error_for_status()?;
        let v: Value = resp.json().await.map_err(|e| ClientError::Protocol(e.to_string()))?;
        if let Some(e) = v.get("error") {
            return Err(ClientError::Rpc { code: e["code"].as_i64().unwrap_or(0), message: e["message"].as_str().unwrap_or("").into() });
        }
        v.get("result").cloned().ok_or_else(|| ClientError::Protocol("no result".into()))
    }

    pub async fn send_message(&self, message: &Message) -> Result<Task, ClientError> {
        let r = self.call("SendMessage", json!({"message": message})).await?;
        serde_json::from_value(r.get("task").cloned().ok_or_else(|| ClientError::Protocol("response has no task".into()))?).map_err(|e| ClientError::Protocol(e.to_string()))
    }

    pub async fn get_task(&self, id: &str) -> Result<Task, ClientError> {
        serde_json::from_value(self.call("GetTask", json!({"id": id})).await?).map_err(|e| ClientError::Protocol(e.to_string()))
    }

    #[allow(dead_code)] // library surface: used by tests and integrators, or pending the settlement-chain decision
    pub async fn cancel_task(&self, id: &str) -> Result<Task, ClientError> {
        serde_json::from_value(self.call("CancelTask", json!({"id": id})).await?).map_err(|e| ClientError::Protocol(e.to_string()))
    }

    /// Poll until the task is terminal or needs input.
    pub async fn wait(&self, id: &str, timeout: Duration) -> Result<Task, ClientError> {
        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            let t = self.get_task(id).await?;
            if t.status.state.is_terminal() || matches!(t.status.state, TaskState::InputRequired | TaskState::AuthRequired) {
                return Ok(t);
            }
            if tokio::time::Instant::now() >= deadline {
                return Err(ClientError::Timeout);
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }

    /// `SendStreamingMessage`: returns every StreamResponse object up to the final status.
    #[allow(dead_code)] // library surface: used by tests and integrators, or pending the settlement-chain decision
    pub async fn send_streaming(&self, message: &Message, timeout: Duration) -> Result<Vec<Value>, ClientError> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let mut req = self.http.post(format!("{}/a2a", self.base)).timeout(timeout).json(&json!({"jsonrpc": "2.0", "id": id, "method": "SendStreamingMessage", "params": {"message": message}}));
        if let Some(b) = &self.bearer {
            req = req.bearer_auth(b);
        }
        let mut resp = req.send().await?.error_for_status()?;
        let mut buf = String::new();
        let mut out = vec![];
        while let Some(chunk) = resp.chunk().await? {
            buf.push_str(&String::from_utf8_lossy(&chunk));
            while let Some(pos) = buf.find("\n\n") {
                let block: String = buf.drain(..pos + 2).collect();
                for line in block.lines() {
                    if let Some(data) = line.strip_prefix("data:") {
                        let v: Value = serde_json::from_str(data.trim()).map_err(|e| ClientError::Protocol(e.to_string()))?;
                        if let Some(e) = v.get("error") {
                            return Err(ClientError::Rpc { code: e["code"].as_i64().unwrap_or(0), message: e["message"].as_str().unwrap_or("").into() });
                        }
                        let result = v["result"].clone();
                        let is_final = result["statusUpdate"]["final"] == true
                            || result["task"]["status"]["state"].as_str().map_or(false, |s| matches!(s, "TASK_STATE_COMPLETED" | "TASK_STATE_FAILED" | "TASK_STATE_CANCELED" | "TASK_STATE_REJECTED"));
                        out.push(result);
                        if is_final {
                            return Ok(out);
                        }
                    }
                }
            }
        }
        Ok(out)
    }
}

/// Exposes an outside A2A agent as a local skill, so a Slakshna agent can
/// delegate to it. Counterparty and spending checks happen in the policy
/// engine before any skill runs.
pub struct RemoteAgentSkill {
    pub local_id: String,
    pub remote_skill: String,
    pub client: A2aClient,
    pub timeout: Duration,
}

#[async_trait]
impl Skill for RemoteAgentSkill {
    fn descriptor(&self) -> crate::agents::a2a::AgentSkill {
        crate::agents::a2a::AgentSkill {
            id: self.local_id.clone(),
            name: format!("Remote: {}", self.remote_skill),
            description: format!("Delegates to an outside A2A agent (skill '{}')", self.remote_skill),
            tags: vec!["remote".into(), "a2a".into()],
            examples: vec![],
            input_modes: vec!["text/plain".into(), "application/json".into()],
            output_modes: vec!["text/plain".into(), "application/json".into()],
        }
    }

    async fn run(&self, _ctx: &SkillCtx, task: &Task, input: &Message) -> Result<Vec<Artifact>, SkillError> {
        let mut m = input.clone();
        // A fresh id per (task, remote) keeps the remote side idempotent across our own retries.
        m.message_id = format!("{}:{}", task.id, self.remote_skill);
        m.task_id = None;
        m.context_id = None;
        m.metadata = Some(json!({"skill": self.remote_skill}));
        let sent = self.client.send_message(&m).await.map_err(|e| SkillError::Failed(e.to_string()))?;
        let done = self.client.wait(&sent.id, self.timeout).await.map_err(|e| SkillError::Failed(e.to_string()))?;
        match done.status.state {
            TaskState::Completed => Ok(done.artifacts),
            s => Err(SkillError::Failed(format!("remote task ended in {s:?}"))),
        }
    }
}
