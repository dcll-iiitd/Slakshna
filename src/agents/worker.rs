//! Rust side of the persistent Python agent worker (A1).
//!
//! Spawns `agent_worker.py` once and talks to it over stdin/stdout JSON lines.
//! Requests are serialized (one in flight), every call is bounded by a
//! timeout, a dead or hung worker is killed and respawned on the next call,
//! and model calls charge the agent's `BudgetTracker` with the token usage the
//! serving endpoint reports.

use crate::agents::budget::{BudgetError, BudgetTracker};
use serde_json::{json, Value};
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::Mutex;

#[derive(Debug, thiserror::Error)]
pub enum WorkerError {
    #[error("worker io: {0}")]
    Io(#[from] std::io::Error),
    #[error("worker timed out after {0:?}")]
    Timeout(Duration),
    #[error("worker exited")]
    Exited,
    #[error("worker error: {0}")]
    Remote(String),
    #[error("bad worker reply: {0}")]
    Protocol(String),
    #[error(transparent)]
    Budget(#[from] BudgetError),
}

struct Proc {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

pub struct AgentWorker {
    python: String,
    script: String,
    env: Vec<(String, String)>,
    proc: Mutex<Option<Proc>>,
    next_id: std::sync::atomic::AtomicU64,
}

impl AgentWorker {
    pub fn new(python: &str, script: &str, env: Vec<(String, String)>) -> Self {
        AgentWorker {
            python: python.into(),
            script: script.into(),
            env,
            proc: Mutex::new(None),
            next_id: 1.into(),
        }
    }

    fn spawn(&self) -> Result<Proc, WorkerError> {
        let mut child = Command::new(&self.python)
            .arg("-u")
            .arg(&self.script)
            .envs(self.env.iter().map(|(k, v)| (k, v)))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .kill_on_drop(true)
            .spawn()?;
        let stdin = child.stdin.take().ok_or(WorkerError::Exited)?;
        let stdout = BufReader::new(child.stdout.take().ok_or(WorkerError::Exited)?);
        Ok(Proc { child, stdin, stdout })
    }

    /// Process id of the live worker, if one is running.
    #[allow(dead_code)] // library surface: used by tests and integrators, or pending the settlement-chain decision
    pub async fn pid(&self) -> Option<u32> {
        self.proc.lock().await.as_ref().and_then(|p| p.child.id())
    }

    pub async fn call(&self, method: &str, params: Value, timeout: Duration) -> Result<Value, WorkerError> {
        let mut guard = self.proc.lock().await;
        // Respawn if the worker has died since the last call.
        let dead = match guard.as_mut() {
            Some(p) => p.child.try_wait()?.is_some(),
            None => true,
        };
        if dead {
            *guard = Some(self.spawn()?);
        }
        let p = guard.as_mut().unwrap();

        let id = self.next_id.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let mut line = serde_json::to_vec(&json!({"id": id, "method": method, "params": params})).unwrap();
        line.push(b'\n');

        let exchange = async {
            p.stdin.write_all(&line).await?;
            p.stdin.flush().await?;
            let mut reply = String::new();
            loop {
                reply.clear();
                if p.stdout.read_line(&mut reply).await? == 0 {
                    return Err(WorkerError::Exited);
                }
                let v: Value = serde_json::from_str(reply.trim())
                    .map_err(|e| WorkerError::Protocol(e.to_string()))?;
                // Skip any stale reply from an earlier request.
                if v["id"].as_u64() == Some(id) {
                    return Ok(v);
                }
            }
        };

        let outcome = tokio::time::timeout(timeout, exchange).await;
        let v = match outcome {
            Ok(Ok(v)) => v,
            Ok(Err(e)) => {
                *guard = None; // kill_on_drop reaps it; next call respawns
                return Err(e);
            }
            Err(_) => {
                *guard = None; // hung: kill, respawn next time
                return Err(WorkerError::Timeout(timeout));
            }
        };
        if v["ok"].as_bool() == Some(true) {
            Ok(v["result"].clone())
        } else {
            Err(WorkerError::Remote(v["error"].as_str().unwrap_or("unknown").to_string()))
        }
    }

    #[allow(dead_code)] // library surface: used by tests and integrators, or pending the settlement-chain decision
    pub async fn ping(&self) -> Result<Value, WorkerError> {
        self.call("ping", json!({}), Duration::from_secs(10)).await
    }

    /// Ask the federation serving endpoint for a chat completion, enforcing
    /// the agent's budget: refused up front if `max_tokens` cannot be
    /// afforded, then charged the usage the server actually reports.
    pub async fn chat(
        &self,
        budget: &Arc<BudgetTracker>,
        messages: Value,
        max_tokens: u64,
    ) -> Result<String, WorkerError> {
        budget.check()?;
        let (tokens_left, _) = budget.remaining();
        if max_tokens > tokens_left {
            // Hard stop: charging more than is left latches the tracker.
            budget.charge_tokens(max_tokens)?;
        }
        let timeout = budget.wall_remaining().min(Duration::from_secs(120));
        let r = self
            .call(
                "chat",
                json!({"messages": messages, "max_tokens": max_tokens, "timeout_s": timeout.as_secs_f64()}),
                timeout + Duration::from_secs(5),
            )
            .await?;
        let used = r["total_tokens"].as_u64().ok_or_else(|| WorkerError::Protocol("no token usage".into()))?;
        budget.charge_tokens(used)?;
        Ok(r["text"].as_str().unwrap_or_default().to_string())
    }

    #[allow(dead_code)] // library surface: used by tests and integrators, or pending the settlement-chain decision
    pub async fn shutdown(&self) {
        let mut g = self.proc.lock().await;
        if let Some(mut p) = g.take() {
            let _ = p.stdin.write_all(b"{\"id\":0,\"method\":\"shutdown\"}\n").await;
            let _ = tokio::time::timeout(Duration::from_secs(2), p.child.wait()).await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agents::budget::Budget;
    use tokio::io::AsyncReadExt;
    use tokio::net::TcpListener;

    fn worker(env: Vec<(String, String)>) -> AgentWorker {
        let script = format!("{}/agent_worker.py", env!("CARGO_MANIFEST_DIR"));
        AgentWorker::new("python3", &script, env)
    }

    /// Minimal OpenAI-compatible server returning a fixed completion.
    async fn serve_once(total_tokens: u64) -> String {
        let l = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = l.local_addr().unwrap();
        tokio::spawn(async move {
            loop {
                let Ok((mut s, _)) = l.accept().await else { return };
                tokio::spawn(async move {
                    let mut buf = vec![0u8; 65536];
                    let mut n = 0;
                    // read until the JSON body is complete (headers + content-length)
                    loop {
                        let r = s.read(&mut buf[n..]).await.unwrap_or(0);
                        if r == 0 { break; }
                        n += r;
                        let text = String::from_utf8_lossy(&buf[..n]).to_string();
                        if let Some(h) = text.find("\r\n\r\n") {
                            let cl = text[..h].to_lowercase().split("content-length:").nth(1)
                                .and_then(|x| x.lines().next()).and_then(|x| x.trim().parse::<usize>().ok()).unwrap_or(0);
                            if n >= h + 4 + cl { break; }
                        }
                    }
                    let body = serde_json::to_string(&json!({
                        "choices": [{"message": {"role": "assistant", "content": "pong"}}],
                        "usage": {"total_tokens": total_tokens}
                    })).unwrap();
                    let resp = format!("HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}", body.len(), body);
                    let _ = s.write_all(resp.as_bytes()).await;
                });
            }
        });
        format!("http://{}", addr)
    }

    /// A server that behaves like vLLM: lists its model, and answers 404 to any other model name.
    async fn strict_server(model: &'static str) -> String {
        let l = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = l.local_addr().unwrap();
        tokio::spawn(async move {
            loop {
                let Ok((mut s, _)) = l.accept().await else { return };
                tokio::spawn(async move {
                    let mut buf = vec![0u8; 65536];
                    let mut n = 0;
                    loop {
                        let r = s.read(&mut buf[n..]).await.unwrap_or(0);
                        if r == 0 { break; }
                        n += r;
                        let text = String::from_utf8_lossy(&buf[..n]).to_string();
                        if let Some(h) = text.find("\r\n\r\n") {
                            let cl = text[..h].to_lowercase().split("content-length:").nth(1).and_then(|x| x.lines().next()).and_then(|x| x.trim().parse::<usize>().ok()).unwrap_or(0);
                            if n >= h + 4 + cl { break; }
                        }
                    }
                    let text = String::from_utf8_lossy(&buf[..n]).to_string();
                    let (status, body) = if text.starts_with("GET /v1/models") {
                        ("200 OK", json!({"data": [{"id": model}]}).to_string())
                    } else if text.replace(": ", ":").contains(&format!("\"model\":\"{model}\"")) {
                        ("200 OK", json!({"choices": [{"message": {"content": "hello"}}], "usage": {"total_tokens": 5}}).to_string())
                    } else {
                        ("404 Not Found", "{\"error\":\"model not found\"}".to_string())
                    };
                    let resp = format!("HTTP/1.1 {status}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}", body.len(), body);
                    let _ = s.write_all(resp.as_bytes()).await;
                });
            }
        });
        format!("http://{addr}")
    }

    #[tokio::test]
    async fn the_worker_asks_for_the_model_the_server_actually_serves() {
        let url = strict_server("real-model-7b").await;
        let b = Arc::new(BudgetTracker::new(Budget { tokens: 1000, gpu_seconds: 0, wall_clock_secs: 60 }));
        // discovered from /v1/models
        let w = worker(vec![("SLAKSHNA_SERVE_URL".into(), url.clone())]);
        assert_eq!(w.chat(&b, json!([{"role": "user", "content": "hi"}]), 50).await.unwrap(), "hello");
        w.shutdown().await;
        // explicit override wins; a wrong explicit name is the server's 404, reported
        let w = worker(vec![("SLAKSHNA_SERVE_URL".into(), url.clone()), ("SLAKSHNA_SERVE_MODEL".into(), "real-model-7b".into())]);
        assert_eq!(w.chat(&b, json!([]), 50).await.unwrap(), "hello");
        w.shutdown().await;
        let w = worker(vec![("SLAKSHNA_SERVE_URL".into(), url), ("SLAKSHNA_SERVE_MODEL".into(), "other".into())]);
        assert!(matches!(w.chat(&b, json!([]), 50).await, Err(WorkerError::Remote(m)) if m.contains("404")));
        w.shutdown().await;
    }

    #[tokio::test]
    async fn worker_is_persistent_across_calls() {
        let w = worker(vec![]);
        let a = w.ping().await.unwrap();
        let b = w.ping().await.unwrap();
        assert_eq!(a["pid"], b["pid"], "same process must serve both calls");
        assert!(b["uptime_s"].as_f64().unwrap() >= a["uptime_s"].as_f64().unwrap());
        w.shutdown().await;
    }

    #[tokio::test]
    async fn unknown_method_is_a_remote_error_and_worker_survives() {
        let w = worker(vec![]);
        let e = w.call("nope", json!({}), Duration::from_secs(5)).await.unwrap_err();
        assert!(matches!(e, WorkerError::Remote(m) if m.contains("unknown method")));
        assert!(w.ping().await.is_ok());
        w.shutdown().await;
    }

    #[tokio::test]
    async fn crashed_worker_is_respawned() {
        let w = worker(vec![]);
        let pid1 = w.ping().await.unwrap()["pid"].as_u64().unwrap();
        unsafe { libc_kill(pid1 as i32) };
        tokio::time::sleep(Duration::from_millis(200)).await;
        let pid2 = w.ping().await.unwrap()["pid"].as_u64().unwrap();
        assert_ne!(pid1, pid2);
        w.shutdown().await;
    }

    unsafe fn libc_kill(pid: i32) {
        let _ = std::process::Command::new("kill").arg("-9").arg(pid.to_string()).status();
    }

    #[tokio::test]
    async fn chat_charges_reported_usage() {
        let url = serve_once(42).await;
        let w = worker(vec![("SLAKSHNA_SERVE_URL".into(), url)]);
        let b = Arc::new(BudgetTracker::new(Budget { tokens: 100, gpu_seconds: 0, wall_clock_secs: 60 }));
        let text = w.chat(&b, json!([{"role": "user", "content": "ping"}]), 50).await.unwrap();
        assert_eq!(text, "pong");
        assert_eq!(b.remaining().0, 58);
        w.shutdown().await;
    }

    #[tokio::test]
    async fn chat_refused_when_budget_cannot_cover_max_tokens() {
        let url = serve_once(1).await;
        let w = worker(vec![("SLAKSHNA_SERVE_URL".into(), url)]);
        let b = Arc::new(BudgetTracker::new(Budget { tokens: 10, gpu_seconds: 0, wall_clock_secs: 60 }));
        let e = w.chat(&b, json!([]), 50).await.unwrap_err();
        assert!(matches!(e, WorkerError::Budget(BudgetError::Tokens { .. })));
        // hard stop: nothing further goes through
        let e = w.chat(&b, json!([]), 1).await.unwrap_err();
        assert!(matches!(e, WorkerError::Budget(BudgetError::Stopped)));
        w.shutdown().await;
    }

    #[tokio::test]
    async fn unreachable_endpoint_is_reported() {
        let w = worker(vec![("SLAKSHNA_SERVE_URL".into(), "http://127.0.0.1:1".into())]);
        let b = Arc::new(BudgetTracker::new(Budget { tokens: 100, gpu_seconds: 0, wall_clock_secs: 60 }));
        let e = w.chat(&b, json!([]), 5).await.unwrap_err();
        assert!(matches!(e, WorkerError::Remote(m) if m.contains("unreachable")));
        w.shutdown().await;
    }
}
