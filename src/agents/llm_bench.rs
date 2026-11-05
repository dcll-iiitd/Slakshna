//! Real-model evaluation of the multi-agent design (M7 on GPUs).
//!
//! The honest agents answer GSM8K word problems with real LLMs served over an OpenAI-compatible
//! endpoint (vLLM on the GPUs); the adversaries and the three ways of organizing the work are those
//! of `testbed.rs`. Results are pooled over several seeds and reported with bootstrap 95%
//! confidence intervals, together with token and compute cost.

use crate::agents::testbed::{Behavior, Metrics, Mode, Solved, Solver, TaskItem, Testbed};
use async_trait::async_trait;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

const SYSTEM: &str = "You solve grade-school math word problems. Reason step by step, briefly, then finish with a final line of the form 'Answer: <number>' containing only the number.";

/// The final numeric answer of a model reply: after "Answer:", else after "####", else the last number.
pub fn parse_final_number(text: &str) -> Option<String> {
    fn first_number(s: &str) -> Option<String> {
        let b: Vec<char> = s.chars().collect();
        let mut i = 0;
        while i < b.len() {
            if b[i].is_ascii_digit() || (b[i] == '-' && i + 1 < b.len() && b[i + 1].is_ascii_digit()) {
                let mut j = i + 1;
                while j < b.len() && (b[j].is_ascii_digit() || b[j] == ',' || (b[j] == '.' && j + 1 < b.len() && b[j + 1].is_ascii_digit())) {
                    j += 1;
                }
                return Some(normalize(&b[i..j].iter().collect::<String>()));
            }
            i += 1;
        }
        None
    }
    fn last_number(s: &str) -> Option<String> {
        let mut last = None;
        let mut rest = s;
        while let Some(n) = first_number(rest) {
            // advance past this number
            let pos = rest.find(|c: char| c.is_ascii_digit()).unwrap_or(0);
            let tail = &rest[pos..];
            let end = tail.find(|c: char| !(c.is_ascii_digit() || c == ',' || c == '.')).unwrap_or(tail.len());
            rest = &tail[end..];
            last = Some(n);
        }
        last
    }
    let lower = text.to_lowercase();
    for marker in ["answer:", "answer is", "####"] {
        if let Some(pos) = lower.rfind(marker) {
            if let Some(n) = first_number(&text[pos + marker.len()..]) {
                return Some(n);
            }
        }
    }
    last_number(text)
}

/// "1,234.50" -> "1234.5"; "18.0" -> "18".
pub fn normalize(n: &str) -> String {
    let mut s = n.replace(',', "");
    if s.contains('.') {
        s = s.trim_end_matches('0').trim_end_matches('.').to_string();
    }
    if s == "-0" || s.is_empty() {
        s = "0".into();
    }
    s
}

pub struct LlmSolver {
    pub http: reqwest::Client,
    pub base_url: String,
    pub model: String,
    pub temperature: f64,
    pub max_tokens: u64,
}

impl LlmSolver {
    pub fn new(base_url: &str, model: &str, temperature: f64) -> Self {
        LlmSolver { http: reqwest::Client::builder().timeout(Duration::from_secs(300)).build().expect("client"), base_url: base_url.trim_end_matches('/').into(), model: model.into(), temperature, max_tokens: 512 }
    }
}

#[async_trait]
impl Solver for LlmSolver {
    async fn solve(&self, input: &Value) -> Result<Solved, String> {
        let q = input["question"].as_str().ok_or("question required")?;
        let body = json!({
            "model": self.model,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}],
            "temperature": self.temperature, "max_tokens": self.max_tokens,
        });
        let t0 = Instant::now();
        let resp = self.http.post(format!("{}/v1/chat/completions", self.base_url)).json(&body).send().await.map_err(|e| e.to_string())?;
        let v: Value = resp.error_for_status().map_err(|e| e.to_string())?.json().await.map_err(|e| e.to_string())?;
        let text = v["choices"][0]["message"]["content"].as_str().ok_or("no completion")?;
        let tokens = v["usage"]["total_tokens"].as_u64().unwrap_or(0);
        // An unparsable reply is a wrong answer, not an error: the agent did answer.
        Ok(Solved { answer: parse_final_number(text).unwrap_or_else(|| "none".into()), tokens, compute_ms: t0.elapsed().as_millis() as u64 })
    }
}

pub fn load_gsm8k(path: &str) -> Result<Vec<TaskItem>, String> {
    let v: Vec<Value> = serde_json::from_slice(&std::fs::read(path).map_err(|e| format!("{path}: {e}"))?).map_err(|e| format!("{path}: {e}"))?;
    v.iter()
        .map(|r| Ok(TaskItem { input: json!({"question": r["question"].as_str().ok_or("missing question")?}), truth: normalize(r["answer"].as_str().ok_or("missing answer")?) }))
        .collect()
}

#[derive(Debug, Clone)]
pub struct ModelServer {
    pub name: String,
    pub url: String,
    pub model: String,
}

#[derive(Debug, Clone)]
pub struct BenchConfig {
    pub servers: Vec<ModelServer>,
    pub workers: usize,
    pub ks: Vec<usize>,
    pub adversaries: Vec<Behavior>,
    pub problems: usize,
    pub seeds: Vec<u64>,
    pub redundancy: usize,
    pub temperature: f64,
    pub concurrency: usize,
    /// Vote share the winning answer needs (0 = plurality).
    pub quorum: f64,
    /// Probes before the first real task in probed mode.
    pub calibration: usize,
}

#[derive(Debug, Clone)]
pub struct Cell {
    pub mode: Mode,
    pub adversary: Behavior,
    pub k: usize,
    pub seed: u64,
    pub metrics: Metrics,
}

fn sample<T: Clone>(pool: &[T], n: usize, seed: u64) -> Vec<T> {
    let mut idx: Vec<usize> = (0..pool.len()).collect();
    idx.shuffle(&mut ChaCha20Rng::seed_from_u64(seed));
    idx.into_iter().take(n).map(|i| pool[i].clone()).collect()
}

/// Run every (adversary, k, seed) cell; cells run concurrently (up to `concurrency`), each on its
/// own population of `workers` agents whose honest members use the configured model servers round-robin.
pub async fn run(cfg: &BenchConfig, test: &[TaskItem], train: &[TaskItem]) -> Result<Vec<Cell>, Box<dyn std::error::Error + Send + Sync>> {
    let sem = Arc::new(tokio::sync::Semaphore::new(cfg.concurrency.max(1)));
    let mut handles = vec![];
    // k = 0 is the same for every adversary type: run it once.
    let mut combos: Vec<(Behavior, usize)> = vec![];
    for &k in &cfg.ks {
        if k == 0 {
            if combos.iter().all(|(_, kk)| *kk != 0) {
                combos.push((cfg.adversaries.first().copied().unwrap_or(Behavior::Colluding), 0));
            }
        } else {
            combos.extend(cfg.adversaries.iter().map(|a| (*a, k)));
        }
    }
    for (adv, k) in combos {
        for &seed in &cfg.seeds {
            let (cfg, sem) = (cfg.clone(), sem.clone());
            let tasks = sample(test, cfg.problems, seed);
            let probes = sample(train, 48, seed + 1_000_003);
            handles.push(tokio::spawn(async move {
                let _permit = sem.acquire_owned().await.expect("semaphore");
                let mut behaviors = vec![Behavior::Honest; cfg.workers];
                for j in 0..k {
                    behaviors[(j * cfg.workers) / k.max(1) + (cfg.workers / (2 * k.max(1)))] = adv;
                }
                let solvers: Vec<Arc<dyn Solver>> = (0..cfg.workers)
                    .map(|i| {
                        let s = &cfg.servers[i % cfg.servers.len()];
                        Arc::new(LlmSolver::new(&s.url, &s.model, cfg.temperature)) as Arc<dyn Solver>
                    })
                    .collect();
                let mut tb = Testbed::start_with(&behaviors, solvers, Duration::from_secs(240)).await.map_err(|e| e.to_string())?;
                tb.quorum = cfg.quorum;
                tb.calibration = cfg.calibration;
                // The strongest single agent for reference: the last server's first honest worker.
                let best = (0..cfg.workers).filter(|i| behaviors[*i] == Behavior::Honest && i % cfg.servers.len() == cfg.servers.len() - 1).next().unwrap_or(0);
                let mut out = vec![];
                for mode in [Mode::Single(best), Mode::Centralized, Mode::Decentralized, Mode::DecentralizedProbed] {
                    let m = tb.run(mode, adv, k, &tasks, &probes, cfg.redundancy, seed).await;
                    eprintln!("[{:?} k={k} seed={seed}] {:?}: {}/{}", adv, mode, m.successes, m.tasks);
                    out.push(Cell { mode, adversary: adv, k, seed, metrics: m });
                }
                Ok::<_, String>(out)
            }));
        }
    }
    let mut cells = vec![];
    for h in handles {
        cells.extend(h.await.map_err(|e| e.to_string())??);
    }
    Ok(cells)
}

/// Percentile bootstrap 95% CI of the mean of 0/1 outcomes.
pub fn bootstrap_ci(outcomes: &[bool], resamples: usize, seed: u64) -> (f64, f64, f64) {
    let n = outcomes.len();
    if n == 0 {
        return (0.0, 0.0, 0.0);
    }
    let mean = outcomes.iter().filter(|o| **o).count() as f64 / n as f64;
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let mut means: Vec<f64> = (0..resamples).map(|_| (0..n).filter(|_| outcomes[rng.gen_range(0..n)]).count() as f64 / n as f64).collect();
    means.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (mean, means[(resamples as f64 * 0.025) as usize], means[((resamples as f64 * 0.975) as usize).min(resamples - 1)])
}

fn mode_name(m: Mode) -> &'static str {
    match m {
        Mode::Single(_) => "single (largest model)",
        Mode::Centralized => "centralized orchestrator",
        Mode::Decentralized => "decentralized, votes only",
        Mode::DecentralizedProbed => "decentralized + probes",
    }
}

fn mode_order(m: Mode) -> u8 {
    match m { Mode::Single(_) => 0, Mode::Centralized => 1, Mode::Decentralized => 2, Mode::DecentralizedProbed => 3 }
}

pub fn report(cells: &[Cell]) -> String {
    type Key = (String, usize, u8);
    let mut groups: BTreeMap<Key, Vec<&Cell>> = BTreeMap::new();
    for c in cells {
        groups.entry((format!("{:?}", c.adversary), c.k, mode_order(c.mode))).or_default().push(c);
    }
    let mut s = String::from("| adversary | k of 9 | mode | accuracy [95% CI] | tasks | tokens/task | agent-seconds/task | msgs/task | latency ms |\n|---|---|---|---|---|---|---|---|---|\n");
    for ((adv, k, _), cs) in &groups {
        let outcomes: Vec<bool> = cs.iter().flat_map(|c| c.metrics.correct.iter().copied()).collect();
        let (m, lo, hi) = bootstrap_ci(&outcomes, 2000, 99);
        let avg = |f: &dyn Fn(&Metrics) -> f64| cs.iter().map(|c| f(&c.metrics)).sum::<f64>() / cs.len() as f64;
        s += &format!(
            "| {} | {} | {} | {:.1}% [{:.1}, {:.1}] | {} | {:.0} | {:.2} | {:.1} | {:.0} |\n",
            if *k == 0 { "none".to_string() } else { adv.clone() }, k, mode_name(cs[0].mode), m * 100.0, lo * 100.0, hi * 100.0, outcomes.len(),
            avg(&|m| m.tokens_per_task), avg(&|m| m.compute_ms_per_task) / 1000.0, avg(&|m| m.messages_per_task), avg(&|m| m.mean_latency_ms),
        );
    }
    s
}

/// Raw per-cell results as JSON (for plotting and reproduction).
pub fn to_json(cells: &[Cell]) -> Value {
    json!(cells.iter().map(|c| json!({
        "mode": mode_name(c.mode), "adversary": format!("{:?}", c.adversary), "k": c.k, "seed": c.seed,
        "tasks": c.metrics.tasks, "successes": c.metrics.successes, "correct": c.metrics.correct,
        "tokens_per_task": c.metrics.tokens_per_task, "compute_ms_per_task": c.metrics.compute_ms_per_task,
        "messages_per_task": c.metrics.messages_per_task, "mean_latency_ms": c.metrics.mean_latency_ms, "cost_per_task": c.metrics.cost_per_task,
    })).collect::<Vec<_>>())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn final_numbers_are_extracted_from_typical_model_replies() {
        let cases = [
            ("So 3 + 4 = 7.\nAnswer: 7", "7"),
            ("... Answer: $1,234.50", "1234.5"),
            ("therefore the total is 18.00 dollars. Answer: 18.0", "18"),
            ("The answer is -5.", "-5"),
            ("steps...\n#### 42", "42"),
            ("She has 12 apples and eats 4, leaving 8 apples", "8"),
            ("Answer: 0.50", "0.5"),
            ("first 5 then 10 then Answer: 15 (not 99)", "15"),
        ];
        for (text, want) in cases {
            assert_eq!(parse_final_number(text).as_deref(), Some(want), "{text}");
        }
        assert_eq!(parse_final_number("no digits here"), None);
        assert_eq!(normalize("007"), "007", "leading zeros untouched; only decimals/commas normalize");
        assert_eq!(normalize("-0.0"), "0");
    }

    #[test]
    fn bootstrap_interval_brackets_the_mean_and_shrinks_with_data() {
        let few: Vec<bool> = (0..20).map(|i| i % 2 == 0).collect();
        let many: Vec<bool> = (0..2000).map(|i| i % 2 == 0).collect();
        let (m1, lo1, hi1) = bootstrap_ci(&few, 2000, 1);
        let (m2, lo2, hi2) = bootstrap_ci(&many, 2000, 1);
        assert!((m1 - 0.5).abs() < 1e-9 && (m2 - 0.5).abs() < 1e-9);
        assert!(lo1 < 0.5 && hi1 > 0.5 && lo2 < 0.5 + 1e-9 && hi2 > 0.5 - 1e-9);
        assert!((hi2 - lo2) < (hi1 - lo1) / 3.0, "{} vs {}", hi2 - lo2, hi1 - lo1);
        assert_eq!(bootstrap_ci(&[], 10, 1), (0.0, 0.0, 0.0));
        let all_true = vec![true; 50];
        assert_eq!(bootstrap_ci(&all_true, 100, 1), (1.0, 1.0, 1.0));
    }

    #[test]
    fn sampling_is_deterministic_per_seed_and_sized() {
        let pool: Vec<u32> = (0..100).collect();
        assert_eq!(sample(&pool, 10, 5), sample(&pool, 10, 5));
        assert_ne!(sample(&pool, 10, 5), sample(&pool, 10, 6));
        assert_eq!(sample(&pool, 500, 5).len(), 100);
    }

    /// The same pipeline as the GPU run, with a stand-in solver, so the harness itself is tested without a GPU.
    #[tokio::test]
    async fn the_harness_runs_cells_and_reports_with_intervals() {
        use crate::agents::testbed::{sum_tasks, SumSolver};
        let behaviors = vec![Behavior::Honest; 6];
        let solvers: Vec<Arc<dyn Solver>> = (0..6).map(|_| Arc::new(SumSolver) as Arc<dyn Solver>).collect();
        let tb = Testbed::start_with(&behaviors, solvers, Duration::from_secs(5)).await.unwrap();
        let tasks = sum_tasks(10, 1);
        let m = tb.run(Mode::Decentralized, Behavior::Colluding, 0, &tasks, &[], 3, 1).await;
        assert_eq!((m.successes, m.correct.len()), (10, 10));
        let cell = Cell { mode: Mode::Decentralized, adversary: Behavior::Colluding, k: 0, seed: 1, metrics: m };
        let table = report(&[cell.clone()]);
        assert!(table.contains("100.0% [100.0, 100.0]") && table.contains("decentralized, votes only"), "{table}");
        assert_eq!(to_json(&[cell])[0]["successes"], 10);
    }
}
