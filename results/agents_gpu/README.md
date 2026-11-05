# Agents / multi-agent on real GPUs (gpuserver2, 8x NVIDIA H200)

Everything here was produced by the code in this repository on real hardware; raw outputs are in this folder.

## 1. Container sandbox (A1/A7), verified on real Docker + GPUs
`cargo test agents::sandbox`: the container runs as the calling user (not root), only its worktree is writable,
inputs and the root filesystem are read-only, the network is off unless policy enables it (and the same probe
reaches out once enabled), the wall clock kills the *container* (not just the client), output floods are killed
promptly, and `gpu = true` shows all 8 GPUs while `gpu_devices = "0"` shows exactly one.

## 2. Multi-agent robustness with real LLM agents (M3/M4/M7), `e1_gsm8k.json`, `figs/llm_bench.png`
9 agents backed by Qwen2.5-0.5B / 1.5B / 3B-Instruct served by vLLM on H200s, answering GSM8K test problems.
k of 9 agents are malicious and are the cheapest bidders (colluders tell one shared wrong answer; liars each tell
their own). 5 seeds x 60 problems = 300 problems per row, bootstrap 95% CIs. Agent answers come from real GPU inference
(median 0.4 s, ~200 tokens per answer).

| adversary | k | best single agent (3B)* | centralized orchestrator | decentralized, votes only | decentralized + probes |
|---|---|---|---|---|---|
| none | 0 | 79.3% [75.0, 83.7] | 55.7% [50.0, 61.0] | 61.3% [55.7, 67.0] | 76.3% [71.7, 81.3] |
| colluding | 2 | 78.3% | 47.3% [41.7, 53.0] | **0.0%** | 75.3% [70.7, 80.0] |
| colluding | 3 | 77.7% | 35.0% [29.7, 40.7] | **0.0%** | 80.7% [76.3, 85.0] |
| lying | 2 | 77.7% | 45.3% [39.7, 51.0] | 74.7% [69.3, 79.7] | 74.7% [69.7, 79.7] |
| lying | 3 | 79.0% | 35.0% [29.7, 40.7] | 58.7% [53.0, 64.3] | 81.0% [76.3, 85.3] |

\* an oracle: the requester is assumed to know which agent is best. The decentralized system is not told.
Cost of verification (no attackers): ~950 tokens/task vs ~200 for one agent, ~30 vs 2 messages per task.

What this supports: pooled answers with reputation anchored to ground-truth probes track the best single agent
(within CI) without knowing in advance which agent that is, and stay there under attack, while an unverified
central orchestrator falls to 35%. What it does not show: it does not beat the best single agent, and the
adversaries are simple, simulated behaviours.

Two design findings from this experiment (both fixed in code, tests added):
* agreement-based reputation rewards *consistency*, not correctness: perfectly consistent colluders out-vote noisy
  honest agents and, in the first version, regained trust even after probes. Probed mode now learns **only** from
  ground-truth probes (`testbed.rs`).
* vote-only pooling is captured completely by cheap colluders (0%).

Reproduce: `iiitd llm-bench --servers "small=URL|Qwen/Qwen2.5-0.5B-Instruct,mid=URL|Qwen/Qwen2.5-1.5B-Instruct,large=URL|Qwen/Qwen2.5-3B-Instruct" --test gsm8k_test.json --train gsm8k_train.json --problems 60 --seeds 5 --ks 0,2,3 --adversaries colluding,lying --calibration 8 --out e1_gsm8k.json`;
servers: `serve.sh` (vLLM 0.32). Figure: `scripts/plot_llm_bench.py`.

## 3. Federated research program on 7 real node processes + GPU containers (A1-A6, M1-M2), `research_gpu.json`
`scripts/e2e_research_gpu.py`: 3 planners/executors, 3 data owners (each scores on its own private 150-problem GSM8K split,
inside a GPU container, publishing only accuracy + a Merkle commitment) and an auditor. Planner and candidate-writer are
Qwen2.5-3B (vLLM). Candidates may change only `prompt.json` of a Qwen2.5-0.5B GSM8K program (`research_project/`);
every run and every evaluation executes in the `slakshna-gpu` Docker image (torch 2.5.1+cu121) with `--network=none`,
read-only root, one GPU. 5 generations finished in 654 s; 8 candidates, 24 private findings.

Held-out check (869 problems no evaluator saw, same container), paired exact McNemar test vs the baseline prompt:

| prompt | held-out accuracy [95% CI] | delta | p |
|---|---|---|---|
| baseline | 40.7% [37.5, 44.0] | | |
| top-1 by private score | 43.5% [40.2, 46.8] | +2.8 | 0.14 |
| top-2 | 39.8% [36.6, 43.1] | -0.9 | 0.64 |
| top-3 | 42.1% [38.9, 45.4] | +1.4 | 0.48 |

**Honest reading:** the loop ran end to end on real GPUs (VRF committee, claims, GPU-container runs, private
evaluation, spot-audit machinery, Shapley credit with an exactly-summing payout table), but this run did **not**
find a prompt that significantly beats the baseline. Candidates looked better on the private splits (up to 45-47%)
than on held-out data: selecting the maximum over few candidates scored on 150 problems is a winner's-curse effect, and
a 3B planner/writer is a weak research agent. An earlier, shorter run (`research_gpu_prelim.json`, 2 generations)
had its top candidate at 45.8% [42.5, 49.1] held-out vs 40.7%, unpaired, and is not reproduced here. Claims about
research *quality* need more programs, seeds and a stronger model; claims about the *system* (it runs, is
verifiable and attributable, survives failures) are what this run supports.
