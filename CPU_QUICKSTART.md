# Running SLAKSHNA on a CPU (laptop) — no GPU required

The original code path needs CUDA (Ray Train + FSDP2/DDP + NCCL, a 7B model, Muon with 8-bit
CUDA kernels). This repo now also has a **CPU path** that trains a real LoRA fine-tune of a small
LLM on a laptop, with the same federated logic on top (sparsified int8 deltas, trust-weighted
aggregation, error feedback, poisoning defense). The GPU path is unchanged.

## What runs where

| | GPU path (unchanged) | CPU path (new) |
|---|---|---|
| Selected by | CUDA present / `device = "cuda"` | `[node] device = "cpu"` (or `"auto"` with no GPU) |
| Template | `node_template.yaml` | `node_template.cpu.yaml` |
| Tokenizer | `bhaskera.launcher.tokenize` (Ray Data) | `bhaskera.launcher.tokenize_cpu` |
| Trainer | `bhaskera.launcher.train` (Ray Train, FSDP2, NCCL) | `bhaskera.launcher.train_cpu` (one process, fp32) |
| Optimizer | Muon / GaLore / AdamW | AdamW (Muon/GaLore need CUDA kernels) |
| Model | OLMo-2-7B | **SmolLM2-135M-Instruct** (default) |
| Examples | `examples/{2,4,6}node` | `examples/cpu/{2,4,6}node` |

## 1. Setup (once)

```powershell
# Windows
powershell -ExecutionPolicy Bypass -File .\setup_cpu.ps1
```
```bash
# Linux / macOS / Git-Bash
bash setup_cpu.sh
```

This creates `.venv` with CPU-only PyTorch (no CUDA wheels), installs `requirements-cpu.txt` and
Bhaskera, downloads the model into `./hf_cache`, tokenizes the dataset once into `./data_cpu_cache`
(shared by all nodes), and runs a 2-step smoke test.

## 2. Run — choose one

**A. Python only (no Rust needed).** Runs exactly what the Rust node does each epoch, minus the
P2P transport. Good for verifying the ML side and for experiments:

```powershell
.\.venv\Scripts\python.exe scripts\simulate_cpu_cluster.py --config-dir examples\cpu\2node --rounds 3 --reset
.\.venv\Scripts\python.exe scripts\simulate_cpu_cluster.py --config-dir examples\cpu\4node --rounds 8 --malicious node-2
```

**B. Full P2P nodes (Rust).** Needs Rust (rustup), the MSVC "Desktop development with C++" build
tools and LLVM/clang on Windows (RocksDB + Iroh), then:

```powershell
cargo build --release
.\start_cluster.ps1 -Nodes 4          # examples\cpu\4node
.\stop_cluster.ps1
```
(`./start_cluster.sh 4 cpu` on Linux/macOS.) Allow `iiitd.exe` through Windows Firewall on private
networks. Nodes find each other with mDNS, so no peer addresses are needed.

Per-node logs: `logs/sim_<id>.log` (option A) or `logs/node<N>_cluster.log` (option B).
Loss curve: `logs/epoch_loss_tracking_cpu.csv`. Trust scores: `logs/trust_scores_new.csv`.

## 3. Choosing the client setup

| Setup | threads / node | round budget | notes |
|---|---|---|---|
| `examples/cpu/2node` | 8 | epoch 600 s, deadline 480 s | could afford a 360M model |
| `examples/cpu/3node` | 5 | epoch 600 s, deadline 480 s | verified: all 3 nodes join the mesh |
| `examples/cpu/4node` | 4 | epoch 600 s, deadline 480 s | recommended default |
| `examples/cpu/6node` | 2 | epoch 900 s, deadline 720 s | slowest; 6 nodes share 16 threads |

Each node gets `logical_cores // nodes` threads (`cpu_threads` in the TOML overrides). N processes
each using *all* cores thrashes and is far slower than N processes using a slice each.

Training stops early at 70 % of `sync_deadline_secs` (`cpu.max_train_seconds`), so a slow machine
yields fewer steps per round instead of missing the barrier. Raise `epoch_duration_secs` /
`sync_deadline_secs` rather than shrinking the model if you want full rounds on slower hardware.

## 4. Model choice

Default **`HuggingFaceTB/SmolLM2-135M-Instruct`**: 135 M params (0.9 M trainable with LoRA r=16),
~0.5 GB RAM per node in fp32, ChatML chat template (matches the dataset's `chatml` format), Llama
architecture (works with Bhaskera's introspection/LoRA unchanged). Alternatives in
`node_template.cpu.yaml`: SmolLM2-360M-Instruct, Qwen2.5-0.5B-Instruct (2 nodes only). After changing
the model delete `data_cpu_cache/` and run `python prepare_cpu.py`.

## 5. Behaviours worth knowing

* **Data is sharded per node** (`data.partition: true`): node *i* of *N* trains on rows `i::N`, so
  simulated clients hold different data. The slice comes from the node id (`node-3` → slice 2).
* **Long conversations are dropped on CPU.** ~25 % of `South_Asia.jsonl` are 700–950-token
  document-grounded prompts; at `seq_len: 256` they are truncated before the answer and would
  contribute no supervised tokens (and produce `NaN` loss). The CPU tokenizer drops rows with no
  supervised token. Raise `seq_len` (slower) to keep them.
* **CPU supports LoRA fine-tuning only.** `lora.enabled: false` (full-parameter / GaLore) exits with a
  clear error; use the GPU path for that.
* **The GPU examples still work on GPU machines.** On a machine with no GPU, `device = "auto"` (the
  default when `[node]` has no `device`) falls back to the CPU path and its small-model template
  automatically — but the GPU examples' epoch timings (600 s) are far too short for a CPU, so use
  the `examples/cpu` configs.
