# SLAKSHNA — Decentralized, Serverless Federated Inference on the Edge

A **Peer-to-Peer Federated Inference Framework** built in **Rust** (Iroh QUIC mesh, gossip metadata dissemination, BLAKE3 blob transfer) and integrated with a high-performance Python inference sidecar (**Bhaskera-serve** + Hugging Face / PEFT). 

**SLAKSHNA** enables decentralized, privacy-preserving, collaborative inference across heterogeneous edge nodes without centralized multi-LoRA servers, cloud coordinators, or prompt leakage. Any participating node can answer complex user queries with the most capable specialist adapter across the entire federation—fetching weights on demand, caching locally, or ensembling multi-node posteriors with zero central infrastructure.

---

## Key Highlights & Architectural Innovations

- **100% Prompt Privacy by Design (Zero Prompt Leakage)**  
  Traditional cloud and multi-tenant servers (e.g., vLLM multi-LoRA) require clients to transmit sensitive user queries across the internet. In SLAKSHNA, **prompts never leave the local node (0 bytes transferred)**. Instead of sending data to the model, SLAKSHNA routes and fetches lightweight LoRA adapters to the edge device holding the prompt.

- **Separation of Metadata Gossip vs. Weight Transfer**  
  Metadata is lightweight; model weights are not. Nodes continuously sign (Ed25519) and broadcast tiny adapter announcements (~100 bytes) over `iroh-gossip`. Large LoRA weights (~4–33 MB) are never broadcast: they are fetched point-to-point via `iroh-blobs` over encrypted QUIC streams **strictly on demand** when a query actually requests that adapter.

- **Content-Addressed In-Memory LRU Caching**  
  Peer adapters are verified against advertised cryptographic SHA-256 hashes before GPU attachment and cached in an on-device LRU pool. The network fetch penalty is paid **only once** (~55.9 ms over 1 Gbps LAN). All subsequent warm queries execute at native GPU speed with zero network traffic.

- **Dynamic Query-Aware Semantic Routing**  
  Under non-IID client distributions, individual client adapters specialize in narrow sub-domains (e.g., Business vs. Sci/Tech), suffering catastrophic accuracy drops on unfamiliar topics. SLAKSHNA matches prompt embeddings against peer descriptor centroids, dynamically routing queries to the best specialist adapter and recovering full multi-domain accuracy (84.3% on AG News, matching centralized upper bounds).

- **Weight-Level vs. Output-Level Collaborative Ensembling**  
  Supports both **Adapter-Level Merging** (averaging parameter matrices $\Delta W_{\text{merged}} = \frac{1}{K}\sum \Delta W_k$ in a single forward pass) and **Output-Level Federated Mixes** (arithmetic probability averaging, log-probability averaging, confidence weighting, and majority voting).

- **High Availability & Graceful Byzantine Fallback**  
  No single point of failure. If a peer node crashes or goes offline, SLAKSHNA automatically falls through to the next best peer adapter or the local base model (retaining **100% query availability**). Cryptographic hash mismatches or corrupt weights are rejected immediately before reaching the model.

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                             Client Application / API                             │
│                      POST /infer   |   POST /decide   |   GET /adapters          │
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         │
┌────────────────────────────────────────▼─────────────────────────────────────────┐
│                          Axum REST & WebSocket Server                            │
│                     (Authorization, Routing, Status Telemetry)                   │
├──────────────────────────────────────────────────────────────────────────────────┤
│                          Rust P2P Engine (`src/`)                                │
│                                                                                  │
│   • Adapter Registry (RocksDB)       • Strategy Selector (`src/inference/`)      │
│     - Tracks peer adapter metadata     - self (personalized local adapter)       │
│     - Validates Ed25519 signatures     - trust (highest peer trust score)        │
│     - Resolves hashes & availability   - latest (newest training round)          │
│                                        - query_aware (semantic centroid match)   │
│                                                                                  │
│   • Iroh QUIC & Gossip Protocol (`src/network/`)                                 │
│     - `iroh-gossip`: Tiny signed metadata announcements over QUIC                │
│     - `iroh-blobs`: On-demand P2P weight downloads (BLAKE3 verified)             │
│     - SHA-256 Integrity Verification: Rejects tampered weights                   │
│     - Local Disk Cache: `{data_dir}/adapters/<sha256>/`                          │
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         │ Loopback HTTP (127.0.0.1:8765)
┌────────────────────────────────────────▼─────────────────────────────────────────┐
│                    Python Inference Sidecar (`inference_engine.py`)              │
│                           (Bhaskera-serve + HF PEFT)                             │
│                                                                                  │
│   • Shared Base Model loaded once in GPU memory (e.g., Qwen2.5-1.5B / OLMo-7B)   │
│   • In-Memory PEFT LoRA Hot-Swapping (LRU Cache of active adapters)              │
│   • Generation Engine (Token streaming, TTFT/TPOT telemetry, Chat Templates)     │
│   • Option-Logits Scoring (System One single forward pass, Option Permutations)  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Inference Execution Pipeline

Federated inference executes in six coordinated stages:

```
  1. Adapter Export ──► 2. Signed Gossip ──► 3. Local Registry ──► 4. Strategy Selection
     (PEFT weights)       (Metadata only)      (RocksDB storage)     (Query / Trust / Self)
                                                                            │
                                         ┌──────────────────────────────────┴───────────────┐
                                         │ Is selected adapter in local cache?              │
                                         └───────────────┬──────────────────┬───────────────┘
                                                    No   │                  │ Yes
                                                         ▼                  │
                                              5. On-Demand P2P Fetch        │
                                                 (iroh-blobs via QUIC)      │
                                                         │                  │
                                                         ▼                  │
                                                 SHA-256 Check OK?          │
                                                         │                  │
                                                         ▼                  ▼
                                              6. Sidecar Local Forward Pass
                                                 (Attach LoRA -> Generate / Score on GPU)
```

1. **Export & Hashing:** When an adapter is trained, it is saved into a content-addressed directory named by its SHA-256 hash (`{data_dir}/adapters/<sha256>/`).
2. **Metadata Gossip:** The node announces `{node_id, round, sha256, blake3, size, base_model, rank, trust_score}` via `iroh-gossip`. No weights are broadcast.
3. **Registry Maintenance:** Every node maintains an on-disk RocksDB table of all active adapters in the swarm (`GET /adapters`).
4. **Query-Time Selection:** Upon receiving a query, the node selects an adapter locally based on the configured strategy (`trust`, `self`, `latest`, or `query_aware`).
5. **On-Demand Retrieval & Verification:** If the adapter is not in the local cache, it is retrieved directly from the peer over QUIC using `iroh-blobs`. The downloaded file is checked against the announced SHA-256 hash. Corrupted files trigger graceful fallback to the base model.
6. **On-Device Forward Pass:** The Python sidecar hot-swaps the verified adapter onto the shared base model and performs the forward pass locally on the node's GPU.

---

## Two Inference Paradigms: Generative vs. Fast Decision

SLAKSHNA provides two distinct inference modalities tailored for different task requirements:

### 1. Generative Inference (System Two / Text Generation — `/infer`)
- **Mechanism:** Autoregressive token-by-token text generation.
- **Prompt Templating:** Wraps inputs into model-specific chat templates (`format_prompt`) to ensure robust instruction-following without conversational drift.
- **Evaluation:** Scored via **BERTScore-F1** (semantic similarity against reference completions), **ROUGE-L** (sequence overlap), and token streaming telemetry (Time-to-First-Token, Time-per-Output-Token).
- **Primary Benchmark:** Databricks Dolly-15k open-ended instruction following.

### 2. Fast Decision Scoring (System One / Option Logits — `/decide`)
- **Mechanism:** Evaluates multiple-choice questions or structured decisions in a **single forward pass** by extracting the model's next-token logits over valid option tokens ($A, B, C, D$).
- **Zero Parsing Errors:** Completely eliminates syntax and format hallucination failures (**0.0% unparseable answers** compared to up to 2.2% in text generation).
- **Permutation Invariance:** Automatically evaluates cyclic permutations of option orderings to eliminate letter/position bias (e.g., standard LLM preference for option "D").
- **Probabilistic Calibration:** Outputs normalized, honest probability distributions, enabling evaluation of Expected Calibration Error (ECE), Negative Log-Likelihood (NLL), Brier score, and AURC.

---

## Adapter Selection & Aggregation Taxonomy

| Strategy | Training Regime | Aggregation Level | GPU Passes | Key Characteristic |
| :--- | :---: | :---: | :---: | :--- |
| **Base Model** | Zero-Shot | None | 1 | Shared foundation model (Qwen2.5-1.5B / OLMo-7B) without adapters. |
| **Own Adapter Only (`self`)** | Local Skew | None | 1 | Zero network traffic; high on local distribution (95.2%), poor on unseen domains (52.2%). |
| **Random Peer Adapter** | Federated | None | 1 | Naive decentralized baseline picking an arbitrary peer adapter. |
| **Query-Aware Routing (`query_aware`)** | Federated | **Dynamic Dispatch** | **1** | Matches prompt embedding against peer descriptor centroids; routes to the single best specialist. |
| **Merged Average (`merged_avg`)** | Federated | **Adapter-Level (Weights)** | **1** | Parameters blended before inference: $\Delta W = \frac{1}{K}\sum \Delta W_k$. Fast, multi-task generalist. |
| **Merged TIES (`merged_ties`)** | Federated | **Adapter-Level (Weights)** | **1** | TIES-SVD parameter merging resolving interference across client weight updates. |
| **Plain Average (`prob_mean`)** | Federated | **Output-Level (Probabilities)** | **$K$** | Arithmetic mean of predicted probability vectors $\mathbf{p} = \frac{1}{K}\sum \mathbf{p}_k$. |
| **Log Average (`logit_mean`)** | Federated | **Output-Level (Logits)** | **$K$** | Geometric probability mean / average of raw logits $\mathbf{z} = \frac{1}{K}\sum \mathbf{z}_k$. Optimal calibration (ECE 0.030). |
| **Confidence-Weighted** | Federated | **Output-Level (Probabilities)** | **$K$** | Weighted by each member's maximum prediction certainty: $w_k \propto \max(\mathbf{p}_k)$. |
| **Majority Vote** | Federated | **Output-Level (Labels)** | **$K$** | Plurality vote over discrete class predictions across peer adapters. |
| **Centralized Baseline** | Pooled (Non-Private) | Centralized | 1 | Upper-bound reference trained on all data in one place. |

---

## Repository Structure (`inference` branch)

```
├── src/                               # Rust P2P Core & Networking Engine
│   ├── main.rs                        # Node entry point, P2P orchestrator
│   ├── api.rs                         # Axum REST API endpoints (/infer, /adapters, /peers)
│   ├── config.rs                      # Configuration loader ([inference], [network], [node])
│   ├── registry.rs                    # Content-addressed adapter registry (RocksDB persistence)
│   ├── identity.rs                    # Ed25519 cryptographic keypairs
│   ├── inference/
│   │   ├── mod.rs                     # Inference service: lifecycle, P2P fetch, sidecar bridge
│   │   └── selector.rs                # Selection policies (trust, self, latest, query_aware)
│   └── network/
│       ├── blobs.rs                   # iroh-blobs wrapper for P2P adapter weight streaming
│       ├── mesh.rs                    # iroh-gossip mesh for metadata propagation
│       └── mod.rs                     # Network abstraction layer
│
├── inference_engine.py                # Python HTTP sidecar: base model + PEFT hot-swapping
├── adapter_export.py                  # Standalone tool to package LoRAs into PEFT directories
├── setup_serve.sh                     # Virtualenv bootstrap script for Bhaskera-serve
├── Bhaskera-serve/                    # Lightweight model serving submodule (branch `serve`)
│
├── experiments/                       # Comprehensive Empirical Evaluation Suite
│   ├── run_nodes.sh                   # Cluster launch script (train vs serve mode)
│   ├── run_benchmarks.sh              # Master benchmark pipeline (offline, API, serving, vLLM)
│   ├── eval_api.py                    # Evaluates real API queries (cold vs warm, selection strategies)
│   ├── eval_offline.py                # Standalone test set scoring across adapters and models
│   ├── bench_serving.py               # TTFT, TPOT, MFU, and concurrency stress benchmarks
│   ├── bench_fetch.py                 # P2P adapter download latency vs LoRA rank/file size
│   ├── compare_systems.py             # Analytical & empirical comparisons vs vLLM and Petals
│   ├── fault_test.py                  # Node failure & crash resilience evaluations
│   ├── aggregate.py                   # Aggregates raw logs into numerical CSV summary tables
│   ├── plot_results.py                # Generates publication figures & documentation
│   ├── results/                       # Evaluation benchmarks (AG News, Dolly-15k, System One)
│   └── tikzfigs/                      # Standalone, publication-ready TikZ/PGFPlots figures
│
├── examples/
│   ├── 2node-serve/                   # Ready-to-run 2-node serving configurations
│   └── 4node-serve/                   # 4-node heterogeneous serving configuration templates
└── tests/                             # Unit tests for adapter export, sidecar, and transport
```

---

## Getting Started: Running Federated Inference

### 1. Prerequisites & Environment Setup

Ensure you have **Rust 1.75+**, **Python 3.10+**, and CUDA-enabled PyTorch installed.

```bash
# Clone the repository with submodules
git clone --recursive https://github.com/dcll-iiitd/Slakshna.git -b inference
cd Slakshna

# Build the Rust P2P node binary (release mode)
cargo build --release

# Set up the Python inference sidecar virtual environment
./setup_serve.sh
```

### 2. Configuration (`node1.toml`)

Configure each node to enable inference mode and point to the shared base model:

```toml
[federation]
id = "slakshna-inference-cluster"
name = "Slakshna Edge Mesh"

[node]
id = "node-1"
data_dir = "./data-node1"
gpu_id = 0

[network]
host = "0.0.0.0"
p2p_port = 9000
api_port = 8545

[inference]
enabled = true
base_model = "Qwen/Qwen2.5-1.5B-Instruct"  # Shared base model
selection = "trust"                         # Default selection: trust | self | latest | query_aware
min_trust = 0.0                             # Minimum trust score required to use a peer adapter
sidecar_port = 8765                         # Local loopback port for the Python sidecar
sidecar_python = "Bhaskera-serve/.venv/bin/python"
serve_bhaskera_path = "Bhaskera-serve"
spawn_sidecar = true                        # Automatically launches inference_engine.py
serve_only = true                           # Pure inference mode (skips local training loop)
cache_max_adapters = 8                      # Maximum adapters retained in local GPU/RAM cache
max_adapter_bytes = 2147483648              # 2 GB cache storage limit
```

### 3. Launching the Local Multi-Node Serving Cluster

Start a 2-node or 4-node serving cluster with pre-configured templates:

```bash
# Start 2 nodes in serve mode (node1 on port 8545, node2 on port 8546)
./start_cluster.sh 2 examples/2node-serve

# Monitor the logs of node 1
tail -f logs/node1_cluster.log
```

Wait until the logs confirm that both nodes have connected via Iroh and the sidecar is initialized:
```text
INFO 🔑 Iroh EndpointId: a65a49db0894467a...
INFO [inference_engine] loading base model Qwen/Qwen2.5-1.5B-Instruct ...
INFO [inference_engine] ready on 127.0.0.1:8765
INFO 🚀 Axum API server listening on 0.0.0.0:8545
```

### 4. Inspecting the Adapter Registry

Query any node's local registry to inspect all available adapters in the federation:

```bash
curl -s http://127.0.0.1:8545/adapters | jq
```

*Example Response:*
```json
{
  "success": true,
  "adapters": [
    {
      "node_id": "node-1",
      "round": 3,
      "sha256": "8f3b21c4a...",
      "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
      "rank": 16,
      "size": 4371840,
      "trust_score": 1.0,
      "cached": true,
      "is_self": true
    },
    {
      "node_id": "node-2",
      "round": 3,
      "sha256": "1e9a78d0f...",
      "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
      "rank": 16,
      "size": 4371840,
      "trust_score": 0.94,
      "cached": false,
      "is_self": false
    }
  ]
}
```

### 5. Running Inference Queries (`POST /infer`)

Send an inference request to `node-1`. If `node-2`'s adapter is selected or requested, `node-1` downloads it over P2P, verifies its SHA-256 hash, attaches it to the local GPU, and generates the response on-device:

```bash
curl -s http://127.0.0.1:8545/infer -H 'Content-Type: application/json' -d '{
  "prompt": "Explain the concept of decentralized consensus in simple terms.",
  "max_tokens": 128,
  "temperature": 0.7,
  "selection": "trust"
}' | jq
```

*Example Response:*
```json
{
  "success": true,
  "result": {
    "text": "Decentralized consensus is a way for a group of independent computers to agree on a single source of truth without relying on any central authority or server...",
    "adapter": {
      "node_id": "node-2",
      "round": 3,
      "sha256": "1e9a78d0f...",
      "cache_hit": false
    },
    "timing": {
      "select_ms": 0.04,
      "fetch_ms": 55.90,
      "attach_ms": 180.80,
      "model_ttft_ms": 78.40,
      "ttft_ms": 315.14,
      "tpot_ms": 30.54,
      "total_ms": 4224.26,
      "n_tokens": 128,
      "n_prompt_tokens": 18
    },
    "note": null
  }
}
```

*Subsequent Request (Warm Cache Hit):*
Repeating the query returns with `"cache_hit": true`, `fetch_ms: 0.0`, and native GPU TTFT (~92 ms).

### 6. Graceful Cluster Shutdown

```bash
./stop_cluster.sh
pkill -f inference_engine.py
```

---

## REST API Reference

| Endpoint | Method | Description | Request Payload | Response Key Fields |
| :--- | :---: | :--- | :--- | :--- |
| `/status` | `GET` | Node health, round, active Iroh peer count | None | `federation_id`, `round`, `peers`, `endpoint_id` |
| `/peers` | `GET` | Active gossip neighbours & remembered peers | None | `connected`, `known`, `endpoint_id` |
| `/adapters` | `GET` | Current local adapter registry | None | `adapters`: array of metadata, `cached`, `is_self` |
| `/infer` | `POST` | Federated generative inference | `{prompt, max_tokens, temperature, selection, from_node, system, raw}` | `text`, `adapter` info, detailed `timing` breakdown |
| `/leaderboard`| `GET` | Current peer trust scores (`alpha` / `w_i`) | None | `leaderboard`: list of `node` and `trust_score` |
| `/health` (sidecar)| `GET` | Sidecar GPU process status (`:8765`) | None | `{"status": "ok"}` |
| `/stats` (sidecar) | `GET` | GPU memory telemetry (`:8765`) | None | `allocated_mb`, `peak_allocated_mb`, `adapters_loaded` |

### `/infer` Parameter Details

- `prompt` *(string, required)*: The input query.
- `max_tokens` *(int, default: 256)*: Maximum new tokens to generate.
- `temperature` *(float, default: 0.7)*: Sampling temperature (0.0 for greedy decoding).
- `selection` *(string, optional)*: Override node strategy (`"trust"`, `"self"`, `"latest"`, `"query_aware"`).
- `from_node` *(string, optional)*: Pin request to a specific node's adapter (for testing/benchmarks).
- `raw` *(bool, default: false)*: When true, skips the model chat template and passes the prompt verbatim.
- `min_tokens` *(int, optional)*: Forces a minimum output length (used for TPOT benchmarking).

---

## Reproducing Empirical Benchmarks

The `experiments/` directory contains automated runners to benchmark quality, latency, concurrency, and fault tolerance:

```bash
# Run the complete benchmark suite on a trained run (e.g. AG News 4-node skew)
./experiments/run_benchmarks.sh ag_news_n4_a0.1_s0

# Individual benchmark steps:
# 1. Evaluate offline quality across all test splits and methods
Bhaskera-serve/.venv/bin/python experiments/eval_offline.py --tag ag_news_n4_a0.1_s0 --gpu 0 --merge

# 2. Evaluate live HTTP API (cold vs warm TTFT, selection strategies)
Bhaskera-serve/.venv/bin/python experiments/eval_api.py --tag ag_news_n4_a0.1_s0 --n 200

# 3. Serving stress benchmark (TTFT, TPOT, MFU, Concurrency)
Bhaskera-serve/.venv/bin/python experiments/bench_serving.py --backend slakshna --url http://127.0.0.1:8545

# 4. Fault tolerance under peer node crashes
Bhaskera-serve/.venv/bin/python experiments/fault_test.py --tag ag_news_n4_a0.1_s0 --n 30

# 5. P2P download speed vs LoRA rank (r=8 to r=128)
Bhaskera-serve/.venv/bin/python experiments/bench_fetch.py --tag ag_news_n4_a0.1_s0

# 6. Aggregate numerical tables and compile LaTeX TikZ plots
Bhaskera-serve/.venv/bin/python experiments/aggregate.py --tag ag_news_n4_a0.1_s0
Bhaskera-serve/.venv/bin/python experiments/plot_results.py --tag ag_news_n4_a0.1_s0
```

---

## Empirical Benchmark Findings Summary

All figures are compiled in `experiments/results/` and `experiments/tikzfigs/`:

1. **Accuracy on AG News (4 Nodes, Skew $\alpha = 0.1$):**
   - Relying solely on a client's **own adapter** yields only **52.2%** accuracy globally due to severe domain blind spots (down to 0.0% on unseen classes).
   - **Query-aware routing** achieves **84.3%** accuracy (Macro-F1: 0.841), eliminating blind spots by routing to the specialist peer.
   - **Merged weights (`merged_avg`)** reaches **83.9%** (Macro-F1: 0.841) in a single forward pass.
   - *(Non-private centralized upper bound: 87.2%).*

2. **Open-Ended Generation Quality (Databricks Dolly-15k):**
   - Base foundation model achieves **0.204 BERTScore-F1**.
   - Single local adapter achieves **0.294**.
   - **Query-aware routing** and **Merged average** both reach **0.316** (+0.112 over base, +0.022 over local adapter).

3. **System Latency & Hardware Utilization (NVIDIA RTX A6000):**
   - **Warm TTFT:** **113.9 ms** (median 92.6 ms) with 0 bytes network communication.
   - **Cold Fetch & Verification:** Completed in just **55.9 ms** over P2P QUIC streams for standard rank-16 adapters (4.17 MB).
   - **Hardware Efficiency (MFU):** Slakshna achieves **41.9% Model FLOPs Utilization** (64.9 TFLOP/s) at 4k prompt lengths, matching and slightly exceeding centralized vLLM (38.7% MFU).

4. **Zero Prompt Exposure & Communication Comparison:**
   - **Slakshna:** **0.0 Bytes** prompt exposure off-device; **0.0 Bytes** network traffic on warm queries.
   - **Central vLLM:** Exposes entire prompt text (389.5 B/query) over the network; single point of failure.
   - **Petals (Layer-Split):** Leaks hidden states (254.5 KB/query); transfers >1.3 MB per query with ~200 ms network latency on every generated token.

5. **Fault Tolerance:**
   - Slakshna edge nodes maintain **100% request completion** during peer crashes with graceful fallback to surviving adapters or base models, avoiding total outages.

---

## 📄 License

This project is licensed under the **Apache License 2.0** — see the [LICENSE](LICENSE) file for details.
