# SLAKSHNA — System One: Fast, Federated Decision Scoring on the Edge

A **Peer-to-Peer Federated Decision Framework** built in **Rust** (Iroh QUIC mesh, gossip metadata, BLAKE3 blob transfer) and integrated with a high-performance Python inference sidecar (**Bhaskera-serve** + PyTorch / PEFT).

**SLAKSHNA System One** introduces fast, deterministic, closed-set probabilistic decision-making on the edge. Rather than relying on slow, autoregressive generative decoding with high variance and unparseable outputs (System Two), System One extracts next-token logits over closed option labels in a **single forward pass** ($O(1)$ decoding steps). Peer nodes collaboratively ensemble their local specialist LoRA adapters via cryptographic peer-to-peer weight transfers, cyclic option debiasing, and trust-weighted posterior mixing—all with **zero prompt leakage**.

---

## System One vs. System Two: Dual-Process Collaborative AI

| Dimension | System One (Fast, Typed Decision) | System Two (Slow, Generative Decoding) |
|---|---|---|
| **Paradigm** | Closed-set probability distribution over options | Open-ended autoregressive token generation |
| **Decoding Steps** | **1 forward pass** ($O(1)$ next-token logit extraction) | $N$ sequential forward passes ($O(N)$ autoregressive tokens) |
| **Output Type** | Structured probabilities, confidence, margin, entropy | Unstructured free-form text or JSON string |
| **Unparseable Rate** | **0.0%** (mathematically impossible to fail format) | **0.6% – 1.9%** (syntax errors, malformed tags, hallucinations) |
| **Option Order Bias** | **0.0%** (fully cancelled via cyclic permutations) | High (10.7% – 21.5% flip rate under option reordering) |
| **Consensus Signal** | Exact posterior agreement $\sum_{k} w_k \mathbb{I}(\hat{y}_k = \hat{y})$ | Approximate semantic clustering (BLEU / BERTScore) |
| **Escalation Path** | Resolves high-confidence routine queries instantly; triggers System Two on low agreement | Invoked only when System One uncertainty / margin demands deep chain-of-thought |

---

## Key Highlights & Architectural Innovations

- **100% Prompt Privacy (Zero Prompt Leakage)**  
  User prompts never leave the local edge node (**0 bytes transmitted**). When ensembling across the federation, client nodes fetch lightweight peer LoRA adapters (~4–33 MB) on-demand over encrypted QUIC streams, executing all scoring passes entirely on local hardware.

- **Option-Logit Scoring (Zero Token Generation)**  
  Given program state and a closed set of candidate choices, System One restricts the model's vocabulary logits to single-token option labels (`" A"`, `" B"`, `" C"`, `" D"`). Softmax over these candidate tokens yields a rigorous, typed probability distribution without running multi-token autoregressive generation.

- **0.0% Unparseable Format Failures**  
  Generative models frequently fail downstream execution pipelines by outputting invalid JSON, preamble conversational filler ("Sure, the answer is..."), or unparseable labels. System One constrains the output space directly at the logit layer, guaranteeing **100% valid, typed decisions**.

- **Cyclic Permutations & Letter Debiasing**  
  LLMs exhibit strong positional and letter biases (e.g. disproportionately preferring option "D" or earlier-listed items). System One automatically evaluates options under evenly spread cyclic shifts:
  $$\text{order}_p = [(i + s_p) \bmod n]$$
  Probabilities are mapped back to option identities and averaged. This completely cancels positional bias (reducing flip rate under reordering from 21.5% to **0.0%**).

- **Two-Stage Chunked Tournament Scaling (up to 255 Options)**  
  Single-token label sets are bounded by tokenizer vocabularies (e.g. 26 Latin letters A–Z). For massive closed option sets (e.g., Banking77 with 77 customer intent classes), System One automatically executes a two-stage tournament:
  1. *Stage 1 (Chunking):* Options are partitioned into chunks of size $\le 26$. The federation scores each chunk and retains the top candidates.
  2. *Stage 2 (Finalist Pass):* Finalists from all chunks compete in a decisive final forward pass.

- **Trust-Weighted Byzantine Defense**  
  Peer distributions are mixed using trust scores derived from historical validation:
  $$w_k = \frac{\exp(\alpha_k / T_{\text{trust}})}{\sum_j \exp(\alpha_j / T_{\text{trust}})}$$
  Under an active Byzantine attack where 2 of 4 nodes submit poisoned adapters (97% probability on incorrect options), trust-weighted mixing maintains **79.9% accuracy**, whereas naive averaging collapses to **24.1%**.

- **Consensus Agreement as Unsupervised Confidence & Escalation Trigger**  
  The agreement metric—the weighted fraction of peer adapters whose individual top choice matches the ensemble decision—acts as an intrinsic, unsupervised confidence metric. When 4 of 4 peers agree, accuracy reaches **96.6%**. Queries with low agreement or narrow margin ($p_{(1)} - p_{(2)} < 0.15$) can be seamlessly escalated to System Two generative reasoning.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                 Client Application                                      │
│                POST /decide        |        POST /infer        |      GET /adapters     │
└────────────────────────────────────────────┬────────────────────────────────────────────┘
                                             │
┌────────────────────────────────────────────▼────────────────────────────────────────────┐
│                              Axum REST Server (`src/api.rs`)                            │
│                         (Request validation, JSON serialization)                        │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                             Rust Core Engine (`src/inference/`)                         │
│                                                                                         │
│   • Top-k Selector (`selector.rs`):                                                     │
│     - Ranks peer adapters by local trust score: w_k ∝ exp(α_k / T_trust)               │
│     - Limits ensembling to top-k peers (default: k=3)                                   │
│                                                                                         │
│   • Peer-to-Peer QUIC Networking (`src/network/`):                                      │
│     - `iroh-gossip`: Receives signed Ed25519 adapter announcements (~100 B)             │
│     - `iroh-blobs`: Fetches missing weights point-to-point on demand                    │
│     - Cryptographic SHA-256 verification against announced digest                       │
│     - Local Disk Cache: `{data_dir}/adapters/<sha256>/`                                 │
└────────────────────────────────────────────┬────────────────────────────────────────────┘
                                             │ Loopback HTTP (127.0.0.1:8765)
┌────────────────────────────────────────────▼────────────────────────────────────────────┐
│                       Python Inference Sidecar (`inference_engine.py`)                  │
│                                 (Bhaskera-serve + HF PEFT)                              │
│                                                                                         │
│   • Shared Base Model loaded in GPU VRAM (e.g., Qwen2.5-1.5B / Qwen2.5-7B)              │
│   • In-Memory LoRA Cache & Fast Adapter Swapping                                        │
│   • System One Decision Engine (`federated_inference/system_one.py`):                   │
│     1. Label Tokenizer Mapping: Single-token label vocabulary (spaced: " A", " B"...)   │
│     2. Prompt Formatting: Program state + Question + Option List + "Answer:"            │
│     3. Cyclic Permutations: Rotates options across passes to cancel letter bias         │
│     4. Next-Token Logit Restriction: Softmax over candidate label IDs                   │
│     5. Two-Stage Tournament: Chunked tournament when options > 26 (e.g. Banking77)      │
│     6. Posterior Mixing: Arithmetic (`prob_mean`) or Geometric (`logit_mean`) ensembling│
│     7. Uncertainty Metrics: Confidence, Margin, Entropy, Format Mass, Member Agreement  │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Mathematical Formulation

### 1. Label Extraction & Option Logit Restriction
Let $\mathcal{V}$ denote the model's vocabulary and $\mathcal{L} = \{l_1, l_2, \dots, l_n\} \subset \mathcal{V}$ be the single-token representation of labels (e.g., `[" A", " B", " C", " D"]`). Given prompt context $x$, the model computes next-token logits $\mathbf{z} \in \mathbb{R}^{|\mathcal{V}|}$.

The option-restricted logits are scaled by calibration temperature $T_{\text{cal}}$:
$$\tilde{z}_i = \frac{z_{l_i}}{T_{\text{cal}}}, \quad i \in \{1, \dots, n\}$$

The normalized probability vector over candidate options for a single prompt ordering is:
$$p(y = i \mid x) = \frac{\exp(\tilde{z}_i)}{\sum_{j=1}^n \exp(\tilde{z}_j)}$$

### 2. Format Mass Integrity Check
To verify that the base model is adhering to the decision prompt format rather than allocating probability mass to conversational drift, System One computes the **format mass**:
$$\text{format\_mass} = \frac{\sum_{i=1}^n \exp(z_{l_i})}{\sum_{v \in \mathcal{V}} \exp(z_v)}$$
If $\text{format\_mass}$ falls below a safe threshold (e.g. $< 0.80$), the decision is flagged as low format integrity.

### 3. Cyclic Option Debiasing
To eliminate positional bias, System One generates $P$ cyclic index shifts $s \in \mathcal{S}$:
$$\text{order}_s = \Big( (i + s) \bmod n \Big)_{i=0}^{n-1}$$
For each permutation, candidate options are shuffled into $\text{order}_s$, evaluated, and the resulting probabilities are mapped back to their canonical option index:
$$p_{\text{debiased}}(y = i) = \frac{1}{|\mathcal{S}|} \sum_{s \in \mathcal{S}} p^{(s)}(y = i)$$

### 4. Federated Posterior Mixing Rules
Given $K$ peer adapters with normalized trust weights $\mathbf{w} = (w_1, \dots, w_K)^T$ where $\sum_k w_k = 1$:

- **Arithmetic Mean (`prob_mean`, Default):**
  $$p_{\text{ens}}(y = i) = \sum_{k=1}^K w_k p_k(y = i)$$
  *Robustness property:* The influence of an adversarial or corrupted adapter is strictly bounded by its weight $w_k$.

- **Log-Probability / Geometric Mean (`logit_mean`):**
  $$\log \tilde{p}_{\text{ens}}(y = i) = \sum_{k=1}^K w_k \log \max(p_k(y = i), \epsilon)$$
  $$p_{\text{ens}}(y = i) = \frac{\exp(\log \tilde{p}_{\text{ens}}(y = i))}{\sum_j \exp(\log \tilde{p}_{\text{ens}}(y = j))}$$
  *Sharpness property:* Highly confident peer consensus dominates unconfident peers.

### 5. Two-Stage Tournament for Large Option Spaces ($N > 26$)
When the number of options $N$ exceeds single-token label capacity $L = |\mathcal{L}| \approx 26$:
1. **Partitioning:** Partition options into $C = \lceil N / L \rceil$ interleaved chunks $\mathcal{C}_1, \dots, \mathcal{C}_C$.
2. **Chunk Elimination:** Evaluate each chunk independently across all $K$ adapters. The ensemble selects the top $M = \max(1, \lfloor L / C \rfloor)$ candidates from each chunk.
3. **Finalist Round:** Pool finalists $\mathcal{F} = \bigcup_{c} \text{top}(\mathcal{C}_c)$ into a final option set ($|\mathcal{F}| \le L$) and execute a decisive scoring pass.

### 6. Uncertainty & Agreement Telemetry
For ensemble decision $\hat{y} = \arg\max_i p_{\text{ens}}(y = i)$:
- **Confidence:** $c = p_{\text{ens}}(\hat{y})$
- **Margin:** $m = p_{\text{ens}}(\hat{y}) - \max_{j \ne \hat{y}} p_{\text{ens}}(j)$
- **Entropy (nats):** $H = -\sum_{i=1}^n p_{\text{ens}}(i) \ln p_{\text{ens}}(i)$
- **Peer Agreement:**
  $$A = \sum_{k=1}^K w_k \cdot \mathbb{I}\left(\arg\max_j p_k(j) = \hat{y}\right)$$

---

## Repository Structure (System One Focus)

```
slakshna/
├── federated_inference/
│   ├── __init__.py
│   └── system_one.py             # Pure-PyTorch System One implementation:
│                                 # label extraction, prompt building, cyclic debiasing,
│                                 # two-stage tournament chunking, mixing rules (prob/logit mean)
├── src/
│   ├── api.rs                    # Axum REST routes: POST /decide, POST /infer, GET /adapters
│   ├── config.rs                 # [inference] configuration parser & validation
│   └── inference/
│       ├── mod.rs                # P2P inference service: adapter fetching, verification,
│       │                         # sidecar lifecycle, DecideRequest/Response orchestration
│       └── selector.rs           # Top-k selection, trust-weight softmax, query routing
├── inference_engine.py           # Loopback HTTP sidecar (port 8765):
│                                 # Loads shared base model, manages PEFT LoRA hot-swapping,
│                                 # executes system_one.decide(...)
├── experiments/
│   └── results/system_one/       # Benchmark figures, tables, and empirical evaluations:
│       ├── ag_news_n4_a0.1_s0_s1/     # AG News 4-class evaluation
│       └── banking77_n4_a0.1_s0_s1/   # Banking77 77-class two-stage tournament
├── examples/
│   └── 2node-serve/              # Cluster TOML configurations for federated serving
│       ├── node1.toml
│       └── node2.toml
├── setup_serve.sh                # Automated environment setup (venv, torch, PEFT, transformers)
├── start_cluster.sh              # Multi-node local cluster launcher
└── stop_cluster.sh               # Cluster shutdown script
```

---

## Quickstart Guide

### 1. Environment Setup
Install the Python inference environment (`Bhaskera-serve`):
```bash
# Clone and build the Rust engine
cargo build --release

# Set up the Python sidecar virtual environment
./setup_serve.sh
```

### 2. Configure Node Serving
Configure each node's `TOML` configuration (e.g. `examples/2node-serve/node1.toml`):
```toml
[federation]
id = "slakshna-fl-1"
name = "SLAKSHNA Federation"

[node]
id = "node-1"
data_dir = "./data-node1"
gpu_id = 0

[network]
p2p_port = 9000
ws_port = 8546
api_port = 8545

[inference]
enabled = true
base_model = "Qwen/Qwen2.5-1.5B-Instruct"
selection = "trust"                      # trust | self | latest
decision_k = 3                           # number of peer adapters to mix
combine = "prob_mean"                    # prob_mean | logit_mean
weight_temperature = 0.1                 # trust-to-weight softmax sharpness
permutations = 0                         # 0 = auto cyclic shifts, 1 = single pass
calibration_temperature = 1.0            # logit scaling before softmax
sidecar_port = 8765
sidecar_python = "Bhaskera-serve/.venv/bin/python"
serve_bhaskera_path = "Bhaskera-serve"
spawn_sidecar = true
serve_only = true                        # serve saved adapters without training
```

### 3. Launch Cluster
Start a 2-node local federation:
```bash
./start_cluster.sh 2 examples/2node-serve
```
Wait until the sidecar reports readiness in `logs/node1_cluster.log`:
```
[inference_engine] ready on http://127.0.0.1:8765
```

### 4. Inspect Local Adapter Registry
Query node 1's live view of advertised peer adapters across the mesh:
```bash
curl -s http://localhost:8545/adapters | jq '.adapters[] | {node_id, round, cached, is_self, trust_score}'
```

---

## REST API Reference

### `POST /decide`
Executes a System One closed-set probabilistic decision.

#### Request Schema
```json
{
  "state": "User profile: Software Engineer. Current account balance: $4,200. Pending transaction: $150 at AWS.",
  "question": "What is the primary topic of this transaction?",
  "options": [
    "Personal Electronics",
    "Cloud Computing Services",
    "Dining and Groceries",
    "Travel and Transport"
  ],
  "k": 3,
  "combine": "prob_mean",
  "permutations": 0,
  "calibration_temperature": 1.0,
  "from_nodes": null
}
```

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `state` | string | No | `""` | Background context, database record, sensor state, or chat history. |
| `question` | string | **Yes** | — | The decision query to be answered. |
| `options` | array[string] | **Yes** | — | Closed list of 2 to 255 distinct, non-empty options. |
| `k` | integer | No | `decision_k` (3) | Number of top peer adapters to ensemble. |
| `combine` | string | No | `"prob_mean"` | Posterior mixing rule: `"prob_mean"` (arithmetic) or `"logit_mean"` (geometric). |
| `permutations` | integer | No | `0` | Cyclic debiasing passes: `0` (auto, up to 6), `1` (single pass), or max `8`. |
| `calibration_temperature`| float | No | `1.0` | Softmax temperature dividing candidate logits prior to probability extraction. |
| `from_nodes` | array[string] | No | `null` | Optional list of specific node IDs to restrict adapter selection to (benchmarks/testing). |

#### Response Schema
```json
{
  "success": true,
  "result": {
    "decision": {
      "choice": 1,
      "label": "Cloud Computing Services",
      "probabilities": [
        0.0124,
        0.9632,
        0.0141,
        0.0103
      ],
      "confidence": 0.9632,
      "margin": 0.9491,
      "entropy": 0.1874,
      "agreement": 1.0,
      "two_stage": false,
      "finalists": [],
      "min_format_mass": 0.9841,
      "members": [
        {
          "name": "c20df84b25f4625b...",
          "weight": 0.45,
          "probabilities": [0.008, 0.975, 0.010, 0.007],
          "format_mass": 0.988
        },
        {
          "name": "a89e13f419b8813a...",
          "weight": 0.35,
          "probabilities": [0.015, 0.958, 0.017, 0.010],
          "format_mass": 0.982
        },
        {
          "name": "74f1b8c08272ea29...",
          "weight": 0.20,
          "probabilities": [0.019, 0.946, 0.021, 0.014],
          "format_mass": 0.981
        }
      ],
      "forward_passes": 12,
      "model_ms": 178.4
    },
    "timing": {
      "select_ms": 0.42,
      "fetch_ms": 0.0,
      "sidecar_ms": 182.5,
      "model_ms": 178.4,
      "forward_passes": 12,
      "total_ms": 183.2
    },
    "sources": [
      {
        "node_id": "node-1",
        "round": 10,
        "sha256": "c20df84b25f4625b...",
        "weight": 0.45,
        "cache_hit": true
      },
      {
        "node_id": "node-2",
        "round": 10,
        "sha256": "a89e13f419b8813a...",
        "weight": 0.35,
        "cache_hit": true
      },
      {
        "node_id": "node-3",
        "round": 10,
        "sha256": "74f1b8c08272ea29...",
        "weight": 0.20,
        "cache_hit": true
      }
    ],
    "note": null
  }
}
```

---

### `GET /adapters`
Returns the active adapter registry on this node:
```bash
curl -s http://localhost:8545/adapters | jq
```

---

### `POST /infer` (Generative System Two Fallback)
For tasks requiring open-ended text or reasoning when System One agreement is low:
```bash
curl -s http://localhost:8545/infer -H 'Content-Type: application/json' -d '{
  "prompt": "Explain why AWS charges are categorized as cloud computing:",
  "max_tokens": 128,
  "selection": "trust"
}' | jq
```

---

## Empirical Benchmarks & Evaluation

Evaluated across distributed edge clusters running **Qwen2.5-1.5B-Instruct** under Dirichlet non-IID label skew ($\alpha = 0.1$, seed 0).

### 1. Decision Accuracy across Selection Strategies (AG News, 4 Classes)

| Selection Method | All Topics (Global Test) | Node's Own Topic Mix (Local Test) | Operational Property |
|---|---|---|---|
| **Own Adapter Only** | 75.8% | **94.2%** | High local fidelity, severely overfits narrow domain |
| **Random Peer Adapter** | 52.4% | 52.6% | Fails when peer specializes in divergent topic |
| **Base Model (Zero-Shot)** | 87.2% | 64.6% | General capability, unspecialized |
| **Majority Vote** | 70.5% | 69.9% | Hard label voting discards confidence information |
| **Local-Only Training** | 55.1% | 95.2% | Non-federated baseline; vulnerable to drift |
| **Merged Average (Weights)** | 83.9% | 84.4% | Parameter matrix averaging: single forward pass |
| **Query-Aware Semantic Routing** | 88.4% | 88.7% | Routes query to specialist adapter centroid |
| **Federated Mix (`prob_mean`, ours)** | **88.7%** | 87.4% | Robust arithmetic mean of posteriors |
| *Centralized (Upper Bound)* | *89.4%* | *87.6%* | Non-private pooled training reference |

---

### 2. Decision Probability Quality & Calibration

System One decisions provide trustworthy confidence values. Evaluated across standard calibration metrics:

| Method | Calibration (ECE $\downarrow$) | Prob Error (Brier $\downarrow$) | Confident Mistakes (NLL $\downarrow$) | Ranking Quality (AURC $\downarrow$) |
|---|---|---|---|---|
| **Local-Only** | 0.180 | 0.440 | 1.060 | 0.140 |
| **Own Adapter** | 0.152 | 0.390 | 0.893 | 0.125 |
| **Base Model** | 0.086 | 0.220 | 0.724 | 0.065 |
| **Merged Average** | 0.058 | 0.200 | 0.430 | 0.048 |
| **Fed Mix (`prob_mean`)** | 0.138 | 0.215 | 0.417 | **0.040** |
| **Fed Mix (`logit_mean`)** | **0.030** | 0.200 | 0.370 | 0.043 |
| **Fed Mix (Query-Aware)** | 0.033 | **0.190** | **0.360** | 0.041 |
| *Centralized** | *0.086* | *0.180* | *0.360* | *0.034* |

- **Logit Mean (`logit_mean`)** achieves an ultra-low **ECE of 0.030**, yielding near-perfect calibration.
- **AURC of 0.040** proves that confidence scores accurately separate correct answers from incorrect ones, enabling effective selective prediction.

---

### 3. Agreement vs. Accuracy: Unsupervised Escalation

The fraction of peer adapters that agree on top-1 choice serves as an unsupervised confidence signal:

| Consensus Agreement | Distribution of Decisions | Empirical Accuracy | Recommended Action |
|---|---|---|---|
| **1 of 4 agree** | 1% of queries | 57.0% | **Escalate to System Two** (Reasoning/CoT) |
| **2 of 4 agree** | 20% of queries | 70.0% | Escalate or request user confirmation |
| **3 of 4 agree** | 38% of queries | 88.0% | Accept System One decision |
| **4 of 4 agree** | 41% of queries | **96.6%** | **Instant high-confidence execution** |

---

### 4. Byzantine & Poisoning Robustness

Evaluated when 1 or 2 peer adapters are maliciously modified to output 97% probability on wrong classes:

| Ensembling Method | Clean (0 Poisoned) | 1 Poisoned Peer | 2 Poisoned Peers | Behavior |
|---|---|---|---|---|
| **Equal Weights (`prob_mean`)** | 88.7% | 80.6% | 24.1% | Linear weight pollution causes collapse |
| **Equal Weights (`logit_mean`)** | 87.5% | 81.8% | 51.4% | Geometric mean resists single outlier |
| **Trust-Weighted Ensembling** | **88.7%** | **86.4%** | **79.9%** | **Trust scores downweight malicious nodes to near 0** |

---

### 5. Scaling to Massive Option Sets: Banking77 (77 Options)

Evaluated on Banking77 customer intent queries with 77 distinct options via the two-stage chunked tournament:

| Evaluation Dimension | Generative Decoding (System Two) | System One (Two-Stage Decision) |
|---|---|---|
| **Classification Accuracy** | 50.5% | **59.7%** (+9.2% improvement) |
| **Unparseable Output Rate** | **1.9%** (failed regex/JSON matching) | **0.0%** (mathematically guaranteed valid choice) |
| **Tournament Survival** | — | **84.2%** of correct options survive Stage 1 to finalists |
| **Total Forward Passes** | Variable (20–60 tokens/pass) | 72 deterministic forward passes (18 chunks $\times$ 4 passes) |
| **Cold Latency** | 3,840 ms | 4,236 ms (includes 61 ms P2P QUIC download) |
| **Warm Latency** | 2,750 ms | 3,122 ms |

---

### 6. End-to-End Latency & Throughput Profile

- **Single Query Latency ($k=1$ adapter, 4 options):** **181 ms** warm end-to-end (179 ms GPU forward passes).
- **Federated Ensemble Latency ($k=4$ adapters, 4 options, auto-permutations):** **677 ms** warm end-to-end (636 ms GPU forward passes across 16 cyclic orderings).
- **Network Fetch Penalty:** Incurred **only on the first cold request** (~64 ms over 1 Gbps LAN to fetch missing LoRA weights via `iroh-blobs`). Warm queries run at zero network overhead from local memory cache.
- **Determinism:** 50 repeated executions of the identical $k=4$ decision yield $0.00\times 10^0$ probability divergence (100.0% identical choice index).

---

## Shutdown & Teardown

```bash
# Stop Rust P2P nodes
./stop_cluster.sh

# Terminate Python sidecar processes
pkill -f inference_engine.py
```

---

## License

This project is licensed under the **Apache License 2.0** — see the [LICENSE](LICENSE) file for details.
