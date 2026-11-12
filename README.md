# SLAKSHNA System One: Fast Federated Decisions on the Edge

SLAKSHNA is a peer-to-peer federated learning system written in **Rust** (Iroh QUIC mesh, gossip, BLAKE3 blob transfer). Each node trains a LoRA adapter on its own data.

**System One** is the inference side. Given a situation and a **closed list of allowed answers**, it returns a probability for every answer. It does not generate text. Each node scores the options locally with several peers' LoRA adapters and mixes the results with trust weights.

- The query (state, question, options) **never leaves the node**. Only adapter files move between nodes.
- Every decision is a valid, typed answer. Nothing has to be parsed.
- Each decision also reports confidence, margin, entropy and how many adapters agreed.

---

## System One vs. System Two

| | System One (`POST /decide`) | System Two (`POST /infer`) |
|---|---|---|
| Task | Pick one of 2–255 given options | Generate free text |
| Output | Probabilities over the options | Text |
| Decoding | Reads next-token logits once per forward pass. No token loop | Normal autoregressive generation |
| Adapters used | Up to `k` peers' adapters, mixed | One adapter |
| Can fail to parse | No | Yes (0.6% – 1.9% in our runs) |

`/infer` does **not** run automatically when System One is unsure. The client decides whether to escalate, for example when `agreement` or `margin` is low.

---

## How a decision works

1. **Publish.** After each training round the node exports its aggregated LoRA adapter, hashes it (SHA-256), registers it in iroh-blobs (BLAKE3) and gossips a small **signed** announcement (`AdapterMeta`). No weights travel over gossip.
2. **Registry.** Every node keeps the latest announcements of all nodes (last 3 rounds per node, stored in RocksDB, so it survives restarts).
3. **Select.** On `POST /decide` the node picks up to `k` adapters from its local registry. Selection uses only local data and no network:
   - only adapters for the same `base_model`;
   - peers must reach `min_trust`;
   - **the node's own adapter is always included**;
   - identical weights from two nodes count once;
   - weight = softmax(local trust / `weight_temperature`).
4. **Fetch and verify.** Missing adapters are downloaded in parallel by hash from the advertising node (or any other node holding the same hash). The SHA-256 is checked before use. A slow, failed or tampered adapter is dropped (`fetch_timeout_secs`, default 30 s). A node that serves a wrong file is marked rejected. If no adapter is available, the base model decides.
5. **Score locally.** The Python sidecar loads the base model once and swaps LoRA adapters in and out. For each adapter it builds a prompt where options are labelled ` A`, ` B`, ..., runs a forward pass, and takes the next-token logits of the label tokens. A softmax over them gives the probabilities.
6. **Debias.** The options are rotated (cyclic shifts) and scored once per rotation. Probabilities are mapped back and averaged, which cancels letter and position bias.
7. **Mix.** The adapters' distributions are combined with their trust weights (`prob_mean` or `logit_mean`).
8. **More options than labels.** A two-stage tournament is used (see below).

### Architecture

```
Client ── POST /decide · POST /infer · GET /adapters
            │
┌───────────▼─────────────────────────────────────────────┐
│ Rust node (iiitd)                                       │
│  src/api.rs            Axum REST API                    │
│  src/inference/        select adapters, fetch + verify, │
│                        call the sidecar                 │
│  src/registry.rs       signed adapter registry          │
│  src/network/          iroh gossip (metadata) +         │
│                        iroh-blobs (adapter weights)     │
│  src/history.rs        peer reviews → local trust       │
└───────────┬─────────────────────────────────────────────┘
            │ loopback HTTP 127.0.0.1:<sidecar_port> (default 8765)
┌───────────▼─────────────────────────────────────────────┐
│ Python sidecar (inference_engine.py)                    │
│  Bhaskera-serve HFEngineWrapper + HF PEFT               │
│  Base model loaded once; LoRA adapters kept in a small  │
│  LRU (SLAKSHNA_INFER_MAX_ADAPTERS, default 4)           │
│  federated_inference/system_one.py: labels, prompt,     │
│  cyclic debiasing, two-stage tournament, mixing, metrics│
└─────────────────────────────────────────────────────────┘
```

### Where trust comes from

Trust is **local**. Each node scores its peers' model updates during training (peer reviews). `rank_trust` in `src/history.rs` turns the reviews into one score per peer:

- a node's review of itself is ignored;
- older rounds count less (decay 0.9 per round);
- each reviewer gets one vote.

The `trust_score` inside an announcement is only informational. Receivers use their own scores. Peer reviews are saved in RocksDB, so trust survives a restart.

---

## Method details

### 1. Option logits and probabilities
Let `l_1..l_n` be the single-token labels (for example ` A`, ` B`, ` C`, ` D`) and `z` the next-token logits. With calibration temperature `T_cal`:

$$p(y=i \mid x) = \frac{\exp(z_{l_i}/T_{\text{cal}})}{\sum_{j=1}^{n}\exp(z_{l_j}/T_{\text{cal}})}$$

Label set: the tokenizer's single-token labels from `A–Z a–z 0–9`, in the spaced form (` A`) when it offers at least 26 of them. Qwen offers 52 spaced labels, so up to 52 options fit in one stage.

### 2. Format mass
The share of the full next-token probability that sits on the label tokens (before restricting):

$$\text{format\_mass} = \frac{\sum_{i}\exp(z_{l_i})}{\sum_{v\in\mathcal{V}}\exp(z_v)}$$

A low value means the model was not following the answer format. The response reports `min_format_mass`. If it is below 0.5, the node adds a warning to `note`.

### 3. Cyclic debiasing
For `n` options and shifts `s`, position `j` shows option `(j + s) mod n`. The probabilities of all rotations are mapped back and averaged.

- `permutations = 0` (auto): **every** shift when `n ≤ 6`, otherwise 6 evenly spread shifts.
- `permutations = 1`: one pass, no debiasing.
- Maximum is 8.
- Cost: one forward pass per adapter per rotation.

With every shift used, each option visits every position equally often, so the answer no longer depends on the option order. With more than 6 options the cancellation is only partial.

### 4. Mixing rules
With normalised weights `w_k` (`Σ w_k = 1`):

- **`prob_mean`** (default): $p(i)=\sum_k w_k\,p_k(i)$. A bad adapter's influence is limited by its weight.
- **`logit_mean`**: $p(i)\propto\exp\big(\sum_k w_k \log\max(p_k(i),10^{-12})\big)$. A confident adapter can dominate.

### 5. Two-stage tournament (more options than labels)
Let `L` be the number of labels (52 for Qwen) and `N > L` the number of options.

1. Split the options into `C = ceil(N / L)` interleaved chunks.
2. Score every chunk with all adapters and mix. Keep the top `max(1, floor(L / C))` options of each chunk.
3. Score the finalists in one more round and take the final decision.

Example, Banking77 (77 options, Qwen): 2 chunks, 26 finalists from each, 52 in the final round. That is (2 chunks + 1 final) × 6 rotations = 18 forward passes per adapter, or 72 for 4 adapters.

### 6. Reported metrics
For the final distribution `p` and choice `ŷ = argmax p`:

- `confidence = p(ŷ)`
- `margin = p(ŷ) − (second-highest probability)`
- `entropy = −Σ p ln p` (nats)
- `agreement = Σ_k w_k · 1[argmax p_k = ŷ]`, the weight share of adapters whose own top choice matches

---

## Repository layout (System One)

```
.
├── federated_inference/
│   ├── system_one.py          # labels, prompt, debiasing, tournament, mixing, metrics
│   └── adapter_export.py      # saves the aggregated adapter as a PEFT dir named by SHA-256
├── inference_engine.py        # Python sidecar: /health, /generate, /decide
├── src/
│   ├── api.rs                 # REST routes (/decide, /infer, /adapters, ...)
│   ├── config.rs              # [inference] settings and validation
│   ├── registry.rs            # signed AdapterMeta + registry (RocksDB-backed)
│   ├── history.rs             # peer reviews and local trust (rank_trust)
│   ├── network/               # iroh gossip + iroh-blobs
│   └── inference/
│       ├── mod.rs             # publish, fetch + verify, sidecar launch, decide()/infer()
│       └── selector.rs        # strategy selection and top-k trust weights
├── tests/test_system_one.py   # unit tests for system_one.py
├── examples/
│   ├── 2node/ 4node/ 6node/   # training configs
│   └── 2node-serve/           # serve-only configs (no training)
├── experiments/               # evaluation scripts; results/system_one/ holds tables and figures
├── scripts/setup_serve.sh     # sets up the serve virtualenv (Bhaskera-serve)
├── setup.sh                   # sets up the training environment
├── start_cluster.sh           # starts N local nodes: ./start_cluster.sh <2|4|6> [config_dir]
└── stop_cluster.sh            # stops them
```

---

## Quickstart

### Requirements
- Linux with NVIDIA GPU(s), Rust (`cargo`), Python 3.
- Git submodules `Bhaskera` (training) and `Bhaskera-serve` (inference): `git submodule update --init`.
- The base model must be the same on every node. The example configs use `Qwen/Qwen2.5-7B-Instruct`.

### 1. Build and set up
```bash
cargo build --release        # builds target/release/iiitd
./setup.sh                   # training environment
./scripts/setup_serve.sh     # serve environment: Bhaskera-serve/.venv with PEFT
```

### 2. Get adapters
Nodes publish the adapters found in `<data_dir>/adapters/`. Create them by training with an example config that has `[inference] enabled = true`, and stop the cluster when training is done:
```bash
./start_cluster.sh 2 examples/2node
./stop_cluster.sh
```
Training and serving use the same `data_dir` (`./data-node1`, `./data-node2`).

### 3. Serve and query
```bash
./start_cluster.sh 2 examples/2node-serve
```
`examples/2node-serve` sets `serve_only = true`. The nodes skip training, publish their newest adapter on disk and answer queries. Node 1 starts a sidecar (`spawn_sidecar = true`, port 8765). Node 2 only hosts its adapter.

Wait for the sidecar to print this in `logs/node1_cluster.log`:
```
[inference_engine] ready on 127.0.0.1:8765
```

List the adapters node 1 knows about:
```bash
curl -s http://localhost:8545/adapters | jq '.adapters[] | {node_id, round, cached, is_self, local_trust}'
```

Ask for a decision:
```bash
curl -s http://localhost:8545/decide -H 'Content-Type: application/json' -d '{
  "state": "Pending transaction: $150 at AWS.",
  "question": "What is the primary topic of this transaction?",
  "options": ["Personal Electronics", "Cloud Computing Services", "Dining and Groceries", "Travel and Transport"]
}' | jq
```

### 4. Stop
```bash
./stop_cluster.sh
```
This stops the nodes. If a sidecar is left running, stop it with `pkill -f inference_engine.py`.

---

## Configuration (`[inference]`)

Put this section in each node's TOML. See `examples/2node-serve/node1.toml` for a full file. Inference is **off by default**.

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `false` | Turn federated inference on |
| `base_model` | — (required when enabled) | Base model for the sidecar. Adapters for other base models are ignored |
| `selection` | `"trust"` | Strategy for `/infer`: `trust`, `self` or `latest` |
| `min_trust` | `0.0` | Peers below this local trust are ignored. Your own adapter is always allowed |
| `decision_k` | `3` | Adapters mixed per decision (1–16) |
| `combine` | `"prob_mean"` | `prob_mean` or `logit_mean` |
| `weight_temperature` | `0.1` | Sharpness of the trust-to-weight softmax. Lower = more weight on the most trusted |
| `permutations` | `0` | Option rotations. `0` = auto, `1` = single pass, max `8` |
| `calibration_temperature` | `1.0` | Divides the label logits before the softmax. `> 1` softens an over-confident model |
| `fetch_timeout_secs` | `30` | How long to wait for a missing adapter before dropping it |
| `max_adapter_bytes` | 2 GiB | Refuse to download larger adapters |
| `cache_max_adapters` | `16` | Downloaded adapters kept on disk (oldest used evicted first, your own never) |
| `sidecar_port` | `8765` | Loopback port of the Python sidecar |
| `sidecar_python` | `"python"` | Python of the serve virtualenv, e.g. `Bhaskera-serve/.venv/bin/python` |
| `serve_bhaskera_path` | `"Bhaskera-serve"` | Path to the `Bhaskera-serve` checkout |
| `spawn_sidecar` | `true` | Start the sidecar with the node. Set `false` to run it yourself or to make the node an adapter host only |
| `serve_only` | `false` | Skip the training loop. Only publish existing adapters and answer queries |

**Sidecar environment variable.** `SLAKSHNA_INFER_MAX_ADAPTERS` (default `4`) is how many adapters the sidecar keeps loaded. A `/decide` that mixes more adapters than this fails. Export a larger value before starting the node if you use `k > 4`.

---

## REST API

All inference endpoints return `503 inference_disabled` when `[inference] enabled = false`, and `502` (`{"success": false, "error": ..., "message": ...}`) when a decision or generation fails.

### `POST /decide`

Request:
```json
{
  "state": "Pending transaction: $150 at AWS.",
  "question": "What is the primary topic of this transaction?",
  "options": ["Personal Electronics", "Cloud Computing Services", "Dining and Groceries", "Travel and Transport"],
  "k": 3,
  "combine": "prob_mean",
  "permutations": 0,
  "calibration_temperature": 1.0,
  "from_nodes": null
}
```

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `question` | string | yes | — | The question to answer |
| `options` | string[] | yes | — | 2–255 options. Whitespace is normalised. They must be distinct and non-empty |
| `state` | string | no | `""` | Context: a record, log, message, history |
| `k` | integer | no | `decision_k` | Adapters to mix (clamped to 1–16) |
| `combine` | string | no | `combine` | `prob_mean` or `logit_mean` |
| `permutations` | integer | no | `permutations` | `0` auto, `1` single pass, max `8` |
| `calibration_temperature` | float | no | `calibration_temperature` | Must be finite and positive |
| `from_nodes` | string[] | no | `null` | Only use adapters from these node IDs (for tests and benchmarks). Your own adapter is always kept |

Response (shortened; values are illustrative):
```json
{
  "success": true,
  "result": {
    "decision": {
      "choice": 1,
      "label": "Cloud Computing Services",
      "probabilities": [0.0124, 0.9632, 0.0141, 0.0103],
      "confidence": 0.9632,
      "margin": 0.9491,
      "entropy": 0.1874,
      "agreement": 1.0,
      "two_stage": false,
      "finalists": [],
      "min_format_mass": 0.9841,
      "members": [
        {"name": "c20df84b25f4625b...", "weight": 0.45, "probabilities": [0.008, 0.975, 0.010, 0.007], "format_mass": 0.988}
      ],
      "forward_passes": 12,
      "model_ms": 178.4
    },
    "timing": {"select_ms": 0.42, "fetch_ms": 0.0, "sidecar_ms": 182.5, "model_ms": 178.4, "forward_passes": 12, "total_ms": 183.2},
    "sources": [
      {"node_id": "slakshna1...", "round": 10, "sha256": "c20df84b25f4625b...", "weight": 0.45, "cache_hit": true}
    ],
    "note": null
  }
}
```

- `choice` is the index into `options`. `label` is the cleaned option text.
- `members[i].name` is the adapter's SHA-256. `sources[i]` is the node that adapter came from (same order). The weights are re-normalised over the adapters that were actually available.
- `forward_passes` = adapters × rotations (× the extra stages for two-stage decisions).
- `finalists` lists the option indices that reached the final round (two-stage only).
- `note` reports dropped adapters, base-model fallback or low format mass.

### `GET /adapters`
The node's registry: the newest adapter announced by each node.
```bash
curl -s http://localhost:8545/adapters | jq
```
Each entry has `node_id`, `round`, `sha256`, `size`, `base_model`, `rank`, `advertised_trust` (the node's own claim), `local_trust` (this node's score, used for weights), `cached` and `is_self`.

### `POST /infer` (System Two, free text)
Generates text with one adapter chosen by `selection`. If no adapter fits or the fetch fails, the base model answers and `note` says so.
```bash
curl -s http://localhost:8545/infer -H 'Content-Type: application/json' -d '{
  "prompt": "Explain why AWS charges are cloud computing costs:",
  "max_tokens": 128,
  "selection": "trust"
}' | jq
```

| Field | Required | Default | Description |
|---|---|---|---|
| `prompt` | yes | — | Text to complete |
| `max_tokens` | no | 256 | Maximum new tokens |
| `temperature` | no | 0.7 | Sampling temperature |
| `selection` | no | `selection` | `trust`, `self` or `latest` |
| `from_node` | no | — | Force the adapter from this node (tests) |

---

## Evaluation results

Setup: Qwen2.5-1.5B-Instruct, 4 nodes, Dirichlet label skew α = 0.1, seed 0, adapters trained for 3 rounds on the System One prompt. All numbers come from a single run (one seed, one skew level).

Full tables, figures and per-figure explanations: [`experiments/results/system_one/`](experiments/results/system_one/README.md) ([AG News](experiments/results/system_one/ag_news_n4_a0.1_s0_s1/README.md), [Banking77](experiments/results/system_one/banking77_n4_a0.1_s0_s1/README.md)).

### 1. Accuracy
Plain-average (`prob_mean`) mixing with equal weights on every adapter. "Own adapter" is the mean over the 4 nodes.

| Method | AG News (4 options) | Banking77 (77 options) |
|---|---|---|
| Base model, no adapter | 0.872 | 0.524 |
| Own adapter only | 0.758 | 0.608 |
| **Federated mix (`prob_mean`)** | **0.887** | **0.626** |
| Federated mix (`logit_mean`) | 0.875 | 0.632 |
| All adapters merged into one | 0.878 | 0.626 |
| Centralised training (not private) | 0.894 | 0.628 |

Mixing more peers helps. AG News accuracy by number of adapters mixed: k=1 0.758, k=2 0.821, k=3 0.863, k=4 0.887. An adapter is still better on its own topics (0.942 vs 0.874 for the mix on each node's own topic mix), so the mix trades some local accuracy for global accuracy.

### 2. Probability quality (AG News, query node 1, lower is better)

| Method | ECE | Brier | NLL | AURC |
|---|---|---|---|---|
| Base model | 0.086 | 0.216 | 0.724 | 0.065 |
| All adapters merged | 0.058 | 0.198 | 0.432 | 0.048 |
| Mix, `prob_mean` | 0.138 | 0.215 | 0.417 | 0.040 |
| Mix, `logit_mean` | 0.030 | 0.199 | 0.374 | 0.043 |
| Centralised | 0.086 | 0.177 | 0.357 | 0.034 |

The `prob_mean` mix is under-confident (high ECE). `logit_mean` is better calibrated. Fitting `calibration_temperature` helps only a little (ECE 0.138 → 0.109). Two more mixes in the result files (confidence-weighted and question-aware) were computed **offline** and are not options of the node.

### 3. Agreement and accuracy
How many of the 4 adapters chose the same option as the mix:

| Agreement | Share of questions | Accuracy |
|---|---|---|
| 1 of 4 | 1% | 57% |
| 2 of 4 | 20% | 70% |
| 3 of 4 | 38% | 88% |
| 4 of 4 | 41% | 97% |

So `agreement` is a useful signal for deciding when to ask for confirmation or call `/infer`. The node does not act on it automatically.

### 4. Poisoned peers (simulation)
One or two of the 4 adapters are replaced by one that puts 97% on a wrong option (AG News).

| Mixing | 0 poisoned | 1 poisoned | 2 poisoned |
|---|---|---|---|
| Equal weights, `prob_mean` | 0.887 | 0.806 | 0.241 |
| Equal weights, `logit_mean` | 0.875 | 0.818 | 0.514 |
| Trust-weighted, `prob_mean` | 0.887 | 0.861 | 0.799 |

The trust-weighted row **assumes trust has already flagged the bad peers** (trust 0.05 vs 0.25, weights = softmax(trust / 0.1)). It is not a live attack. Robustness depends on trust really identifying bad peers.

### 5. Deciding vs generating (same adapters, mean accuracy)

| | Generate | Decide (System One) |
|---|---|---|
| AG News | 0.814 | 0.797 |
| Banking77 | 0.505 | 0.597 |
| Unparseable outputs | 0.6% / 1.9% | 0% |

On Banking77 (two-stage) a decision is better. On AG News, with only 4 options, generating was slightly more accurate. System One's advantage there is guaranteed valid output.

### 6. Position bias
With a single option order, the answer changes when the options are reordered for 10.7% (federated mix) to 21.5% (one node's own adapter) of AG News questions. Scoring all rotations makes this 0%.

### 7. Latency and throughput (AG News, 4 options, one decision at a time)

| Case | End-to-end |
|---|---|
| k=1, warm | 181 ms |
| k=4, warm (16 forward passes, 636 ms model time) | 677 ms |
| k=4, cold (adapters downloaded, 64 ms of transfer) | 1774 ms |
| Banking77, k=4, warm (72 forward passes) | 3122 ms |

- The sidecar runs one forward pass at a time. At 16 decisions in flight one node handled 1.4 decisions/s. A central vLLM server holding the same adapters handled 17.6/s.
- The same k=4 decision repeated 50 times returned identical probabilities.
- With 1 or 2 of 4 peers down every decision was still answered with fewer members (AG News accuracy 0.900 → 0.867 on 30 questions), but slower while the dead peer's download failed.
- On Banking77, 99.4% of correct options reached the final round of the tournament.

---

---

## License

Apache License 2.0. See [LICENSE](LICENSE).
