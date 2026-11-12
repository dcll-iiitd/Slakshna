# SLAKSHNA: Serverless Federated Inference on the Edge

SLAKSHNA is a peer-to-peer federated learning system written in **Rust** (Iroh QUIC mesh, gossip, BLAKE3 blob transfer). Each node trains a LoRA adapter on its own data.

This branch (`inference`) adds **federated inference**. Any node can answer a prompt with an adapter trained by another node, with no central server:

- Nodes announce their adapters with small signed messages over gossip.
- A node picks an adapter from its **local** registry, downloads it from a peer **only if it is not already cached**, checks its SHA-256 hash, attaches it to a shared base model and generates the answer **on its own GPU**.
- The prompt never leaves the node. Only adapter files move between nodes.

Generation runs in a Python sidecar built on **Bhaskera-serve** and Hugging Face **PEFT**.

> **Scope.** This branch serves **one adapter per query** (`POST /infer`, free-text generation). Mixing several adapters into a typed probability answer (`POST /decide`, "System One") lives on the `system-one` branch.

---

## Key ideas

- **Prompt privacy.** The prompt is only sent to the node's own loopback sidecar (`127.0.0.1`). In our comparison a central server receives the full prompt (389 B per query in the measured request) and a layer-split design sends hidden states (about 254 KB per query, analytical estimate). SLAKSHNA sends 0 bytes of prompt data off the node.
- **Metadata over gossip, weights point-to-point.** Each announcement (`AdapterMeta`) is signed with Ed25519 and measured about 1 KB per adapter. It includes the adapter's PEFT config. Weights are never broadcast. They are fetched over `iroh-blobs` (encrypted QUIC) when a query needs them.
- **Content addressing.** An adapter is identified by the SHA-256 of its `adapter_model.safetensors`. A downloaded file is checked against the announced hash before it is used. A peer that serves a wrong file is marked rejected, and the node tries another holder of the same hash.
- **Cache.** Fetched adapters are kept on disk at `{data_dir}/adapters/<sha256>/` (default: 16, least recently used removed first, your own never). The sidecar keeps a few of them attached to the GPU model (default 4). The network cost is paid once per adapter.
- **Fallback.** If the chosen adapter is unavailable (down, timed out, wrong hash), the node tries the next best adapter (up to 3 in total), then answers with the base model. The response `note` says what happened.

---

## How a query is answered

```
  Training round ends
   1. Export     adapter saved as a PEFT dir named by its SHA-256
   2. Publish    blob added to iroh-blobs (BLAKE3), signed AdapterMeta gossiped
   3. Registry   every node keeps the latest announcements (RocksDB)

  POST /infer
   4. Select     rank adapters from the local registry (trust | self | latest). No network
   5. Fetch      cached?  yes -> use it
                          no  -> download from the advertiser (or another holder),
                                 verify SHA-256, cache it
   6. Generate   sidecar attaches the adapter and generates on the local GPU
```

```
Client ── POST /infer · GET /adapters · ...
          │
┌─────────▼───────────────────────────────────────────────────┐
│ Rust node (target/release/iiitd)                            │
│  src/api.rs         Axum REST API                           │
│  src/inference/     rank, fetch + verify, call the sidecar  │
│  src/registry.rs    signed adapter registry (RocksDB)       │
│  src/network/       iroh-gossip (metadata), iroh-blobs      │
│  src/history.rs     peer reviews -> local trust scores      │
└─────────┬───────────────────────────────────────────────────┘
          │ loopback HTTP 127.0.0.1:<sidecar_port> (default 8765)
┌─────────▼───────────────────────────────────────────────────┐
│ Python sidecar (inference_engine.py)                        │
│  Bhaskera-serve HFEngineWrapper + PEFT                      │
│  Base model loaded once; LoRA adapters hot-swapped          │
│  Streaming generation with TTFT / TPOT timing               │
└─────────────────────────────────────────────────────────────┘
```

### Adapter selection (`selection`)

The node ranks the adapters in its local registry and tries them in order. Only adapters for the node's `base_model` are used. Peers must reach `min_trust` (your own adapter is always allowed).

| Strategy | Order |
|---|---|
| `trust` (default) | Highest **local** trust score first, then newest round, then your own adapter |
| `self` | Only your own adapter, newest round first |
| `latest` | Newest round first, then trust, then your own adapter |

Trust is computed locally from peer reviews made during training (`rank_trust` in `src/history.rs`). It is not taken from the announcement, which is informational only. Until trust scores separate the adapters, `trust` effectively picks the newest round and the response `note` says so.

### Methods compared in the evaluation

The result tables also contain methods that the node does **not** implement. They are computed offline by `experiments/eval_offline.py` and `experiments/aggregate.py` to show what is possible:

| Method | What it is | In the node? |
|---|---|---|
| Own adapter | The node's own adapter | yes (`self`) |
| Random peer | A random node's adapter | no (offline) |
| Query-aware | Choose the node whose training-data centroid (mean sentence embedding) is closest to the question | no. Computed offline. The centroid is "the descriptor a node could advertise" |
| Merged average / TIES | All adapters merged into one adapter with PEFT (`add_weighted_adapter`) | no (offline) |
| Majority vote | Plurality over all adapters' predictions | no (offline) |
| Oracle | Best adapter per question | no (upper bound) |
| Centralised | Training on pooled data | reference, not private |

---

## Repository layout

```
.
├── src/
│   ├── main.rs, config.rs, state.rs, identity.rs
│   ├── api.rs                 # REST routes (/infer, /adapters, /status, /peers, ...)
│   ├── registry.rs            # signed AdapterMeta + registry (RocksDB-backed)
│   ├── history.rs             # peer reviews and local trust (rank_trust)
│   ├── inference/
│   │   ├── mod.rs             # publish, fetch + verify, sidecar launch, infer()
│   │   └── selector.rs        # rank(): trust / self / latest
│   └── network/               # blobs.rs (iroh-blobs), mesh.rs (iroh-gossip)
├── inference_engine.py        # Python sidecar: /health, /stats, /generate
├── adapter_export.py          # used by ml_engine.py: saves an adapter as a PEFT dir named by SHA-256
├── setup.sh                   # training environment
├── setup_serve.sh             # serve environment (Bhaskera-serve virtualenv with PEFT)
├── start_cluster.sh           # ./start_cluster.sh <2|4|6> [config_dir]
├── stop_cluster.sh
├── Bhaskera/                  # training submodule
├── Bhaskera-serve/            # serving submodule (branch `serve`)
├── examples/
│   ├── 2node/ 4node/ 6node/   # training configs
│   └── 2node-serve/           # serve-only configs (node1 = query node, node2 = adapter host)
├── experiments/               # evaluation suite (see "Reproducing the benchmarks")
│   ├── setup_run.py, run_nodes.sh, wait_training.sh, run_benchmarks.sh
│   ├── eval_offline.py, eval_api.py, bench_serving.py, bench_fetch.py, fault_test.py
│   ├── aggregate.py, plot_results.py, compare_systems.py, run_vllm.sh
│   ├── results/               # ag_news_n4_a0.1_s0, dolly_n4_a0.1_s0 (tables, figures, raw)
│   └── tikzfigs/              # TikZ/PGFPlots figures
└── tests/                     # incl. test_inference_engine.py, test_adapter_export.py
```

---

## Quickstart

### Requirements
- Linux with NVIDIA GPU(s), a recent stable Rust toolchain (the Docker image uses 1.93), Python 3.
- Submodules: `git submodule update --init` (`Bhaskera` for training, `Bhaskera-serve` for inference).
- The base model must be identical on every node.

### 1. Build and set up
```bash
git clone --recursive https://github.com/dcll-iiitd/Slakshna.git -b inference
cd Slakshna

cargo build --release     # builds target/release/iiitd
./setup.sh                # training environment
./setup_serve.sh          # serve environment (Bhaskera-serve/.venv with PEFT)
```

### 2. Get adapters by training
A node can only serve adapters that exist in `{data_dir}/adapters/`. Train once with a config that has `[inference] enabled = true`, then stop the cluster:
```bash
./start_cluster.sh 2 examples/2node     # nodes train and publish an adapter each round
./stop_cluster.sh
```
Training and serving must use the same `data_dir` (`./data-node1`, `./data-node2`).

### 3. Serve and query
```bash
./start_cluster.sh 2 examples/2node-serve
tail -f logs/node1_cluster.log
```
`examples/2node-serve` sets `serve_only = true`: no training, the nodes publish their newest on-disk adapter and answer queries. Node 1 (API port 8545) runs a sidecar (`spawn_sidecar = true`). Node 2 (API port 8555) only hosts its adapter.

Wait for the sidecar line in the log:
```
[inference_engine] ready on 127.0.0.1:8765
```

List the adapters node 1 knows:
```bash
curl -s http://127.0.0.1:8545/adapters | jq
```
Ask a question:
```bash
curl -s http://127.0.0.1:8545/infer -H 'Content-Type: application/json' -d '{
  "prompt": "Explain the concept of decentralized consensus in simple terms.",
  "max_tokens": 128,
  "selection": "trust"
}' | jq
```

### 4. Stop
```bash
./stop_cluster.sh
pkill -f inference_engine.py    # only if a sidecar is left running
```

---

## Configuration (`[inference]`)

Inference is **off by default**. See `examples/2node-serve/node1.toml` for a complete file (the example uses `allenai/OLMo-2-1124-7B`; the benchmark runs used `Qwen/Qwen2.5-1.5B-Instruct`).

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `false` | Turn federated inference on |
| `base_model` | required when enabled | Base model the sidecar loads. Adapters for other base models are ignored. Must match `model.name` in `node_template.yaml` |
| `selection` | `"trust"` | Default strategy: `trust`, `self` or `latest` |
| `min_trust` | `0.0` | Peers below this local trust are ignored (your own adapter is always allowed) |
| `sidecar_port` | `8765` | Loopback port of the sidecar |
| `sidecar_python` | `"python"` | Python of the serve virtualenv, e.g. `Bhaskera-serve/.venv/bin/python` |
| `serve_bhaskera_path` | `"Bhaskera-serve"` | Path to the `Bhaskera-serve` checkout |
| `spawn_sidecar` | `true` | Start the sidecar with the node. `false` = run it yourself, or make the node an adapter host only |
| `serve_only` | `false` | Skip the training loop. Only publish existing adapters and answer queries |
| `cache_max_adapters` | `16` | Fetched adapters kept **on disk** (oldest used removed first, your own never) |
| `max_adapter_bytes` | 2 GiB | Refuse to download any single adapter larger than this |
| `fetch_timeout_secs` | `30` | Wait this long for a missing adapter, then try the next candidate |
| `request_timeout_secs` | `300` | Wait this long for the sidecar to answer |
| `max_new_tokens_limit` | `1024` | Hard cap on generated tokens per query |
| `max_prompt_chars` | `32768` | Longest accepted prompt |
| `chat_template` | `true` | Wrap prompts in the model's chat template (the adapters were trained on chat-formatted data) |
| `api_token` | unset | If set, `POST /infer` requires `Authorization: Bearer <token>` |

**Sidecar environment variables** (export before starting the node): `SLAKSHNA_INFER_MAX_ADAPTERS` (adapters kept attached at once, default 4) and `SLAKSHNA_INFER_MAX_NEW_TOKENS` (default 2048).

The API listens on `network.host` (usually `0.0.0.0`) and every query costs GPU time. Set `api_token` if the node is reachable by others.

---

## REST API

| Endpoint | Method | Description |
|---|---|---|
| `/infer` | `POST` | Generate text with a federated adapter |
| `/adapters` | `GET` | Local adapter registry |
| `/status` | `GET` | `federation_id`, `federation_name`, `round`, `peers`, `browsers`, `endpoint_id` |
| `/peers` | `GET` | Connected gossip neighbours and every peer ever reached |
| `/leaderboard` | `GET` | This node's local trust score per peer (`node`, `trust_score`) |
| `/updates`, `/updates/latest`, `/updates/:index` | `GET` | Training update history |
| `/ws` | `GET` | WebSocket (welcome message only) |

Sidecar (loopback only, default `:8765`): `GET /health`, `GET /stats` (GPU memory and adapters loaded), `GET /stats/reset`, `POST /generate`. The Rust node calls `/generate`. Clients should call the node.

`POST /infer` and `GET /adapters` return `503 inference_disabled` if `[inference] enabled = false`.

### `POST /infer`

| Field | Required | Default | Description |
|---|---|---|---|
| `prompt` | yes | — | The query. Must be non-empty and at most `max_prompt_chars` |
| `max_tokens` | no | 256 | Capped by `max_new_tokens_limit` (the response `note` says when it was reduced) |
| `temperature` | no | 0.7 | Between 0 and 2. `0` = greedy |
| `selection` | no | `selection` | `trust`, `self` or `latest` |
| `from_node` | no | — | Use only this node's adapter (tests and benchmarks). Falls back to normal selection if that node has none |
| `system` | no | — | System message for the chat template |
| `raw` | no | `false` | Skip the chat template and send the prompt as is |
| `min_tokens` | no | — | Force a minimum output length (benchmarks) |

Errors: `400 invalid_request` (bad input), `401 unauthorized` (bad or missing token), `502 inference_failed` (node or sidecar failure), `503 inference_disabled`.

Response (values illustrative):
```json
{
  "success": true,
  "result": {
    "text": "Decentralized consensus is a way for a group of independent computers to agree...",
    "adapter": {
      "node_id": "slakshna1...",
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

- `adapter` is `null` when the base model answered. `note` explains fallbacks and dropped adapters.
- `ttft_ms` = `select_ms` + `fetch_ms` + `attach_ms` + `model_ttft_ms`, the delay before the first token.
- Repeating the query gives `cache_hit: true` and `fetch_ms` near 0.

### `GET /adapters`
The newest adapter announced by each node (the registry keeps the last 3 rounds per node).
```json
{
  "success": true,
  "adapters": [
    {
      "node_id": "slakshna1...",
      "round": 3,
      "sha256": "8f3b21c4a...",
      "size": 4373040,
      "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
      "rank": 16,
      "advertised_trust": 0.0,
      "local_trust": 0.0,
      "cached": true,
      "is_self": true
    }
  ]
}
```
`advertised_trust` is the owner's own claim. `local_trust` is this node's score and is the one used for ranking.

---

## Reproducing the benchmarks

Everything lives in `experiments/`. A run is identified by a tag such as `ag_news_n4_a0.1_s0` (dataset, 4 nodes, Dirichlet label skew α = 0.1, seed 0).

```bash
# 1. Prepare data and configs for the run (tokenises each node's data, writes train/ and serve/ configs)
python experiments/prepare_data.py --dataset ag_news --nodes 4 --alpha 0.1 --seed 0
python experiments/setup_run.py --tag ag_news_n4_a0.1_s0 --model Qwen/Qwen2.5-1.5B-Instruct --rounds 3

# 2. Train, then wait until every node published the last round
./experiments/run_nodes.sh ag_news_n4_a0.1_s0 train
./experiments/wait_training.sh ag_news_n4_a0.1_s0 3 4
./experiments/stop_nodes.sh ag_news_n4_a0.1_s0 train

# 3. All evaluation steps (offline, api, serving, vllm, fault, fetch, tables)
./experiments/run_benchmarks.sh ag_news_n4_a0.1_s0
```

Individual steps (run with `Bhaskera-serve/.venv/bin/python`):
```bash
python experiments/eval_offline.py --tag ag_news_n4_a0.1_s0 --gpu 0 --merge   # every adapter on every test set
python experiments/eval_api.py     --tag ag_news_n4_a0.1_s0 --n 200            # live /infer: strategies, cold vs warm
python experiments/bench_serving.py --backend slakshna --url http://127.0.0.1:8545   # TTFT, TPOT, MFU, concurrency
python experiments/fault_test.py   --tag ag_news_n4_a0.1_s0 --n 30             # peers down
python experiments/bench_fetch.py  --tag ag_news_n4_a0.1_s0                    # fetch time vs LoRA rank
python experiments/aggregate.py    --tag ag_news_n4_a0.1_s0                    # CSV tables
python experiments/plot_results.py --tag ag_news_n4_a0.1_s0                    # figures
```
The vLLM comparison needs its own environment (`experiments/run_vllm.sh`, `experiments/.venv-vllm`).

---

## Results

Setup: Qwen2.5-1.5B-Instruct, 4 nodes, Dirichlet label skew α = 0.1, seed 0, 3 training rounds, one GPU per node (NVIDIA RTX A6000 for the speed numbers). One run, so treat small differences with care. Full tables, figures and explanations: [`experiments/results/`](experiments/results).

### 1. AG News accuracy (4 topics, 1,000 balanced test questions)
In AG News the answer is parsed from the generated text. "Own adapter" is the mean over the 4 nodes.

| Method | Accuracy | In the node? |
|---|---|---|
| Base model, no adapter | 71.3% | — |
| Own adapter only | 52.4% | yes (`self`) |
| Majority vote of all adapters | 70.5% | no (offline) |
| Merged adapters, TIES | 73.9% | no (offline) |
| Merged adapters, average | 83.9% | no (offline) |
| Query-aware choice | 84.3% | no (offline) |
| Centralised training (not private) | 87.2% | reference |
| Oracle (best adapter per question) | 98.7% | upper bound |

Under label skew each adapter is an expert on its own topics (it reaches 90.8% to 99.6% on its own local test set) but collapses on topics it never saw (0.0% on some). Combining peers' knowledge recovers most of the gap, but the node itself currently serves a single adapter per query.

### 2. Dolly-15k open-ended generation (BERTScore-F1, 200 questions)

| Method | BERTScore-F1 |
|---|---|
| Base model | 0.204 |
| Own adapter only | 0.294 |
| Merged average | 0.316 |
| Query-aware choice | 0.316 |
| Merged TIES | 0.319 |

### 3. Latency (AG News run, time to first token)

| Case | Mean | Median |
|---|---|---|
| Warm (adapter cached) | 113.9 ms | 92.6 ms |
| Cold (adapter fetched first) | 550.4 ms | 300.6 ms |

A cold query adds about 55.9 ms to fetch and verify a rank-16 adapter (4.17 MB) over a local network, plus 180.8 ms to attach it to the GPU model. Choosing the adapter takes under 0.05 ms. Fetch time grows with adapter size: 39 ms at rank 8 (2.1 MB) up to 344 ms at rank 128 (33.3 MB).

### 4. Speed compared with a central vLLM multi-LoRA server

| | SLAKSHNA sidecar | vLLM |
|---|---|---|
| TTFT, short prompts (128 to 512 tokens) | 55.0 ms | 27.2 ms |
| Model FLOPs utilisation at 4,096 tokens | 41.9% | 38.7% |
| Decode speed | 30.5 ms/token | 7.3 ms/token |
| Throughput, 16 concurrent requests | about 33 tokens/s | 1,692 tokens/s |

The sidecar handles one request at a time. Under heavy concurrency vLLM is far faster, because it batches. The design assumes each node serves its own local user.

### 5. Fault tolerance (30 queries per case)
With 1 or 2 peers killed, 100% of queries were still answered. With `trust` the node fell back to a surviving peer's adapter, at about 1.16 s latency (instead of about 0.19 s) while the dead peer's fetch failed. With `self` nothing changed (about 0.12 s). Note that fallback changes which adapter answers, so accuracy can change too.


---

## License

Apache License 2.0. See [LICENSE](LICENSE).
