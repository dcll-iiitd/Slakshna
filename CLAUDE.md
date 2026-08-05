# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

SLAKSHNA is a decentralized federated-learning system: a **Rust P2P node** (Iroh QUIC + gossip) that periodically shells out to a **Python ML engine** (`ml_engine.py`), which drives **Bhaskera** (vendored under `Bhaskera/`, a Ray-native LLM training framework) to do local LoRA fine-tuning. Nodes exchange sparsified model deltas over gossip and aggregate them with per-peer trust weights.

The core data structure is `UpdateHistory` (`src/history.rs`): `peer_updates: HashMap<node_id, Vec<UpdateRecord>>`, an in-memory append-only log per participant holding `RecordKind::ModelUpdate` (a compressed delta) and `RecordKind::PeerReview` (a trust score). It is not persisted — only the node keypair lives in RocksDB. Records are hash-chained per node (`prev_hash`) purely as tamper-evident sequencing.

This codebase was originally forked from a blockchain and has since been reframed to native federated-learning terminology. If you find remaining blockchain vocabulary anywhere, it is a leftover, not a design intent — see "Known leftovers" below.

## Build & run

```bash
# Python env (do this first — SLAKSHNA's setup.sh expects it)
cd Bhaskera && bash setup.sh && source bhaskera-activate.sh && cd ..

bash setup.sh                 # pip deps + creates data/ logs/ ml_models/ ml_states/ + cargo build --release
cargo build --release         # binary is target/release/iiitd (bin name != package name `slakshna`)
cargo check                   # fast iteration on the Rust side

# rocksdb -> zstd-sys -> bindgen needs libclang. If the build dies with
# "Unable to find libclang", point it at the installed LLVM:
export LIBCLANG_PATH=/usr/lib/llvm-18/lib

./target/release/iiitd --config node1.toml    # arg parsing is positional: exactly `--config <path>`, else config.toml
```

Run the Python engine standalone (no Rust node) — this is the fastest way to test the FL logic:

```bash
python ml_engine.py <my_id> <peer_id_1> <peer_id_2> ...
MALICIOUS_NODES="node-2" python ml_engine.py node-2 node-1     # poisoning mode (forces lr=1.0)
```

Experiment drivers, all run **from the repo root** (they resolve `ml_engine.py` and `logs/` relative to CWD):

```bash
python scripts/run_ml_sweep.py     # 50 nodes × 40 rounds × malicious-count sweep, writes logs/ml_performance_*.csv
python orchestrator.py             # spawns N local Rust nodes from config.toml into benchmark_configs/
python plots_script/plot_loss.py   # plots read logs/*.csv, write results/*.png
```

There is no test suite beyond three unit tests in `src/identity.rs` (`cargo test`).

## Architecture

### The epoch loop (`src/main.rs`)

One `tokio::spawn`ed task drives everything, aligned to the wall clock (`now % epoch_duration_secs`) so independently started nodes converge on the same epoch boundary:

1. Scan `peer_updates` for each peer's latest `ModelUpdate`; write its base64 payload to `{data_dir}/network_deltas/{peer_id}_delta.b64`.
2. Spawn `python ml_engine.py <my_id> <peer_ids...>` with `IIITD_DATA_DIR={data_dir}` and `CUDA_VISIBLE_DEVICES={node.gpu_id}`.
3. Parse the **last stdout line** as `MLEngineOutput` JSON; build a `ModelUpdate` record from `compressed_delta` plus one `PeerReview` record per peer weight; append locally and gossip-broadcast.
4. Barrier until `epoch_start + sync_deadline_secs` or until `peer_updates.len() >= expected_peers`, then log the top-trusted cohort (`top_trusted_peers`, `COHORT_SIZE`) — currently informational only, it drives nothing.

Cadence knobs live in `[training]` in the node TOML (`epoch_duration_secs`, `sync_deadline_secs`, `expected_peers`), all with serde defaults matching the previous hardcoded 600/300/6.

### The Rust ↔ Python contract

This is the fragile seam. `ml_engine.py` must print exactly one JSON object as its **final stdout line**:

```json
{"validation_score": f64, "model_hash": str, "weights": {peer_id: f64}, "metadata": str, "compressed_delta": "<base64 torch.save of the sparse delta>"}
```

Anything printed to stdout after it breaks parsing silently (the Rust side just skips record creation). **All diagnostics in `ml_engine.py` go to `stderr`.** Deltas travel *inside* gossip messages as base64 — `Gossip::builder().max_message_size(10 MB)` in `mesh.rs` is the ceiling on delta size.

### Node identity

`main.rs` overwrites the TOML `node.id` with a bech32 `NodeId` (hrp `slakshna`) derived from an Ed25519 keypair persisted in `{data_dir}/rocksdb` (`state.rs::get_or_create_node_identity`). So the `id` in the TOML is cosmetic; the real ID is stable per `data_dir` and **deleting `data_dir` gives the node a new identity**, orphaning its `ml_models/ckpt_*` and `ml_states/*` files. This ID is what's passed to Python and what keys every file under `ml_models/`, `ml_states/`, and `logs/`.

Separately, the **Iroh EndpointId** is what peers dial. It is a different Ed25519 key, but no longer a random one: `mesh.rs` derives it from the same persisted keypair via `Keypair::transport_seed()` (`sha256("slakshna/iroh-endpoint/v1" || signing_key)`), so it is **stable across restarts** as long as `data_dir` survives. That stability is what makes a bootstrap-free federation possible — cached peer entries and published `peers` values stay valid. It's printed at startup, served at `GET /status`, and goes in a peer's `peers` list as a bare id (or `<endpoint_id>@<ip:port>` to pin an address).

### Networking (`src/network/mesh.rs`)

**There is no bootstrap node and no node type.** Every participant is symmetric; membership is decided by the shared `[federation] id`, which is hashed into both the gossip topic (`sha256(federation.id)`) and the mDNS service label (`sl<hex10>`, kept under the 15-char DNS-SD limit). All nodes must share a `[federation] id` to see each other.

`start()` layers four independent ways to find peers, none of which needs a designated host:

1. **mDNS** (`iroh-mdns-address-lookup`) — LAN/same-host discovery. Its `subscribe()` stream is drained by a background task that calls `GossipSender::join_peers` and caches the peer.
2. **Peer cache** — every peer seen via mDNS or gossip `NeighborUp` is written to RocksDB under `peer:{endpoint_id}` (`state.rs::remember_peer` / `known_peers`) and re-offered on the next start.
3. **Gossip peer exchange** — iroh-gossip's own membership layer, once any single peer is reachable.
4. **Mainline DHT + pkarr/DNS** (`iroh-mainline-address-lookup`, `presets::N0`) — global EndpointId → address resolution.

Each is switchable in `[discovery]` (`mdns`/`dht`/`dns`/`relay`, all default true). Addresses pinned as `<endpoint_id>@<ip:port>` in `network.peers` go into a `MemoryLookup` (still supported for Playit.gg-style tunnels; see README).

Two details that matter:

- `gossip.subscribe(...)`, **not** `subscribe_and_join` — the latter blocks until a neighbour exists, which would make a lone node hang forever. A node must be able to start first and be found later.
- A `REJOIN_INTERVAL` (60 s) task re-offers the whole remembered peer set to the membership layer. This is what heals partitions with no coordinator.

`network.peers` accepts `boot_nodes` as a serde alias, so older config files still load.

The legacy `star.rs` WebSocket star topology has been deleted — a star is the thing this design is removing — along with `network.topology` and `[network.star]`. `P2PMessage` now lives in `src/network/mod.rs`.

### `ml_engine.py` pipeline

`main()` runs, per invocation (one epoch, one node): load `ml_states/{id}_state.json` → softmax `alpha` into trust weights `w_i` → write `config_{id}.yaml` from `node_template.yaml` → tokenize (cached) → `python -m bhaskera.launcher.train` as a subprocess, streaming stdout to parse `[epoch N][step M] loss=X` into `logs/epoch_loss_tracking.csv` → `delta = new_adapter - old_sd` → DP clip → add error-feedback residual → top-k sparsify (residual saved back to `ml_states/{id}_error_feedback.pth`) → base64 → decode + `validate_peer_delta` each peer's `.b64` → trust-weighted sum → write back to the checkpoint *and* `ml_models/{id}_base_lora.pth` → update `alpha` via cosine similarity → print JSON.

`TRAINING_MODE=finetuning` (default) does LoRA deltas; `pretraining` does full-parameter deltas against `ml_models/{id}_base_full` and switches the optimizer to GaLore. `OPTIMIZER` env var (`galore`/`muon`) rewires `plugins.optimizers` in the generated YAML.

Multi-node-on-one-host isolation is all derived from `md5(node_id) % 1000` offsets: Ray ports, dashboard port, and `RAY_TMPDIR=/tmp/sl_t_<last 6 chars of id>` (kept short because AF_UNIX paths cap at 107 chars).

## Landmines

- **`ml_engine.py:574` hardcodes `env["CUDA_VISIBLE_DEVICES"] = "1,2,3"`** for the Bhaskera child process, overriding the GPU pin the Rust node passed in. Fails on any machine without GPUs 1–3.
- Constants in code diverge from the README: sparsity is `0.1` at the call site (`sparsify_tensor` defaults to `0.01`), DP runs with `max_norm=100.0, noise_multiplier=0.0` (noise off, deliberately, for convergence).
- **`validate_peer_delta(max_allowed_norm=10.0)` vs. own clipping at `max_norm=100.0`** — a node's own deltas can exceed the threshold its peers use to reject them.
- `RecordKind` serializes into the gossip payload by variant and field name, so renaming anything in `src/history.rs` is a **wire-format break**: every node in a federation must run the same build.
- `benchmark_configs/*.toml` are generated by `orchestrator.py` from `config.toml` (not tracked here) and get wiped on each run.
- `setup.sh` runs `sed -i "s|/mnt/disk1/slakshna|$BASE_DIR|g" ml_engine.py` — it rewrites tracked source in place.
- `Bhaskera/` is a **vendored copy, not a git submodule**; edits there are edits to this repo. It has its own `README.md` + `ARCHITECTURE.md` and per-package `README.md`s that are the reference for config schema and the FSDP/DCP internals.
- `ml_engine.py` writes `config_{id}.yaml` and `bhaskera_crash.log` into the repo root.
- Docker (`Dockerfile`/`docker-compose.yml`/`deploy.sh`) builds only the Rust binary behind nginx + certbot — no Python, no GPU. It cannot run the FL loop; it's leftover public-API deployment.

## Known leftovers

These still carry pre-fork blockchain framing and were deliberately left alone:

- `scripts/benchmark*.py`, `scripts/test_shield_*.py`, and `plots_script/diag.py` target REST endpoints (`/wallet/new`, `/faucet`, `/nonce/pending`, `/tx/sign`, `/blocks`, shielded transactions) that **do not exist** in `src/api.rs`. They are dead, but they produced the CSVs behind `results/latency_vs_throughput.png`, so they were kept as experiment provenance. Do not use them as an API reference.
- The binary is still named `iiitd` (`[[bin]]` in `Cargo.toml`) and the Docker network is `iiitd` — the institute abbreviation, kept so every documented command and SLURM script keeps working.
- `deploy.sh` points at the live DNS name `iiitd-chain.duckdns.org`; renaming it would invalidate the issued certificate.
