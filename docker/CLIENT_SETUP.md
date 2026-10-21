# SLAKSHNA node: client setup

One image runs a complete SLAKSHNA node: the Rust P2P layer (networking, weight transport,
API) and the Python ML engine (GPU training) in the same container.

```bash
docker pull ghcr.io/himanshu4nagdev/slakshna:latest      # or the pinned tag :v1
```

| | |
|---|---|
| Image | `ghcr.io/himanshu4nagdev/slakshna` (public, no login needed) |
| Tags | `latest`, `v1` (same image) |
| Digest | `sha256:8192d231a498ce80177ab8c9dde0ddc0ed3d1799e8802ed18206ebd61b763fea` |

## 1. What you need

| | |
|---|---|
| Machine | Linux x86_64 with an NVIDIA GPU |
| NVIDIA driver | **>= 525.60.13** (the image ships PyTorch 2.5.1 built for CUDA 12.1) |
| Docker | Engine 24+ and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) |
| Disk | image about 12 GB unpacked (about 3.8 GB to download) plus the model cache and checkpoints; budget at least 40 GB |
| Files you provide | `node.toml` (section 4) and `node_template.yaml` (the training configuration: model, dataset path, hyper-parameters) |
| Data | your own dataset directory, mounted read-only at `/data`; `data.train_path` in the template must point inside `/data` |

The image does **not** contain a sample node configuration or a usable training template:

* There is no sample `node.toml` in the image. **The example in section 4 of this file is the
  reference.**
* The built-in `/app/node_template.yaml` points at dataset paths that do not exist on your
  machine. Always mount your own template.

## 2. Get the image

The package is **public**: no account or token is needed.

```bash
docker pull ghcr.io/himanshu4nagdev/slakshna:latest
docker image inspect ghcr.io/himanshu4nagdev/slakshna:latest --format '{{index .RepoDigests 0}}'
# expect: ghcr.io/himanshu4nagdev/slakshna@sha256:8192d231a498ce80177ab8c9dde0ddc0ed3d1799e8802ed18206ebd61b763fea
```

No registry access (air-gapped machine)? Use an offline copy:

```bash
# on a machine that has the image
docker save ghcr.io/himanshu4nagdev/slakshna:latest | gzip > slakshna.tar.gz
# on the target machine
gunzip -c slakshna.tar.gz | docker load
```

## 3. What is inside the image

| Path | What |
|---|---|
| `/usr/local/bin/iiitd` | Rust node (P2P networking, weight transport, REST/WebSocket API) |
| `/usr/local/bin/entrypoint.sh` | Container entrypoint: runs the preflight check, then starts the node |
| `/usr/local/bin/preflight.py` | Startup checks (config, writable directories, dataset, GPU, driver vs CUDA) |
| `/usr/local/bin/healthcheck.py` | Docker health check (`GET /status` on the API port) |
| `/opt/venv/` | Python 3.11 environment: PyTorch 2.5.1+cu121, Ray, Transformers, PEFT, Bhaskera |
| `/app/ml_engine.py`, `/app/federated_communication/` | Python ML engine started by the Rust node every round |
| `/app/node_template.yaml` | Built-in training template (lab paths; replace it with your own mount) |
| `/app/config.toml` | Where the node reads its configuration (`SLAKSHNA_CONFIG`); you mount your `node.toml` here |

The container runs as a non-root user (uid 1012).

## 4. Node configuration (`node.toml`)

This is the complete configuration. It is valid as written; change the values marked `CHANGE`.

The same template is also kept in the repository as
[`docker/config/node.example.toml`](https://github.com/himanshu4nagdev/Slakshna/blob/docker-setup/docker/config/node.example.toml)
(branch `docker-setup`); you can copy it from there instead of from this page.

```toml
[federation]
id = "CHANGE-me-federation-id"       # CHANGE: identical on every node of the federation
name = "SLAKSHNA Federation"

[training]
epoch_duration_secs = 600
sync_deadline_secs = 300
expected_peers = 2                    # CHANGE: total number of nodes in the federation

[compression]
enabled = true
sparsity = 0.1                        # must be in (0, 1]
quantization = "symmetric_int8"       # the only supported value
allow_legacy_delta_format = false
max_payload_bytes = 7340032           # 1 .. 2000000000
max_tensor_elements = 2000000000      # > 0

[node]
id = "client-node-1"                  # CHANGE: a readable name for this node
data_dir = "/app/node-data"           # do not change: node identity (keypair) and known peers live here
gpu_id = 0                            # GPUs given to the container are renumbered from 0 inside it
num_gpus = 1

[network]
host = "127.0.0.1"                    # API / WebSocket bind address (the API is unauthenticated)
p2p_port = 9000                       # Iroh QUIC, UDP
ws_port = 8546
api_port = 8545
peers = []                            # across networks: ["<EndpointId>@<public-ip>:9000"]
# allowed_peers = []                  # optional whitelist of EndpointIds; empty or absent = anyone with the federation id

[discovery]
mdns = true                           # same LAN / same host only
dht = true                            # BitTorrent mainline DHT, works across the internet
dns = true                            # n0 pkarr/DNS lookup (third-party service); false = self-contained
relay = false                         # true = fall back to public relays when direct hole punching fails

[logging]
level = "info"
```

| Section | Required? | If omitted |
|---|---|---|
| `[federation]` `id`, `name` | required | node does not start |
| `[node]` `id`, `data_dir` | required | node does not start; `gpu_id`/`num_gpus` default to unset / 1 GPU |
| `[network]` `host`, `p2p_port`, `ws_port`, `api_port` | required | node does not start; `peers`, `allowed_peers` default to empty |
| `[logging]` `level` | required | node does not start |
| `[training]` | optional | defaults 600 / 300 / **6 expected peers**, so set it explicitly |
| `[compression]` | optional | defaults as in the example |
| `[discovery]` | optional | all four default to **true** (including `relay`) |

Rules that matter:

* **Every node uses the same `[federation] id`.** Otherwise the nodes do not join the same federation.
* `expected_peers` is the **total number of nodes** in the federation.
* Never delete the `slakshna-node-data` volume: it holds the node identity.

## 5. Run

```bash
docker run -d --name slakshna-node --init --restart unless-stopped \
  --network host --gpus '"device=0"' --shm-size 16g --stop-timeout 60 \
  -v $PWD/node.toml:/app/config.toml:ro \
  -v $PWD/node_template.yaml:/app/node_template.yaml:ro \
  -v /path/to/your/data:/data:ro \
  -v slakshna-node-data:/app/node-data \
  -v slakshna-models:/app/ml_models \
  -v slakshna-states:/app/ml_states \
  -v slakshna-work:/app/data \
  -v slakshna-logs:/app/logs \
  -v slakshna-cache:/app/cache \
  -v slakshna-hf:/app/hf_cache \
  ghcr.io/himanshu4nagdev/slakshna:latest
```

| Flag | Why |
|---|---|
| `--gpus '"device=0"'` | GPU access (`--gpus all` also works). Needs the NVIDIA Container Toolkit |
| `--shm-size 16g` | Ray / PyTorch shared memory; Docker's default 64 MB is too small |
| `--network host` | Lets the node advertise the machine's real addresses for QUIC hole punching and mDNS. This is the tested configuration |
| `--init`, `--stop-timeout 60` | Reaps Ray/Python child processes and gives them time to exit on `docker stop` |
| `slakshna-node-data` volume | Persists the node identity and peer list across restarts |
| `-e HF_TOKEN=...` | Add it only if your model is gated on Hugging Face |

The files you bind-mount (`node.toml`, template, data) must be readable by uid 1012
(for example `chmod a+r node.toml node_template.yaml`).

First start downloads the model into `slakshna-hf` (network access to Hugging Face). To
pre-download or to run offline, populate that volume once and add `-e HF_HUB_OFFLINE=1`.

Publishing the port instead of host networking (`-p 9000:9000/udp`, without `--network host`)
is possible but **not yet tested across networks**; see section 9.

Check the node:

```bash
docker ps                                   # STATUS should become "healthy"
docker logs -f slakshna-node
curl -s http://127.0.0.1:8545/status        # round, peers, endpoint_id
curl -s http://127.0.0.1:8545/peers         # connected / known peers
```

The first training round starts at the next epoch boundary (the node aligns to the wall
clock, so a start can wait up to `epoch_duration_secs`).

## 6. Startup checks (preflight)

Before the node starts, `preflight.py` checks the configuration, writable directories, the
dataset path from the template, GPU visibility and the NVIDIA driver version. If a check fails
the container exits with code 1 and lists the reasons, for example:

```
[preflight] FAILED - the node was NOT started:
  - Dataset not found: data.train_path = /data/... (mount your dataset read-only at that path)
  - CUDA is not available inside the container. Is the NVIDIA runtime enabled (compose 'deploy.resources.reservations.devices' / --gpus)?
```

**`SLAKSHNA_SKIP_PREFLIGHT=1` (troubleshooting / advanced only, do not use in production).**
It bypasses all of these checks, including the GPU and dataset checks. It is useful only to
inspect the P2P layer on a machine without a GPU (section 8). A node started this way cannot
train, and problems that preflight would have reported appear later as engine errors in the logs.

## 7. Finding the other nodes

mDNS only works on the same LAN. Across networks pick one:

1. Open or forward the machine's **UDP** `p2p_port` (9000) so peers can reach it, with `dht = true`.
2. Set `relay = true`: public relays forward encrypted QUIC traffic when direct connections fail.
3. Give each node explicit seeds: `peers = ["<EndpointId>@<public-ip>:9000"]`.

A node's id: `curl -s http://127.0.0.1:8545/status` (field `endpoint_id`). The startup log also
prints "Any peer can join this federation with: peers = [...]".

## 8. How the weight exchange works

Checked against the repository code (`src/main.rs`, `src/network/mesh.rs`, `ml_engine.py`):

1. Every epoch (aligned to the wall clock) the Rust node starts `python ml_engine.py`, which
   trains LoRA adapters on the GPU.
2. The engine returns a sparse, int8-quantised, base64-encoded weight delta as JSON on its last
   stdout line.
3. The Rust layer records it in its local history and broadcasts it over an `iroh-gossip`
   topic derived from the federation id (maximum message size 10 MiB).
4. Deltas received from peers are written by the Rust layer to
   `<data_dir>/network_deltas/<peer>_delta.b64`.
5. In the next round the engine decodes each peer delta, validates it (norm limit), scores it
   (cosine similarity / validation loss), updates that peer's trust weight and aggregates the
   deltas with those weights.
6. The sync barrier ends when `expected_peers` participants have reported, or at
   `sync_deadline_secs`.

If the ML engine fails in a round, the Rust node logs the error and keeps running its P2P loop;
it does not crash.

Not covered here: the exact record format of reviews/trust records inside gossip and the
meaning of the "top trusted cohort" constant (`src/history.rs`). Confirm those from the
repository documentation.

When `federated.total_epochs` in the training template is reached, the node process exits with
code 0 (the container stops). With `--restart unless-stopped` Docker will start it again.

## 9. Verification

### 9.1 GPU server (RTX A6000, driver 580)

| Area | Status | Notes |
|---|---|---|
| Rust binary + Python + Bhaskera + GPU libraries in one image | ✅ Tested | PyTorch 2.5.1+cu121, CUDA matmul, flash-attn 2.8.3 |
| Container start with the config mounted at runtime | ✅ Tested | preflight, API, health check |
| P2P listener (Iroh QUIC) bound on UDP 9000 inside the container | ✅ Tested | |
| Python engine starts and trains on the GPU inside the container | ✅ Tested | small model, one GPU |
| Several nodes on **one host**, host networking, mDNS discovery | ✅ Tested | 3 nodes, one GPU each |
| Delta exchange and trust weighting between nodes (2 rounds) | ✅ Tested | peers' deltas received and aggregated in round 2 |
| Restart keeps the node identity | ✅ Tested | same EndpointId after restart |
| Graceful stop (SIGTERM) leaves no orphan processes | ✅ Tested | stop took about 1.4 s, exit code 143 |
| 7B template (`batch_size 8`, `seq_len 2048`) on a single 48 GB GPU | ❌ CUDA out of memory | lower `batch_size` / `seq_len` in your template or use more GPU memory |

### 9.2 Non-GPU clean-machine test (image pulled from GHCR)

| Check | Result |
|---|---|
| Pull from GHCR without login | ✅ Image is public; anonymous pull works |
| Digest | ✅ Pulled image Id / RepoDigest is `sha256:8192d231a498…`: bit-for-bit the image tested on the GPU server |
| Contents | ✅ `/usr/local/bin/iiitd` (ELF64 x86-64, 38.7 MB), Python 3.11.17, torch 2.5.1+cu121 (cuDNN 90100), `ml_engine.py`, `federated_communication/` and `node_template.yaml` in `/app`; `docker history` shows 28 layers, not truncated |
| PyTorch without a GPU | ✅ `torch.cuda.is_available()` is `False` (expected); CPU matmul works |
| Preflight without a GPU | ✅ Fails cleanly (exit 1) with "CUDA is not available inside the container. Is the NVIDIA runtime enabled (--gpus)?" and "Dataset not found" |
| P2P layer without a GPU (`SLAKSHNA_SKIP_PREFLIGHT=1`) | ✅ Iroh QUIC mesh, mDNS and Mainline DHT initialise; "Iroh router listening on port 9000"; UDP 9000 bound on the host; `http://127.0.0.1:8545/status` returns live JSON |
| Fault tolerance | ✅ When the ML engine failed, the Rust node did not crash: it caught the error and kept its P2P loop running (container stayed healthy) |
| CUDA training | ⏸ Not testable without a GPU (covered by 9.1) |
| Real weight exchange / convergence between peers | ⏸ Not testable without a GPU |
| GPU preflight driver-version check | ⏸ Not testable without a GPU (covered by 9.1) |
| End-to-end multi-node round | ⏸ Not testable without a GPU |

### 9.3 Not yet tested

| Area | Notes |
|---|---|
| Nodes on **different networks / behind NAT** | Needs validation: UDP reachability, DHT, relay and seed peers |
| Publishing the port with `-p 9000:9000/udp` (bridge networking) across networks | The node would advertise the container address; use seed peers or host networking |
| `relay = true` fallback | |
| Multi-GPU node (`num_gpus` > 1) | |
| Fully offline run (`HF_HUB_OFFLINE=1` with a pre-seeded cache) | The preflight check reports a missing model |
| Other GPU models / driver versions | Only RTX A6000 with driver 580 |
| One node via `docker compose` (`compose.node.yaml`) | Syntax-validated only; the `docker run` command above is the tested form |

Cross-network operation therefore still needs validation on your side. Plan a short test with
two nodes on different networks before running a long job.

## 10. Known gaps

| Gap | Detail |
|---|---|
| Root `docker-compose.yml` | Belongs to a Rust-only deployment and references an `nginx/` folder that is not in the repository. It is not needed for this image |
| UDP port reachability | For peers on other networks the UDP `p2p_port` must be reachable (port forwarding or relay) |
| API exposure | `/status`, `/peers`, ... have no authentication. Keep `host = "127.0.0.1"` or put it behind a proxy/firewall |
| No sample config in the image | Use section 4 of this file |
| Container user | The image runs as uid 1012. Files you bind-mount must be readable by that uid; named volumes need no action |
| Image label | `org.slakshna.commit` carries the commit the image was built from |
| Container storage space | Writes outside the volumes go to the container's writable layer. If the host filesystem that holds Docker's snapshots is full you will see "No space left on device" |
