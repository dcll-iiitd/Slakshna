# SLAKSHNA on Docker

One container = one federated-learning node: the Rust P2P node (`iiitd`) and the
Python ML engine (`ml_engine.py` + Bhaskera) run together, on one GPU (or several).

| You want to... | Use |
|---|---|
| Try the whole thing on one GPU server (3 nodes, 1 GPU each) | `compose.yaml` |
| Same, with 2 nodes | `compose.yaml` + `compose.2node.yaml` |
| Run **one node on this machine** and federate with other machines | `compose.node.yaml` |

All paths below are relative to the `docker/` directory unless noted.

## 1. Requirements

| | |
|---|---|
| OS / arch | Linux x86_64 |
| NVIDIA driver | **>= 525.60.13** (the image ships PyTorch 2.5.1 built for CUDA 12.1) |
| Docker | Engine 24+ with Compose v2 and Buildx |
| GPU support in Docker | [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html); check with `docker info \| grep -i nvidia` |
| Disk | Image is about 12 GB unpacked (about 3.8 GB compressed). Add the model cache (a 7B model in bf16 is about 15 GB) and checkpoints: budget at least 40 GB free. Docker's image store **and its containerd snapshot store** must both have room (see Troubleshooting) |
| GPU memory | Reference setup: one 48 GB GPU per node for the 7B template. The smoke template runs on far less |

## 2. Get the code (the submodule is required)

Bhaskera is a git submodule. A plain `git clone` leaves `Bhaskera/` **empty** and the build
will stop with a clear error.

```bash
git clone --recursive https://github.com/himanshu4nagdev/Slakshna.git
cd Slakshna
# already cloned without --recursive?
git submodule update --init --recursive
ls Bhaskera/pyproject.toml          # must exist
```

## 3. Build the image

Run from the **repository root** (the build context is the root, the Dockerfile is `docker/Dockerfile`).

```bash
SHA=$(git rev-parse --short HEAD)
docker buildx build -f docker/Dockerfile \
  --build-arg UID=$(id -u) --build-arg GID=$(id -g) \
  --build-arg GIT_COMMIT=$SHA \
  --build-arg BHASKERA_COMMIT=$(git -C Bhaskera rev-parse HEAD) \
  -t slakshna:$SHA -t slakshna:dev --load .
```

* `UID`/`GID` make the container user match yours, so mounted data stays readable and
  volumes are owned correctly. Build as the same user that will run compose.
* The build needs internet access to: Docker Hub, `pypi.org`, `download.pytorch.org`,
  `github.com` (flash-attn wheel), `crates.io` and `deb.debian.org` (HTTPS).
* Python dependencies are pinned in `requirements.lock.txt` (identical to the lab training
  environment); base images are pinned by digest. Rebuilding gives the same result.
* Which commits an image was built from:
  `docker inspect slakshna:dev --format '{{json .Config.Labels}}'`

## 4. Configure

```bash
cp env.example .env       # .env is git-ignored; it may hold tokens
```

| Variable | Meaning |
|---|---|
| `SLAKSHNA_IMAGE` | Image to run (default `slakshna:dev`) |
| `NODE1_GPU` `NODE2_GPU` `NODE3_GPU` | Host GPU index per demo node |
| `GPU_DEVICE` | Host GPU index for the single production node |
| `SHM_SIZE` | Shared memory per container (default `16g`) |
| `DATASET_DIR` | Directory with training data, mounted read-only at `/data` |
| `NODE_TEMPLATE` | Training template: `config/node_template.yaml` (7B model) or `config/node_template.smoke.yaml` (tiny model) |
| `NODE_CONFIG` | Production only: path to your node `.toml` |
| `HF_TOKEN`, `HF_HUB_OFFLINE` | Hugging Face token (gated models) / offline mode |
| `WANDB_MODE`, `WANDB_API_KEY` | `offline` by default; set `online` + key to sync |

The training data is **never baked into the image**. Mount your own; it stays on your machine.
The template's `data.train_path` must point inside `/data`.

## 5. Quick pipeline check (3-node demo on one server)

Use the tiny-model template first (set `NODE_TEMPLATE=./config/node_template.smoke.yaml`, the default in `env.example`).

```bash
docker compose --profile init run --rm model-init   # one-time: download the model into the shared cache
docker compose up -d
docker compose ps                                   # wait until all three are "healthy"
curl -s http://127.0.0.1:8545/peers                 # node1: should list the other two (also :8555, :8565)
docker compose logs -f node1                        # look for "Local Training Complete"
docker compose stop                                 # graceful stop (up to 60 s)
```

Why `model-init`: it downloads the model once into the **shared** `hf_cache` volume, so the
nodes do not all download the same weights at the same moment.

Real run: set `NODE_TEMPLATE=./config/node_template.yaml`, run `model-init` again, then `up -d`.
2 nodes instead of 3: `docker compose -f compose.yaml -f compose.2node.yaml up -d node1 node2`.

## 6. Production: one node per machine

Every machine runs one node; nodes find each other and form the federation.

```bash
cp config/node.example.toml config/node.toml        # edit: federation id, node id, expected_peers, ports
cp env.example .env                                 # set NODE_CONFIG=./config/node.toml, DATASET_DIR=/path/to/data, GPU_DEVICE=0
docker compose -f compose.node.yaml --profile init run --rm model-init
docker compose -f compose.node.yaml up -d
docker compose -f compose.node.yaml ps
```

Rules that matter:

* **Same `[federation] id` on every node**, and `expected_peers` = number of nodes in the
  federation. A smaller number than reality ends epochs early; a larger one makes every epoch
  wait the full `sync_deadline_secs`.
* **Never delete the `node-data` volume**: it holds the node's identity (keypair) and known
  peers. Losing it gives the node a new identity.
* GPUs are renumbered inside the container. Keep `gpu_id = 0` and set `num_gpus` to the
  number of GPUs you assigned in compose (`device_ids`).

## 7. Networking

The containers use **host networking**, so the ports in the node `.toml` are the real ports on the machine.

| Port | Protocol | Purpose | Must be reachable by peers? |
|---|---|---|---|
| `p2p_port` (9000) | **UDP** | Iroh QUIC P2P | **Yes** |
| `api_port` (8545) | TCP | REST API (`/status`, `/peers`, ...) | No (bound to 127.0.0.1) |
| `ws_port` (8546) | TCP | WebSocket feed | No |

The API is **unauthenticated**. Keep `host = "127.0.0.1"` unless it sits behind a firewall or reverse proxy.

Peer discovery (`[discovery]`): `mdns` finds nodes on the same LAN or host; `dht` (BitTorrent
mainline DHT) and `dns` find nodes across the internet. `relay = false` by default, so when
nodes sit behind different NATs pick one:

1. open the UDP `p2p_port` on the machine's firewall / port-forward it, or
2. set `relay = true` in `[discovery]` (public relays forward encrypted traffic only), or
3. give each node its peers explicitly: `peers = ["<EndpointId>@<ip>:<port>"]`.
   A node's id: `curl -s http://127.0.0.1:8545/status` -> `endpoint_id`.

## 8. Offline / air-gapped model cache

The model downloads from Hugging Face on first use (the 7B template uses
`trust_remote_code`, so remote code is fetched with the weights). For machines without
internet:

1. On a machine with internet run `model-init` once (this fills the `hf_cache` volume).
2. Copy the volume to the target machine, e.g.
   `docker run --rm -v slakshna_hf_cache:/c -v "$PWD":/o alpine tar -C /c -czf /o/hf_cache.tgz .`
   and restore it into the target volume the same way.
3. Set `HF_HUB_OFFLINE=1` in `.env`. The preflight check tells you if the model is missing.

## 9. Moving the image to another machine

| Method | Commands |
|---|---|
| Build locally | Section 3 on each machine |
| Tar file | `docker save slakshna:dev \| zstd -T0 -o slakshna.tar.zst` then on the target: `zstd -dc slakshna.tar.zst \| docker load` |
| Registry | `docker tag slakshna:dev registry.example.com/slakshna:TAG && docker push registry.example.com/slakshna:TAG`, then set `SLAKSHNA_IMAGE=registry.example.com/slakshna:TAG` |

## 10. Operating it

| Task | Command |
|---|---|
| Logs | `docker compose logs -f node1` (production: `-f compose.node.yaml logs -f node`) |
| Stop, keep data | `docker compose stop` / `docker compose down` |
| Start again | `docker compose up -d` |
| Upgrade | rebuild the image (section 3), then `docker compose up -d` |
| **Delete all data** | `docker compose down -v` (removes model checkpoints **and node identities**) |

What is persisted (named volumes): `node-data` (identity, peer DB), `ml_models` (checkpoints),
`ml_states`, `data` (work files), `logs`, `cache` (tokenized data), and the shared `hf_cache`.

## 11. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Build stops: `Bhaskera/pyproject.toml is missing` | Submodule not initialised: `git submodule update --init --recursive` |
| Build: `403 Forbidden` from `deb.debian.org` | A firewall blocks plain-HTTP package downloads. The Dockerfile already uses HTTPS; check your proxy allows `deb.debian.org:443` |
| Build / pull: `x509: certificate signed by unknown authority` on a layer | Intermittent on some networks. Retry; `docker pull` of the base images first helps |
| Container exits with `[preflight] FAILED` | Read the listed reasons: config path, unwritable dir, dataset path, GPU, driver |
| `CUDA is not available inside the container` | NVIDIA Container Toolkit not configured, or the GPU index in `.env` does not exist |
| `driver ... is too old for CUDA 12.1` | Update the NVIDIA driver to >= 525.60.13 |
| `Directory not writable by uid ...` | Image built with a different `UID` than the user running compose; rebuild with `--build-arg UID=$(id -u)` |
| `No space left on device` inside a container (e.g. `mkdtemp('/tmp')`) while the data volumes have room | A container's writable layer lives in containerd's snapshot store, which can be on a different (full) filesystem than `Docker Root Dir`. Check with `docker run --rm --entrypoint df slakshna:dev -h /` and free space there, or move containerd's `root` to a larger disk |
| Nodes never see each other | Section 7: UDP port, `relay`, or seed `peers`; make sure `[federation] id` is identical |
| Every epoch waits ~5 min | `expected_peers` is larger than the number of running nodes |
| Out of GPU memory | Lower `batch_size`/`seq_len` in the template, or use the smoke template to verify the pipeline first |

## 12. Security notes

* Membership of the `docker` group is equivalent to root on the host.
* Keep `.env` out of git (it is ignored) and out of images.
* The REST/WebSocket API has no authentication: do not expose it to untrusted networks.
* P2P traffic is encrypted end-to-end (QUIC/TLS 1.3, Ed25519 identities); relays and tunnels only see ciphertext.
