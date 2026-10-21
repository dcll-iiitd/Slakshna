# SLAKSHNA Node: User Manual

This manual takes you from an empty machine to a running SLAKSHNA node, one step at a time.
It is the friendly companion to `CLIENT_SETUP.md`, which remains the detailed technical
reference. If the two ever disagree, `CLIENT_SETUP.md` is authoritative.

**What you will end up with:** one Docker container on your GPU machine that trains a model on
your own data and exchanges model updates (not data) with the other nodes of your federation.

**Time needed:** about 30 minutes plus download time (about 3.8 GB for the image, plus the model).

---

## Step 1. Check the prerequisites

Run each command on the machine that will host the node.

| # | Requirement | Check with | You should see |
|---|---|---|---|
| 1 | Linux on x86_64 | `uname -sm` | `Linux x86_64` |
| 2 | An NVIDIA GPU | `nvidia-smi -L` | at least one `GPU 0: ...` line |
| 3 | NVIDIA driver **525.60.13 or newer** | `nvidia-smi --query-gpu=driver_version --format=csv,noheader` | a version >= 525.60.13 |
| 4 | Docker Engine **24 or newer** | `docker --version` | `Docker version 24.x` or higher |
| 5 | NVIDIA Container Toolkit | `docker info \| grep -i nvidia` | a line mentioning `nvidia` (runtime) |
| 6 | About **40 GB** free disk where Docker stores data | `df -h "$(docker info --format '{{.DockerRootDir}}')"` | 40 GB or more available |

If item 5 shows nothing, install the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
before continuing. Without it the container cannot see the GPU.

---

## Step 2. Pull the image

The image is public; no login or token is needed.

```bash
docker pull ghcr.io/himanshu4nagdev/slakshna:latest
```

Confirm you received exactly the published image:

```bash
docker image inspect ghcr.io/himanshu4nagdev/slakshna:latest --format '{{index .RepoDigests 0}}'
```

Expected output:

```
ghcr.io/himanshu4nagdev/slakshna@sha256:8192d231a498ce80177ab8c9dde0ddc0ed3d1799e8802ed18206ebd61b763fea
```

Confirm the container can use your GPU:

```bash
docker run --rm --gpus all --entrypoint python ghcr.io/himanshu4nagdev/slakshna:latest \
  -c "import torch; print('GPU available:', torch.cuda.is_available())"
```

Expected output: `GPU available: True`. If it prints `False`, revisit Step 1, items 2, 3 and 5.

`:latest` and `:v1` are the same image today. Use `:v1` if you want a tag that will never move.

---

## Step 3. Prepare your three input files

Create a working folder, for example `~/slakshna`, and put these files in it:

| File | What it is | Where it goes in the container |
|---|---|---|
| `node.toml` | Who this node is and how it finds its peers | `/app/config.toml` |
| `node_template.yaml` | The training recipe: model, data path, batch size, number of rounds | `/app/node_template.yaml` |
| your dataset folder | Your training data, never shared with other nodes | `/data` (read-only) |

The image contains **no sample `node.toml`** and its built-in training template is
lab-specific. You always provide both files yourself; you never rebuild the image.

### 3a. `node.toml`: the node configuration

Copy the block below into `~/slakshna/node.toml`. It is complete and valid as written.
Change only the three lines marked `CHANGE`.

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
sparsity = 0.1
quantization = "symmetric_int8"
allow_legacy_delta_format = false
max_payload_bytes = 7340032
max_tensor_elements = 2000000000

[node]
id = "client-node-1"                  # CHANGE: a readable name for this node
data_dir = "/app/node-data"           # do not change
gpu_id = 0                            # do not change (the GPU you pass in always appears as 0)
num_gpus = 1

[network]
host = "127.0.0.1"
p2p_port = 9000
ws_port = 8546
api_port = 8545
peers = []

[discovery]
mdns = true
dht = true
dns = true
relay = false

[logging]
level = "info"
```

The three values to change:

1. **`[federation] id`**: a name for your federation. Every node that should train together
   must use **exactly the same id**. Agree on it with the other participants. Pick something
   specific (not just `test`), because nodes with the same id can find each other over the internet.
2. **`[training] expected_peers`**: the **total number of nodes** in the federation, including
   this one. If it is wrong, every round waits the full `sync_deadline_secs` (5 minutes).
3. **`[node] id`**: any readable name for this machine, for example `hospital-a-node`.

The same file is kept in the repository as
[`docker/config/node.example.toml`](https://github.com/himanshu4nagdev/Slakshna/blob/docker-setup/docker/config/node.example.toml)
(branch `docker-setup`). `CLIENT_SETUP.md`, section 4, explains every field.

### 3b. `node_template.yaml`: the training template

**Why you need it.** The image has a built-in template at `/app/node_template.yaml`, but it
points to dataset paths that exist only on the developers' lab server. You replace it by
mounting your own file at the same path (Step 4 does this). No rebuild is needed.

**Where to start.** Use one of the two templates in the repository (branch `docker-setup`):

| Template | Model | Use it for |
|---|---|---|
| [`docker/config/node_template.smoke.yaml`](https://github.com/himanshu4nagdev/Slakshna/blob/docker-setup/docker/config/node_template.smoke.yaml) | `Qwen/Qwen2.5-0.5B` (small), 4 steps, 1 round | A first end-to-end check that everything works. Tested |
| [`docker/config/node_template.yaml`](https://github.com/himanshu4nagdev/Slakshna/blob/docker-setup/docker/config/node_template.yaml) | `allenai/OLMo-2-1124-7B`, 16 steps, 8 rounds | The full-size recipe. See the GPU memory note below before using it |

**What is inside.** These are the fields of the repository template. The ones you are most
likely to change are in bold.

| Field | Repository value | Meaning |
|---|---|---|
| `federated.total_epochs` | **8** | Number of federated rounds. When reached, the node stops (the container exits with code 0) |
| `federated.optimizer` | `muon` | Optimizer used with LoRA. Leave as is |
| `federated.mode` | `finetuning` | Leave as is |
| `model.name` | **`allenai/OLMo-2-1124-7B`** | Hugging Face model to fine-tune |
| `model.dtype` | `bfloat16` | Leave as is |
| `data.name` | `local` | Your own files. Leave as is |
| `data.train_path` | **`/data/South_Asia.jsonl`** | Your training file. **Must start with `/data/`** |
| `data.format` | `chatml` | Chat-style records (see 3c) |
| `data.seq_len` | **2048** | Tokens per training sequence |
| `data.max_train_samples` | 10000 | Sample cap passed to the data pipeline; its exact effect is not covered here. Leave as is unless the training framework documentation says otherwise |
| `data.tokenized_path` | `/app/cache/tokenized` | Where the tokenized copy of your data is cached. Use a new folder name if you change the dataset or `seq_len` |
| `lora.enabled`, `rank`, `alpha`, `dropout`, `target_modules` | `true`, 16, 64, 0.03, `["q_proj","v_proj"]` | LoRA adapter settings |
| `training.batch_size` | **8** | Sequences per step. The main lever for GPU memory |
| `training.gradient_accumulation_steps` | **4** | Steps accumulated before an update |
| `training.max_steps` | **16** | Training steps per round |
| `training.learning_rate` | `3e-4` | |
| `logging.tracker` | `wandb` | Logs stay local (offline mode is the default in the image) |

You do **not** need to set checkpoint locations: the engine sets the checkpoint and cache
directories itself.

**GPU memory: read this before a long run.**

| Configuration | Result in our tests |
|---|---|
| Small template (`Qwen2.5-0.5B`, `seq_len` 2048, `batch_size` 2) on one 48 GB GPU | Works. Peak about 13.6 GB |
| 7B template (`batch_size` 8, `seq_len` 2048) on one 48 GB GPU | **Fails: CUDA out of memory** |

If your GPU has 48 GB or less, do one of the following before using the 7B template:
lower `training.batch_size` (and raise `gradient_accumulation_steps` if you want the same
effective batch), lower `data.seq_len`, or choose a smaller `model.name`. Which exact values
fit a 48 GB GPU for the 7B model has **not yet been tested**; start small and increase.

Do not set `data.seq_len` very low. With packed chat data a short window can contain only the
prompt part of each conversation, which leaves nothing to learn from and shows up as
`loss=nan` in the logs. Keep 2048 unless you have a reason to change it.

### 3c. Your dataset

* Put your training file(s) in one folder on the host, for example `/srv/slakshna-data/`.
* The folder is mounted **read-only** at `/data`, so a host file
  `/srv/slakshna-data/train.jsonl` appears in the container as `/data/train.jsonl`. Set
  `data.train_path` to that container path.
* Your data never leaves your machine; only model updates are exchanged.

**Format.** The image was tested with JSON Lines (one JSON object per line) where each record
has a `messages` list of `{role, content}` entries with the roles `system`, `user` and
`assistant`:

```json
{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "A question"}, {"role": "assistant", "content": "The answer"}]}
```

Training learns from the `assistant` replies. Extra fields on a record are allowed (the
tested dataset carries extra fields). Other layouts have not been verified with this image.

---

## Step 4. Run the node

Make your files readable by the container user (uid 1012):

```bash
chmod a+r ~/slakshna/node.toml ~/slakshna/node_template.yaml
chmod -R a+rX /srv/slakshna-data
```

Start the node (replace `/srv/slakshna-data` with your data folder):

```bash
cd ~/slakshna
docker run -d --name slakshna-node --init --restart unless-stopped \
  --network host --gpus '"device=0"' --shm-size 16g --stop-timeout 60 \
  -v $PWD/node.toml:/app/config.toml:ro \
  -v $PWD/node_template.yaml:/app/node_template.yaml:ro \
  -v /srv/slakshna-data:/data:ro \
  -v slakshna-node-data:/app/node-data \
  -v slakshna-models:/app/ml_models \
  -v slakshna-states:/app/ml_states \
  -v slakshna-work:/app/data \
  -v slakshna-logs:/app/logs \
  -v slakshna-cache:/app/cache \
  -v slakshna-hf:/app/hf_cache \
  ghcr.io/himanshu4nagdev/slakshna:latest
```

What each part does:

| Part | Purpose |
|---|---|
| `--name slakshna-node` | A fixed name so the later commands work |
| `--init` | Cleans up the training processes correctly |
| `--restart unless-stopped` | Starts the node again after a reboot or after it finishes its rounds |
| `--network host` | Uses the machine's network directly, so peers can reach the node. This is the tested setup |
| `--gpus '"device=0"'` | Gives the container GPU 0 (`--gpus all` also works) |
| `--shm-size 16g` | Shared memory for the training framework; the Docker default is too small |
| `--stop-timeout 60` | Gives training up to 60 seconds to shut down on `docker stop` |
| `-v .../node.toml:/app/config.toml:ro` | Your node configuration |
| `-v .../node_template.yaml:/app/node_template.yaml:ro` | Your training template (replaces the built-in one) |
| `-v /srv/slakshna-data:/data:ro` | Your dataset, read-only |
| `-v slakshna-node-data:...` | **The node identity.** Never delete this volume |
| `-v slakshna-models`, `-states`, `-work`, `-logs`, `-cache` | Checkpoints, state, logs and the tokenized-data cache, kept across restarts |
| `-v slakshna-hf:/app/hf_cache` | Downloaded model files, so they are fetched only once |

If your model is gated on Hugging Face, add `-e HF_TOKEN=<your token>` to the command.

On first start the node downloads the model from Hugging Face. This needs internet access
and can take several minutes for a large model.

---

## Step 5. Check that it is running

| # | Command | What to look for |
|---|---|---|
| 1 | `docker ps --filter name=slakshna-node` | `STATUS` becomes `Up ... (healthy)` within a few minutes |
| 2 | `docker logs -f slakshna-node` | `[preflight] all checks passed`, then `Iroh router listening on port 9000` and `API: http://127.0.0.1:8545` |
| 3 | `curl -s http://127.0.0.1:8545/status` | JSON with your `federation_id`, the current `round`, `peers` and this node's `endpoint_id` |
| 4 | `curl -s http://127.0.0.1:8545/peers` | `connected` / `known` lists the other nodes once they are found |

Training does not start the moment the container starts. The node waits for the next round
boundary (up to `epoch_duration_secs`, 10 minutes by default). When a round has run, the logs
show `Local Training Complete! Score: ...`. In later rounds you will also see
`Network Delta Extracted` lines: those are model updates received from your peers.

To stop the node: `docker stop slakshna-node`. To start it again: `docker start slakshna-node`.

---

## Step 6. Connect several nodes

Every node in the federation needs:

1. the **same `[federation] id`**,
2. `expected_peers` set to the **total number of nodes**,
3. its own `node.id`, its own data and its own GPU.

**Nodes on the same local network** find each other automatically (mDNS). Nothing else to do.

**Nodes on different networks** (different sites, behind routers/NAT) need one of these:

| Option | What to do |
|---|---|
| A. Open the UDP port | Allow or forward **UDP port 9000** to the node's machine on each site (keep `dht = true`) |
| B. Use relays | Set `relay = true` in `[discovery]`. Public relays forward the (encrypted) traffic when a direct connection is not possible |
| C. Seed peers | Put the other nodes' addresses in `peers`, for example `peers = ["<EndpointId>@203.0.113.10:9000"]`. Get a node's `EndpointId` from `curl -s http://127.0.0.1:8545/status` on that node |

> **Not yet tested:** running nodes across different networks (options A, B and C) has not been
> validated yet. Before a long job, run a short test with two nodes on different networks and
> confirm in `curl -s http://127.0.0.1:8545/peers` that they see each other.

---

## Step 7. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Container exits; logs show `CUDA is not available inside the container` | The container cannot see the GPU | Install/configure the NVIDIA Container Toolkit; keep `--gpus` in the command; re-run the GPU check in Step 2 |
| Logs show `NVIDIA driver ... is too old for CUDA 12.1` | Old driver | Update the NVIDIA driver to 525.60.13 or newer |
| Logs show `Dataset not found: data.train_path = ...` | The path in the template does not match the mount | Make sure `data.train_path` starts with `/data/` and the file exists in the folder you mounted with `-v ...:/data:ro` |
| Logs show `Node config not found` or a TOML error | `node.toml` missing, wrong path, or a typo | Check the `-v $PWD/node.toml:/app/config.toml:ro` part and compare your file with Step 3a |
| Logs show `Directory not writable` or `Permission denied` on your files | File permissions | Run the `chmod` commands from Step 4 |
| `CUDA out of memory` in the logs | Model too large for the GPU at these settings | Lower `training.batch_size` and/or `data.seq_len`, or use a smaller model (Step 3b) |
| `loss=nan` in the training logs | Training windows contain no answer text | Keep `data.seq_len` at 2048 (Step 3b) and use a fresh `data.tokenized_path` folder |
| `No space left on device` | The disk that holds Docker's data is full | Free space there (Step 1, item 6) |
| Model download fails | No access to Hugging Face, or a gated model | Allow internet access, add `-e HF_TOKEN=...` for gated models. For offline machines see `CLIENT_SETUP.md` (offline mode is not yet tested) |
| Each round waits about 5 minutes | `expected_peers` is larger than the number of running nodes | Set it to the real node count |
| Nodes never see each other | Different federation ids, or a network barrier | Check that the ids match exactly; for different networks see Step 6 |

**Advanced only, not for production:** setting `-e SLAKSHNA_SKIP_PREFLIGHT=1` skips all startup
checks, including the GPU and dataset checks. It exists only for diagnosing the network layer on
a machine without a GPU. A node started this way cannot train.

---

## Step 8. What has been tested

| Area | Status |
|---|---|
| Image integrity: same image digest after pulling from the registry on a clean machine | Tested |
| Image contents: networking node, Python training engine and GPU libraries in one image | Tested |
| Training on a GPU inside the container | Tested (one GPU, RTX A6000) |
| Several nodes on one machine finding each other and exchanging model updates | Tested (3 nodes, 2 rounds) |
| Node identity kept after a restart; clean shutdown | Tested |
| Behaviour without a GPU: clear preflight error; networking layer still starts when checks are skipped | Tested |
| 7B template at `batch_size` 8, `seq_len` 2048 on one 48 GB GPU | Fails (out of memory) |
| Nodes on different networks / behind NAT | **Not yet tested** |
| More than one GPU per node | **Not yet tested** |
| Fully offline operation | **Not yet tested** |
| Other GPU models and driver versions | **Not yet tested** |

The full list with details is in `CLIENT_SETUP.md`, section 9.
