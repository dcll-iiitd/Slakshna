# SLAKSHNA — Decentralized Geo-Localised Personalized Federated Learning

A **Peer-to-Peer Federated Learning Framework** built in **Rust** and integrated with a Python Machine Learning Engine (**Bhaskera**). **SLAKSHNA** enables decentralized, privacy-preserving, weighted-aggregation Federated Learning (FL) without centralized aggregators or synchronous blocking rounds. It runs across geo-localized machines and institutional clusters (including SLURM-managed supercomputers, kubernetes managed clusters) separated by complex firewalls, securely sharing compressed model updates without any central coordinator.

---

## Key Features & Architectural Highlights

- **Asynchronous P2P Training**  
  Instead of traditional synchronous FL rounds waiting for slow participants, SLAKSHNA operates asynchronously. Nodes continuously train on local data, broadcast compressed model deltas to the network, and evaluate peers dynamically.

- **Iroh QUIC Mesh & Gossip Network (`iroh-gossip`)** 
  Built on **Iroh v1.0.2**, the framework utilizes **QUIC (Quick UDP (User Datagram Protocol) Internet Connections)** transport, direct NAT (Network Address Translation) traversal (STUN/DERP), and `iroh-gossip` topic swarms. Nodes discover peers dynamically using cryptographic Ed25519 `EndpointId` public keys.

- **No Bootstrap Node**  
  Every participant is identical: there is no coordinator, no first node, and no node type. Membership follows from the shared `[federation] id` alone. Nodes find each other through **mDNS** on the local network, the **BitTorrent mainline DHT** and pkarr/DNS globally, and **gossip peer exchange** once connected — and each node remembers every peer it has ever met in its own RocksDB, so a restarted federation reassembles itself with no configuration and no privileged host.

- **Universal Firewall & VPN Traversal (`Playit.gg`)**  
  Academic and enterprise networks (such as university campus firewalls or remote VPNs) often block inbound UDP/TCP hole-punching and standard DERP relay traffic. SLAKSHNA natively supports static public UDP/TCP tunneling via **Playit.gg**, providing fixed, persistent public addresses (`<ip>:<port>`) for nodes across different cities without requiring root/sudo access or complex router configurations.

- **Bhaskera ML Engine (`ml_engine.py`)**  
  A robust Python engine bridging the Rust networking layer with distributed GPU/CPU training. Powered by **Ray Train (`TorchTrainer`)**, **PyTorch**, and **parameter efficient training algorithms**, it executes local pre-training, and fine-tuning on tokenized datasets while streaming real-time epoch loss tracking. During training it can offload the optimizer states to perform **concurrent evals** on the model.

- **SLURM Supercomputer & Multi-Core Cluster Support**  
  Fully compatible with HPC SLURM clusters (`srun` / `sbatch`). SLURM isolates allocated GPUs seamlessly and maps to cluster-assigned resources without port collisions or resource deadlocks.

- **Weighted Aggregation**  
  Peers asynchronously evaluate incoming model updates by computing cosine similarity against their local gradient direction and tracking validation loss improvements. Nodes dynamically update peer weights (`state["alpha"]` and normalized `w_i` weights) and aggregate updates based on these weights. It checks not only malicious updates but also provides foundation for mitigating catastrophic forgetting.

- **Sparsification and Compression**  
  Before broadcasting over the P2P network, local weight updates are sparsified to retain only the most significant weights (e.g., `sparsity=0.01`). The sparse tensors are encoded (e.g., `fp16`, `fp8`) and base64 compressed, reducing network bandwidth requirements by over 98%.

- **Post-training RL (GRPO)**  
  Besides SFT, a node can improve the shared LoRA adapter with reinforcement learning against a verifiable reward (e.g., a correct final answer). Only the sparse LoRA delta is shared. See [Post-training RL](#post-training-rl-grpo--branch-rl).

- **Differential Privacy (DP)**  
  L2 norm clipping, Gaussian noise etc. augmented to ensure Differential Privacy for local gradients protected against membership inference and model inversion attacks. Our differential privacy component also allows integrating `opacus` (`PrivacyEngine`) and `opt-einsum`.

---

## Security & Privacy Architecture

SLAKSHNA is built from the ground up to operate securely over untrusted public networks, proxies, and shared supercomputers:

1. **End-to-End Cryptographic Transport (`TLS 1.3 over QUIC`)**  
   Every node generates an `Ed25519` cryptographic keypair upon startup (`src/network/mesh.rs`). All communication across the Iroh mesh, whether sent directly via local IPs or routed across public internet tunnels like `Playit.gg`, is wrapped in end-to-end **TLS 1.3** encryption.
   - **Zero-Trust Tunnels:** Public proxy services (`Playit.gg`) act purely as raw packet forwarders. They cannot read, decrypt, or tamper with model weights because they do not hold the private keys. This mechanism consistently saves both subscription and storage on services such as Cloudflare.

2. **Poisoning Defense**  
   To prevent adversarial nodes from ruining the global model (`Model Poisoning`), SLAKSHNA does not use simple averaging. When a node receives a peer's delta, `ml_engine.py` evaluates the update against local validation metrics (`Cosine Similarity` & `Validation Loss`). If a node submits poisoned or erratic updates, its trust score (`alpha`) drops, rendering its weight in the Federated Averaging formula close to `0.0`.

3. **Differential Privacy against Data Reconstruction**  
   By combining sparsification/compression with differential privacy, raw local dataset samples (e.g., chat dataset, patient records, etc.) can never be reconstructed by eavesdroppers or peer nodes.

---

## System Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Axum HTTP & WS Server                           │
│                     (Node Status, Leaderboard)                         │
├──────────────────────────────────┬─────────────────────────────────────┤
│         Rust P2P Engine          │           Python ML Engine          │
│                                  │                                     │
│  • Iroh Mesh & Gossip Protocol   │  • ml_engine.py Bridge              │
│  • Decentralized Sync            │  • Bhaskera (Ray Train / PyTorch)   │
│  • Local State Persistence       │  • LoRA Fine-Tuning & SparseLoCo    │
│  • Asynchronous Evaluation       │  • Differential Privacy             │
├──────────────────────────────────┴─────────────────────────────────────┤
│                    Iroh Network (`iroh-gossip`)                        │
│          (QUIC / Ed25519 TLS 1.3 / mDNS / STUN / Playit.gg)            │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Technology Stack

| Layer                    | Technologies Used                                                                                                                                            |
| :----------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Networking Core**      | **Rust** (`edition = 2021`), **Tokio** async runtime                                                                                                         |
| **P2P Communication**    | **Iroh** (`iroh v1.0.2`, `iroh-gossip`, `iroh-relay`), **QUIC**, **Ed25519 TLS 1.3**, **Playit.gg** (Static Tunnels)                                         |
| **Peer Discovery**       | **mDNS** (`iroh-mdns-address-lookup`), **BitTorrent mainline DHT** (`iroh-mainline-address-lookup`), **pkarr/DNS**, gossip peer exchange, on-disk peer cache |
| **API & WebSockets**     | **Axum 0.7**, **Hyper**, **tokio-tungstenite** (`WebSocket`), **Serde / Serde JSON**                                                                         |
| **ML Engine & FL**       | **Python 3.11+**, **PyTorch**, **Ray / Ray Train** (`ray.train.torch.TorchTrainer`), **setproctitle**                                                        |
| **Transformers & PEFT**  | **HuggingFace Transformers**, **PEFT** (`LoRA`), **PyArrow** (Parquet caching), **PyYAML**                                                                   |
| **Differential Privacy** | **Gradient clipping**, **Noice injection**, **Opacus** (`PrivacyEngine`), **opt-einsum**                                                                     |

---

## Repository Structure

| Path                       | Description                                                                                                                     |
| :------------------------- | :------------------------------------------------------------------------------------------------------------------------------ |
| `src/main.rs`              | Rust Node entry point, federated epoch loop, P2P orchestrator, and network broadcast                                            |
| `src/history.rs`           | Local update history — the per-peer append-only log of model updates and peer reviews                                           |
| `src/identity.rs`          | Ed25519 keypair and the node's stable federation identity (`NodeId`)                                                            |
| `src/network/`             | Iroh QUIC + Gossip network implementation (`mesh.rs`, `mod.rs`) — peer discovery, gossip swarm, and secure peer synchronization |
| `src/api.rs`               | Axum HTTP REST endpoints and real-time WebSocket broadcast server for dashboards                                                |
| `src/config.rs`            | TOML configuration loader for federation, training cadence, network ports, IDs, and storage paths                               |
| `ml_engine.py`             | Python ML bridge executing Bhaskera distributed LoRA training, DP clipping, sparsification, and peer evaluation                 |
| `setup.sh`                 | Main installation script for system dependencies and virtual environments                                                       |
| `Bhaskera/`                | Submodule / embedded repository containing the distributed LLM training framework                                               |
| `node_template.yaml`       | Base YAML template for HuggingFace / Ray / PEFT training arguments                                                              |
| `node1.toml`               | Node-1 configuration file                                                                                                       |
| `node2.toml`, `node3.toml` | Peer node configuration files                                                                                                   |
| `logs/`                    | Directory containing runtime communication logs and real-time epoch loss tracking CSVs                                          |
| `plots_script/`            | Python scripts for visualizing training metrics, trust scores, and system performance                                           |
| `examples/2node-rl/`       | Two-node federated RL (GRPO) demo configs; `examples/4node-sft-olmo/` holds the 4-node OLMo-2 SFT configs                       |
| `experiments/`             | `fed_sim.py` multi-node simulator and `run_experiments.py` job runner for the RL/SFT federation experiments                     |
| `slakshna_eval.py`         | Held-out NLL / perplexity / bits-per-byte for chat fine-tuning                                  |
| `tests/`                   | RL integration, simulator, norm-policy and evaluation tests                                                                     |
| `results/`                 | Output directory containing generated plots and evaluation metric graphs                                                        |
---

## Environment & Prerequisites Setup

When setting up on a machine where Rust, Cargo, or Python are installed in custom directories (such as `/mnt/disk1/...` or scratch drives), export your environment variables before compiling or running:

```bash
# 1. Point to your Rust & Cargo installation
export CARGO_HOME=/mnt/disk1/slakshna/rust/.cargo
export RUSTUP_HOME=/mnt/disk1/slakshna/rust/.rustup
export PATH=$CARGO_HOME/bin:$PATH

# 2. Activate Python Environment (e.g., using uv, poetry, etc.)
if [ -f "/mnt/disk1/slakshna/Bhaskera/bhaskera-activate.sh" ]; then
    source /mnt/disk1/slakshna/Bhaskera/bhaskera-activate.sh
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi
```

### Installation & Build

Follow these exact steps in sequence to set up the environment and build the project:

```bash
# 1. Initialize the Bhaskera submodule
git submodule update --init --recursive

# 2. Navigate to the Bhaskera ML engine directory
cd Bhaskera

# 3. Run the Bhaskera setup script (installs python dependencies)
bash setup.sh

# 4. Activate the Bhaskera Python virtual environment
source bhaskera-activate.sh

# 5. Move back to the SLAKSHNA-FL root directory
cd ..

# 6. Run the SLAKSHNA root setup script
bash setup.sh

# 7. Build the Rust P2P node binary in release mode
# (Make sure you have Rust and Cargo installed!)
cargo build --release

# 8. You are now ready to run your nodes!
# (e.g. ./target/release/iiitd --config config.toml)
```

---

## Running a Local Multi-Node Cluster

To make local testing and simulations easy, you can use the provided cluster scripts. We have pre-configured 2-node, 4-node, and 6-node configurations in the `examples/` directory.

### Requirements
- **Rust / Cargo**: Required to compile the node binary (done automatically by the script if not found).
- **Python / Bhaskera**: The Python environment must be set up properly as shown in the Installation steps. The script will automatically source `.venv` or `bhaskera-activate.sh`.

### Starting the Cluster

To start an entire cluster with a single command, use `start_cluster.sh` and pass the number of nodes (2, 4, or 6):

```bash
./start_cluster.sh 4
```

This will run all 4 nodes in the background. The script outputs logs directly into the `logs/` directory.
You can monitor the logs of any specific node like this:

```bash
tail -f logs/node1_cluster.log
```

### Stopping the Cluster

To safely shut down all the nodes started by the script, simply run:

```bash
./stop_cluster.sh
```

---

## TOML Configuration Breakdown

Every node requires its own `.toml` configuration file (`config.toml`, `node2.toml`, etc.).

Every node file has the same shape. **There is no bootstrap node and no node type** —
membership is decided entirely by the shared `[federation] id`.

```toml
[federation]
id = "slakshna-fl-1"        # Nodes sharing this id derive the same gossip topic and train together
name = "SLAKSHNA Federation"

[training]
epoch_duration_secs = 600   # Length of one federated epoch (aligned to the wall clock)
sync_deadline_secs = 300    # How far into the epoch to wait for peer updates before aggregating
expected_peers = 6          # Release the sync barrier early once this many nodes have reported

[node]
id = "node-1"
data_dir = "./data-node1"   # Dedicated delta storage directory
gpu_id = 0                  # GPU assigned to this node for local training

[network]
host = "0.0.0.0"
p2p_port = 9000             # Iroh QUIC router listening port
api_port = 8545             # Axum HTTP REST API port
ws_port = 8546              # WebSocket port
peers = []                  # Optional seeds; empty is fine — see "Joining a federation".

[discovery]
mdns = true                 # Zero-config discovery of federation members on the local network
dht = true                  # Serverless global lookup over the BitTorrent mainline DHT
dns = true                  # Number 0's public pkarr/DNS lookup
relay = true                # Public relays as a fallback transport when hole punching fails

[logging]
level = "info"
```

> Every node in a federation must share the same `[federation] id` — it is hashed into the
> 32-byte gossip topic *and* into the mDNS service label, so a mismatch silently puts nodes
> in different swarms.

### Joining a federation

A node never depends on a particular other node being up. Start it with `peers = []`
and it comes up alone, advertises itself, and merges into the swarm as soon as any
member appears. Four independent mechanisms find peers, in rough order of speed:

| Mechanism                                                       | Scope                 | Depends on                                                      |
| :-------------------------------------------------------------- | :-------------------- | :-------------------------------------------------------------- |
| **mDNS** (`discovery.mdns`)                                     | Same LAN or same host | Nothing — multicast only                                        |
| **Remembered peers**                                            | Anywhere              | `{data_dir}/rocksdb`; every peer ever met is redialled on start |
| **Gossip peer exchange**                                        | Anywhere              | One reachable member; the swarm hands over the rest             |
| **Mainline DHT / pkarr DNS** (`discovery.dht`, `discovery.dns`) | Internet              | Public DHT; no server we operate                                |

On a LAN, that means nothing to configure: launch the nodes in any order and they find
each other. Across the internet, hand any one member's EndpointId to a joining node —
any member, not a designated one:

```toml
[network]
peers = ["a65a49db0894467a3b6d95eda3924c309a5589e265f734332f2b65100364be90"]
```

Each node prints its own EndpointId at startup and serves it at `GET /status`;
`GET /peers` lists both live neighbours and the remembered set. The EndpointId is
derived from the keypair in `{data_dir}/rocksdb`, so **it is stable across restarts** —
publish it once and it stays valid.

Peers may still be pinned to an explicit address with `<EndpointId>@<host>:<port>` when
no discovery mechanism can reach them (see the tunnelling section below).

---

## Managing the Bhaskera Submodule (Contributor Guide)

The `Bhaskera` ML engine is included as a Git submodule.

### 1. Cloning & Pulling Updates
Whenever you start working or pull updates from teammates, you must update *both* the parent repo and the submodule.

**Cloning for the first time:**
```bash
git clone --recursive https://github.com/dcll-iiitd/Slakshna.git
```

**Pulling updates from a branch:**
To avoid "untracked files" or "unstaged changes" errors, always update the submodule after pulling or switching branches:
```bash
git pull
git submodule update --init --recursive
```
*(You can configure git to do this automatically by using `git pull --recurse-submodules`)*

### 2. Making changes OUTSIDE Bhaskera (e.g., Rust code, config files)
If you are only editing files in the main `Slakshna` directory (like `README.md` or `node_template.yaml`), the workflow is standard:
```bash
git add .
git commit -m "Update README"
git push
```

### 3. Making changes INSIDE Bhaskera
If you edit code *inside* the `Bhaskera` folder, you must push the code twice: once for the submodule to save the actual code, and once for the parent to "bookmark" that new commit.

**Step 1: Push the actual code (Inside Bhaskera):**
```bash
cd Bhaskera
git add .
git commit -m "Fixed bug"
git push origin Slakshna
```
(here `Slakshna` is the branch name of Bhaskera; the RL work lives on Bhaskera's `slakshna-rl` branch, so on the `RL` branch of this repo push `git push origin slakshna-rl` instead)

**Step 2: Tell the parent about it (Back in Slakshna):**
```bash
cd ..
git add Bhaskera
git commit -m "Update Bhaskera to latest commit"
git push
```
> **⚠️ The Golden Rule of Submodules:** Never push the Parent if you haven't pushed the Child. Always commit and push inside `Bhaskera` first. If you push the parent repo pointing to a commit in Bhaskera that hasn't been pushed to GitHub yet, your teammates will get a `fatal: not our ref` error.

### 4. Switching Branches (Troubleshooting)
If you switch to a branch where `Bhaskera` is not yet a submodule (like an older `main` branch), Git may refuse to checkout because of conflicting files. You can safely move the directory out of the way temporarily:
```bash
mv Bhaskera Bhaskera_backup
git checkout main
git submodule update --init --recursive
rm -rf Bhaskera_backup
```

---

## Model & Training Configuration (`node_template.yaml`)

While the `.toml` files control the Rust node's networking and federation behavior, the **`node_template.yaml`** file configures the underlying Python ML Engine (`Bhaskera`). 

At runtime, `ml_engine.py` dynamically reads `node_template.yaml`, overwrites node-specific dynamic paths (like `tokenized_path` or `save_directory`), and generates a `config_{node-id}.yaml` for Ray Train to execute.

Developers should modify `node_template.yaml` to adjust:
- **Model Selection & Architecture:** Change `model.name` (e.g., `NousResearch/Llama-2-7b-hf`), data types, or enable Liger Kernels.
- **Training Paradigm (LoRA vs. Full-Parameter):** Toggle `lora.enabled`. For LoRA, you can configure `rank`, `alpha`, and target modules. For full-parameter fine-tuning, disable LoRA and switch to an optimizer like `galore_muon`.
- **Optimizers & Hyperparameters:** Change `federated.optimizer`, adjust `batch_size`, `learning_rate`, `gradient_accumulation_steps`, or `max_steps`.
- **Distributed Strategy:** Configure FSDP settings (`distributed.strategy: "fsdp"`) including activation checkpointing and sharding behavior.
- **Dataset Configuration:** Change `data.dataset_name` and rendering formats (e.g., `chatml`).

> **Note:** Any changes made to `node_template.yaml` will apply uniformly across all nodes in your local simulation unless explicitly overwritten by `ml_engine.py`.

---

## Dataset Tokenization

Before starting the federated training process, you must tokenize your dataset. The tokenization process converts your raw JSONL data into packed Parquet shards which the training engine loads directly.

1. **Configure Tokenization:**
   Navigate to the `Bhaskera` submodule and open the provided tokenization config:
   `configs/tokenize_south_asia.yaml`.
   Update the `data.path` to point to your raw JSONL dataset and `data.cache_dir` to your desired output directory.

2. **Run the Tokenizer:**
   Run the following command from inside the `Bhaskera` directory:
   ```bash
   bhaskera-tokenize --config configs/tokenize_south_asia.yaml
   ```
   
3. **Update Training Config:**
   The command above will output a `tokenized_path:` snippet at the end. Copy this path and paste it into your `node_template.yaml` under `data.tokenized_path`.

---

## Running the System across Geo-Localized Machines

Try it without a tunnel first: relays plus DHT/pkarr lookup traverse most NATs on their
own, and no node needs a public address. The steps below are the fallback for machines
behind strict university or corporate firewalls (NAT/Deep Packet Inspection) that block
hole punching and relay traffic outright.

Note that the tunnelled node is **not** a bootstrap node — it is an ordinary member that
happens to have a reachable address. Any member with a working address can serve as an
entry point, and once a joining node is in the swarm it learns every other member
through gossip.

**What is Playit.gg?**  
[Playit.gg](https://playit.gg) is a service that creates a secure outbound tunnel from your local machine to a public cloud server. It gives your local node a static public IP address on the internet, completely bypassing incoming firewall restrictions. Because SLAKSHNA uses Iroh (End-to-End Encryption), passing data through Playit's public servers is 100% secure.

### Step 1: Start the Playit Tunnel (Main Machine)
*You must run this on your "Main Machine" (e.g., Delhi server) **before** starting the SLAKSHNA node.*

1. Install `playit` on the main machine.
2. Start the Playit daemon (e.g., `cd ~/playit && ./playit start`).
3. Follow the CLI prompt to create a tunnel. Create a **UDP/TCP tunnel** pointing to your local Iroh `p2p_port` (e.g., `9000` or `9001` based on your config).
4. Playit will assign you a public endpoint. **Note down this IP and Port** (e.g., `147.185.221.225:42060`).

### Step 2: Start the Tunnelled Node (Main Machine)
With the tunnel running in the background, start your node:
```bash
./target/release/iiitd --config node1.toml
```
When started, the node prints its Iroh `EndpointId` (public key) — stable across restarts:
```
INFO 🔑 Iroh EndpointId: a65a49db0894467a3b6d95eda3924c309a5589e265f734332f2b65100364be90
INFO 🤝 Any peer can join this federation with: peers = ["a65a49db…64be90"]
```

### Step 3: Connect Other Nodes (e.g., Mumbai Machine)
On the other machines, open their TOML configuration file (e.g., `node2.toml`).

Combine the **EndpointId** (from Step 2) and the **Playit Public IP:Port** (from Step 1)
in the form `<EndpointId>@<playit_ip>:<playit_port>` to pin the address, bypassing
discovery entirely:

```toml
[network]
# Format: ["<EndpointId>@<Playit_IP>:<Playit_Port>"]
peers = ["a65a49db0894467a3b6d95eda3924c309a5589e265f734332f2b65100364be90@147.185.221.225:42060"]
```

Now, start the other node:
```bash
./target/release/iiitd --config node2.toml
```
It dials the public Playit IP, encrypts the traffic against the EndpointId, and joins the
swarm. From there gossip peer exchange introduces it to every other member, and the peers
it meets are cached in its own `data_dir` — so on the next start it no longer needs this
entry at all.

---

## Running on Academic SLURM Supercomputers

When deploying SLAKSHNA on a SLURM cluster login node:
1. **Never run directly on the login node without a GPU allocation**, as `torch.cuda.is_available()` will fail (`no GPUs found!`).
2. **Set `gpu_id = 0` in your `.toml` file.** When SLURM allocates a physical GPU (`rpgpu[...]`) to your job, it maps that card inside the container to `CUDA_VISIBLE_DEVICES=0`.
3. **Launch the node using `srun` on the GPU partition:**
   ```bash
   srun -p gpu --gres=gpu:1 --time=04:00:00 ./target/release/iiitd --config config.toml
   ```

---

## HTTP REST & WebSocket API

The node exposes an Axum-powered API for monitoring trust evaluations and system status:

| Method | Endpoint                 | Description                                                                         |
| :----- | :----------------------- | :---------------------------------------------------------------------------------- |
| `GET`  | `/status`                | Returns federation id, completed round, active Iroh P2P peer count, and node status |
| `GET`  | `/updates`               | Returns every model update and peer review on record                                |
| `GET`  | `/updates/latest`        | Returns the most recent record in the local update history                          |
| `GET`  | `/updates/:index`        | Returns what each participant contributed at that position in its log               |
| `GET`  | `/leaderboard`           | Returns node trust score rankings (`alpha` / `w_i`)                                 |
| `WS`   | `ws://localhost:8546/ws` | Live WebSocket stream emitting peer evaluation updates                              |

---

## Testing Model Poisoning & Defense

You can simulate a malicious node attempting to poison the Federated Learning by setting the `MALICIOUS_NODES` environment variable:

```bash
MALICIOUS_NODES="node-2" ./target/release/iiitd --config node2.toml
```

When `node-2` runs in malicious mode, it injects a destructive learning rate (`learning_rate = 1.0`). When `node-1` receives `node-2`'s model delta, `ml_engine.py` computes cosine similarity and observes negative alignment. `node-1` automatically slashes `node-2`'s trust score and down-weights its updates in the final model aggregation.

---

## Post-training RL (GRPO) — branch `RL`

This branch adds **reinforcement-learning post-training** as a second training mode next to supervised fine-tuning (SFT). Each node improves the shared LoRA adapter with **GRPO** (Group Relative Policy Optimization) against a **verifiable reward** — a plain function that checks the answer, not a learned reward model. Everything after local training is unchanged: the node produces a sparse int8 LoRA delta, gossips it, and aggregates peer deltas with trust weights.

```
local prompts + checker ─► GRPO (Bhaskera) ─► LoRA delta ─► clip / sparsify / int8 ─► gossip ─► trust-weighted sum
```

### What was built

| Component | Where | What it does |
| :-- | :-- | :-- |
| GRPO trainer | `Bhaskera/src/bhaskera/rl/` (Bhaskera branch `slakshna-rl`) | Samples `group_size` completions per prompt, scores them with the reward function, computes group-relative advantages, and takes a PPO-clipped policy step with an optional KL penalty to the base model. No value network, no reward model; the base model is frozen and only LoRA weights train. |
| Reward | `bhaskera/rl/rewards.py` | Built-in `math_boxed`: 1.0 if the last `\boxed{...}` equals the gold answer (exact fraction comparison), else 0.0. New rewards are added with `@register_reward`. |
| CLI | `bhaskera-rl --config <yaml>` | Standalone single-GPU RL run, no Slakshna needed. |
| Slakshna integration | `ml_engine.py`, `federated_communication/` | `federated.mode: "rl"` runs the RL trainer instead of SFT each federated round; resumes from the aggregated adapter; continues the prompt stream across rounds; reports mean reward in the update metadata. |
| Poisoned-delta screening | `federated_communication/` | A peer delta is rejected if non-finite, above `SLAKSHNA_MAX_DELTA_NORM` (default 100, +5% headroom), or more than `SLAKSHNA_MAX_PEER_NORM_RATIO` (default 5) times this node's own delta norm. Rejections are logged as `peer_delta_rejected` in `logs/runtime_comm.log`. |
| Example config | `examples/2node-rl/` | Two-node federated RL demo on GSM8K with disjoint data shards. |
| Experiments | `experiments/fed_sim.py`, `experiments/run_experiments.py`, `plots_script/analyze_fed_sim.py` | Multi-node simulator, job runner and analysis used for the federated results below. |
| Tests | `tests/test_rl_integration.py`, `tests/test_fed_sim.py`, `tests/test_norm_policy.py`, `tests/test_slakshna_eval.py`; `Bhaskera/src/bhaskera/rl/tests/` | Unit and integration tests (see "Run the tests"). |

**Privacy:** prompts, gold answers, the reward function and all generated text stay on the node. Only the sparse LoRA delta leaves, exactly as in SFT mode.

### Commands

All commands run from the Slakshna root unless stated. Complete the *Installation & Build* steps above first (the Bhaskera submodule must be on the commit recorded by this branch: `git submodule update --init --recursive`).

**1. Prepare GSM8K prompts** (downloads `openai/gsm8k` from Hugging Face; `--num-shards 2` gives each of 2 nodes a disjoint training shard):

```bash
source Bhaskera/bhaskera-activate.sh
python Bhaskera/scripts/prepare_gsm8k_rl.py --out-dir data/gsm8k --num-shards 2
# writes data/gsm8k/train.jsonl (7,473), test.jsonl (1,319), train_0.jsonl, train_1.jsonl
```

**2. Single-node RL, no federation** (one GPU; `Qwen2.5-0.5B-Instruct`, 100 GRPO steps by default):

```bash
cd Bhaskera
bhaskera-rl --config configs/rl_gsm8k.yaml
```

Output is one line per step, e.g. `[epoch 0][step 7] loss=… reward=0.3828 kl=0.02002 …`, plus periodic greedy `[eval][step N] eval_reward=…` on the held-out test file. LoRA checkpoints are written to `checkpoint.save_dir`.

**3. Federated RL across two real Slakshna nodes** (node 1 on GPU 0, node 2 on GPU 1; each trains on its own shard):

```bash
cargo build --release
./start_cluster.sh 2 examples/2node-rl      # per-node configs: node{1,2}.toml + node{1,2}.yaml
tail -f logs/node1_cluster.log
./stop_cluster.sh
```

The example is a short demo: 3 federated rounds of 6 GRPO steps each (`federated.total_epochs`, `training.max_steps` in `node*.yaml`); each node exits after its last round. Peers discover each other by mDNS on the same host/LAN; no addresses need configuring.

**4. Enable RL in your own config** — in `node_template.yaml` (or a per-node yaml):

```yaml
federated:
  mode: "rl"            # was "finetuning"
lora:
  enabled: true         # required; RL trains only the adapter
training:
  learning_rate: 5.0e-5
  max_steps: 20         # GRPO steps per federated round
rl:
  train_path: data/gsm8k/train_0.jsonl     # {"prompt": ..., "answer": ...} per line
  eval_path: data/gsm8k/test.jsonl         # optional
  reward: math_boxed
  group_size: 8
  prompts_per_step: 16
  max_new_tokens: 320
  kl_coef: 0.04
```

Relative `rl.*_path` values resolve against the Slakshna directory. All `rl:` options: `Bhaskera/src/bhaskera/rl/README.md`; integration details: [docs/rl_training.md](docs/rl_training.md).

**5. Reproduce the multi-node experiments** (single-process simulator that reuses the real trainer and delta pipeline; needs GPUs):

```bash
python experiments/run_experiments.py --dry-run        # list the jobs
python experiments/run_experiments.py --gpus 0,2,3 --slots 3 --seeds 0,1,2 --out results/fed_sim
python plots_script/analyze_fed_sim.py results/fed_sim   # tables, paired tests, figures
```

**6. Run the tests:**

```bash
pytest tests/test_rl_integration.py tests/test_fed_sim.py tests/test_norm_policy.py tests/test_slakshna_eval.py -v
cd Bhaskera && pytest src/bhaskera/rl/tests src/bhaskera/tests -v      # trainer unit tests
```

`test_rl_integration.py` runs the real `ml_engine.py` on a tiny model: round 1 (reward rises, delta is non-zero and valid), round 2 (resumes the adapter and the sample counter), and a second node that verifies and aggregates the first node's RL delta. `pytest` is a dev extra; install it into the Bhaskera environment first.

### Measured results

Task: GSM8K, reward = final `\boxed{}` answer correct, evaluation = greedy decoding on the **full 1,319-problem test set**. Raw logs, configs and figures are in `results/rl_long_run_2026-10-22/` and `results/fed_sim/`.

**Single node, 240 GRPO steps** (16 prompts × 8 completions per step, `max_new_tokens` 512, one run per configuration, so no seed variance is available):

| Model | LR | KL coef | Accuracy step 0 | step 80 | step 160 | step 240 |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| Qwen2.5-0.5B-Instruct | 5e-5 | 0.04 | 44.1% | 48.3% | 49.1% | 49.6% |
| Qwen2.5-0.5B-Instruct | 1e-4 | 0 | 44.1% | 45.3% | 47.7% | 49.1% |
| Qwen2.5-1.5B-Instruct | 2e-5 | 0.04 | 68.8% | 71.0% | 73.6% | 73.5% |
| Qwen2.5-1.5B-Instruct | 5e-5 | 0.04 | 68.8% | 71.0% | 73.2% | 73.6% |
| Qwen2.5-1.5B-Instruct | 5e-5 | 0 | 68.8% | 74.8% | 75.8% | 74.7% |

RL gives roughly +5 points over the untrained model for both sizes. With one run per row, differences *between* configurations (about 1 point) are within noise; the gain over step 0 is the finding.

**Federated vs. isolated vs. centralized** (`experiments/fed_sim.py`, Qwen2.5-0.5B-Instruct, N = 4 simulated nodes, 8 rounds × 10 steps, 5 seeds IID / 3 seeds non-IID; untrained model = 34.9%):

| Method | IID accuracy (mean ± std) | Non-IID accuracy | MB sent per run |
| :-- | :-- | :-- | :-- |
| Isolated RL (no sharing) | 40.8 ± 0.6 | 42.4 ± 0.8 | 0 |
| **Federated RL (Slakshna delta pipeline)** | **41.6 ± 0.7** | **41.6 ± 2.0** | 147 |
| Centralized RL (upper reference) | 42.6 ± 1.6 | — | 0 |
| Federated SFT | 31.6 ± 1.2 | 30.1 ± 2.1 | 147 |

What this does and does not show:

* **Supported:** federated RL beats federated SFT on this task by about 10 points (IID: +10.0, McNemar p = 1.6e-13; non-IID: +11.5). SFT on GSM8K solutions made the model *worse* than untrained (31.6% vs 34.9%) while RL improved it. Federated RL also reduces the disagreement between nodes' adapters (consensus gap 0.62 vs 1.90 for isolated, IID).
* **Not supported:** that federation improves accuracy over each node training alone. Federated RL vs isolated RL is +0.8 points IID (McNemar p = 0.84, seed t-test p = 0.14) and −0.7 points non-IID (p = 0.47). Federated RL is about 1 point below centralized RL (McNemar p = 0.003; seed t-test p = 0.18). Full statistics: `results/fed_sim/analysis.md`.
* **Simulator, not the network:** `fed_sim.py` runs nodes sequentially in one process with the real trainer and delta/aggregation code; it does not exercise the Rust network, gossip or discovery. Its absolute accuracies are not comparable with the single-node table above (different protocol).

**Real 2-node Rust run** (`results/rl_long_run_2026-10-22/federated_2node/`): two nodes, 6 federated rounds × 20 GRPO steps, disjoint data shards, `max_new_tokens` 320, evaluated on only the first 200 test problems (about ±3.4 points of noise). It demonstrates that the full pipeline — RL training, sparse delta exchange over gossip, trust-weighted aggregation, resume — runs end to end on real nodes. The evaluation set is too small to claim an accuracy improvement from this run.

### Known limits

* Peer trust is cosine similarity between deltas. Nodes training on very different private tasks may look dissimilar; scoring a peer's delta by held-out reward is not implemented.
* Per-node record signatures are placeholders in `src/main.rs`, so authorship of an update is not cryptographically proven. The norm screening stops oversized poisoning but not a poisoned delta of normal size (sign flip, backdoor); cosine trust weighting is the only defense against those, and robustness experiments (E3 in `run_experiments.py`) have no results yet.
* RL is single-GPU per node, and rollouts use plain Hugging Face `generate` (correct, not fast).
* `validation_score` in the node's JSON is still the last logged loss, which is not a quality measure for GRPO; use `reward` in `metadata`.
* Only a verifiable-answer reward (`math_boxed`) is provided. Rewards that execute generated code are not included (they need a sandbox).
* Results are on one task (GSM8K) with Qwen2.5 0.5B/1.5B models.

---

## 📄 License

This project is licensed under the **Apache License 2.0** — see the [LICENSE](file:///mnt/disk1/slakshna/slakshnaFL/SLAKSHNA/LICENSE) file for details.

