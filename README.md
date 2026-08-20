# SLAKSHNA — Decentralized Geo-Localised Personalized Federated Learning

A **Peer-to-Peer Federated Learning Framework** built in **Rust** and integrated with a high-performance Python Machine Learning Engine (**Bhaskera**). **SLAKSHNA** enables decentralized, privacy-preserving, weighted-aggregation Federated Learning (FL) without centralized aggregators or synchronous blocking rounds. It runs across geo-localized machines and institutional clusters (including SLURM-managed supercomputers, kubernetes managed clusters) separated by complex firewalls, securely sharing compressed model updates without any central coordinator.

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
peers = []                  # Optional seeds; empty is fine — see "Joining a federation"

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

## Managing the Bhaskera Submodule

The `Bhaskera` ML engine is included as a Git submodule. When cloning or pulling updates, you must ensure the submodule is synced.

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

**Switching Branches:**
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

## 📄 License

This project is licensed under the **Apache License 2.0** — see the [LICENSE](file:///mnt/disk1/slakshna/slakshnaFL/SLAKSHNA/LICENSE) file for details.

