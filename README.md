# SLAKSHNA — Decentralized Geo-Localised Personalized Federated Learning

A **Peer-to-Peer Federated Learning Framework** built in **Rust** and integrated with a high-performance Python Machine Learning Engine (**Bhaskera**). **SLAKSHNA** enables decentralized, privacy-preserving, weighted-aggregation Federated Learning (FL) without centralized aggregators or synchronous blocking rounds. It runs across geo-localized machines and institutional clusters (including SLURM-managed supercomputers, kubernetes managed clusters) separated by complex firewalls, securely sharing compressed model updates without any central coordinator.

---

## Key Features & Architectural Highlights

- **Asynchronous P2P Training**  
  Instead of traditional synchronous FL rounds waiting for slow participants, SLAKSHNA operates asynchronously. Nodes continuously train on local data, broadcast compressed model deltas to the network, and evaluate peers dynamically.

- **Iroh QUIC Mesh & Gossip Network (`iroh-gossip`)** 
  Built on **Iroh v1.0.2**, the framework utilizes **QUIC (Quick UDP (User Datagram Protocol) Internet Connections)** transport, direct NAT (Network Address Translation) traversal (STUN/DERP), and `iroh-gossip` topic swarms. Nodes discover peers dynamically using cryptographic Ed25519 `NodeId` public keys.

- **Universal Firewall & VPN Traversal (`Playit.gg`)**  
  Academic and enterprise networks (such as university campus firewalls or remote VPNs) often block inbound UDP/TCP hole-punching and standard DERP relay traffic. SLAKSHNA natively supports static public UDP/TCP tunneling via **Playit.gg**, providing fixed, persistent public addresses (`<ip>:<port>`) for nodes across different cities without requiring root/sudo access or complex router configurations.

- **Bhaskera ML Engine (`ml_engine.py`)**  
  A robust Python engine bridging the Rust networking layer with distributed GPU/CPU training. Powered by **Ray Train (`TorchTrainer`)**, **PyTorch**, and **parameter efficient training algorithms**, it executes local pre-training, and fine-tuning on tokenized datasets while streaming real-time epoch loss tracking. During training it can offload the optimizer states to perform **concurrent evals** on the model.

- **SLURM Supercomputer & Multi-Core Cluster Support**  
  Fully compatible with HPC SLURM clusters (`srun` / `sbatch`). SLURM isolates allocated GPUs seamlessly and maps to cluster-assigned resources without port collisions or resource deadlocks.

- **Weighted Aggregation**  
  Peers asynchronously evaluate incoming model proposals by computing cosine similarity against their local gradient direction and tracking validation loss improvements. Nodes dynamically update peer weights (`state["alpha"]` and normalized `w_i` weights) and aggregate updates based on these weights. It checks not only malicious updates but also provides foundation for mitigating catastrophic forgetting.

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
   To prevent adversarial nodes from ruining the global model (`Model Poisoning`), SLAKSHNA does not use simple averaging. When a node receives a peer's delta, `ml_engine.py` evaluates the proposal against local validation metrics (`Cosine Similarity` & `Validation Loss`). If a node submits poisoned or erratic updates, its trust score (`alpha`) drops, rendering its weight in the Federated Averaging formula close to `0.0`.

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

| Layer | Technologies Used |
| :--- | :--- |
| **Networking Core** | **Rust** (`edition = 2021`), **Tokio** async runtime |
| **P2P Communication** | **Iroh** (`iroh v1.0.2`, `iroh-gossip`, `iroh-relay`), **QUIC**, **Ed25519 TLS 1.3**, **Playit.gg** (Static Tunnels) |
| **API & WebSockets** | **Axum 0.7**, **Hyper**, **tokio-tungstenite** (`WebSocket`), **Serde / Serde JSON** |
| **ML Engine & FL** | **Python 3.11+**, **PyTorch**, **Ray / Ray Train** (`ray.train.torch.TorchTrainer`), **setproctitle** |
| **Transformers & PEFT** | **HuggingFace Transformers**, **PEFT** (`LoRA`), **PyArrow** (Parquet caching), **PyYAML** |
| **Differential Privacy** | **Gradient clipping**, **Noice injection**, **Opacus** (`PrivacyEngine`), **opt-einsum**|

---

## Repository Structure

| Path | Description |
| :--- | :--- |
| `src/main.rs` | Node entry point, phase execution, ML process orchestration, and P2P broadcast |
| `src/network/` | Iroh QUIC + Gossip network implementation (`mesh.rs`, `mod.rs`, `star.rs`) for peer synchronization |
| `src/api.rs` | Axum HTTP REST endpoints and real-time WebSocket broadcast server |
| `src/config.rs` | TOML configuration loader for network ports and storage paths |
| `ml_engine.py` | Python bridge executing Bhaskera distributed LoRA training, sparsification (`SparseLoCo`), and evaluation |
| `Bhaskera/` | Submodule / embedded repository containing the Bhaskera distributed LLM training framework |
| `config.toml` | Master/Node-1 configuration file |
| `node2.toml` / `node3.toml` | Peer node configuration files |

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
# 1. Run the SLAKSHNA root setup script
bash setup.sh
./setup.sh

# 2. Navigate to the Bhaskera subdirectory and run its setup script
cd Bhaskera
bash setup.sh
./setup.sh

# 3. Activate the Bhaskera Python virtual environment
source bhaskera-activate.sh

# 4. Move back to the SLAKSHNA root directory
cd ..

# 5. Build the Rust P2P node binary in release mode
# (Make sure you have Rust and Cargo installed)
cargo build --release
```

---

## TOML Configuration Breakdown

Every node requires its own `.toml` configuration file (`config.toml`, `node2.toml`, etc.).

### Master Node (`config.toml`)
```toml
[node]
id = "node-1"
type = "master"
data_dir = "./data-node1"   # Dedicated delta storage directory
gpu_id = 0                  # GPU assigned to this node for local training

[network]
topology = "mesh"
host = "0.0.0.0"
p2p_port = 9000             # Iroh QUIC router listening port
api_port = 8545             # Axum HTTP REST API port
ws_port = 8546              # WebSocket port
boot_nodes = []             # Master has no initial boot nodes
```

### Remote Peer Node (`node2.toml` / `node3.toml`)
When connecting a remote node over the internet or across campuses, point `boot_nodes` directly to the Master Node's **Ed25519 `NodeId`** (printed by the master node upon startup) or its public static tunnel (`Playit.gg`):

```toml
[node]
id = "node-2"
type = "full"
data_dir = "./data-node2"   # MUST be unique per node
gpu_id = 0                  # Set to 0 if running inside SLURM (--gres=gpu:1), or 1 if multi-GPU server

[network]
topology = "mesh"
host = "0.0.0.0"
p2p_port = 9001             
api_port = 8555             
ws_port = 8547              

# Point boot_nodes to the Master Node's Iroh PublicKey (NodeId):
# Iroh automatically discovers the route via direct IP, mDNS, STUN, or public Playit tunnel
boot_nodes = ["<MASTER_IROH_PUBLIC_KEY>@<tunnel address>"]
```

---

## Running the System across Geo-Localized Machines

If your machines are located in different cities (e.g., Delhi $\leftrightarrow$ Mumbai) and are separated by strict university or corporate firewalls (NAT/Deep Packet Inspection) that block peer-to-peer discovery, you must use a reverse proxy tunnel.

**What is Playit.gg?**  
[Playit.gg](https://playit.gg) is a service that creates a secure outbound tunnel from your local machine to a public cloud server. It gives your local node a static public IP address on the internet, completely bypassing incoming firewall restrictions. Because SLAKSHNA uses Iroh (End-to-End Encryption), passing data through Playit's public servers is 100% secure.

### Step 1: Start the Playit Tunnel (Main Machine)
*You must run this on your "Main Machine" (e.g., Delhi server) **before** starting the SLAKSHNA node.*

1. Install `playit` on the main machine.
2. Start the Playit daemon (e.g., `cd ~/playit && ./playit start`).
3. Follow the CLI prompt to create a tunnel. Create a **UDP/TCP tunnel** pointing to your local Iroh `p2p_port` (e.g., `9000` or `9001` based on your config).
4. Playit will assign you a public endpoint. **Note down this IP and Port** (e.g., `147.185.221.225:42060`).

### Step 2: Start the Master Node (Main Machine)
With the tunnel running in the background, start your node:
```bash
./target/release/iiitd --config config.toml
```
When started, the node will output its unique cryptographic Iroh `NodeId` (Public Key):
```
INFO 🔑 Iroh NodeId: a65a49db0894467a3b6d95eda3924c309a5589e265f734332f2b65100364be90
```

### Step 3: Connect Peer Nodes (e.g., Mumbai Machine)
On your secondary machines, open their TOML configuration file (e.g., `node2.toml`).

You need to tell this machine exactly how to reach the Main Machine. Combine the **NodeId** (from Step 2) and the **Playit Public IP:Port** (from Step 1) using the format `<node_id>@<playit_ip>:<playit_port>`.

Update the `boot_nodes` field:
```toml
[network]
# Format: ["<NodeId>@<Playit_IP>:<Playit_Port>"]
boot_nodes = ["a65a49db0894467a3b6d95eda3924c309a5589e265f734332f2b65100364be90@147.185.221.225:42060"]
```

Now, start the peer node:
```bash
./target/release/iiitd --config node2.toml
```
The peer node will dial the public Playit IP, encrypt the traffic using the NodeId, and establish a direct connection to the main machine!

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

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/status` | Returns active Iroh P2P peer count and node status |
| `GET` | `/leaderboard` | Returns active node reputation and trust score rankings (`alpha` / `w_i`) |
| `WS` | `ws://localhost:8546/ws` | Live WebSocket stream emitting peer evaluation updates |

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

