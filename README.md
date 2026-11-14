# SLAKSHNA-RAG: Verifiable Peer-to-Peer Federated RAG

This branch adds **federated retrieval-augmented generation (RAG)** to SLAKSHNA, the peer-to-peer federated learning framework. Each participant keeps its documents on its own machine. When someone asks a question, the node searches its own documents, asks its peers for relevant passages over an encrypted peer-to-peer connection, **checks with cryptographic proofs** that every passage a peer returns is genuine, and has a local language model answer from the combined evidence.

There is no central server, no shared index and no bootstrap node.

---

## Contents

1. [What is RAG?](#1-what-is-rag)
2. [Why federated RAG?](#2-why-federated-rag)
3. [How SLAKSHNA-RAG works](#3-how-slakshna-rag-works)
4. [Requirements](#4-requirements)
5. [Build](#5-build)
6. [Run](#6-run)
7. [Configuration](#7-configuration)
8. [HTTP API](#8-http-api)
9. [Tests](#9-tests)
10. [Comparing with Flower FedRAG](#10-comparing-with-flower-fedrag)
11. [Repository layout](#11-repository-layout)
12. [Limits](#12-limits)

---

## 1. What is RAG?

A large language model (LLM) only knows what was in its training data. Ask it about a document it has never seen, or about something that happened after it was trained, and it either refuses or makes up a plausible answer.

**Retrieval-augmented generation** fixes this by looking things up first:

1. **Index.** Documents are split into short passages ("chunks"). An *embedding model* turns each chunk into a vector, a list of numbers that captures its meaning, so chunks about similar things get similar vectors. The vectors go into a *vector index* for fast similarity search.
2. **Retrieve.** The question is embedded with the same model, and the index returns the chunks whose vectors are closest to it.
3. **Generate.** The retrieved chunks are pasted into the LLM's prompt as context, and the model answers from them.

The model doesn't need retraining when the documents change. You re-index them instead, and every answer can be traced back to the passages it came from.

## 2. Why federated RAG?

Ordinary RAG assumes all documents sit in **one index on one machine**. Often they can't:

- **Privacy and regulation.** Hospitals, universities and companies hold data they may not copy to a shared server (patient records, internal reports, unpublished research).
- **Ownership.** Each group wants to keep control over its data, and be able to add, update or withdraw documents without asking a central operator.
- **No single point of trust or failure.** A central index is a single target for attack, a single bill, and a single thing that can go down.

**Federated RAG** keeps every document collection where it is. A question is sent to the participants, each searches its own collection, and only the few passages relevant to that question leave a participant's machine.

Most federated RAG systems, such as Flower's FedRAG example, still use a **central server** that sends questions out and merges answers, and they **trust** whatever each client sends back. SLAKSHNA-RAG removes both assumptions:

| | Typical federated RAG | SLAKSHNA-RAG |
| :-- | :-- | :-- |
| Coordinator | Central server | None: every node is equal and can ask or answer |
| Finding peers | Configured server address | mDNS on a LAN, the mainline DHT, DNS, or listed peers (Iroh QUIC mesh) |
| Trust in peers | Passages and scores taken as sent | Every passage carries a Merkle proof against a signed commitment; scores are recomputed by the asker |
| Which peers are asked | All of them | Only peers whose topics match the question (two-round and directory modes) |

## 3. How SLAKSHNA-RAG works

### 3.1 The pieces

```
           your question (HTTP)
                  │
   ┌──────────────▼──────────────┐                  ┌──────────────────────────────┐
   │ Rust node  (iiitd)          │  /slakshna/rag/2 │ Rust node of a peer          │
   │  src/api.rs   POST /rag/... │◄────────────────►│  answers hello / centroids / │
   │  src/rag/     query pipeline│   Iroh QUIC      │  chunks / search / directory │
   │               proof checks  │   (encrypted)    │                              │
   └──────────────┬──────────────┘                  └──────────────┬───────────────┘
                  │ HTTP on 127.0.0.1                              │
   ┌──────────────▼──────────────┐                  ┌──────────────▼───────────────┐
   │ Python sidecar (rag_engine) │                  │ Python sidecar of the peer   │
   │  PDF/text loading, chunking │                  │  its own documents, index    │
   │  embeddings, FAISS HNSW     │                  │  and Merkle tree             │
   │  k-means clusters, Merkle   │                  └──────────────────────────────┘
   │  tree, answer via Ollama    │
   └─────────────────────────────┘
```

- The **Rust node** (`src/`) handles identity, networking, the HTTP API and all checks on what peers send.
- The **Python sidecar** (`rag_engine.py`, package `rag/`) is a local helper process the node starts for you. It does the machine-learning work: reading documents, embedding, searching, and calling [Ollama](https://ollama.com) to write the answer.
- Peers talk over a dedicated Iroh protocol, `/slakshna/rag/2`: one QUIC stream per request, with JSON messages.

### 3.2 Indexing: committing to your documents

When a node ingests documents (PDF, `.txt`, `.md`, or pre-chunked MedRAG `.jsonl`), the sidecar:

1. splits them into chunks and embeds each chunk (default model `sentence-transformers/all-MiniLM-L6-v2`);
2. stores the vectors in a **FAISS HNSW** index, a fast approximate nearest-neighbour search;
3. groups the chunks into **topics** with spherical k-means. Each topic (cluster) is summarised by its **centroid**, the average direction of its chunks' vectors;
4. builds a **two-level Merkle tree**. A Merkle tree is a tree of hashes in which each parent is the hash of its children, so the single hash at the top (the *root*) changes if anything below it changes. Here the lower level hashes each cluster's chunks, and the upper level hashes each cluster's id, centroid and chunk-tree root. The 32-byte root therefore pins down every chunk, every centroid, and which cluster each chunk belongs to.

The Rust node then **signs** the root, a version number that only increases, and the cluster count with its Ed25519 key. That key is also the node's network identity (its Iroh `EndpointId`), so peers can check the signature without any extra key exchange.

### 3.3 Querying: three retrieval modes

Every node fetches every peer's signed root in the background (`hello`, once a minute). A query then runs in one of three modes:

| Mode | Round 1 | Round 2 | When to use |
| :-- | :-- | :-- | :-- |
| `two_round` (default) | Ask **every** peer for its centroids closest to the question; keep the best clusters overall | Ask only the owners of the winning clusters for chunks from those clusters | General use; peers with unrelated topics aren't asked for chunks |
| `flat` | — | Ask every peer for its best chunks directly | Baseline; fewest round trips |
| `directory` | **None.** Clusters are chosen locally from a verified copy of every peer's centroids, fetched once per root version | Same as `two_round` | Many peers; saves a network round per question. Needs `[rag] directory = true` |

In every mode the asking node:

- checks each centroid's and chunk's **Merkle proof** against the peer's signed root (cluster proofs also by their position in the tree, so a peer can't pass one cluster off as another);
- **re-embeds every peer chunk** and uses its own score, so a peer can't inflate its relevance;
- drops anything that fails, and records a flag for the peer (`bad_chunk_proof`, `score_mismatch`, `bad_centroid_proof`, `incomplete_directory`, `equivocation`, …);
- merges verified peer chunks with its own, keeps the top `final_k`, and passes them to the LLM.

A peer caught signing two different roots under the same version (**equivocation**) is refused from then on. A peer that is down is marked `skipped_stale` so it doesn't cost a timeout on every question, and comes back at the next successful refresh.

## 4. Requirements

| Needed for | What |
| :-- | :-- |
| The node | [Rust](https://rustup.rs) (stable, edition 2021), a C compiler |
| The sidecar | Python 3.11 and [uv](https://docs.astral.sh/uv/) (or any way to make a virtual environment) |
| Answer writing | [Ollama](https://ollama.com) with a small model, e.g. `qwen2.5:1.5b`. Not needed with `generator = "none"`, which returns the retrieved passages only |

RAG runs on an ordinary CPU machine. No GPU and no Bhaskera training environment are needed when `[training] enabled = false`, as in the RAG example configs.

## 5. Build

```bash
git clone -b RAG https://github.com/dcll-iiitd/Slakshna.git
cd Slakshna

# 1. The Rust node -> target/release/iiitd
cargo build --release

# 2. The Python sidecar's own environment (CPU build of torch)
uv venv .venv-rag --python 3.11
uv pip install --python .venv-rag/bin/python --torch-backend cpu -r requirements-rag.txt

# 3. An answer model (skip if you only want retrieved passages)
ollama pull qwen2.5:1.5b
```

The `Bhaskera` submodules are only needed for federated *training*; you can skip `git submodule update` for RAG.

Always run commands **from the repository root**. The configs and scripts use paths relative to it.

## 6. Run

### 6.1 Two nodes on one machine

The repository includes a small test corpus (`corpus/`, Wikipedia articles as PDFs). Node 1 holds articles about Indian space missions, and node 2 holds articles about scientists and places, so a question about node 2's topics can only be answered with node 2's help.

```bash
./target/release/iiitd --config examples/rag-2node/node2.toml     # terminal 1
./target/release/iiitd --config examples/rag-2node/node1.toml     # terminal 2
```

Each node starts its sidecar, ingests its `corpus_dir`, signs its root and joins the mesh. On one machine or one LAN the nodes find each other through mDNS. Wait about 10 seconds after both are up for the first background refresh.

Check that node 1 can see node 2 and that node 2's signature checks out:

```bash
curl localhost:8545/rag/debug/hello      # node 2 should show status "ok" and signature_valid: true
curl localhost:8545/rag/status           # sidecar, own root, peer registry, directory
```

### 6.2 Ask a question

```bash
curl localhost:8545/rag/query -H 'content-type: application/json' \
     -d '{"query": "In which town was A. P. J. Abdul Kalam born and raised?"}'
```

The answer comes back with the `contexts` it was built from. Each context shows its `node`, `score` and `verification` (`local`, `verified`, `verified_new_root` or `failed_proof`). There is also a `peers` list showing what each peer did (`ok`, `timeout`, `skipped_stale`, `refused`, …, plus any flags and bytes sent), and `timings_ms`.

Optional request fields:

| Field | Values |
| :-- | :-- |
| `retrieval` | `"two_round"`, `"flat"`, `"directory"` (needs `[rag] directory = true`) |
| `scope` | `"all"` (default), `"local"` (own index only), `"peers"` (peers only) |
| `top_k` | chunks taken from each index |
| `final_k` | chunks kept after merging and given to the LLM |

To run all 15 test questions in `corpus/questions.jsonl` and save the full responses:

```bash
python3 scripts/rag_run_questions.py --api http://localhost:8545 --retrieval two_round --out /tmp/two_round.jsonl
```

### 6.3 Add documents while running

```bash
curl localhost:8545/rag/ingest -H 'content-type: application/json' -d '{"path": "path/to/folder"}'
curl localhost:8545/rag/ingest -H 'content-type: application/json' \
     -d '{"documents": [{"id": "note.txt", "text": "..."}]}'
```

Adding documents re-clusters the index and produces a new signed root with a higher version. Peers pick it up at their next refresh, or immediately if they notice the version changed during a query.

### 6.4 Nodes on different machines

1. Use the same `[federation] id` on every node. Nodes with different ids never see each other.
2. Build **the same commit** everywhere; the wire format must match.
3. If mDNS can't cross your network, list the other node in `[network] peers`. You can get a node's `EndpointId` without starting it:

   ```bash
   ./target/release/iiitd --config examples/rag-2node/node2.toml --print-endpoint-id
   ```

   and then in node 1's config: `peers = ["<EndpointId>@<ip of machine 2>:9001"]`. Turning on `dht`, `dns` and `relay` in `[discovery]` lets nodes find each other across the internet and behind NAT.

### 6.5 Many nodes on one machine

`scripts/rag_make_configs.py` writes configs for N local nodes (unique ports and data directories, every node listing every other), and `scripts/m9/slakshna_cluster.py` starts them and waits until all of them see each other:

```bash
python3 scripts/rag_make_configs.py --nodes 5 --name demo --out configs/demo --data-root data-demo
python3 scripts/m9/slakshna_cluster.py up   --configs configs/demo
python3 scripts/m9/slakshna_cluster.py down --configs configs/demo
```

### 6.6 The sidecar on its own

The Python side can also run without any Rust node, which is handy for debugging an index:

```bash
.venv-rag/bin/python rag_engine.py ingest --data-dir ./data-rag-node1 --path corpus/node1
.venv-rag/bin/python rag_engine.py serve  --data-dir ./data-rag-node1 --port 8600 --generator none
curl localhost:8600/health
```

## 7. Configuration

A RAG node is an ordinary SLAKSHNA config with `[training] enabled = false` and a `[rag]` section. See `examples/rag-2node/node1.toml` for a complete, commented example. The most useful `[rag]` keys:

| Key | Default | Meaning |
| :-- | :-- | :-- |
| `enabled` | `false` | Turn RAG on for this node |
| `corpus_dir` | none | Folder ingested at startup (already-indexed files are skipped) |
| `sidecar_port` | `8600` | Local port of the sidecar; unique per node on one machine |
| `python` | `.venv-rag/bin/python` | Interpreter with `requirements-rag.txt` installed |
| `embedding_model` | `sentence-transformers/all-MiniLM-L6-v2` | **Must be the same on every node** |
| `generator` / `generator_model` | `"ollama"` / `qwen2.5:1.5b` | `"none"` returns passages without an answer |
| `retrieval` | `"two_round"` | Default mode; a query can override it |
| `top_k`, `final_k` | `5`, `5` | Chunks per index / chunks kept after merging |
| `n_clusters` | `0` (= √chunks) | k-means topics per node, at most 64 |
| `centroids_per_peer`, `max_clusters`, `min_centroid_score` | `3`, `5`, `0.2` | Round-1 cluster selection |
| `chunks_per_cluster` | `3` | Round-2 chunks per chosen cluster |
| `directory` | `false` | Serve own centroids and keep peers' directory (enables `directory` mode). It reveals this node's topic map to allowed peers |
| `peer_timeout_ms`, `select_timeout_ms` | `5000`, `2000` | Per-peer timeouts for chunks / for `hello` and round 1 |
| `root_refresh_secs`, `peer_stale_secs` | `60`, `180` | Background root refresh; when a silent peer is skipped |
| `require_prior_root` | `false` | Reject chunks proven against a root first seen during the same query |
| `adversary` | `"none"` | **Experiments only:** make this node misbehave towards peers (`tamper_text`, `inflate_score`, `fake_centroid`, `equivocate`, `hide_cluster`) to test detection |

`[network] allowed_peers` restricts which `EndpointId`s may query this node.

## 8. HTTP API

| Route | What it does |
| :-- | :-- |
| `POST /rag/query` | Answer a question (section 6.2) |
| `POST /rag/ingest` | Add documents: `{"path": ...}` or `{"documents": [{"id", "text"}]}` |
| `GET /rag/status` | Sidecar health, this node's signed root, the peer registry (roots, flags, evidence), directory state |
| `GET /rag/debug/hello` | Fetch every peer's signed root now and check each signature |

The federated-learning routes (`/status`, `/peers`, `/updates`, `/leaderboard`, `/ws`) are unchanged.

## 9. Tests

```bash
# Rust: Merkle proofs, signed roots, protocol messages, registry, retrieval checks, simulated attackers, directory
cargo test --release

# Directory scale test: 1000 peers x 13 centroids, prints timings
cargo test --release directory_scale -- --ignored --nocapture

# Python sidecar: store, Merkle tree, HTTP routes
.venv-rag/bin/python -m unittest tests/test_rag_store.py tests/test_rag_merkle.py tests/test_rag_server.py
```

`tests/fixtures/merkle_vectors.json` is checked by **both** the Python and the Rust tests, so the two languages can't drift apart on the exact bytes they hash. Regenerate it with `scripts/rag_merkle_vectors.py` only when the format changes on purpose.

On the two-node test corpus, all six `node1_only` and all six `node2_only` questions retrieved the right source in every mode, and every remote context verified. Each simulated attack was caught with its expected flag, and no tampered text ever reached a context or an answer. The three `cross_node` questions, which need one passage from each node, still fail: the top passages tend to come from one side.

## 10. Comparing with Flower FedRAG

`scripts/m9/` holds a head-to-head comparison with [Flower's FedRAG example](https://github.com/flwrlabs/flower/tree/main/examples/fedrag) on the MedRAG Textbooks corpus and MIRAGE benchmark questions, sharded across 5 nodes. Both systems' retrieved passages are answered by the **same** LLM with the same prompt (`answer.py`), so only retrieval differs.

| Step | Script |
| :-- | :-- |
| Download and shard the data, embed once | `prepare_textbooks.py`, `embed_cache.py` |
| Central "one big index" reference for recall | `global_baseline.py` |
| Slakshna: start N nodes, ask every question in one mode | `slakshna_cluster.py`, `run_slakshna.py` |
| Flower: pinned FedRAG copy, SuperLink + SuperNodes | `flower/setup_flower.sh`, `flower/flower_cluster.py` |
| Answer every contexts file with one LLM; compute tables | `answer.py`, `evaluate.py` |

Typical sequence:

```bash
.venv-rag/bin/python scripts/m9/prepare_textbooks.py --out datasets/textbooks
python3 scripts/rag_make_configs.py --nodes 5 --name book --out m9-configs/book --data-root m9-data/book
python3 scripts/m9/slakshna_cluster.py up --configs m9-configs/book
python3 scripts/m9/run_slakshna.py --configs m9-configs/book --data datasets/textbooks \
        --retrieval directory --out results/m9/raw/book/slakshna_directory.jsonl
bash    scripts/m9/flower/setup_flower.sh --data datasets/textbooks
python3 scripts/m9/flower/flower_cluster.py up ...      # then run / down
```

Run each script with `--help` for its options.

## 11. Repository layout

Files added or changed for RAG:

```
rag_engine.py              sidecar entry point: `ingest` and `serve`
rag/                       loaders, chunking, embedder, FAISS store, k-means, Merkle tree, HTTP server, Ollama generator
requirements-rag.txt       sidecar dependencies
src/rag/mod.rs             RagService: query pipeline, merging, own signed root
src/rag/sidecar.rs         starts the sidecar and talks to it over local HTTP
src/rag/protocol.rs        the /slakshna/rag/2 peer protocol (both sides)
src/rag/retrieval.rs       two_round / flat / directory pipelines and every check on peer data
src/rag/merkle.rs          Merkle proof verification and signed roots
src/rag/registry.rs        peers' signed roots, staleness, equivocation evidence
src/rag/directory.rs       verified copy of peers' centroids (directory mode)
src/rag/adversary.rs       simulated misbehaving peers, for experiments only
src/api.rs, config.rs,     /rag/* routes, [rag] config section,
src/main.rs, identity.rs,  sidecar startup and --print-endpoint-id,
src/network/mesh.rs        transport key for signing, second Iroh protocol
examples/rag-2node/        two-node RAG configs
corpus/                    two-node test corpus and 15 questions
scripts/rag_*.py           corpus builder, config generator, question runner, Merkle test vectors
scripts/m9/                Slakshna vs Flower FedRAG evaluation
tests/test_rag_*.py        sidecar tests; tests/fixtures/merkle_vectors.json shared with Rust
```

The federated-learning side of SLAKSHNA (`ml_engine.py`, `federated_communication/`, the Bhaskera submodule, the epoch loop in `src/main.rs`) is unchanged and still runs when `[training] enabled` is left on. The `main` branch README describes it.

## 12. Limits

The proofs bind a peer to **what it committed**, not to whether that content is true or complete:

- **Bad content committed in advance** passes every check. A peer can ingest misleading documents before signing.
- **Withholding.** A proof shows a chunk is in the corpus, not that it is the best one. A peer can leave relevant chunks out. Hiding a whole cluster is caught in directory mode (`incomplete_directory`) but not in `two_round`.
- **First contact.** The first root seen from a peer is accepted as is, and the peer registry is kept in memory, so it is lost on restart.
- **Privacy.** Peers see the raw question. Centroids reveal a node's topics, which is why `directory` is off by default. Chunks travel as plain text inside the encrypted QUIC connection.
- **Flags are recorded, not acted on.** Apart from equivocation, a flagged peer is still asked next time; turning flags into trust scores is future work.
- **Membership.** A node queries the peers it knows (gossip neighbours plus remembered and listed peers). Very large federations will need a proper membership mechanism.

---

## Licence

Apache License 2.0. See [LICENSE](LICENSE).
