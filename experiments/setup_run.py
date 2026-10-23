"""
Prepare one experiment run end to end (everything under experiments/):

  1. patch node_template.yaml for the experiment model / rounds / per-round training budget
  2. tokenize each node's own partition (experiments/data/<tag>/nodeK/train.jsonl -> nodeK/tokenized)
  3. write node configs:  experiments/runs/<tag>/train/nodeK.toml   (training, no model server)
                          experiments/runs/<tag>/serve/nodeK.toml   (serve_only, every node answers queries)

State of a run (RocksDB, adapters, blobs) lives in experiments/runs/<tag>/data-nodeK, so runs never
touch each other or the repo-level data-node* folders.

    python experiments/setup_run.py --tag ag_news_n4_a0.1_s0 --model Qwen/Qwen2.5-1.5B-Instruct --rounds 3
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(REPO, "experiments")


def patch_template(model: str, rounds: int, steps: int, batch: int, accum: int, lr: str, seq_len: int):
    path = os.path.join(REPO, "node_template.yaml")
    with open(path) as f:
        t = yaml.safe_load(f)
    t["model"]["name"] = model
    t["federated"]["total_epochs"] = rounds
    t["data"]["seq_len"] = seq_len
    t["data"]["max_train_samples"] = 1000000
    t["training"]["batch_size"] = batch
    t["training"]["gradient_accumulation_steps"] = accum
    t["training"]["max_steps"] = steps
    t["training"]["learning_rate"] = float(lr)
    with open(path, "w") as f:
        yaml.safe_dump(t, f, sort_keys=False)
    return t


def toml_node(i: int, n: int, tag: str, model: str, gpus: list[int], mode: str, epoch_s: int, deadline_s: int,
              run: str, sparsity: float, isolated: bool, port_offset: int) -> str:
    gpu = gpus[(i - 1) % len(gpus)]
    serve = mode == "serve"
    off = port_offset if mode == "train" else 0  # serve runs happen one at a time on the default ports
    fed = f"{run}-node{i}" if isolated else run  # isolated: every node is its own one-member federation
    return f'''[federation]
id = "slakshna-exp-{fed}"
name = "SLAKSHNA experiment {fed}"

[training]
epoch_duration_secs = {epoch_s}
sync_deadline_secs = {deadline_s}
expected_peers = {1 if isolated else n}

[compression]
enabled = true
sparsity = {sparsity}
quantization = "symmetric_int8"
allow_legacy_delta_format = false
max_payload_bytes = 2000000000
max_tensor_elements = 2000000000

[node]
id = "node-{i}"
data_dir = "./experiments/runs/{run}/data-node{i}"
gpu_id = {gpu}
num_gpus = 1

[network]
host = "0.0.0.0"
p2p_port = {9000 + off + i - 1}
ws_port = {8600 + off + i - 1}
api_port = {8545 + off + i - 1}
peers = []

[discovery]
mdns = true
dht = false
dns = false
relay = false

[inference]
enabled = true
base_model = "{model}"
selection = "trust"
min_trust = 0.0
sidecar_port = {8765 + off + i - 1}
sidecar_python = "Bhaskera-serve/.venv/bin/python"
serve_bhaskera_path = "Bhaskera-serve"
spawn_sidecar = {"true" if serve else "false"}
serve_only = {"true" if serve else "false"}
fetch_timeout_secs = 60
request_timeout_secs = 600

[logging]
level = "info"
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=30, help="optimizer steps per round")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", default="3e-4")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--gpus", default="0,1,2", help="GPUs to spread nodes over, round-robin")
    ap.add_argument("--epoch-secs", type=int, default=480)
    ap.add_argument("--deadline-secs", type=int, default=420)
    ap.add_argument("--skip-tokenize", action="store_true")
    ap.add_argument("--run", default=None, help="run name (default = tag); lets several runs share one data tag")
    ap.add_argument("--sparsity", type=float, default=0.1, help="fraction of each update sent to peers (1.0 = full FedAvg)")
    ap.add_argument("--isolated", action="store_true", help="no federation: every node trains alone (local-only baseline)")
    ap.add_argument("--port-offset", type=int, default=0, help="shift training ports so runs can train at the same time")
    a = ap.parse_args()

    data_root = os.path.join(EXP, "data", a.tag)
    meta = json.load(open(os.path.join(data_root, "meta.json")))
    n = meta["n_nodes"]
    gpus = [int(x) for x in a.gpus.split(",")]

    tmpl = patch_template(a.model, a.rounds, a.steps, a.batch, a.accum, a.lr, a.seq_len)
    print(f"node_template.yaml -> model={a.model} rounds={a.rounds} steps/round={a.steps} "
          f"batch={a.batch}x{a.accum} lr={a.lr} seq_len={a.seq_len}")

    if not a.skip_tokenize:
        tok_py = os.path.join(REPO, "scripts", "tokenize_dataset.py")
        for k in range(1, n + 1):
            part = os.path.join(data_root, f"node{k}")
            t = yaml.safe_load(open(os.path.join(REPO, "node_template.yaml")))
            t["data"]["train_path"] = os.path.join(part, "train.jsonl")
            t["data"]["tokenized_path"] = os.path.join(part, "tokenized")
            tpath = os.path.join(part, "template.yaml")
            yaml.safe_dump(t, open(tpath, "w"), sort_keys=False)
            print(f"tokenizing node{k} ...", flush=True)
            r = subprocess.run([sys.executable, tok_py, "--template", tpath, "--force"],
                               cwd=REPO, capture_output=True, text=True)
            tail = "\n".join((r.stdout + r.stderr).strip().splitlines()[-3:])
            if r.returncode != 0:
                print(tail)
                raise SystemExit(f"tokenization of node{k} failed")
            print("  " + tail.splitlines()[-1])

    run = a.run or a.tag
    run_dir = os.path.join(EXP, "runs", run)
    for mode in ("train", "serve"):
        d = os.path.join(run_dir, mode)
        os.makedirs(d, exist_ok=True)
        for i in range(1, n + 1):
            with open(os.path.join(d, f"node{i}.toml"), "w") as f:
                f.write(toml_node(i, n, a.tag, a.model, gpus, mode, a.epoch_secs, a.deadline_secs,
                                  run, a.sparsity, a.isolated, a.port_offset))
    # serve_fault: serve configs with a short fetch timeout, for experiments/fault_test.py
    os.makedirs(os.path.join(run_dir, "serve_fault"), exist_ok=True)
    for i in range(1, n + 1):
        txt = open(os.path.join(run_dir, "serve", f"node{i}.toml")).read().replace("fetch_timeout_secs = 60", "fetch_timeout_secs = 10")
        open(os.path.join(run_dir, "serve_fault", f"node{i}.toml"), "w").write(txt)
    with open(os.path.join(run_dir, "run.json"), "w") as f:
        json.dump({"run": run, "data_tag": a.tag, "sparsity": a.sparsity, "isolated": a.isolated,
                   "rounds": a.rounds, "steps": a.steps, "model": a.model}, f, indent=2)
    print(f"configs written under {run_dir}/{{train,serve}}/")


if __name__ == "__main__":
    main()
