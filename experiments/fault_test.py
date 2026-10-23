"""
Fault tolerance of federated inference: what happens to queries when peers go down?

The script starts and stops the nodes itself, from the serve_fault configs (fetch_timeout_secs = 10, made by
setup_run.py), so a dead peer costs 10 s instead of 60 s per attempt. Queries go to node1 (the query node). Phases:
  healthy        : all nodes up
  1 peer down    : node2 killed (node2 holds the adapter "trust" selection prefers in the pilot)
  2 peers down   : node2 and node3 killed
Before each (phase, method) all nodes are restarted with emptied caches (experiments/reset_caches.sh removes the
blob store and fetched adapter folders), then the phase's peers are killed, so a dead peer's adapter really has to be
fetched over the network, and that fails. Each query uses the normal selection (trust, then fallback down
the ranking, then the base model).

Output: <out>/raw/fault_tolerance.csv, one row per query:
  phase, peers_down, method, qid, ok, correct, adapter_node, cache_hit, fell_back, total_ms, note

    Bhaskera-serve/.venv/bin/python experiments/fault_test.py --tag ag_news_n4_a0.1_s0 --n 30
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import signal
import subprocess
import sys
import time

import requests

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "experiments"))
from eval_offline import SYSTEM, parse_label, read_jsonl  # noqa: E402


def pid_on_port(port):
    out = subprocess.run(["ss", "-ltnp"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if f":{port} " in line and "pid=" in line:
            return int(line.split("pid=")[1].split(",")[0])
    return None


def kill_node(tag, k, mode):
    """Kill node k's Rust process and its model sidecar."""
    pids = [int(x) for x in open(os.path.join(REPO, "experiments", "runs", tag, f"{mode}.pids")).read().split()]
    for p in (pids[k - 1], pid_on_port(8765 + k - 1)):
        if p:
            try:
                os.kill(p, signal.SIGKILL)
            except ProcessLookupError:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--methods", default="trust,self")
    ap.add_argument("--out", default=None)
    ap.add_argument("--mode", default="serve_fault", help="config folder the nodes were started from")
    a = ap.parse_args()
    out = a.out or os.path.join(REPO, "experiments", "results", a.tag)
    os.makedirs(os.path.join(out, "raw"), exist_ok=True)
    meta = json.load(open(os.path.join(REPO, "experiments", "data", a.tag, "meta.json")))
    labels = meta["labels"]
    tests = read_jsonl(os.path.join(REPO, "experiments", "data", a.tag, "test_global.jsonl"))
    random.Random(3).shuffle(tests)
    api = "http://127.0.0.1:8545"

    def restart_cold():
        # A downloaded adapter also stays in the node's blob store, so "cold" needs a restart with an emptied store.
        subprocess.run(["./experiments/stop_nodes.sh", a.tag, a.mode], cwd=REPO, capture_output=True)
        subprocess.run(["./experiments/reset_caches.sh", a.tag], cwd=REPO, capture_output=True)
        subprocess.run(["./experiments/run_nodes.sh", a.tag, a.mode, "local"], cwd=REPO, capture_output=True)
        t0 = time.time()
        while time.time() - t0 < 600:
            try:
                ok = all(requests.get(f"http://127.0.0.1:{8765 + k - 1}/health", timeout=3).status_code == 200
                         for k in range(1, meta["n_nodes"] + 1))
                if ok and len(requests.get(api + "/adapters", timeout=3).json()["adapters"]) >= meta["n_nodes"]:
                    return
            except Exception:
                pass
            time.sleep(3)
        raise SystemExit("nodes did not come back")

    restart_cold()
    ids = {}
    for k in range(1, meta["n_nodes"] + 1):
        for x in requests.get(f"http://127.0.0.1:{8545 + k - 1}/adapters").json()["adapters"]:
            if x["is_self"]:
                ids[x["node_id"]] = f"node{k}"

    f = open(os.path.join(out, "raw", "fault_tolerance.csv"), "w", newline="")
    w = csv.writer(f)
    w.writerow(["phase", "peers_down", "method", "qid", "ok", "correct", "adapter_node", "cache_hit", "fell_back",
                "total_ms", "note"])
    phases = [("healthy", []), ("1 peer down", [2]), ("2 peers down", [2, 3])]
    first = True
    for name, down in phases:
        for method in a.methods.split(","):
            if not first:
                restart_cold()
            first = False
            for k in down:
                kill_node(a.tag, k, a.mode)
            time.sleep(3)
            for q in tests[: a.n]:
                body = {"prompt": q["prompt"], "max_tokens": 8, "temperature": 0.0, "system": SYSTEM, "selection": method}
                t0 = time.time()
                try:
                    r = requests.post(api + "/infer", json=body, timeout=600).json()
                except Exception as e:
                    r = {"success": False, "message": repr(e)}
                ms = (time.time() - t0) * 1e3
                if not r.get("success"):
                    w.writerow([name, len(down), method, q["id"], 0, 0, "", "", "", ms, str(r.get("message", ""))[:100]])
                    continue
                res = r["result"]
                ad = res.get("adapter") or {}
                p = parse_label(res["text"], labels)
                note = res.get("note") or ""
                w.writerow([name, len(down), method, q["id"], 1, int(p == q["label"]),
                            ids.get(ad.get("node_id"), "base model") if ad else "base model",
                            ad.get("cache_hit", ""), int("unavailable" in note or "timed out" in note), ms, note[:100]])
            f.flush()
            print(f"{name} / {method} done", flush=True)
    f.close()
    subprocess.run(["./experiments/stop_nodes.sh", a.tag, a.mode], cwd=REPO, capture_output=True)
    print("wrote", os.path.join(out, "raw", "fault_tolerance.csv"))


if __name__ == "__main__":
    main()
