"""
Exercise the REAL federated-inference system through its HTTP API and record one row per query.

Run this with the nodes in serve mode (experiments/run_nodes.sh <tag> serve). Two phases per query node:

  fetch   : for every other node s, drop the cached copy of s's adapter, then send one COLD query
            (adapter must be fetched by hash, verified, attached) and several WARM queries (cache hit).
            -> fetch/attach/TTFT/TPOT for cold vs warm.
  strategy: N held-out questions answered under each selection strategy (self / trust / latest / random),
            greedy decoding, same prompt and system turn the nodes were trained with.
            -> accuracy per strategy through the actual system, plus per-query latency.

Output: experiments/results/<tag>/raw/api_queries.csv
  columns: tag, phase, query_node, method, source_node, qid, label, label_text, pred, correct, invalid, raw,
           adapter_node, adapter_round, cache_hit, select_ms, fetch_ms, attach_ms, model_ttft_ms, ttft_ms,
           tpot_ms, total_ms, n_tokens, n_prompt_tokens, note, ts

    Bhaskera-serve/.venv/bin/python experiments/eval_api.py --tag ag_news_n4_a0.1_s0 --n 200
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import sys
import threading
import time

import requests

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "experiments"))
from eval_offline import SYSTEM, parse_label, read_jsonl  # noqa: E402

COLS = ["tag", "phase", "query_node", "method", "source_node", "qid", "label", "label_text", "pred", "correct",
        "invalid", "raw", "adapter_node", "adapter_round", "cache_hit", "select_ms", "fetch_ms", "attach_ms",
        "model_ttft_ms", "ttft_ms", "tpot_ms", "total_ms", "n_tokens", "n_prompt_tokens", "note", "ts"]


def api(k: int) -> str:
    return f"http://127.0.0.1:{8545 + k - 1}"


def wait_ready(tag, n, timeout=900):
    """All nodes up, all sidecars healthy, every node's registry lists all n adapters."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            ok = True
            for k in range(1, n + 1):
                ad = requests.get(api(k) + "/adapters", timeout=5).json().get("adapters", [])
                ok &= len(ad) >= n
                h = requests.get(f"http://127.0.0.1:{8765 + k - 1}/health", timeout=5)
                ok &= h.status_code == 200
            if ok:
                return
        except Exception:
            pass
        time.sleep(5)
    raise SystemExit("nodes did not become ready (registries not complete or a sidecar is down)")


class Recorder:
    def __init__(self, path):
        self.f = open(path, "w", newline="", encoding="utf-8")
        self.w = csv.writer(self.f)
        self.w.writerow(COLS)
        self.lock = threading.Lock()

    def add(self, row):
        with self.lock:
            self.w.writerow([row.get(c, "") for c in COLS])
            self.f.flush()


def ask(k, q, labels, selection=None, from_node=None, max_tokens=8):
    body = {"prompt": q["prompt"], "max_tokens": max_tokens, "temperature": 0.0, "system": SYSTEM}
    if selection:
        body["selection"] = selection
    if from_node:
        body["from_node"] = from_node
    t0 = time.time()
    r = requests.post(api(k) + "/infer", json=body, timeout=900).json()
    wall = (time.time() - t0) * 1000.0
    if not r.get("success"):
        return None, wall, r
    return r["result"], wall, None


def to_row(tag, phase, k, method, src, q, res, labels, ids):
    raw = res["text"].strip().replace("\n", " ")[:80]
    pred = parse_label(res["text"], labels)
    ad, tm = res.get("adapter") or {}, res.get("timing", {})
    return {
        "tag": tag, "phase": phase, "query_node": f"node{k}", "method": method, "source_node": src,
        "qid": q["id"], "label": q["label"], "label_text": q["label_text"], "pred": pred,
        "correct": int(pred == q["label"]), "invalid": int(pred < 0), "raw": raw,
        "adapter_node": ids.get(ad.get("node_id"), ad.get("node_id", "base")) if ad else "base",
        "adapter_round": ad.get("round", 0), "cache_hit": ad.get("cache_hit", ""),
        "select_ms": tm.get("select_ms"), "fetch_ms": tm.get("fetch_ms"), "attach_ms": tm.get("attach_ms"),
        "model_ttft_ms": tm.get("model_ttft_ms"), "ttft_ms": tm.get("ttft_ms"), "tpot_ms": tm.get("tpot_ms"),
        "total_ms": tm.get("total_ms"), "n_tokens": tm.get("n_tokens"), "n_prompt_tokens": tm.get("n_prompt_tokens"),
        "note": (res.get("note") or "")[:120], "ts": int(time.time()),
    }


def worker(k, a, meta, tests, ids_by_node, rec, errors):
    labels, n = meta["labels"], meta["n_nodes"]
    rng = random.Random(1000 + k)
    try:
        # ---- fetch phase: cold vs warm -----------------------------------------------------------------
        for s in range(1, n + 1):
            if s == k:
                continue
            nid = ids_by_node[s]
            adapters = requests.get(api(k) + "/adapters").json()["adapters"]
            sha = next(x["sha256"] for x in adapters if x["node_id"] == nid)
            cached = os.path.join(REPO, "experiments", "runs", a.tag, f"data-node{k}", "adapters", sha)
            if os.path.isdir(cached):
                shutil.rmtree(cached)  # force a real fetch by hash
            probe = tests[:1 + a.warm]
            for i, q in enumerate(probe):
                res, wall, err = ask(k, q, labels, from_node=nid)
                if res is None:
                    print(f"node{k}<-node{s} query failed: {err}", file=sys.stderr)
                    continue
                rec.add(to_row(a.tag, "fetch", k, "cold" if i == 0 else "warm", f"node{s}", q, res, labels, ids_by_node_inv))
        # ---- strategy phase ------------------------------------------------------------------------------
        for method in a.methods:
            for q in tests[: a.n]:
                if method == "random":
                    s = rng.choice([x for x in range(1, n + 1)])
                    res, wall, err = ask(k, q, labels, from_node=ids_by_node[s])
                else:
                    res, wall, err = ask(k, q, labels, selection=method)
                if res is None:
                    print(f"node{k} {method} failed: {err}", file=sys.stderr)
                    continue
                rec.add(to_row(a.tag, "strategy", k, method, "", q, res, labels, ids_by_node_inv))
            print(f"node{k} finished method={method}", flush=True)
    except Exception as e:  # keep other nodes going
        errors.append((k, repr(e)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n", type=int, default=200, help="questions per (query node, method)")
    ap.add_argument("--warm", type=int, default=5)
    ap.add_argument("--methods", default="self,trust,latest,random")
    ap.add_argument("--query-nodes", default="all")
    a = ap.parse_args()
    a.methods = a.methods.split(",")

    data_root = os.path.join(REPO, "experiments", "data", a.tag)
    meta = json.load(open(os.path.join(data_root, "meta.json")))
    n = meta["n_nodes"]
    tests = read_jsonl(os.path.join(data_root, "test_global.jsonl"))
    random.Random(7).shuffle(tests)

    wait_ready(a.tag, n)
    global ids_by_node_inv
    ids_by_node = {}
    for k in range(1, n + 1):
        for x in requests.get(api(k) + "/adapters").json()["adapters"]:
            if x["is_self"]:
                ids_by_node[k] = x["node_id"]
    ids_by_node_inv = {v: f"node{k}" for k, v in ids_by_node.items()}
    print("node ids:", {f"node{k}": v for k, v in ids_by_node.items()}, flush=True)

    out_dir = os.path.join(REPO, "experiments", "results", a.tag)
    os.makedirs(out_dir, exist_ok=True)
    json.dump({f"node{k}": v for k, v in ids_by_node.items()}, open(os.path.join(out_dir, "node_ids.json"), "w"), indent=2)
    os.makedirs(os.path.join(out_dir, "raw"), exist_ok=True)
    rec = Recorder(os.path.join(out_dir, "raw", "api_queries.csv"))
    qn = range(1, n + 1) if a.query_nodes == "all" else [int(x) for x in a.query_nodes.split(",")]
    errors, threads = [], []
    for k in qn:
        t = threading.Thread(target=worker, args=(k, a, meta, tests, ids_by_node, rec, errors))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    if errors:
        print("errors:", errors, file=sys.stderr)
    print("done ->", os.path.join(out_dir, "raw", "api_queries.csv"))


if __name__ == "__main__":
    main()
