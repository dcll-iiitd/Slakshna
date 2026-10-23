"""
Cost of fetching a peer's adapter as a function of its size (LoRA rank 8 / 16 / 64 / 128).

Works on a COPY of a run (experiments/runs/<tag>_fetchbench) because it publishes synthetic adapters that would
otherwise stay in the run's registry. For each rank: node2 gets a synthetic adapter of that rank (same layers as the
real one, LoRA B = 0 so it does not change answers) under a new round number, node2 is restarted so it re-publishes it,
then node1 asks for node2's adapter several times, each time restarted with an empty blob store (= a real cold fetch).

Output: experiments/results/<tag>/raw/fetch_vs_adapter_size.csv
  lora_rank, adapter_mb, rep, fetch_ms, attach_ms, ttft_ms, total_ms

    Bhaskera-serve/.venv/bin/python experiments/bench_fetch.py --tag ag_news_n4_a0.1_s0
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time

import requests

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "experiments"))
from fault_test import pid_on_port  # noqa: E402


def make_adapter(template_dir, out_root, rank, node_id, rnd):
    import torch
    from safetensors.torch import load_file, save_file
    src = load_file(os.path.join(template_dir, "adapter_model.safetensors"))
    t = {}
    for k, v in src.items():
        if "lora_A" in k:
            t[k] = (torch.randn(rank, v.shape[1]) * 0.01).to(torch.bfloat16)
        elif "lora_B" in k:
            t[k] = torch.zeros(v.shape[0], rank, dtype=torch.bfloat16)
    tmp = os.path.join(out_root, f".synthetic_r{rank}")
    os.makedirs(tmp, exist_ok=True)
    save_file(t, os.path.join(tmp, "adapter_model.safetensors"))
    sha = hashlib.sha256(open(os.path.join(tmp, "adapter_model.safetensors"), "rb").read()).hexdigest()
    cfg = json.load(open(os.path.join(template_dir, "adapter_config.json")))
    cfg["r"], cfg["lora_alpha"] = rank, 2 * rank
    json.dump(cfg, open(os.path.join(tmp, "adapter_config.json"), "w"))
    m = json.load(open(os.path.join(template_dir, "adapter_meta.json")))
    m.update(node_id=node_id, round=rnd, rank=rank, alpha=2 * rank, kind="local",
             adapter_model_safetensors_sha256=sha, created_at=int(time.time()))
    json.dump(m, open(os.path.join(tmp, "adapter_meta.json"), "w"))
    final = os.path.join(out_root, sha)
    if os.path.exists(final):
        shutil.rmtree(final)
    os.replace(tmp, final)
    return sha, os.path.getsize(os.path.join(final, "adapter_model.safetensors")) / 2**20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ranks", default="8,16,64,128")
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    src = os.path.join(REPO, "experiments", "runs", a.tag)
    copy = src + "_fetchbench"
    if os.path.exists(copy):
        shutil.rmtree(copy)
    shutil.copytree(src, copy, ignore=shutil.ignore_patterns("logs", "*.pids", "train"))
    os.makedirs(os.path.join(copy, "logs"), exist_ok=True)
    for p in glob.glob(os.path.join(copy, "serve", "*.toml")):
        txt = open(p).read().replace(f"experiments/runs/{a.tag}/", f"experiments/runs/{a.tag}_fetchbench/")
        open(p, "w").write(txt)
    env = dict(os.environ, SLAKSHNA_EXPORT_KIND="local")

    def start(k):
        log = open(os.path.join(copy, "logs", f"node{k}_serve.log"), "a")
        return subprocess.Popen(["./target/release/iiitd", "--config", os.path.join(copy, "serve", f"node{k}.toml")],
                                cwd=REPO, stdout=log, stderr=log, env=env)

    def stop(proc, k):
        proc.send_signal(signal.SIGKILL)
        p = pid_on_port(8765 + k - 1)
        if p:
            os.kill(p, signal.SIGKILL)
        time.sleep(3)

    procs = {k: start(k) for k in (1, 2)}  # node1 queries, node2 publishes; that is all this test needs
    api1 = "http://127.0.0.1:8545"

    def wait_health(k, timeout=600):
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                if requests.get(f"http://127.0.0.1:{8765 + k - 1}/health", timeout=3).status_code == 200:
                    return
            except Exception:
                pass
            time.sleep(3)
        raise SystemExit(f"node{k} sidecar not ready")

    wait_health(1)
    wait_health(2)
    me2 = next(x for x in requests.get("http://127.0.0.1:8546/adapters").json()["adapters"] if x["is_self"])
    node2_id = me2["node_id"]
    tmpl = next(d for d in glob.glob(os.path.join(copy, "data-node2", "adapters", "*"))
                if os.path.isfile(os.path.join(d, "adapter_meta.json")) and json.load(open(os.path.join(d, "adapter_meta.json"))).get("kind") == "local")
    out = os.path.join(REPO, "experiments", "results", a.tag, "raw")
    os.makedirs(out, exist_ok=True)
    f = open(os.path.join(out, "fetch_vs_adapter_size.csv"), "w", newline="")
    w = csv.writer(f)
    w.writerow(["lora_rank", "adapter_mb", "rep", "fetch_ms", "attach_ms", "ttft_ms", "total_ms"])
    q = json.loads(open(os.path.join(REPO, "experiments", "data", a.tag, "test_global.jsonl")).readline())
    for i, r in enumerate(int(x) for x in a.ranks.split(",")):
        rnd = 100 + i
        sha, mb = make_adapter(tmpl, os.path.join(copy, "data-node2", "adapters"), r, node2_id, rnd)
        stop(procs[2], 2)
        procs[2] = start(2)
        t0 = time.time()
        while time.time() - t0 < 300:  # node1 must see the new round in its registry
            ads = requests.get(api1 + "/adapters").json()["adapters"]
            if any(x["node_id"] == node2_id and x["round"] == rnd for x in ads):
                break
            time.sleep(2)
        for rep in range(a.reps):
            # Truly cold: node1 restarts with an empty blob store and no fetched adapters
            # (a downloaded adapter also stays in the blob store, so deleting its folder is not enough).
            stop(procs[1], 1)
            shutil.rmtree(os.path.join(copy, "data-node1", "blobs"), ignore_errors=True)
            for ad in glob.glob(os.path.join(copy, "data-node1", "adapters", "*")):
                if not os.path.isfile(os.path.join(ad, "adapter_meta.json")):
                    shutil.rmtree(ad, ignore_errors=True)
            procs[1] = start(1)
            wait_health(1)
            t0 = time.time()
            while time.time() - t0 < 300:
                try:
                    ads = requests.get(api1 + "/adapters", timeout=3).json()["adapters"]
                    if any(x["node_id"] == node2_id and x["round"] == rnd for x in ads):
                        break
                except Exception:
                    pass
                time.sleep(2)
            res = requests.post(api1 + "/infer", json={"prompt": q["prompt"], "max_tokens": 4, "temperature": 0.0,
                                                       "from_node": node2_id}, timeout=600).json()["result"]
            tm = res["timing"]
            if (res.get("adapter") or {}).get("sha256") != sha:
                print(f"rank {r}: got a different adapter: {res.get('note')}")
                continue
            w.writerow([r, round(mb, 2), rep, tm["fetch_ms"], tm["attach_ms"], tm["ttft_ms"], tm["total_ms"]])
        f.flush()
        print(f"rank {r}: {mb:.1f} MB done", flush=True)
    f.close()
    for k, p in procs.items():
        stop(p, k)
    print("wrote", os.path.join(out, "fetch_vs_adapter_size.csv"))


if __name__ == "__main__":
    main()
