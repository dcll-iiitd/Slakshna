"""
System benchmark for System One decisions (one row per decision in raw/decide_<backend>.csv).

  --backend slakshna : our node's /decide (local registry -> fetch missing adapters by hash -> one forward pass per
                       adapter per option ordering on the node's own GPU), with the node's own timing breakdown.
  --backend vllm     : a central vLLM server holding the same adapters. One decision = one completion request per
                       member per cyclic ordering, max_tokens=1, logprobs restricted to the option letters with
                       allowed_token_ids (the same restricted next-token distribution System One uses); requests of a
                       decision are sent concurrently so vLLM can batch them. The mix is computed with system_one.

Sweeps:
  members      k = 1..4 adapters mixed (all orderings)                   -> latency vs k
  permutations 1 / 2 / all orderings, k = 4                              -> latency vs debiasing cost
  options      2..n options (random subsets incl. the gold), k = 1 and 4 -> latency vs options (two-stage above 52)
  concurrency  1..16 decisions in flight, k = 4                          -> decisions/s, p50/p95 latency
  determinism  the same decision 50 times (ours)                         -> max probability difference
  cold         ours only: caches reset before the run (reset_caches.sh), first k=4 decision fetches 3 adapters

    Bhaskera-serve/.venv/bin/python experiments/bench_decisions.py --tag ag_news_n4_a0.1_s0_s1 --backend slakshna
    experiments/.venv-vllm/bin/python experiments/bench_decisions.py --tag ... --backend vllm --url http://127.0.0.1:8000
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault("HF_HOME", os.path.join(REPO, "hf_cache"))

COLS = ["backend", "sweep", "k", "permutations", "n_options", "concurrency", "rep", "ok", "correct", "choice", "gold",
        "confidence", "total_ms", "select_ms", "fetch_ms", "sidecar_ms", "model_ms", "forward_passes", "members_used",
        "cache_misses", "max_prob_diff", "t_start", "t_end", "note"]


class Ours:
    def __init__(self, url, node_ids):
        self.url, self.node_ids = url.rstrip("/"), node_ids

    def decide(self, q, options, gold, k, perms):
        body = {"state": q["state"], "question": q["question"], "options": options, "k": k, "permutations": perms}
        t0 = time.perf_counter()
        r = requests.post(self.url + "/decide", json=body, timeout=900).json()
        ms = (time.perf_counter() - t0) * 1e3
        if not r.get("success"):
            return {"ok": 0, "total_ms": ms, "note": str(r.get("message", ""))[:100]}
        res = r["result"]
        d, tm = res["decision"], res.get("timing", {})
        return {"ok": 1, "correct": int(d["choice"] == gold), "choice": d["choice"], "gold": gold,
                "confidence": d["confidence"], "total_ms": ms, "select_ms": tm.get("select_ms"),
                "fetch_ms": tm.get("fetch_ms"), "sidecar_ms": tm.get("sidecar_ms"), "model_ms": tm.get("model_ms"),
                "forward_passes": tm.get("forward_passes"), "members_used": len(res.get("sources", [])),
                "cache_misses": sum(1 for s in res.get("sources", []) if not s.get("cache_hit")),
                "probs": d["probabilities"], "note": (res.get("note") or "")[:100]}


    def generate(self, q, options, gold):
        """The same question answered by generation through /infer (own adapter), the option text parsed back."""
        prompt = (f"{q['state']}\n\nQuestion: {q['question']}\nOptions:\n" + "\n".join(f"- {o}" for o in options)
                  + "\nAnswer with the exact text of one option and nothing else.")
        body = {"prompt": prompt, "max_tokens": 16, "temperature": 0.0, "selection": "self",
                "system": "You are a helpful AI assistant"}
        t0 = time.perf_counter()
        r = requests.post(self.url + "/infer", json=body, timeout=900).json()
        ms = (time.perf_counter() - t0) * 1e3
        if not r.get("success"):
            return {"ok": 0, "total_ms": ms, "note": str(r.get("message", ""))[:100]}
        t = r["result"]["text"].strip().strip("\"'`.*- ").lower().replace(" ", "_")
        pred = next((i for i in sorted(range(len(options)), key=lambda i: -len(options[i]))
                     if t.startswith(options[i].lower().replace(" ", "_"))), -1)
        return {"ok": 1, "correct": int(pred == gold), "choice": pred, "gold": gold, "total_ms": ms,
                "note": "unparseable" if pred < 0 else ""}


class VLLM:
    """System One decisions computed by a central vLLM server (single stage only)."""

    def __init__(self, url, members, tok):
        from federated_inference import system_one as so
        self.so, self.url, self.members, self.tok = so, url.rstrip("/"), members, tok
        self.labels = so.label_vocab(tok)
        self.pool = ThreadPoolExecutor(256)

    def _one(self, member, prompt, ids):
        body = {"model": member, "prompt": prompt, "max_tokens": 1, "temperature": 0.0, "logprobs": len(ids),
                "allowed_token_ids": ids}
        r = requests.post(self.url + "/v1/completions", json=body, timeout=600).json()
        top = r["choices"][0]["logprobs"]["top_logprobs"][0]
        return top

    def decide(self, q, options, gold, k, perms):
        import torch
        so = self.so
        n = len(options)
        if n > len(self.labels):
            return {"ok": 0, "note": "two-stage decisions are not implemented for the vLLM baseline"}
        chars = [c for c, _ in self.labels[:n]]
        ids = [i for _, i in self.labels[:n]]
        shifts = so._shifts(n, perms)
        t0 = time.perf_counter()
        jobs = []
        for m in self.members[:k]:
            for s in shifts:
                order = [(j + s) % n for j in range(n)]
                prompt = so.build_prompt(q["state"], q["question"], [options[i] for i in order], chars)
                jobs.append((m, s, self.pool.submit(self._one, m, prompt, ids)))
        rows = {m: torch.zeros(n, dtype=torch.float64) for m in self.members[:k]}
        for m, s, fut in jobs:
            top = fut.result()
            lp = torch.full((n,), -1e9, dtype=torch.float64)
            for j, c in enumerate(chars):
                key = next((t for t in top if t.strip() == c and t.startswith(" ")), None) or (" " + c if (" " + c) in top else None)
                if key is not None:
                    lp[j] = top[key]
            p = torch.softmax(lp, -1)
            for j in range(n):
                rows[m][(j + s) % n] += p[j]
        stack = torch.stack([rows[m] / len(shifts) for m in self.members[:k]])
        mixed = so.mix_probs(stack, so.normalise_weights([1.0] * k), "prob_mean")
        ms = (time.perf_counter() - t0) * 1e3
        ch = int(mixed.argmax())
        return {"ok": 1, "correct": int(ch == gold), "choice": ch, "gold": gold, "confidence": float(mixed[ch]),
                "total_ms": ms, "forward_passes": len(jobs), "members_used": k, "probs": mixed.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--backend", choices=["slakshna", "vllm"], required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8545")
    ap.add_argument("--n", type=int, default=30, help="decisions per setting")
    ap.add_argument("--sweeps", default="members,permutations,options,concurrency,generate,determinism")
    ap.add_argument("--cold", action="store_true", help="first: one cold k=4 decision per question set (caches reset beforehand)")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    a = ap.parse_args()
    data = os.path.join(REPO, "experiments", "data", a.tag)
    meta = json.load(open(os.path.join(data, "meta.json")))
    names = meta["labels"]
    qs = [json.loads(l) for l in open(os.path.join(data, "test_global.jsonl"))]
    random.Random(5).shuffle(qs)
    out = os.path.join(REPO, "experiments", "results", "system_one", a.tag, "raw")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"decide_{a.backend}.csv")
    f = open(path, "w", newline="")
    w = csv.writer(f)
    w.writerow(COLS)
    lock = threading.Lock()

    if a.backend == "slakshna":
        be = Ours(a.url, None)
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.model)
        models = [m["id"] for m in requests.get(a.url + "/v1/models").json()["data"] if m["id"].startswith("node")]
        be = VLLM(a.url, sorted(models), tok)

    def record(sweep, k, perms, n_opt, conc, rep, res, ts):
        res = dict(res)
        res.pop("probs", None)
        row = {"backend": a.backend, "sweep": sweep, "k": k, "permutations": perms, "n_options": n_opt,
               "concurrency": conc, "rep": rep, "t_start": ts, "t_end": time.time(), **res}
        with lock:
            w.writerow([row.get(c, "") for c in COLS])

    def run(sweep, k, perms, n_opt, conc, items):
        todo = list(enumerate(items))

        def worker():
            while True:
                with lock:
                    if not todo:
                        return
                    rep, (q, opts, gold) = todo.pop(0)
                ts = time.time()
                try:
                    res = be.decide(q, opts, gold, k, perms)
                except Exception as e:
                    res = {"ok": 0, "note": repr(e)[:100]}
                record(sweep, k, perms, n_opt, conc, rep, res, ts)
        th = [threading.Thread(target=worker) for _ in range(conc)]
        [t.start() for t in th]
        [t.join() for t in th]
        f.flush()

    full = [(q, names, q["label"]) for q in qs]
    if a.cold:
        run("cold", 4, 0, len(names), 1, full[:1])
    run("warmup", 4, 0, len(names), 1, full[:3])
    sweeps = a.sweeps.split(",")
    if "members" in sweeps:
        for k in (1, 2, 3, 4):
            run("members", k, 0, len(names), 1, full[: a.n])
            print(f"[{a.backend}] members k={k} done", flush=True)
    if "permutations" in sweeps:
        for p in (1, 2, 0):
            run("permutations", 4, p, len(names), 1, full[: a.n])
        print(f"[{a.backend}] permutations done", flush=True)
    if "options" in sweeps:
        rng = random.Random(9)
        counts = [c for c in (2, 4, 8, 16, 32, 52, 77) if c <= len(names)]
        for c in counts:
            items = []
            for q in full[: a.n]:
                others = [i for i in range(len(names)) if i != q[2]]
                sub = sorted([q[2]] + rng.sample(others, c - 1))
                items.append((q[0], [names[i] for i in sub], sub.index(q[2])))
            for k in (1, 4):
                run("options", k, 0, c, 1, items)
            print(f"[{a.backend}] options {c} done", flush=True)
    if "concurrency" in sweeps:
        for c in (1, 2, 4, 8, 16):
            run("concurrency", 4, 0, len(names), c, full[: max(16, 4 * c)])
            print(f"[{a.backend}] concurrency {c} done", flush=True)
    if "generate" in sweeps and a.backend == "slakshna":
        for rep, (q, opts, gold) in enumerate(full[: a.n]):
            ts = time.time()
            record("generate", 1, "", len(names), 1, rep, be.generate(q, opts, gold), ts)
        f.flush()
        print(f"[{a.backend}] generate done", flush=True)
    if "determinism" in sweeps and a.backend == "slakshna":
        q = full[0]
        first = be.decide(q[0], q[1], q[2], 4, 0)
        for rep in range(50):
            ts = time.time()
            r = be.decide(q[0], q[1], q[2], 4, 0)
            r["max_prob_diff"] = max(abs(x - y) for x, y in zip(r.get("probs", []), first.get("probs", []))) if r.get("ok") else ""
            record("determinism", 4, 0, len(names), 1, rep, r, ts)
        print(f"[{a.backend}] determinism done", flush=True)
    f.close()
    print("wrote", path)


if __name__ == "__main__":
    main()
