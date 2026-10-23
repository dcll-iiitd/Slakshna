"""
Standard LLM-serving benchmark (TTFT, TPOT, end-to-end latency, throughput, goodput) for
  --backend slakshna : our node's /infer API (federated path: registry -> adapter -> local sidecar)
  --backend vllm     : a vLLM OpenAI-compatible server with the same LoRA adapters (centralised multi-LoRA server)

Same model, same adapter, synthetic prompts with an exact number of tokens (every request's prompt is different, so
no prompt caching can help either system), greedy decoding, fixed output length. vLLM runs with prefix caching off.

Sweeps (each one row per request in raw/serving_<backend>.csv):
  prompt_len  : prompt 128..4096 tokens, 16 output tokens, 1 request at a time   -> TTFT, prefill TFLOP/s
  output_len  : 256-token prompt, 16..512 output tokens, 1 request at a time     -> TPOT, end-to-end latency
  concurrency : 256-token prompt, 128 output tokens, 1..16 requests in flight     -> throughput, goodput, queueing

How the numbers are measured:
  vllm     : streamed responses; TTFT = first streamed token at the client; TPOT = (last - first) / (tokens - 1).
  slakshna : the API is not streamed, so TTFT/TPOT come from the server's own timing (sidecar streams internally);
             client TTFT under concurrency = client end-to-end - decode time, so it includes time spent queued.

    experiments/.venv-vllm/bin/python experiments/bench_serving.py --backend vllm --url http://127.0.0.1:8000 --adapter node1
    Bhaskera-serve/.venv/bin/python experiments/bench_serving.py --backend slakshna --url http://127.0.0.1:8545
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import threading
import time

import requests

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", os.path.join(REPO, "hf_cache"))

COLS = ["backend", "sweep", "prompt_tokens", "output_tokens", "concurrency", "rep", "ok", "ttft_ms", "tpot_ms",
        "e2e_ms", "n_out", "server_model_ttft_ms", "queue_ms", "t_start", "t_end"]

FILLER = ("Federated learning lets many devices train a shared model without moving their data. "
          "Each device keeps its examples private and only shares model updates with its peers. ")


class Prompts:
    """Prompts of an exact token length, all DIFFERENT: each starts with its own random words, so no request can
    reuse another's cached prompt (vLLM caches shared prefixes by default; our sidecar has no such cache)."""

    def __init__(self, tok, seed=0):
        import random
        self.tok, self.rng = tok, random.Random(seed)
        words = [w for w in "time river light stone paper market signal garden window engine forest letter music silver "
                 "planet bridge harbor winter summer copper valley mirror candle rocket island thunder".split()]
        self.words = words

    def __call__(self, n_tokens: int) -> str:
        head = " ".join(self.rng.choice(self.words) for _ in range(24)) + ". "
        ids = self.tok(head + FILLER * (n_tokens // 10 + 50), add_special_tokens=False)["input_ids"][:n_tokens]
        return self.tok.decode(ids)


class Backend:
    def __init__(self, kind, url, adapter):
        self.kind, self.url, self.adapter = kind, url.rstrip("/"), adapter

    def run(self, prompt, out_tokens):
        """-> dict(ok, ttft_ms, tpot_ms, e2e_ms, n_out, server_model_ttft_ms)."""
        t0 = time.perf_counter()
        if self.kind == "vllm":
            body = {"model": self.adapter, "prompt": prompt, "max_tokens": out_tokens, "min_tokens": out_tokens,
                    "temperature": 0.0, "stream": True, "ignore_eos": True,
                    "stream_options": {"include_usage": True}}
            first, n_out = None, 0
            with requests.post(self.url + "/v1/completions", json=body, stream=True, timeout=600) as r:
                if r.status_code != 200:
                    return {"ok": 0}
                for line in r.iter_lines():
                    if not line or not line.startswith(b"data: ") or line == b"data: [DONE]":
                        continue
                    ev = json.loads(line[6:])
                    if ev.get("usage"):
                        n_out = ev["usage"].get("completion_tokens", n_out)
                    if ev.get("choices") and ev["choices"][0].get("text") and first is None:
                        first = time.perf_counter()
            t1 = time.perf_counter()
            first = first or t1
            return {"ok": 1, "ttft_ms": (first - t0) * 1e3, "e2e_ms": (t1 - t0) * 1e3, "n_out": n_out,
                    "tpot_ms": (t1 - first) * 1e3 / max(n_out - 1, 1), "server_model_ttft_ms": ""}
        body = {"prompt": prompt, "max_tokens": out_tokens, "min_tokens": out_tokens, "temperature": 0.0,
                "raw": True, "selection": "self"}
        r = requests.post(self.url + "/infer", json=body, timeout=900)
        t1 = time.perf_counter()
        j = r.json()
        if not j.get("success"):
            return {"ok": 0}
        tm = j["result"]["timing"]
        e2e = (t1 - t0) * 1e3
        decode = tm["tpot_ms"] * max(tm["n_tokens"] - 1, 0)
        return {"ok": 1, "ttft_ms": e2e - decode, "tpot_ms": tm["tpot_ms"], "e2e_ms": e2e, "n_out": tm["n_tokens"],
                "server_model_ttft_ms": tm["model_ttft_ms"]}


def sweep_requests(be, w, name, prompt_tokens, out_tokens, conc, n_req, make):
    """Run n_req requests with `conc` in flight; write one row each."""
    lock = threading.Lock()
    todo = list(range(n_req))

    def worker():
        while True:
            with lock:
                if not todo:
                    return
                rep = todo.pop(0)
                prompt = make(prompt_tokens)  # a fresh, unique prompt for every request
            ts = time.time()
            try:
                res = be.run(prompt, out_tokens)
            except Exception:
                res = {"ok": 0}
            row = {"backend": be.kind, "sweep": name, "prompt_tokens": prompt_tokens, "output_tokens": out_tokens,
                   "concurrency": conc, "rep": rep, "t_start": ts, "t_end": time.time(), **res}
            with lock:
                w.writerow([row.get(c, "") for c in COLS])

    th = [threading.Thread(target=worker) for _ in range(conc)]
    for t in th:
        t.start()
    for t in th:
        t.join()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["slakshna", "vllm"], required=True)
    ap.add_argument("--url", required=True)
    ap.add_argument("--adapter", default="node1", help="vLLM LoRA module name")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", required=True, help="results folder; writes raw/serving_<backend>.csv")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    be = Backend(a.backend, a.url, a.adapter)
    os.makedirs(os.path.join(a.out, "raw"), exist_ok=True)
    path = os.path.join(a.out, "raw", f"serving_{a.backend}.csv")
    f = open(path, "w", newline="")
    w = csv.writer(f)
    w.writerow(COLS)

    prompt_lens = [128, 512] if a.smoke else [128, 256, 512, 1024, 2048, 4096]
    out_lens = [16, 64] if a.smoke else [16, 64, 128, 256, 512]
    concs = [1, 2] if a.smoke else [1, 2, 4, 8, 16]
    reps = 2 if a.smoke else 5

    make = Prompts(tok)
    sweep_requests(be, w, "warmup", 256, 16, 1, 3, make)
    for L in prompt_lens:
        sweep_requests(be, w, "prompt_len", L, 16, 1, reps, make)
        f.flush()
        print(f"[{a.backend}] prompt {L} done", flush=True)
    for O in out_lens:
        sweep_requests(be, w, "output_len", 256, O, 1, max(3, reps - 2), make)
        f.flush()
        print(f"[{a.backend}] output {O} done", flush=True)
    for C in concs:
        sweep_requests(be, w, "concurrency", 256, 128, C, max(8, 4 * C), make)
        f.flush()
        print(f"[{a.backend}] concurrency {C} done", flush=True)
    f.close()
    print("wrote", path)


if __name__ == "__main__":
    main()
