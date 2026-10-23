"""
Score every trained adapter on every test set and write ONE ROW PER QUESTION.

Everything else (adapter x test-set accuracy, per-method accuracy, oracle, query-aware routing, generation quality)
is derived from this file by aggregate.py, so the model runs once per (adapter, question).

It loads the base model once and hot-swaps the real adapter files the nodes published
(experiments/runs/<run>/data-nodeK/adapters/<sha256>/), decodes greedily, and formats prompts with the same
function as the serving sidecar (inference_engine.format_prompt), so offline and API results are comparable.

Adapters scored:
  base                              the model with no adapter
  <node>:<kind>:r<round>            adapters of the main run (kind = local | aggregated)
  <label>/<node>:<kind>:r<round>    final-round adapters of baseline runs given with --extra-runs
  merged_avg, merged_ties           the main run's final local adapters merged into one (--merge)

Classification tasks (AG News): the answer is parsed against the label names -> pred / correct / invalid.
Generation tasks (Dolly): the full answer is kept in `raw`; aggregate.py scores it against the reference.

Output: experiments/results/<tag>/raw/predictions.csv
  tag, adapter_id, adapter_node, kind, round, sha, test_set, qid, label, label_text, pred, raw, correct, invalid

    Bhaskera-serve/.venv/bin/python experiments/eval_offline.py --tag ag_news_n4_a0.1_s0 --gpu 0 --merge \\
        --extra-runs ag_news_n4_a0.1_s0_fedavg:fedavg,ag_news_n4_a0.1_s0_isolated:local_only,ag_news_n1_a0.1_s0_central:centralised
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault("HF_HOME", os.path.join(REPO, "hf_cache"))

SYSTEM = "You are a helpful AI assistant"  # the system turn the nodes were trained with


def run_nodes(run: str) -> int:
    rj = os.path.join(REPO, "experiments", "runs", run, "run.json")
    tag = json.load(open(rj))["data_tag"] if os.path.isfile(rj) else run
    return json.load(open(os.path.join(REPO, "experiments", "data", tag, "meta.json")))["n_nodes"]


def discover_adapters(run: str, n_nodes: int, prefix: str = "", kinds=None, final_only=False):
    """-> list of dicts for the adapters nodes of `run` exported."""
    found = []
    for k in range(1, n_nodes + 1):
        for d in glob.glob(os.path.join(REPO, "experiments", "runs", run, f"data-node{k}", "adapters", "*")):
            mp = os.path.join(d, "adapter_meta.json")
            if not (os.path.isfile(mp) and os.path.isfile(os.path.join(d, "adapter_model.safetensors"))):
                continue  # fetched copies of peers' adapters have no adapter_meta.json
            m = json.load(open(mp))
            kind = m.get("kind", "aggregated")
            if kinds and kind not in kinds:
                continue
            found.append({"dir": d, "sha": os.path.basename(d), "node": f"node{k}", "kind": kind,
                          "round": int(m["round"]), "id": f"{prefix}node{k}:{kind}:r{m['round']}"})
    if final_only and found:
        last = max(x["round"] for x in found)
        found = [x for x in found if x["round"] == last]
    return sorted(found, key=lambda a: (a["node"], a["kind"], a["round"]))


def parse_label(raw: str, labels: list[str]) -> int:
    """Index of the label the answer starts with (case-insensitive, longest label first), else -1."""
    s = raw.strip().strip("\"'`.*").lower()
    for i in sorted(range(len(labels)), key=lambda i: -len(labels[i])):
        if s.startswith(labels[i].lower()):
            return i
    return -1


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="data tag = main run name")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--max-new", type=int, default=None, help="default 8 (classification) / 128 (generation)")
    ap.add_argument("--n-global", type=int, default=None, help="default 1000 (classification) / 200 (generation)")
    ap.add_argument("--n-local", type=int, default=None, help="default 500 (classification) / 100 (generation)")
    ap.add_argument("--extra-runs", default="", help="run:label,... baseline runs whose final adapters are scored too")
    ap.add_argument("--merge", action="store_true", help="also score the final local adapters merged into one")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from inference_engine import format_prompt

    data_root = os.path.join(REPO, "experiments", "data", a.tag)
    meta = json.load(open(os.path.join(data_root, "meta.json")))
    labels, n_nodes = meta["labels"], meta["n_nodes"]
    gen = meta.get("task") == "generate"
    max_new = a.max_new or (128 if gen else 8)
    batch = a.batch or (32 if gen else 64)
    n_global = a.n_global or (200 if gen else 1000)
    n_local = a.n_local or (100 if gen else 500)
    out_dir = a.out or os.path.join(REPO, "experiments", "results", a.tag)
    os.makedirs(os.path.join(out_dir, "raw"), exist_ok=True)

    tests = {"global": read_jsonl(os.path.join(data_root, "test_global.jsonl"))[:n_global]}
    for k in range(1, n_nodes + 1):
        tests[f"local{k}"] = read_jsonl(os.path.join(data_root, f"test_node{k}.jsonl"))[:n_local]
    if not gen:
        tests["val"] = read_jsonl(os.path.join(data_root, "val.jsonl"))

    adapters = discover_adapters(a.tag, n_nodes)
    if not adapters:
        raise SystemExit(f"no adapters under experiments/runs/{a.tag}/data-node*/adapters")
    last_round = max(x["round"] for x in adapters)
    for x in adapters:
        x["final"] = x["round"] == last_round
    for spec in [s for s in a.extra_runs.split(",") if s]:
        run, label = spec.split(":")
        extra = discover_adapters(run, run_nodes(run), prefix=f"{label}/", final_only=True)
        for x in extra:
            x["final"] = True
        print(f"  + {len(extra)} adapters from {run} ({label})")
        adapters += extra
    print(f"{len(adapters)} adapters, main run last round {last_round}; tests: " +
          ", ".join(f"{k}={len(v)}" for k, v in tests.items()), flush=True)

    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).cuda().eval()

    peft_model = None
    names = {}
    for i, ad in enumerate(adapters):
        name = f"a{i}"
        if peft_model is None:
            peft_model = PeftModel.from_pretrained(base, ad["dir"], adapter_name=name, is_trainable=False)
            peft_model.eval()
        else:
            peft_model.load_adapter(ad["dir"], adapter_name=name, is_trainable=False)
        names[ad["id"]] = name

    jobs = [("base", None)] + [(ad["id"], ad) for ad in adapters]
    if a.merge:
        locals_final = [ad for ad in adapters if "/" not in ad["id"] and ad["kind"] == "local" and ad["final"]]
        src = [names[ad["id"]] for ad in locals_final]
        w = [1.0 / len(src)] * len(src)
        # "cat" keeps every adapter's rank, so the merged update is exactly the average of the LoRA deltas.
        peft_model.add_weighted_adapter(src, w, adapter_name="merged_avg", combination_type="cat")
        peft_model.add_weighted_adapter(src, [1.0] * len(src), adapter_name="merged_ties",
                                        combination_type="ties_svd", density=0.5)
        for m in ("merged_avg", "merged_ties"):
            names[m] = m
            jobs.append((m, {"node": "", "kind": "merged", "round": last_round, "sha": "", "final": True, "id": m}))
    print("adapters loaded", flush=True)

    @torch.no_grad()
    def run(rows):
        outs = []
        for i in range(0, len(rows), batch):
            chunk = rows[i:i + batch]
            texts = [format_prompt(tok, r["prompt"], True, SYSTEM) for r in chunk]
            enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=3072).to("cuda")
            g = peft_model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                    pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
            outs.extend(tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True))
        return outs

    path = os.path.join(out_dir, "raw", "predictions.csv")
    cols = ["tag", "adapter_id", "adapter_node", "kind", "round", "sha", "test_set", "qid", "label",
            "label_text", "pred", "raw", "correct", "invalid"]
    f = open(path, "w", newline="", encoding="utf-8")
    wr = csv.writer(f)
    wr.writerow(cols)
    for adapter_id, ad in jobs:
        t0 = time.time()
        sets = list(tests) if (ad is None or ad["final"]) else ["global"]
        line = []
        for ts in sets:
            rows = tests[ts]
            if ad is None:
                with peft_model.disable_adapter():
                    raws = run(rows)
            else:
                peft_model.set_adapter(names[adapter_id])
                raws = run(rows)
            n_ok = 0
            for r, raw in zip(rows, raws):
                if gen:
                    p, ok, inv, keep = "", "", "", raw.strip()
                else:
                    p = parse_label(raw, labels)
                    ok, inv = int(p == r["label"]), int(p < 0)
                    n_ok += ok
                    keep = raw.strip().replace("\n", " ")[:80]
                wr.writerow([a.tag, adapter_id, ad["node"] if ad else "", ad["kind"] if ad else "",
                             ad["round"] if ad else 0, ad["sha"] if ad else "", ts, r["id"], r["label"],
                             r["label_text"], p, keep, ok, inv])
            line.append(f"{ts}={n_ok / len(rows):.3f}" if not gen else f"{ts}:{len(rows)}")
        f.flush()
        print(f"{adapter_id:<36} {time.time() - t0:5.0f}s  " + " ".join(line), flush=True)
    f.close()
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
