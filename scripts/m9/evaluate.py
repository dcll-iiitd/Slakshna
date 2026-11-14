"""Turn M9's raw contexts and predictions into summary tables (CSV and Markdown).

    .venv-rag/bin/python scripts/m9/evaluate.py --data datasets/textbooks \
        --raw results/m9/raw --global results/m9/raw/global.jsonl --out results/m9/summary

Reads every contexts file under --raw (any *.jsonl that isn't a prediction
file or the answer cache) and every prediction file (*.pred.<model>.jsonl),
and writes to --out:

    retrieval.csv  per contexts file: recall@k against the exact global top k
                   (run 1; matched by text hash, L13), share of perfect
                   questions, overlap with Flower's contexts, contexts per shard,
                   bytes, peers contacted, verification labels
    latency.csv    per contexts file, over all runs: median / p95 / mean wall
                   time, and Slakshna's select / fetch / rescore medians
    accuracy.csv   per prediction file: accuracy over all questions and over
                   answered ones (Flower's definition), with 95% bootstrap intervals
    paired.csv     same scheme and model, pairs of systems on the same questions:
                   accuracy difference, its 95% bootstrap interval, McNemar's exact p
    e0a.csv        check E0(a): Flower's inline answers vs answer.py on the same contexts
    attacks.csv    contexts files from runs with an attacking node: share of
                   queries where the attacker was flagged or refused, FORGED
                   contexts, contexts from the attacker, recall
    env.txt        commit, Flower set-up, versions
    *.png          plots, if matplotlib is installed
"""

import argparse
import csv
import glob
import math
import os
import platform
import subprocess
import sys
from collections import Counter, defaultdict

import numpy as np

from m9_common import ROOT, load_questions, log, read_jsonl, text_hash

RNG = np.random.default_rng(0)
BOOTSTRAP = 2000


def bootstrap_interval(values):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return (float("nan"), float("nan"))
    means = values[RNG.integers(0, len(values), (BOOTSTRAP, len(values)))].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def mcnemar_exact(b, c):
    """Two-sided exact McNemar test on the discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def percentile(values, q):
    return float(np.percentile(values, q)) if len(values) else float("nan")


def shard_maps(data):
    """Per scheme: snippet id -> shard and text hash -> shard, from the partition files."""
    maps = {}
    for scheme_dir in glob.glob(os.path.join(data, "partitions", "*")):
        by_id, by_hash = {}, {}
        for node_dir in glob.glob(os.path.join(scheme_dir, "node*")):
            shard = int(os.path.basename(node_dir)[4:])
            for path in glob.glob(os.path.join(node_dir, "*.jsonl")):
                for row in read_jsonl(path):
                    by_id[row.get("orig_id") or row["id"]] = shard
                    by_hash[text_hash(row["content"])] = shard
        maps[os.path.basename(scheme_dir)] = (by_id, by_hash)
    return maps


def shard_of(context, scheme, maps):
    if isinstance(context.get("node"), int):
        return context["node"]
    by_id, by_hash = maps.get(scheme, ({}, {}))
    if context.get("id") in by_id:
        return by_id[context["id"]]
    return by_hash.get(text_hash(context["text"]))


def find_files(raw):
    contexts, predictions = [], []
    for path in sorted(glob.glob(os.path.join(raw, "**", "*.jsonl"), recursive=True)):
        if os.sep + "answer_cache" + os.sep in path:
            continue
        (predictions if ".pred." in os.path.basename(path) else contexts).append(path)
    return contexts, predictions


def label(path, raw):
    return os.path.relpath(path, raw)[:-len(".jsonl")]


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in row.items()})
    log(f"wrote {path} ({len(rows)} rows)")


def retrieval_rows(files, raw, global_top, maps, k):
    flower = {}  # scheme -> qid -> set of hashes
    loaded = {}
    for path in files:
        rows = [r for r in read_jsonl(path) if r["run"] == 1]
        loaded[path] = rows
        if rows and rows[0]["system"] == "flower" and not rows[0]["extra"].get("adversaries"):
            flower.setdefault(rows[0]["scheme"], {}).update(
                {r["qid"]: {text_hash(c["text"]) for c in r["contexts"]} for r in rows})
    out = []
    for path, rows in loaded.items():
        if not rows:
            continue
        system, scheme = rows[0]["system"], rows[0]["scheme"]
        recalls, overlaps, per_shard = [], [], Counter()
        labels, bytes_, contacted = Counter(), [], []
        for r in rows:
            hashes = [text_hash(c["text"]) for c in r["contexts"]]
            if r["qid"] in global_top:
                recalls.append(len(global_top[r["qid"]] & set(hashes)) / k)
            if system != "flower" and r["qid"] in flower.get(scheme, {}):
                overlaps.append(len(flower[scheme][r["qid"]] & set(hashes)) / k)
            for c in r["contexts"]:
                per_shard[shard_of(c, scheme, maps)] += 1
                labels[c.get("verification", "-")] += 1
            if "bytes" in r["extra"]:
                bytes_.append(r["extra"]["bytes"])
            if "peers_contacted" in r["extra"]:
                contacted.append(r["extra"]["peers_contacted"])
        low, high = bootstrap_interval(recalls)
        out.append({
            "file": label(path, raw), "system": system, "scheme": scheme, "questions": len(rows),
            f"recall@{k}": float(np.mean(recalls)) if recalls else float("nan"),
            "recall_ci_low": low, "recall_ci_high": high,
            "perfect_share": float(np.mean([x == 1.0 for x in recalls])) if recalls else float("nan"),
            "overlap_with_flower": float(np.mean(overlaps)) if overlaps else float("nan"),
            "contexts_per_shard": " ".join(f"{s}:{n / len(rows):.2f}" for s, n in sorted(per_shard.items(), key=str)),
            "bytes_median": percentile(bytes_, 50), "bytes_p95": percentile(bytes_, 95),
            "peers_contacted_mean": float(np.mean(contacted)) if contacted else float("nan"),
            "verification": " ".join(f"{v}:{n}" for v, n in sorted(labels.items())),
        })
    return out


def latency_rows(files, raw):
    out = []
    for path in files:
        rows = read_jsonl(path)
        if not rows:
            continue
        wall = [r["retrieval_s"] for r in rows]
        row = {"file": label(path, raw), "system": rows[0]["system"], "scheme": rows[0]["scheme"],
               "runs": len({r["run"] for r in rows}), "rows": len(rows),
               "median_s": percentile(wall, 50), "p95_s": percentile(wall, 95), "mean_s": float(np.mean(wall))}
        timings = [r["extra"]["timings_ms"] for r in rows if r["extra"].get("timings_ms")]
        for part in ("select", "fetch", "rescore", "local_retrieve"):
            row[f"{part}_ms_median"] = percentile([t[part] for t in timings if part in t], 50)
        out.append(row)
    return out


def accuracy_rows(files, raw, gold):
    out, correct_by = [], {}
    for path in files:
        rows = [r for r in read_jsonl(path) if r["qid"] in gold]
        if not rows:
            continue
        right = [r["predicted"] == gold[r["qid"]] for r in rows]
        answered = [ok for ok, r in zip(right, rows) if r["predicted"] is not None]
        low, high = bootstrap_interval(right)
        key = (rows[0]["scheme"], rows[0]["model"], rows[0]["system"], label(path, raw))
        correct_by[key] = {r["qid"]: ok for r, ok in zip(rows, right)}
        out.append({
            "file": label(path, raw), "system": rows[0]["system"], "scheme": rows[0]["scheme"],
            "model": rows[0]["model"], "questions": len(rows), "answered": len(answered),
            "accuracy_all": float(np.mean(right)), "ci_low": low, "ci_high": high,
            "accuracy_answered": float(np.mean(answered)) if answered else float("nan"),
            "truncated": sum(bool(r.get("truncated")) for r in rows),
            **{f"accuracy_{d}": float(np.mean([ok for ok, r in zip(right, rows) if r["dataset"] == d]))
               for d in sorted({r["dataset"] for r in rows})},
        })
    return out, correct_by


def paired_rows(correct_by):
    out = []
    keys = sorted(correct_by, key=str)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            # Same model; same scheme, or one side is scheme-free (global index, closed book).
            if a[1] != b[1] or (a[0] != b[0] and a[0] is not None and b[0] is not None):
                continue
            common = sorted(set(correct_by[a]) & set(correct_by[b]))
            if not common:
                continue
            x = np.array([correct_by[a][q] for q in common], dtype=float)
            y = np.array([correct_by[b][q] for q in common], dtype=float)
            low, high = bootstrap_interval(x - y)
            only_a, only_b = int(np.sum((x == 1) & (y == 0))), int(np.sum((x == 0) & (y == 1)))
            out.append({"model": a[1], "a": a[3], "b": b[3], "questions": len(common),
                        "accuracy_a": float(x.mean()), "accuracy_b": float(y.mean()),
                        "difference": float((x - y).mean()), "ci_low": low, "ci_high": high,
                        "only_a_right": only_a, "only_b_right": only_b, "mcnemar_p": mcnemar_exact(only_a, only_b)})
    return out


def attack_rows(files, raw, global_top, maps, k):
    out = []
    for path in files:
        rows = read_jsonl(path)
        if not rows or not rows[0]["extra"].get("adversaries"):
            continue
        attackers = {a["endpoint_id"]: a for a in rows[0]["extra"]["adversaries"]}
        flagged = refused = asked = forged = from_attacker = 0
        flags, recalls = Counter(), []
        attacker_nodes = {a["node"] for a in attackers.values()}
        for r in rows:
            for p in r["extra"]["peers"]:
                if p["endpoint_id"] in attackers:
                    asked += 1
                    flags.update(p["flags"] or [])
                    if p["status"] in ("refused", "no_root"):
                        refused += 1
                    if p["flags"] or p["status"] in ("refused", "no_root"):
                        flagged += 1
            forged += sum("FORGED" in c["text"] for c in r["contexts"])
            # Node i holds shard i, so a context from the attacker's shard came from the attacker.
            from_attacker += sum(1 for c in r["contexts"] if shard_of(c, r["scheme"], maps) in attacker_nodes)
            if r["qid"] in global_top:
                recalls.append(len(global_top[r["qid"]] & {text_hash(c["text"]) for c in r["contexts"]}) / k)
        out.append({"file": label(path, raw), "system": rows[0]["system"], "scheme": rows[0]["scheme"],
                    "attack": ",".join(sorted({a["kind"] for a in attackers.values()})), "queries": len(rows),
                    "attacker_flagged_share": flagged / asked if asked else float("nan"),
                    "attacker_refused_share": refused / asked if asked else float("nan"),
                    "flags": " ".join(f"{f}:{n}" for f, n in sorted(flags.items())),
                    "forged_contexts": forged, "contexts_from_attacker_per_query": from_attacker / len(rows),
                    f"recall@{k}": float(np.mean(recalls)) if recalls else float("nan")})
    return out


def e0a_rows(contexts, predictions, raw, gold):
    """Check E0(a): answers Flower's server gave inline vs answer.py on the saved contexts."""
    out = []
    for path in contexts:
        inline = {r["qid"]: r["extra"]["inline_prediction"] for r in read_jsonl(path)
                  if r["run"] == 1 and "inline_prediction" in r["extra"]}
        if not inline:
            continue
        stem = path[:-len(".jsonl")] + ".pred."
        for pred_path in [p for p in predictions if p.startswith(stem)]:
            rows = [r for r in read_jsonl(pred_path) if r["qid"] in inline]
            agree = sum(r["predicted"] == inline[r["qid"]] for r in rows)
            out.append({"contexts": label(path, raw), "predictions": label(pred_path, raw),
                        "backend": rows[0]["backend"] if rows else "", "questions": len(rows), "agree": agree,
                        "inline_accuracy": float(np.mean([inline[r["qid"]] == gold[r["qid"]] for r in rows])) if rows else float("nan"),
                        "answer_py_accuracy": float(np.mean([r["predicted"] == gold[r["qid"]] for r in rows])) if rows else float("nan")})
    return out


def write_env(path, data):
    def run(command):
        try:
            return subprocess.run(command, capture_output=True, text=True, cwd=ROOT, timeout=10).stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            return "unavailable"
    lines = [f"slakshna_commit={run(['git', 'rev-parse', 'HEAD'])}",
             f"slakshna_dirty={'yes' if run(['git', 'status', '--porcelain', '--untracked-files=no']) else 'no'}",
             f"python={platform.python_version()}", f"ollama={run(['ollama', '--version'])}", f"data={data}"]
    setup = os.path.join(ROOT, "flower-fedrag-m9", "M9_SETUP.txt")
    if os.path.exists(setup):
        with open(setup, encoding="utf-8") as f:
            lines += [line.strip() for line in f if line.strip()]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def plots(out, retrieval, accuracy, latency, k):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log("matplotlib not installed: CSVs only (uv pip install --python .venv-rag/bin/python matplotlib)")
        return
    for name, rows, value, low, high, title in [
        ("recall", retrieval, f"recall@{k}", "recall_ci_low", "recall_ci_high", f"recall@{k} vs exact global top {k}"),
        ("accuracy", accuracy, "accuracy_all", "ci_low", "ci_high", "answer accuracy (all questions)"),
    ]:
        rows = [r for r in rows if not math.isnan(r[value])]
        if not rows:
            continue
        fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(rows)), 4))
        names = [r["file"] for r in rows]
        values = [r[value] for r in rows]
        errors = [[v - r[low] for v, r in zip(values, rows)], [r[high] - v for v, r in zip(values, rows)]]
        ax.bar(range(len(rows)), values, yerr=errors, capsize=3, color="#4a78b5")
        ax.set_xticks(range(len(rows)), names, rotation=60, ha="right", fontsize=7)
        ax.set_ylim(0, 1)
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"{name}.png"), dpi=150)
        plt.close(fig)
    if latency:
        fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(latency)), 4))
        ax.bar(range(len(latency)), [r["median_s"] for r in latency], color="#4a78b5", label="median")
        ax.scatter(range(len(latency)), [r["p95_s"] for r in latency], color="#c0504d", zorder=3, label="p95")
        ax.set_xticks(range(len(latency)), [r["file"] for r in latency], rotation=60, ha="right", fontsize=7)
        ax.set_ylabel("seconds per question (retrieval)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out, "latency.png"), dpi=150)
        plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--raw", required=True, help="folder holding contexts and prediction files (searched recursively)")
    p.add_argument("--global", dest="global_path", required=True, help="global_baseline.py output")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    questions = load_questions(os.path.join(args.data, "questions.jsonl"))
    gold = {q["qid"]: q["answer"] for q in questions}
    global_top = {r["qid"]: {text_hash(c["text"]) for c in r["contexts"][:args.k]}
                  for r in read_jsonl(args.global_path)}
    maps = shard_maps(args.data)
    contexts, predictions = find_files(args.raw)
    log(f"{len(contexts)} contexts files, {len(predictions)} prediction files")

    retrieval = retrieval_rows(contexts, args.raw, global_top, maps, args.k)
    latency = latency_rows(contexts, args.raw)
    accuracy, correct_by = accuracy_rows(predictions, args.raw, gold)
    write_csv(os.path.join(args.out, "retrieval.csv"), retrieval)
    write_csv(os.path.join(args.out, "latency.csv"), latency)
    write_csv(os.path.join(args.out, "accuracy.csv"), accuracy)
    write_csv(os.path.join(args.out, "paired.csv"), paired_rows(correct_by))
    write_csv(os.path.join(args.out, "attacks.csv"), attack_rows(contexts, args.raw, global_top, maps, args.k))
    write_csv(os.path.join(args.out, "e0a.csv"), e0a_rows(contexts, predictions, args.raw, gold))
    write_env(os.path.join(args.out, "env.txt"), args.data)
    plots(args.out, retrieval, accuracy, latency, args.k)


if __name__ == "__main__":
    sys.exit(main())
