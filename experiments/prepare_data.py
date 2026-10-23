"""
Build the labelled, non-IID per-node datasets for the federated inference / System One experiments.

For one (dataset, n_nodes, alpha, seed) it writes, under <out>/<tag>/:

    node1/train.jsonl ... nodeN/train.jsonl   chat-format SFT data, label-skewed by a Dirichlet(alpha) split
    test_global.jsonl                          held-out, class-balanced test set (no node ever trains on it)
    test_node<k>.jsonl                         held-out test drawn with node k's own class mix (local distribution)
    val.jsonl                                  small held-out validation set (for calibration / probe checks)
    meta.json                                  labels, prompt template, class histogram per node

Each test row: {"id", "text", "label", "label_text", "prompt", "options"}; `prompt` is exactly the user turn
the nodes were trained on, so training and evaluation use one format.

Low alpha = strong label skew (each node mostly sees a few classes); alpha=10 is nearly IID.

    python experiments/prepare_data.py --dataset ag_news --nodes 4 --alpha 0.1 --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter, defaultdict

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", os.path.join(REPO, "hf_cache"))
import sys  # noqa: E402
sys.path.insert(0, REPO)

SYSTEM = "You are a helpful AI assistant"


def load(dataset: str):
    """-> train rows, test rows, label names. Each row: {text, label (int)}."""
    from datasets import load_dataset

    if dataset == "ag_news":
        d = load_dataset("fancyzhx/ag_news")
        names = ["World", "Sports", "Business", "Sci/Tech"]
        tr = [{"text": r["text"], "label": r["label"]} for r in d["train"]]
        te = [{"text": r["text"], "label": r["label"]} for r in d["test"]]
        instr = "Classify the news article into exactly one of these topics: {opts}."
    elif dataset == "banking77":
        d = load_dataset("mteb/banking77")
        by = {}
        for r in d["train"]:
            by[r["label"]] = r["label_text"]
        names = [by[i] for i in range(len(by))]
        tr = [{"text": r["text"], "label": r["label"]} for r in d["train"]]
        te = [{"text": r["text"], "label": r["label"]} for r in d["test"]]
        instr = "Identify the intent of the banking customer message. Choose exactly one of: {opts}."
    elif dataset == "dolly":
        # Generative task: nodes are skewed by task category; answers are scored against the human reference.
        d = load_dataset("databricks/databricks-dolly-15k")["train"]
        names = sorted(set(d["category"]))
        rows = []
        for r in d:
            prompt = r["instruction"].strip()
            if r["context"].strip():
                prompt += "\n\nContext: " + r["context"].strip()
            if len(prompt) > 3000 or len(r["response"]) > 1500:
                continue  # keep prompts and answers within the training sequence length
            rows.append({"text": prompt, "label": names.index(r["category"]), "reference": r["response"].strip()})
        rng = np.random.default_rng(12345)  # fixed held-out split, independent of --seed
        idx = rng.permutation(len(rows))
        cut = 2000
        te = [rows[i] for i in idx[:cut]]
        tr = [rows[i] for i in idx[cut:]]
        instr = None
    else:
        raise SystemExit(f"unknown dataset {dataset}")
    return tr, te, names, instr


def make_prompt(instr, names: list[str], text: str) -> str:
    if instr is None:  # generative task: the prompt is the instruction itself
        return text
    return f"{instr.format(opts=', '.join(names))}\n\nText: {text.strip()}\n\nAnswer with only the label."


def dirichlet_split(labels: np.ndarray, n_nodes: int, alpha: float, rng: np.random.Generator, min_per_node: int = 50):
    """Per-class Dirichlet allocation (label skew). Retries until every node has min_per_node rows."""
    classes = sorted(set(labels.tolist()))
    for _ in range(200):
        parts: list[list[int]] = [[] for _ in range(n_nodes)]
        for c in classes:
            idx = np.where(labels == c)[0]
            rng.shuffle(idx)
            p = rng.dirichlet([alpha] * n_nodes)
            cuts = (np.cumsum(p) * len(idx)).astype(int)[:-1]
            for k, chunk in enumerate(np.split(idx, cuts)):
                parts[k].extend(chunk.tolist())
        if min(len(p) for p in parts) >= min_per_node:
            return parts
    raise SystemExit("could not build a split with enough rows per node; raise --train-total or alpha")


# System One ("decision") format: the question every decision of a dataset asks.
DECISION_QUESTIONS = {
    "ag_news": "Which topic is this news article about?",
    "banking77": "What does the customer want?",
}
MAX_SHOWN_OPTIONS = 26  # training examples show the gold option plus random distractors, letters A-Z


def decision_example(text, gold, names, question, rng):
    """One training example in the decision format: the System One prompt (federated_inference.system_one.build_prompt)
    with the options shuffled, answered by the gold option's letter. Many-label datasets show the gold plus random
    distractors, so the model learns to pick content, not a position or a letter."""
    from federated_inference.system_one import build_prompt
    others = [i for i in range(len(names)) if i != gold]
    shown = [gold] + list(rng.choice(others, size=min(len(others), MAX_SHOWN_OPTIONS - 1), replace=False))
    rng.shuffle(shown)
    letters = [chr(ord("A") + j) for j in range(len(shown))]
    prompt = build_prompt(f"Text: {text.strip()}", question, [names[i] for i in shown], letters)
    return prompt, letters[shown.index(gold)]


def row(i, r, names, instr, split, question=None):
    return {
        "state": f"Text: {r['text'].strip()}",
        "question": question,
        "id": f"{split}-{i}",
        "text": r["text"],
        "label": int(r["label"]),
        "label_text": names[r["label"]],
        "prompt": make_prompt(instr, names, r["text"]),
        "options": names,
        "reference": r.get("reference", names[r["label"]]),
    }


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ag_news", choices=["ag_news", "banking77", "dolly"])
    ap.add_argument("--nodes", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-total", type=int, default=12000, help="training rows shared out across all nodes")
    ap.add_argument("--test-size", type=int, default=1000)
    ap.add_argument("--val-size", type=int, default=400)
    ap.add_argument("--out", default=os.path.join(REPO, "experiments", "data"))
    ap.add_argument("--format", default="generate", choices=["generate", "system_one"],
                    help="generate: answer = label text; system_one: System One decision prompt, answer = option letter")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    random.seed(a.seed)
    train, test, names, instr = load(a.dataset)
    n_cls = len(names)

    # --- held-out sets come first so no node can ever train on them ---------------------------------
    tr_idx = rng.permutation(len(train))
    val_idx, tr_idx = tr_idx[: a.val_size], tr_idx[a.val_size:]
    tr_idx = tr_idx[: a.train_total]
    train_rows = [train[i] for i in tr_idx]
    val_rows = [train[i] for i in val_idx]

    # class-balanced global test set
    by_cls = defaultdict(list)
    for i, r in enumerate(test):
        by_cls[r["label"]].append(i)
    per = max(1, a.test_size // n_cls)
    g_idx = []
    for c in range(n_cls):
        ids = by_cls[c][:]
        rng.shuffle(ids)
        g_idx.extend(ids[:per])
    rng.shuffle(g_idx)
    test_global = [test[i] for i in g_idx]

    # --- label-skewed partition ---------------------------------------------------------------------
    labels = np.array([r["label"] for r in train_rows])
    parts = dirichlet_split(labels, a.nodes, a.alpha, rng)

    decision = a.format == "system_one"
    question = DECISION_QUESTIONS.get(a.dataset) if decision else None
    if decision and question is None:
        raise SystemExit(f"--format system_one has no decision question for {a.dataset}")
    orng = np.random.default_rng(a.seed + 1000)  # option shuffles; separate so the splits match the generate format
    tag = f"{a.dataset}_n{a.nodes}_a{a.alpha:g}_s{a.seed}" + ("_s1" if decision else "")
    root = os.path.join(a.out, tag)
    hist = {}
    for k, idxs in enumerate(parts, start=1):
        rows = [train_rows[i] for i in idxs]
        rng.shuffle(rows)
        recs = []
        for r in rows:
            if decision:
                user, answer = decision_example(r["text"], r["label"], names, question, orng)
            else:
                user, answer = make_prompt(instr, names, r["text"]), r.get("reference", names[r["label"]])
            recs.append({
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": answer},
                ]
            })
        write_jsonl(os.path.join(root, f"node{k}", "train.jsonl"), recs)
        cnt = Counter(r["label"] for r in rows)
        hist[f"node{k}"] = {names[c]: cnt.get(c, 0) for c in range(n_cls)}

    # --- per-node local test sets: node k's own class mix, drawn from the held-out test pool --------
    pool = defaultdict(list)
    for i in range(len(test)):
        pool[test[i]["label"]].append(i)
    local_sizes = {}
    for k in range(1, a.nodes + 1):
        h = np.array([hist[f"node{k}"][names[c]] for c in range(n_cls)], dtype=float)
        p = h / h.sum()
        draw = rng.choice(n_cls, size=a.test_size, p=p)
        rows = []
        for j, c in enumerate(draw):
            rows.append(row(f"{k}-{j}", test[int(rng.choice(pool[int(c)]))], names, instr, f"local{k}", question))
        write_jsonl(os.path.join(root, f"test_node{k}.jsonl"), rows)
        local_sizes[f"node{k}"] = len(rows)

    write_jsonl(os.path.join(root, "test_global.jsonl"), [row(i, r, names, instr, "global", question) for i, r in enumerate(test_global)])
    write_jsonl(os.path.join(root, "val.jsonl"), [row(i, r, names, instr, "val", question) for i, r in enumerate(val_rows)])
    meta = {
        "tag": tag, "dataset": a.dataset, "n_nodes": a.nodes, "alpha": a.alpha, "seed": a.seed,
        "labels": names, "instruction": instr, "format": a.format, "decision_question": question, "task": "generate" if instr is None else "classify", "system": SYSTEM,
        "train_per_node": {f"node{k}": len(p) for k, p in enumerate(parts, start=1)},
        "class_hist": hist, "n_test_global": len(test_global), "n_val": len(val_rows),
    }
    with open(os.path.join(root, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"wrote {root}")
    for k in range(1, a.nodes + 1):
        h = hist[f"node{k}"]
        top = sorted(h.items(), key=lambda x: -x[1])[:3]
        print(f"  node{k}: {meta['train_per_node'][f'node{k}']:>6} rows  top classes: {top}")


if __name__ == "__main__":
    main()
