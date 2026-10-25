"""Document-level train / validation / test split of the South Asia chat dataset, plus federated shards.

    python scripts/prepare_south_asia_split.py --out-dir data/south_asia --nodes 4

Many questions are generated from the same source document (doc_id). Splitting by example would put
near-duplicate questions in both training and evaluation, so the split is by *document*: every
question of a document goes to exactly one of train / val / test. The training documents are then
dealt round-robin (IID) to the nodes, so no document is split across nodes either.

Outputs: train.jsonl, val.jsonl, test.jsonl, node_0.jsonl ... node_{N-1}.jsonl, split_stats.json
"""
import argparse
import collections
import json
import os
import random


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="Bhaskera/dataset/South_Asia.jsonl")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--nodes", type=int, default=4)
    p.add_argument("--val-frac", type=float, default=0.05, help="fraction of documents for validation")
    p.add_argument("--test-frac", type=float, default=0.05, help="fraction of documents for test")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    by_doc = collections.defaultdict(list)
    with open(args.data, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            by_doc[row["doc_id"]].append(line if line.endswith("\n") else line + "\n")

    docs = sorted(by_doc)
    random.Random(args.seed).shuffle(docs)
    n_val = max(1, round(len(docs) * args.val_frac))
    n_test = max(1, round(len(docs) * args.test_frac))
    split_docs = {"val": docs[:n_val], "test": docs[n_val:n_val + n_test], "train": docs[n_val + n_test:]}

    os.makedirs(args.out_dir, exist_ok=True)
    stats = {"seed": args.seed, "documents": {k: len(v) for k, v in split_docs.items()}, "examples": {}}
    for name, ds in split_docs.items():
        lines = [ln for d in ds for ln in by_doc[d]]
        with open(os.path.join(args.out_dir, f"{name}.jsonl"), "w", encoding="utf-8") as f:
            f.writelines(lines)
        stats["examples"][name] = len(lines)

    # shards: deal training documents to nodes, balancing example counts (largest document first)
    shards = [[] for _ in range(args.nodes)]
    sizes = [0] * args.nodes
    for d in sorted(split_docs["train"], key=lambda d: -len(by_doc[d])):
        k = sizes.index(min(sizes))
        shards[k] += by_doc[d]
        sizes[k] += len(by_doc[d])
    for k, lines in enumerate(shards):
        random.Random(args.seed + k).shuffle(lines)
        with open(os.path.join(args.out_dir, f"node_{k}.jsonl"), "w", encoding="utf-8") as f:
            f.writelines(lines)
    stats["node_examples"] = sizes

    # sanity: no document appears in two splits
    seen = {}
    for name, ds in split_docs.items():
        for d in ds:
            assert d not in seen, f"document {d} in both {seen[d]} and {name}"
            seen[d] = name
    with open(os.path.join(args.out_dir, "split_stats.json"), "w") as f:
        json.dump(stats, f, indent=1)
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
