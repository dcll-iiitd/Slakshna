"""Download MedRAG Textbooks and the MIRAGE questions, and cut the corpus into shards.

    python scripts/m9/prepare_textbooks.py --out datasets/textbooks
    # small local test: two short books, 300 snippets each
    python scripts/m9/prepare_textbooks.py --out datasets/textbooks-mini \
        --books Pathoma_Husain,First_Aid_Step1 --max-per-book 300 --nodes 3

Writes, under --out:
    raw/chunk/<Book>.jsonl, raw/benchmark.json      downloads, kept for re-runs
    corpus.jsonl                                    every snippet: {id, book, content}
    questions.jsonl                                 {qid, dataset, mirage_id, question, options, answer}
    subset_<n>.json                                 fixed random sample of qids (latency and attack runs)
    partitions/<scheme>/node<i>/*.jsonl             MedRAG-format shard files, read by both systems
    flower/<scheme>/textbooks_s<i>/chunk            symlink to the shard, the layout Flower's retriever expects
    stats.json                                      snippets and characters per node, seeds, duplicates

Schemes: `book` puts whole books on nodes (biggest first, each to the node with
the fewest snippets so far); `random` deals snippets to nodes at random with
equal counts. In `random` a book is split, so shard files are renamed
<Book>__s<i>.jsonl with ids <Book>__s<i>_<line>: Flower finds a snippet's text
by reading the line number at the end of its id. The original id is kept as
`orig_id`, which Slakshna uses as the chunk id.
"""

import argparse
import json
import os
import random
import shutil
import urllib.request
from collections import Counter

from m9_common import BOOKS, log, text_hash, write_json, write_jsonl

BOOK_URL = "https://huggingface.co/datasets/MedRAG/textbooks/resolve/main/chunk/{book}.jsonl"
MIRAGE_URL = "https://raw.githubusercontent.com/Teddy-XiongGZ/MIRAGE/main/benchmark.json"
SCHEMES = ("book", "random")


def download(url, path):
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    log(f"downloading {url}")
    tmp = path + ".part"
    with urllib.request.urlopen(url, timeout=120) as response, open(tmp, "wb") as f:
        shutil.copyfileobj(response, f, length=1 << 20)
    os.replace(tmp, path)


def load_book(path, limit):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    for n, row in enumerate(rows):
        # Flower reads snippet n of a file as line n; ids must agree with that.
        if not row["id"].endswith(f"_{n}"):
            raise ValueError(f"{path}: line {n} has id {row['id']}, expected a name ending in _{n}")
    return rows


def assign_books(counts, nodes):
    """Whole books to nodes: biggest book first, each to the node with the fewest snippets."""
    load = [0] * nodes
    owner = {}
    for book in sorted(counts, key=lambda b: (-counts[b], b)):
        node = min(range(nodes), key=lambda i: (load[i], i))
        owner[book] = node
        load[node] += counts[book]
    return owner


def write_book_scheme(books, nodes, out):
    owner = assign_books({b: len(rows) for b, rows in books.items()}, nodes)
    shards = [[] for _ in range(nodes)]
    for book, rows in books.items():
        write_jsonl(os.path.join(out, f"node{owner[book]}", f"{book}.jsonl"), rows)
        shards[owner[book]].extend(rows)
    return shards, {"owner": owner}


def write_random_scheme(books, nodes, out, seed):
    everything = [(book, row) for book, rows in books.items() for row in rows]
    order = list(range(len(everything)))
    random.Random(seed).shuffle(order)
    node_of = [0] * len(everything)
    for rank, index in enumerate(order):
        node_of[index] = rank % nodes
    shards = [[] for _ in range(nodes)]
    per_file = {}
    for index, (book, row) in enumerate(everything):  # keeps each book's original order within a shard
        node = node_of[index]
        name = f"{book}__s{node}"
        lines = per_file.setdefault((node, name), [])
        renamed = dict(row, id=f"{name}_{len(lines)}", orig_id=row["id"])
        lines.append(renamed)
        shards[node].append(renamed)
    for (node, name), lines in per_file.items():
        write_jsonl(os.path.join(out, f"node{node}", f"{name}.jsonl"), lines)
    return shards, {"seed": seed}


def link_for_flower(dataset_dir, scheme, nodes):
    base = os.path.join(dataset_dir, "flower", scheme)
    if os.path.exists(base):
        shutil.rmtree(base)  # any FAISS index here was built from the old shards
    for i in range(nodes):
        corpus = os.path.join(base, f"textbooks_s{i}")
        os.makedirs(corpus)
        # Relative, so the folder still works after rsync to another machine.
        os.symlink(os.path.join("..", "..", "..", "partitions", scheme, f"node{i}"), os.path.join(corpus, "chunk"))


def export_questions(benchmark_path, datasets, out, subset_size, seed):
    with open(benchmark_path, "r", encoding="utf-8") as f:
        benchmark = json.load(f)
    questions = []
    for dataset in datasets:
        entries = benchmark[dataset]
        # Sorted keys, numbered from 1: the order and the ids Flower's server uses.
        for n, key in enumerate(sorted(entries), start=1):
            entry = entries[key]
            questions.append({
                "qid": f"{dataset}_{n}", "dataset": dataset, "mirage_id": key,
                "question": entry["question"], "options": entry["options"], "answer": entry["answer"],
            })
    write_jsonl(os.path.join(out, "questions.jsonl"), questions)
    qids = [q["qid"] for q in questions]
    chosen = set(random.Random(seed).sample(qids, min(subset_size, len(qids))))
    write_json(os.path.join(out, f"subset_{subset_size}.json"), [q for q in qids if q in chosen])
    return questions


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="dataset folder, e.g. datasets/textbooks")
    p.add_argument("--books", default=",".join(BOOKS), help="comma-separated subset of the 18 books")
    p.add_argument("--max-per-book", type=int, default=0, help="first N snippets of each book (0 = all)")
    p.add_argument("--nodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--datasets", default="medqa,mmlu", help="MIRAGE question sets")
    p.add_argument("--subset-size", type=int, default=200)
    args = p.parse_args()

    books_wanted = args.books.split(",")
    unknown = set(books_wanted) - set(BOOKS)
    if unknown:
        p.error(f"unknown books: {sorted(unknown)}")
    raw = os.path.join(args.out, "raw")
    for book in books_wanted:
        download(BOOK_URL.format(book=book), os.path.join(raw, "chunk", f"{book}.jsonl"))
    download(MIRAGE_URL, os.path.join(raw, "benchmark.json"))

    books = {b: load_book(os.path.join(raw, "chunk", f"{b}.jsonl"), args.max_per_book) for b in books_wanted}
    corpus = [{"id": row["id"], "book": book, "content": row["content"]} for book, rows in books.items() for row in rows]
    ids = Counter(row["id"] for row in corpus)
    if len(ids) != len(corpus):
        raise ValueError(f"duplicate snippet ids: {[i for i, c in ids.items() if c > 1][:5]}")
    write_jsonl(os.path.join(args.out, "corpus.jsonl"), corpus)
    hashes = Counter(text_hash(row["content"]) for row in corpus)
    log(f"{len(corpus)} snippets from {len(books)} books; "
        f"{sum(c - 1 for c in hashes.values() if c > 1)} have the same text as an earlier one")

    stats = {"books": {b: len(rows) for b, rows in books.items()}, "snippets": len(corpus),
             "duplicate_texts": sum(c - 1 for c in hashes.values() if c > 1),
             "max_per_book": args.max_per_book, "nodes": args.nodes, "schemes": {}}
    for scheme in SCHEMES:
        out = os.path.join(args.out, "partitions", scheme)
        if os.path.exists(out):
            shutil.rmtree(out)
        if scheme == "book":
            shards, info = write_book_scheme(books, args.nodes, out)
        else:
            shards, info = write_random_scheme(books, args.nodes, out, args.seed)
        link_for_flower(args.out, scheme, args.nodes)
        info["nodes"] = [{"snippets": len(s), "characters": sum(len(r["content"]) for r in s)} for s in shards]
        stats["schemes"][scheme] = info
        log(f"{scheme}: snippets per node {[n['snippets'] for n in info['nodes']]}")
        if any(n["snippets"] == 0 for n in info["nodes"]):
            log(f"WARNING: {scheme} left a node empty; use more books or fewer nodes")

    questions = export_questions(os.path.join(raw, "benchmark.json"), args.datasets.split(","),
                                 args.out, args.subset_size, args.seed)
    stats["questions"] = dict(Counter(q["dataset"] for q in questions))
    stats["subset_size"] = args.subset_size
    stats["seed"] = args.seed
    write_json(os.path.join(args.out, "stats.json"), stats)
    log(f"{len(questions)} questions {stats['questions']}; wrote {args.out}")


if __name__ == "__main__":
    main()
