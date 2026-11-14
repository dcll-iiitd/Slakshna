"""Embed every snippet of a prepared dataset once, for the global baseline,
the routing simulation and `rag_engine.py ingest --vectors`.

    .venv-rag/bin/python scripts/m9/embed_cache.py --data datasets/textbooks --threads 120

Writes <data>/embeddings/<model>/ with vectors.npy (float32, unit rows, in
corpus.jsonl order), ids.json and meta.json; the format rag/vector_cache.py
reads. Work is saved in blocks under parts/, so an interrupted run resumes
where it stopped. Vectors are computed exactly as the sidecar's Embedder does
(same model, normalised), so the ingest spot check passes.
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

from m9_common import DEFAULT_MODEL, ROOT, log, model_slug, read_jsonl, write_json

sys.path.insert(0, ROOT)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="folder written by prepare_textbooks.py")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--threads", type=int, default=0, help="torch threads (0 = torch's default, all cores)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--block", type=int, default=4096, help="snippets per saved block")
    args = p.parse_args()

    if args.threads:
        os.environ["SLAKSHNA_TORCH_THREADS"] = str(args.threads)
    from rag.embedder import Embedder

    corpus = read_jsonl(os.path.join(args.data, "corpus.jsonl"))
    out = os.path.join(args.data, "embeddings", model_slug(args.model))
    parts = os.path.join(out, "parts")
    os.makedirs(parts, exist_ok=True)
    ids = [row["id"] for row in corpus]
    ids_path = os.path.join(parts, "ids.json")
    if os.path.exists(ids_path):
        with open(ids_path, encoding="utf-8") as f:
            if json.load(f) != ids:
                sys.exit(f"{parts} holds blocks for a different corpus; delete {out} and run again")
    else:
        write_json(ids_path, ids)

    log(f"loading {args.model}")
    embedder = Embedder(args.model)
    blocks = range(0, len(corpus), args.block)
    started, done = time.time(), 0
    for number, start in enumerate(blocks):
        path = os.path.join(parts, f"block_{number:05d}.npy")
        if os.path.exists(path):
            continue
        texts = [row["content"] for row in corpus[start:start + args.block]]
        vectors = embedder.model.encode(texts, batch_size=args.batch_size, normalize_embeddings=True,
                                        show_progress_bar=False)
        np.save(path + ".tmp.npy", np.asarray(vectors, dtype=np.float32))
        os.replace(path + ".tmp.npy", path)
        done += len(texts)
        rate = done / (time.time() - started)
        left = len(corpus) - start - len(texts)
        log(f"block {number + 1}/{len(blocks)}: {rate:.1f} snippets/s, about {left / rate / 60:.0f} min left")

    vectors = np.concatenate([np.load(os.path.join(parts, f"block_{n:05d}.npy")) for n in range(len(blocks))])
    assert vectors.shape == (len(corpus), embedder.dim)
    np.save(os.path.join(out, "vectors.npy"), vectors)
    write_json(os.path.join(out, "ids.json"), ids)
    write_json(os.path.join(out, "meta.json"), {"embedding_model": args.model, "dim": embedder.dim,
                                                "count": len(ids), "batch_size": args.batch_size})
    for path in glob.glob(os.path.join(parts, "*")):
        os.remove(path)
    os.rmdir(parts)
    log(f"wrote {vectors.shape[0]} vectors of {vectors.shape[1]} values to {out}")


if __name__ == "__main__":
    main()
