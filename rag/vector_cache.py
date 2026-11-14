"""Pre-computed snippet vectors, so a large pre-chunked corpus is embedded once.

A cache directory (written by scripts/m9/embed_cache.py) holds:
    vectors.npy   float32 [n, dim], unit-length rows, one per snippet
    ids.json      the n snippet ids, in the same order
    meta.json     {"embedding_model", "dim", "count"}

Ingesting with a cache gives the same vectors the embedder would, minus the
hours of CPU time. Each batch is spot-checked: a few of its snippets are
embedded again and must match the cache, so a cache built from other text or
another model is refused instead of silently used.
"""

import json
import os
import random

import numpy as np

# Largest allowed difference between a cached and a freshly computed value.
# Batching changes the padding inside the model, which moves values by ~1e-6.
TOLERANCE = 1e-4
CHECKS_PER_BATCH = 50


class VectorCache:
    def __init__(self, directory, seed=0):
        with open(os.path.join(directory, "meta.json"), "r", encoding="utf-8") as f:
            self.meta = json.load(f)
        with open(os.path.join(directory, "ids.json"), "r", encoding="utf-8") as f:
            ids = json.load(f)
        self.vectors = np.load(os.path.join(directory, "vectors.npy"), mmap_mode="r", allow_pickle=False)
        if self.vectors.shape != (len(ids), self.meta["dim"]):
            raise ValueError(f"vector cache in {directory} is inconsistent: vectors {self.vectors.shape}, "
                             f"{len(ids)} ids, dim {self.meta['dim']}")
        self.row = {chunk_id: i for i, chunk_id in enumerate(ids)}
        self.directory = directory
        self.random = random.Random(seed)
        self.checked = 0
        self.max_difference = 0.0

    def vectors_for(self, chunk_ids, texts, embedder):
        """Vectors for `chunk_ids`, from the cache; ids it lacks are embedded now."""
        if self.meta["embedding_model"] != embedder.model_name or self.meta["dim"] != embedder.dim:
            raise ValueError(f"vector cache in {self.directory} was built with {self.meta['embedding_model']} "
                             f"(dim {self.meta['dim']}), not {embedder.model_name} (dim {embedder.dim})")
        vectors = np.zeros((len(chunk_ids), embedder.dim), dtype=np.float32)
        missing = [i for i, chunk_id in enumerate(chunk_ids) if chunk_id not in self.row]
        found = [i for i, chunk_id in enumerate(chunk_ids) if chunk_id in self.row]
        if found:
            vectors[found] = self.vectors[[self.row[chunk_ids[i]] for i in found]]
        if missing:
            vectors[missing] = embedder.embed([texts[i] for i in missing])

        sample = self.random.sample(found, min(CHECKS_PER_BATCH, len(found)))
        if sample:
            fresh = embedder.embed([texts[i] for i in sample])
            difference = float(np.max(np.abs(fresh - vectors[sample])))
            self.checked += len(sample)
            self.max_difference = max(self.max_difference, difference)
            if difference > TOLERANCE:
                raise ValueError(f"vector cache in {self.directory} does not match this text and model: "
                                 f"re-embedded snippets differ by up to {difference:.2e} (allowed {TOLERANCE:.0e})")
        return vectors
