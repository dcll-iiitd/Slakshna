"""On-disk vector store: a FAISS index plus the chunk text it points at.

Layout under `directory`:
    index.faiss    HNSW graph index over unit vectors, scored by inner product
                   (= cosine similarity); approximate nearest-neighbour search
    chunks.jsonl   one JSON object per chunk, in the same order as the index;
                   each names the k-means cluster it belongs to
    centroids.npy  float32 [k, dim], one unit-length centroid per cluster
    meta.json      embedding model, dimension, the documents already ingested,
                   and the Merkle commitment (root and version) over all of it

Adding documents is two steps: `add_document` chunks, embeds and appends;
`commit` re-clusters everything, rebuilds the Merkle tree and saves. `add_path`
and `add_documents` do both.
"""

import hashlib
import json
import os
import sys
import threading
import time

import faiss
import numpy as np

from .chunking import chunk_text
from .clustering import choose_k, cluster
from .loaders import list_documents, load_document, load_snippets
from .merkle import Commitment

# HNSW settings. M: graph neighbours per vector. efConstruction: candidates
# examined while building the graph. efSearch: candidates examined per query;
# higher is closer to exact search and slower. All three are far above what a
# corpus of a few hundred chunks needs, so results match exact search there.
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 128


def new_index(dim):
    """An empty HNSW index that ranks by inner product."""
    index = faiss.IndexHNSWFlat(dim, HNSW_M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    index.hnsw.efSearch = HNSW_EF_SEARCH
    return index


class VectorStore:
    def __init__(self, directory, embedder, n_clusters=0, vector_cache=None):
        """`n_clusters` is the number of k-means clusters; 0 picks it from the corpus size.

        `vector_cache` (a `VectorCache`) supplies pre-computed vectors for
        snippets ingested with `add_snippets`, instead of embedding them again.
        """
        choose_k(0, n_clusters)  # rejects an out-of-range setting early
        self.directory = directory
        self.embedder = embedder
        self.n_clusters = n_clusters
        self.vector_cache = vector_cache
        self.index = new_index(embedder.dim)
        self.vectors = np.zeros((0, embedder.dim), dtype=np.float32)
        self.chunks = []
        self.documents = {}  # source name -> sha256 of the document text
        self.centroids = np.zeros((0, embedder.dim), dtype=np.float32)
        self.commitment = Commitment([], self.centroids)
        self.version = 0
        # Re-entrant, so add_documents can hold it across its adds and the
        # commit: other threads never see chunks that are not yet committed.
        # Callers hold it too when several reads must come from one version.
        self.lock = threading.RLock()
        os.makedirs(directory, exist_ok=True)
        if os.path.exists(self._path("meta.json")):
            self._load()

    def _path(self, name):
        return os.path.join(self.directory, name)

    def _load(self):
        with open(self._path("meta.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta["embedding_model"] != self.embedder.model_name or meta["dim"] != self.embedder.dim:
            raise ValueError(
                f"index in {self.directory} was built with {meta['embedding_model']} "
                f"(dim {meta['dim']}), not {self.embedder.model_name} (dim {self.embedder.dim}); "
                "delete the directory and ingest again to switch models"
            )
        self.documents = meta["documents"]
        self.index = faiss.read_index(self._path("index.faiss"))
        with open(self._path("chunks.jsonl"), "r", encoding="utf-8") as f:
            self.chunks = [json.loads(line) for line in f if line.strip()]
        if self.index.ntotal != len(self.chunks):
            raise ValueError(f"index in {self.directory} is inconsistent: "
                             f"{self.index.ntotal} vectors, {len(self.chunks)} chunks")
        if self.index.ntotal:
            self.vectors = self.index.reconstruct_n(0, self.index.ntotal)
        if isinstance(self.index, faiss.IndexHNSWFlat):
            self.index.hnsw.efSearch = HNSW_EF_SEARCH
        else:
            # Built before the switch to HNSW (an exact IndexFlatIP). Rebuild it
            # from the same vectors; chunks, clusters and the Merkle root are
            # unaffected, because the root does not cover the index file.
            print(f"Index in {self.directory} uses exact search; rebuilding it as HNSW.", file=sys.stderr)
            self.index = new_index(self.embedder.dim)
            if len(self.vectors):
                self.index.add(np.ascontiguousarray(self.vectors, dtype=np.float32))
            faiss.write_index(self.index, self._path("index.faiss"))

        committed = meta.get("commitment")
        if committed is None:
            print(f"Index in {self.directory} predates clustering; clustering it now.", file=sys.stderr)
            self.commit()
            return

        try:
            centroids = np.load(self._path("centroids.npy"), allow_pickle=False)
            if centroids.ndim != 2 or centroids.shape[1] != self.embedder.dim:
                raise ValueError(f"centroids.npy has shape {centroids.shape}")
            commitment = Commitment(self.chunks, centroids)
        except (OSError, KeyError, ValueError) as e:
            raise ValueError(f"index files in {self.directory} were changed on disk ({e}); "
                             "delete the directory and ingest again") from e
        if commitment.root_hex != committed["root"]:
            raise ValueError(f"index files in {self.directory} were changed on disk: their Merkle root "
                             f"{commitment.root_hex} differs from {committed['root']} in meta.json; "
                             "delete the directory and ingest again")
        self.centroids = centroids
        self.commitment = commitment
        self.version = committed["version"]

        if committed.get("n_clusters_setting", 0) != self.n_clusters:
            print(f"n_clusters changed from {committed.get('n_clusters_setting', 0)} to "
                  f"{self.n_clusters}; re-clustering.", file=sys.stderr)
            self.commit()

    def _save(self):
        faiss.write_index(self.index, self._path("index.faiss"))
        with open(self._path("chunks.jsonl"), "w", encoding="utf-8") as f:
            for chunk in self.chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        np.save(self._path("centroids.npy"), self.centroids, allow_pickle=False)
        meta = {
            "embedding_model": self.embedder.model_name,
            "dim": self.embedder.dim,
            "documents": self.documents,
            "commitment": {
                "root": self.commitment.root_hex,
                "version": self.version,
                "num_clusters": self.commitment.num_clusters,
                "n_clusters_setting": self.n_clusters,
            },
        }
        # meta.json goes last: if saving is cut short, the root check on the
        # next load notices the mismatch instead of serving a mixed index.
        with open(self._path("meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    @property
    def num_chunks(self):
        return len(self.chunks)

    def commitment_summary(self):
        """The fields of this node's root that Rust signs (see `SignedRoot` in src/rag/merkle.rs)."""
        with self.lock:
            return {
                "root": self.commitment.root_hex,
                "version": self.version,
                "num_chunks": self.commitment.num_chunks,
                "num_clusters": self.commitment.num_clusters,
                "dim": self.embedder.dim,
                "embedding_model": self.embedder.model_name,
            }

    def add_document(self, source, text):
        """Chunk, embed and append one document. Returns the number of chunks added.

        Nothing is saved, and the new chunks are not in the Merkle commitment
        until `commit()` runs. A document already stored under the same
        `source` is skipped if its text is unchanged and rejected if it differs.
        """
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        with self.lock:
            if self._already_ingested(source, digest):
                return 0
            pieces = chunk_text(text)
            ids = [f"{source}#{i}" for i in range(len(pieces))]
            return self._append(source, digest, ids, pieces, self.embedder.embed(pieces) if pieces else None)

    def add_snippets(self, source, snippets):
        """Append chunks that were cut beforehand, one per `(chunk_id, text)` pair.

        Used for corpora that ship already chunked, such as MedRAG's JSONL
        snippets, so every system compared on them searches the same pieces.
        Vectors come from `self.vector_cache` when one is set. Same skip and
        reject rules as `add_document`; nothing is committed.
        """
        snippets = list(snippets)
        digest = hashlib.sha256(json.dumps(snippets, ensure_ascii=False).encode("utf-8")).hexdigest()
        with self.lock:
            if self._already_ingested(source, digest):
                return 0
            ids = [chunk_id for chunk_id, _ in snippets]
            texts = [text for _, text in snippets]
            vectors = None
            if snippets:
                if self.vector_cache is not None:
                    vectors = self.vector_cache.vectors_for(ids, texts, self.embedder)
                else:
                    vectors = self.embedder.embed(texts)
            return self._append(source, digest, ids, texts, vectors)

    def _already_ingested(self, source, digest):
        known = self.documents.get(source)
        if known is not None and known != digest:
            raise ValueError(f"document '{source}' is already ingested with different content; "
                             "delete the index directory and ingest again to replace it")
        return known == digest

    def _append(self, source, digest, chunk_ids, texts, vectors):
        if not texts:
            return 0
        self.index.add(vectors)
        self.vectors = np.vstack([self.vectors, vectors])
        for chunk_id, text in zip(chunk_ids, texts):
            self.chunks.append({"chunk_id": chunk_id, "source": source, "text": text})
        self.documents[source] = digest
        return len(texts)

    def commit(self):
        """Re-cluster every chunk, rebuild the Merkle tree, bump the version and save.

        The version is the current Unix time in milliseconds, and always more
        than the previous one, so peers can tell a newer root from an older one.
        """
        with self.lock:
            k = choose_k(len(self.chunks), self.n_clusters)
            if k == 0:
                centroids = np.zeros((0, self.embedder.dim), dtype=np.float32)
                assignments = []
            else:
                centroids, assignments = cluster(self.vectors, k)
            for chunk, c in zip(self.chunks, assignments):
                chunk["cluster"] = int(c)
            self.centroids = centroids
            self.commitment = Commitment(self.chunks, centroids)
            self.version = max(self.version + 1, int(time.time() * 1000)) if self.chunks else 0
            self._save()
            return self.commitment_summary()

    def add_documents(self, documents):
        """Add `(source, text)` pairs, then commit once if anything was added."""
        return self._add_all(lambda: self.add_document(source, text) for source, text in documents)

    def add_path(self, path):
        """Ingest a file, or every supported file under a directory.

        `.jsonl` files hold pre-cut snippets (see `load_snippets`); the rest
        are chunked here.
        """
        files = list_documents(path)
        if not files:
            raise ValueError(f"no PDF, .txt, .md or .jsonl files found at {path}")

        def adder(f):
            source = os.path.basename(f)
            if f.lower().endswith(".jsonl"):
                return lambda: self.add_snippets(source, load_snippets(f))
            return lambda: self.add_document(source, load_document(f))

        return self._add_all(adder(f) for f in files)

    def _add_all(self, adds):
        """Run each add, then commit once if anything was added."""
        with self.lock:
            added = 0
            try:
                for add in adds:
                    added += add()
            finally:
                # Commit what was added even if a later document failed, so the
                # in-memory index and the files on disk stay the same.
                if added:
                    self.commit()
            return added

    def _require_committed(self):
        if self.commitment.num_chunks != len(self.chunks):
            raise RuntimeError("chunks were added without commit(); they have no proofs yet")

    def _query_vector(self, query_vec):
        q = np.asarray(query_vec, dtype=np.float32).reshape(-1)
        if q.shape[0] != self.embedder.dim:
            raise ValueError(f"query vector has {q.shape[0]} values, expected {self.embedder.dim}")
        return q

    def _result(self, index, score):
        """A chunk as returned to callers, without the internal `cluster` field."""
        chunk = {key: value for key, value in self.chunks[index].items() if key != "cluster"}
        chunk["score"] = float(score)
        return chunk

    def _proven_result(self, index, score):
        result = self._result(index, score)
        result["cluster_id"] = self.commitment.chunk_cluster[index]
        result["chunk_proof"] = self.commitment.chunk_proof(index)
        return result

    def search(self, query, top_k, with_proofs=False):
        """Return up to `top_k` chunks most similar to `query`, best first.

        With `with_proofs`, returns `{"chunks", "clusters"}` instead: each chunk
        also carries `cluster_id` and `chunk_proof`, and `clusters` holds the
        entry of every cluster those chunks belong to.
        """
        with self.lock:
            if with_proofs:
                self._require_committed()
            results = []
            if self.chunks and top_k > 0:
                vector = self.embedder.embed([query])
                scores, ids = self.index.search(vector, min(top_k, len(self.chunks)))
                for score, idx in zip(scores[0], ids[0]):
                    if idx < 0:
                        continue
                    make = self._proven_result if with_proofs else self._result
                    results.append(make(int(idx), score))
            if not with_proofs:
                return results
            cluster_ids = sorted({chunk["cluster_id"] for chunk in results})
            return {"chunks": results, "clusters": [self.commitment.cluster_entry(c) for c in cluster_ids]}

    def all_clusters(self):
        """Every cluster entry of the current commitment, ids 0..k-1, in order.

        This is the node's centroid directory. Read
        it under `self.lock` together with `self.version` so both match.
        """
        with self.lock:
            self._require_committed()
            return [self.commitment.cluster_entry(c) for c in range(self.commitment.num_clusters)]

    def nearest_centroids(self, query_vec, n):
        """The `n` cluster entries whose centroids are closest to `query_vec`, best first.

        Each entry also has `best_chunk_score`, the best score of any chunk in
        that cluster.
        """
        with self.lock:
            self._require_committed()
            if n <= 0 or self.commitment.num_clusters == 0:
                return []
            q = self._query_vector(query_vec)
            order = np.argsort(-(self.centroids @ q), kind="stable")[:n]
            entries = []
            for c in order:
                entry = self.commitment.cluster_entry(int(c))
                members = self.commitment.members[c]
                entry["best_chunk_score"] = float(np.max(self.vectors[members] @ q))
                entries.append(entry)
            return entries

    def search_clusters(self, query_vec, cluster_ids, per_cluster):
        """The best `per_cluster` chunks inside each listed cluster, with proofs.

        Returns `{"clusters", "chunks"}`: the entry of each requested cluster,
        and the chunks of all of them sorted best first.
        """
        with self.lock:
            self._require_committed()
            q = self._query_vector(query_vec)
            wanted = list(dict.fromkeys(int(c) for c in cluster_ids))
            for c in wanted:
                if not 0 <= c < self.commitment.num_clusters:
                    raise ValueError(f"no cluster {c}; this index has {self.commitment.num_clusters}")
            chunks = []
            for c in wanted:
                members = np.asarray(self.commitment.members[c])
                scores = self.vectors[members] @ q
                for j in np.argsort(-scores, kind="stable")[:max(per_cluster, 0)]:
                    chunks.append(self._proven_result(int(members[j]), scores[j]))
            chunks.sort(key=lambda chunk: -chunk["score"])
            return {"clusters": [self.commitment.cluster_entry(c) for c in wanted], "chunks": chunks}
