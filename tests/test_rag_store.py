"""Unit tests for RAG chunking and the on-disk vector store.

A fake embedder keeps these tests free of model downloads: it counts a few
keywords, so texts sharing a keyword get similar vectors.
"""

import json
import os
import sys
import tempfile
import unittest

import faiss
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rag.chunking import chunk_text
from rag.clustering import choose_k
from rag.merkle import ZERO_HASH, cluster_hash, fold_proof, leaf_hash
from rag.store import HNSW_EF_SEARCH, VectorStore
from rag.vector_cache import VectorCache

KEYWORDS = ["moon", "mars", "rhino", "river"]


class FakeEmbedder:
    model_name = "fake-keyword-embedder"
    dim = len(KEYWORDS) + 1

    def embed(self, texts):
        rows = []
        for text in texts:
            lowered = text.lower()
            row = [float(lowered.count(k)) for k in KEYWORDS] + [0.1]
            rows.append(row)
        vectors = np.asarray(rows, dtype=np.float32)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


class ChunkingTests(unittest.TestCase):
    def test_windows_overlap_and_cover_all_words(self):
        words = [f"w{i}" for i in range(450)]
        chunks = chunk_text(" ".join(words), chunk_words=200, overlap_words=40)

        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0].split()[-40:], chunks[1].split()[:40])
        self.assertEqual(chunks[-1].split()[-1], "w449")

    def test_short_and_empty_text(self):
        self.assertEqual(chunk_text("just a few words"), ["just a few words"])
        self.assertEqual(chunk_text("   "), [])

    def test_rejects_overlap_not_smaller_than_chunk(self):
        with self.assertRaises(ValueError):
            chunk_text("a b c", chunk_words=5, overlap_words=5)


class VectorStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = VectorStore(self.tmp.name, FakeEmbedder())
        self.store.add_document("moon.txt", "The moon lander touched down near the moon south pole.")
        self.store.add_document("park.txt", "The rhino lives beside the river in the park.")
        self.store.commit()

    def test_search_ranks_matching_document_first(self):
        results = self.store.search("where does the rhino live", top_k=2)

        self.assertEqual(results[0]["source"], "park.txt")
        self.assertEqual(results[0]["chunk_id"], "park.txt#0")
        self.assertGreater(results[0]["score"], results[1]["score"])

    def test_top_k_is_capped_at_index_size(self):
        self.assertEqual(len(self.store.search("moon", top_k=10)), 2)
        self.assertEqual(self.store.search("moon", top_k=0), [])

    def test_index_survives_reload(self):
        reloaded = VectorStore(self.tmp.name, FakeEmbedder())

        self.assertEqual(reloaded.num_chunks, 2)
        self.assertEqual(reloaded.search("moon", top_k=1)[0]["source"], "moon.txt")

    def test_reingesting_same_document_is_a_no_op(self):
        added = self.store.add_document("moon.txt", "The moon lander touched down near the moon south pole.")

        self.assertEqual(added, 0)
        self.assertEqual(self.store.num_chunks, 2)

    def test_changed_document_is_rejected(self):
        with self.assertRaises(ValueError):
            self.store.add_document("moon.txt", "Different text about mars.")

    def test_reload_with_other_model_is_rejected(self):
        other = FakeEmbedder()
        other.model_name = "some-other-model"
        with self.assertRaises(ValueError):
            VectorStore(self.tmp.name, other)


# Twelve one-chunk documents, three per keyword, with a little of a second
# keyword mixed in so the clusters are not trivially identical points.
CLUSTER_DOCS = {
    f"{topic}{i}.txt": f"{(topic + ' ') * (3 + i)} {other} notes number {i}"
    for topic, other in [("moon", "mars"), ("mars", "moon"), ("rhino", "river"), ("river", "rhino")]
    for i in range(3)
}


def verify_chunk(store, chunk, entries):
    """Fold a returned chunk's proofs up to the store's root, as a peer would."""
    entry = entries[chunk["cluster_id"]]
    subtree = fold_proof(leaf_hash(chunk["chunk_id"], chunk["text"]), chunk["chunk_proof"])
    if subtree.hex() != entry["subtree_root"]:
        return False
    centroid = np.frombuffer(bytes.fromhex(entry["centroid"]), dtype="<f4")
    start = cluster_hash(entry["cluster_id"], centroid, subtree)
    return fold_proof(start, entry["cluster_proof"]).hex() == store.commitment_summary()["root"]


class ClusteredStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.corpus = os.path.join(self.tmp.name, "corpus")
        self.index_dir = os.path.join(self.tmp.name, "rag")
        os.makedirs(self.corpus)
        for name, text in CLUSTER_DOCS.items():
            self.write_doc(name, text)
        self.store = VectorStore(self.index_dir, FakeEmbedder())
        self.store.add_path(self.corpus)

    def write_doc(self, name, text):
        with open(os.path.join(self.corpus, name), "w", encoding="utf-8") as f:
            f.write(text)

    def query(self, text):
        return FakeEmbedder().embed([text])[0]

    def test_choose_k(self):
        self.assertEqual(choose_k(0), 0)
        self.assertEqual(choose_k(1), 1)
        self.assertEqual(choose_k(161), 13)
        self.assertEqual(choose_k(100_000), 64)
        self.assertEqual(choose_k(100, requested=8), 8)
        self.assertEqual(choose_k(3, requested=10), 3)
        for bad in (-1, 65):
            with self.assertRaises(ValueError):
                choose_k(10, requested=bad)

    def test_clusters_are_numbered_without_gaps_and_cover_every_chunk(self):
        summary = self.store.commitment_summary()
        k = summary["num_clusters"]
        self.assertEqual(k, 3)  # round(sqrt(12)); no cluster ended up empty here
        self.assertEqual({chunk["cluster"] for chunk in self.store.chunks}, set(range(k)))
        sizes = [self.store.commitment.cluster_entry(c)["size"] for c in range(k)]
        self.assertEqual(sum(sizes), summary["num_chunks"])
        self.assertEqual(summary["num_chunks"], len(CLUSTER_DOCS))
        self.assertEqual(self.store.centroids.shape, (k, FakeEmbedder.dim))
        np.testing.assert_allclose(np.linalg.norm(self.store.centroids, axis=1), 1.0, rtol=1e-5)

    def test_every_chunk_proof_folds_to_root(self):
        k = self.store.commitment.num_clusters
        result = self.store.search_clusters(self.query("moon"), range(k), per_cluster=100)
        entries = {entry["cluster_id"]: entry for entry in result["clusters"]}
        self.assertEqual(len(result["chunks"]), len(CLUSTER_DOCS))
        for chunk in result["chunks"]:
            self.assertTrue(verify_chunk(self.store, chunk, entries), chunk["chunk_id"])

        tampered = dict(result["chunks"][0], text=result["chunks"][0]["text"] + "!")
        self.assertFalse(verify_chunk(self.store, tampered, entries))

    def test_flat_search_with_proofs(self):
        plain = self.store.search("rhino", top_k=4)
        proven = self.store.search("rhino", top_k=4, with_proofs=True)
        self.assertEqual([c["chunk_id"] for c in plain], [c["chunk_id"] for c in proven["chunks"]])
        self.assertNotIn("cluster", plain[0])
        entries = {entry["cluster_id"]: entry for entry in proven["clusters"]}
        self.assertEqual(set(entries), {c["cluster_id"] for c in proven["chunks"]})
        for chunk in proven["chunks"]:
            self.assertTrue(verify_chunk(self.store, chunk, entries))

    def test_nearest_centroids_are_ranked_and_report_best_chunk(self):
        q = self.query("rhino")
        entries = self.store.nearest_centroids(q, 2)
        self.assertEqual(len(entries), 2)
        sims = [float(np.frombuffer(bytes.fromhex(e["centroid"]), dtype="<f4") @ q) for e in entries]
        self.assertGreaterEqual(sims[0], sims[1])
        best = self.store.search_clusters(q, [entries[0]["cluster_id"]], per_cluster=1)["chunks"][0]
        self.assertAlmostEqual(entries[0]["best_chunk_score"], best["score"], places=5)
        self.assertTrue(best["chunk_id"].startswith(("rhino", "river")))
        self.assertEqual(len(self.store.nearest_centroids(q, 50)), self.store.commitment.num_clusters)

    def test_search_clusters_returns_only_requested_clusters(self):
        result = self.store.search_clusters(self.query("mars"), [2, 0, 2], per_cluster=2)
        self.assertEqual([entry["cluster_id"] for entry in result["clusters"]], [2, 0])
        self.assertTrue(all(chunk["cluster_id"] in (0, 2) for chunk in result["chunks"]))
        self.assertLessEqual(len(result["chunks"]), 4)
        scores = [chunk["score"] for chunk in result["chunks"]]
        self.assertEqual(scores, sorted(scores, reverse=True))
        with self.assertRaises(ValueError):
            self.store.search_clusters(self.query("mars"), [99], per_cluster=1)

    def test_root_survives_reload(self):
        before = self.store.commitment_summary()
        reloaded = VectorStore(self.index_dir, FakeEmbedder())
        self.assertEqual(reloaded.commitment_summary(), before)

    def test_hand_edited_chunks_file_fails_to_load(self):
        path = os.path.join(self.index_dir, "chunks.jsonl")
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
        chunk = json.loads(lines[3])
        chunk["text"] = chunk["text"].replace("notes", "nodes")
        lines[3] = json.dumps(chunk) + "\n"
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines)
        with self.assertRaisesRegex(ValueError, "changed on disk"):
            VectorStore(self.index_dir, FakeEmbedder())

    def test_edited_centroids_fail_to_load(self):
        path = os.path.join(self.index_dir, "centroids.npy")
        centroids = np.load(path)
        centroids[0, 0] += 0.001
        np.save(path, centroids)
        with self.assertRaisesRegex(ValueError, "changed on disk"):
            VectorStore(self.index_dir, FakeEmbedder())

    def test_new_document_changes_root_and_raises_version(self):
        before = self.store.commitment_summary()
        self.write_doc("moon_extra.txt", "moon moon base camp")
        self.assertEqual(self.store.add_path(self.corpus), 1)
        after = self.store.commitment_summary()
        self.assertNotEqual(after["root"], before["root"])
        self.assertGreater(after["version"], before["version"])
        self.assertEqual(VectorStore(self.index_dir, FakeEmbedder()).commitment_summary(), after)

    def test_reingesting_same_documents_changes_nothing(self):
        before = self.store.commitment_summary()
        with open(os.path.join(self.index_dir, "meta.json"), encoding="utf-8") as f:
            meta_before = f.read()
        self.assertEqual(self.store.add_path(self.corpus), 0)
        self.assertEqual(self.store.commitment_summary(), before)
        with open(os.path.join(self.index_dir, "meta.json"), encoding="utf-8") as f:
            self.assertEqual(f.read(), meta_before)

    def test_index_without_commitment_is_migrated_on_load(self):
        # Turn the saved index into the pre-clustering format.
        meta_path = os.path.join(self.index_dir, "meta.json")
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        del meta["commitment"]
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f)
        chunks_path = os.path.join(self.index_dir, "chunks.jsonl")
        with open(chunks_path, encoding="utf-8") as f:
            chunks = [json.loads(line) for line in f]
        with open(chunks_path, "w", encoding="utf-8") as f:
            for chunk in chunks:
                del chunk["cluster"]
                f.write(json.dumps(chunk) + "\n")
        os.remove(os.path.join(self.index_dir, "centroids.npy"))

        migrated = VectorStore(self.index_dir, FakeEmbedder())
        summary = migrated.commitment_summary()
        self.assertEqual(summary["num_chunks"], len(CLUSTER_DOCS))
        self.assertEqual(summary["num_clusters"], 3)
        self.assertGreater(summary["version"], 0)
        self.assertEqual(VectorStore(self.index_dir, FakeEmbedder()).commitment_summary(), summary)

    def test_changed_n_clusters_reclusters_on_load(self):
        before = self.store.commitment_summary()
        two = VectorStore(self.index_dir, FakeEmbedder(), n_clusters=2)
        self.assertEqual(two.commitment_summary()["num_clusters"], 2)
        self.assertGreater(two.commitment_summary()["version"], before["version"])

    def test_uncommitted_chunks_have_no_proofs(self):
        self.store.add_document("loose.txt", "mars rover")
        with self.assertRaises(RuntimeError):
            self.store.search("mars", top_k=3, with_proofs=True)
        self.assertEqual(len(self.store.search("mars", top_k=3)), 3)

    def test_all_clusters_is_the_complete_directory(self):
        entries = self.store.all_clusters()
        summary = self.store.commitment_summary()
        self.assertEqual(len(entries), summary["num_clusters"])
        self.assertEqual([e["cluster_id"] for e in entries], list(range(summary["num_clusters"])))
        self.assertEqual(sum(e["size"] for e in entries), summary["num_chunks"])
        for entry in entries:
            self.assertNotIn("best_chunk_score", entry)
            centroid = np.frombuffer(bytes.fromhex(entry["centroid"]), dtype="<f4")
            start = cluster_hash(entry["cluster_id"], centroid, bytes.fromhex(entry["subtree_root"]))
            self.assertEqual(fold_proof(start, entry["cluster_proof"]).hex(), summary["root"])

    def test_empty_store_has_zero_root(self):
        empty = VectorStore(os.path.join(self.tmp.name, "empty"), FakeEmbedder())
        summary = empty.commitment_summary()
        self.assertEqual(summary["root"], ZERO_HASH.hex())
        self.assertEqual((summary["version"], summary["num_clusters"], summary["num_chunks"]), (0, 0, 0))
        self.assertEqual(empty.nearest_centroids(self.query("moon"), 3), [])
        self.assertEqual(empty.search("moon", 3, with_proofs=True), {"chunks": [], "clusters": []})
        self.assertEqual(empty.all_clusters(), [])


class HnswIndexTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = VectorStore(self.tmp.name, FakeEmbedder())
        self.store.add_documents(CLUSTER_DOCS.items())

    def test_new_index_is_hnsw_with_inner_product(self):
        self.assertIsInstance(self.store.index, faiss.IndexHNSWFlat)
        self.assertEqual(self.store.index.metric_type, faiss.METRIC_INNER_PRODUCT)
        reloaded = VectorStore(self.tmp.name, FakeEmbedder())
        self.assertIsInstance(reloaded.index, faiss.IndexHNSWFlat)
        self.assertEqual(reloaded.index.hnsw.efSearch, HNSW_EF_SEARCH)

    def test_search_matches_exact_search(self):
        for query in ["moon", "mars rover", "rhino river", "river", "notes"]:
            q = FakeEmbedder().embed([query])[0]
            exact = sorted((self.store.vectors @ q).tolist(), reverse=True)[:4]
            found = [c["score"] for c in self.store.search(query, top_k=4)]
            np.testing.assert_allclose(found, exact, rtol=1e-5, err_msg=query)

    def test_exact_index_from_before_hnsw_is_rebuilt_on_load(self):
        before = self.store.commitment_summary()
        expected = [c["chunk_id"] for c in self.store.search("rhino", top_k=3)]
        # Write the same vectors in the old exact format.
        flat = faiss.IndexFlatIP(FakeEmbedder.dim)
        flat.add(self.store.vectors)
        faiss.write_index(flat, os.path.join(self.tmp.name, "index.faiss"))

        rebuilt = VectorStore(self.tmp.name, FakeEmbedder())
        self.assertIsInstance(rebuilt.index, faiss.IndexHNSWFlat)
        np.testing.assert_array_equal(rebuilt.vectors, self.store.vectors)
        self.assertEqual(rebuilt.commitment_summary(), before)
        self.assertEqual([c["chunk_id"] for c in rebuilt.search("rhino", top_k=3)], expected)
        # The rebuilt index was saved, so the next load needs no rebuild.
        self.assertIsInstance(faiss.read_index(os.path.join(self.tmp.name, "index.faiss")), faiss.IndexHNSWFlat)


def write_snippets(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


SNIPPETS = [
    {"id": "Book_0", "title": "Book", "content": "The moon orbits the earth.", "contents": "Book. The moon..."},
    {"id": "Book_1", "title": "Book", "content": "A rhino by the river."},
    {"id": "Book__s2_0", "orig_id": "Book_2", "title": "Book", "content": "Mars is red."},
]


class SnippetIngestTests(unittest.TestCase):
    """Pre-cut MedRAG snippets (scripts/m9): one line is one chunk, not re-chunked."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.corpus = os.path.join(self.tmp.name, "corpus")
        os.makedirs(self.corpus)
        write_snippets(os.path.join(self.corpus, "Book.jsonl"), SNIPPETS)
        with open(os.path.join(self.corpus, "notes.txt"), "w", encoding="utf-8") as f:
            f.write("river notes")
        self.index_dir = os.path.join(self.tmp.name, "index")

    def test_snippets_become_chunks_as_given(self):
        store = VectorStore(self.index_dir, FakeEmbedder())
        self.assertEqual(store.add_path(self.corpus), 4)

        by_id = {chunk["chunk_id"]: chunk for chunk in store.chunks}
        self.assertEqual(set(by_id), {"Book_0", "Book_1", "Book_2", "notes.txt#0"})
        self.assertEqual(by_id["Book_0"]["text"], "The moon orbits the earth.")  # `content`, not `contents`
        self.assertEqual(by_id["Book_2"]["source"], "Book.jsonl")
        self.assertEqual(store.search("rhino", top_k=1)[0]["chunk_id"], "Book_1")
        self.assertEqual(store.commitment.num_chunks, 4)

    def test_snippets_survive_reload_and_reingest_is_a_no_op(self):
        VectorStore(self.index_dir, FakeEmbedder()).add_path(self.corpus)
        reloaded = VectorStore(self.index_dir, FakeEmbedder())

        self.assertEqual(reloaded.num_chunks, 4)
        self.assertEqual(reloaded.add_path(self.corpus), 0)

    def test_changed_snippet_file_is_rejected(self):
        store = VectorStore(self.index_dir, FakeEmbedder())
        store.add_path(self.corpus)
        write_snippets(os.path.join(self.corpus, "Book.jsonl"), SNIPPETS[:2])
        with self.assertRaises(ValueError):
            store.add_path(self.corpus)

    def test_snippet_without_content_is_rejected(self):
        write_snippets(os.path.join(self.corpus, "Bad.jsonl"), [{"id": "Bad_0", "text": "no content field"}])
        with self.assertRaises(ValueError):
            VectorStore(self.index_dir, FakeEmbedder()).add_path(self.corpus)


class VectorCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.corpus = os.path.join(self.tmp.name, "corpus")
        os.makedirs(self.corpus)
        write_snippets(os.path.join(self.corpus, "Book.jsonl"), SNIPPETS)
        self.cache = os.path.join(self.tmp.name, "cache")

    def write_cache(self, ids, texts, model_name=FakeEmbedder.model_name, shift=0.0):
        os.makedirs(self.cache, exist_ok=True)
        vectors = FakeEmbedder().embed(texts) + np.float32(shift)
        np.save(os.path.join(self.cache, "vectors.npy"), vectors.astype(np.float32))
        with open(os.path.join(self.cache, "ids.json"), "w") as f:
            json.dump(ids, f)
        with open(os.path.join(self.cache, "meta.json"), "w") as f:
            json.dump({"embedding_model": model_name, "dim": FakeEmbedder.dim, "count": len(ids)}, f)

    def test_cached_vectors_are_used_and_missing_ones_embedded(self):
        # Book_1 is not in the cache, so it must be embedded on the spot.
        self.write_cache(["Book_2", "Book_0"], ["Mars is red.", "The moon orbits the earth."])
        cache = VectorCache(self.cache)
        store = VectorStore(os.path.join(self.tmp.name, "index"), FakeEmbedder(), vector_cache=cache)
        store.add_path(self.corpus)

        expected = FakeEmbedder().embed([row["content"] for row in SNIPPETS])
        np.testing.assert_allclose(store.vectors, expected, atol=1e-6)
        self.assertEqual(cache.checked, 2)

    def test_cache_from_other_text_is_refused(self):
        self.write_cache(["Book_0", "Book_1", "Book_2"], ["mars", "moon", "rhino"])
        store = VectorStore(os.path.join(self.tmp.name, "index"), FakeEmbedder(), vector_cache=VectorCache(self.cache))
        with self.assertRaises(ValueError):
            store.add_path(self.corpus)
        self.assertEqual(store.num_chunks, 0)

    def test_cache_from_other_model_is_refused(self):
        texts = [row["content"] for row in SNIPPETS]
        self.write_cache(["Book_0", "Book_1", "Book_2"], texts, model_name="other-model")
        store = VectorStore(os.path.join(self.tmp.name, "index"), FakeEmbedder(), vector_cache=VectorCache(self.cache))
        with self.assertRaises(ValueError):
            store.add_path(self.corpus)


if __name__ == "__main__":
    unittest.main()
