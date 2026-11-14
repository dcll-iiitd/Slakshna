"""Unit tests for the RAG sidecar's HTTP routes (rag/server.py).

Uses FastAPI's in-process test client and the keyword-counting fake embedder
from test_rag_store.py, so no model or network is needed.
"""

import os
import sys
import tempfile
import unittest
import warnings

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

warnings.filterwarnings("ignore", message=".*httpx.*")
from fastapi.testclient import TestClient  # noqa: E402

from rag.merkle import cluster_hash, fold_proof  # noqa: E402
from rag.server import create_app  # noqa: E402
from rag.store import VectorStore  # noqa: E402
from test_rag_store import CLUSTER_DOCS, FakeEmbedder, verify_chunk  # noqa: E402


class NoGenerator:
    name = "none"
    model = None

    def generate(self, query, chunks):
        return ""


class SidecarRouteTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = VectorStore(self.tmp.name, FakeEmbedder())
        self.store.add_documents(CLUSTER_DOCS.items())
        self.client = TestClient(create_app(self.store, NoGenerator()))

    def post(self, path, body, status=200):
        response = self.client.post(path, json=body)
        self.assertEqual(response.status_code, status, response.text)
        return response.json()

    def test_health_and_commitment_agree(self):
        commitment = self.client.get("/commitment").json()
        health = self.client.get("/health").json()
        self.assertEqual(health["commitment"], {"root": commitment["root"], "version": commitment["version"]})
        self.assertEqual(commitment, {
            "root": self.store.commitment.root_hex,
            "version": self.store.version,
            "num_chunks": len(CLUSTER_DOCS),
            "num_clusters": 3,
            "dim": FakeEmbedder.dim,
            "embedding_model": FakeEmbedder.model_name,
        })

    def test_embed_returns_little_endian_float32_hex(self):
        result = self.post("/embed", {"texts": ["moon base", "rhino"]})
        self.assertEqual(result["dim"], FakeEmbedder.dim)
        decoded = np.array([np.frombuffer(bytes.fromhex(v), dtype="<f4") for v in result["vectors"]])
        np.testing.assert_array_equal(decoded, FakeEmbedder().embed(["moon base", "rhino"]))
        self.assertEqual(self.post("/embed", {"texts": []})["vectors"], [])

    def test_centroids_verify_against_root(self):
        result = self.post("/centroids", {"query": "rhino by the river", "n": 2})
        self.assertEqual(result["root_version"], self.store.version)
        self.assertEqual(len(result["centroids"]), 2)
        for entry in result["centroids"]:
            self.assertIn("best_chunk_score", entry)
            # A one-chunk-per-cluster search returns that cluster's best chunk; its
            # proof checks the centroid entry all the way to the root.
            best = self.post("/search_clusters", {"query": "rhino by the river", "root_version": result["root_version"],
                                                  "clusters": [entry["cluster_id"]], "per_cluster": 1})
            self.assertTrue(verify_chunk(self.store, best["chunks"][0], {entry["cluster_id"]: entry}))

    def test_search_clusters_returns_requested_clusters_with_proofs(self):
        version = self.store.version
        result = self.post("/search_clusters", {"query": "mars", "root_version": version,
                                                "clusters": [1, 0], "per_cluster": 2})
        self.assertEqual(result["root_version"], version)
        entries = {entry["cluster_id"]: entry for entry in result["clusters"]}
        self.assertEqual(set(entries), {0, 1})
        self.assertTrue(result["chunks"])
        for chunk in result["chunks"]:
            self.assertIn(chunk["cluster_id"], (0, 1))
            self.assertTrue(verify_chunk(self.store, chunk, entries))

    def test_clusters_returns_the_whole_directory(self):
        result = self.client.get("/clusters").json()
        self.assertEqual(result["root_version"], self.store.version)
        self.assertEqual([e["cluster_id"] for e in result["clusters"]], [0, 1, 2])
        self.assertEqual(result["clusters"], self.store.all_clusters())
        root = self.store.commitment.root_hex
        for entry in result["clusters"]:
            centroid = np.frombuffer(bytes.fromhex(entry["centroid"]), dtype="<f4")
            start = cluster_hash(entry["cluster_id"], centroid, bytes.fromhex(entry["subtree_root"]))
            self.assertEqual(fold_proof(start, entry["cluster_proof"]).hex(), root)

    def test_search_clusters_rejects_stale_root_and_unknown_cluster(self):
        stale = self.post("/search_clusters", {"query": "mars", "root_version": self.store.version - 1,
                                               "clusters": [0], "per_cluster": 1}, status=409)
        self.assertEqual(stale["detail"]["error"], "stale_root")
        self.assertEqual(stale["detail"]["root"]["root"], self.store.commitment.root_hex)
        self.post("/search_clusters", {"query": "mars", "root_version": self.store.version,
                                       "clusters": [99], "per_cluster": 1}, status=400)

    def test_retrieve_is_unchanged_without_proofs(self):
        result = self.post("/retrieve", {"query": "moon", "top_k": 3})
        self.assertEqual(set(result), {"chunks", "embedding_model", "retrieve_ms"})
        self.assertEqual(set(result["chunks"][0]), {"chunk_id", "source", "text", "score"})

    def test_retrieve_with_proofs(self):
        result = self.post("/retrieve", {"query": "moon", "top_k": 3, "with_proofs": True})
        self.assertEqual(result["root_version"], self.store.version)
        entries = {entry["cluster_id"]: entry for entry in result["clusters"]}
        self.assertEqual(len(result["chunks"]), 3)
        for chunk in result["chunks"]:
            self.assertTrue(verify_chunk(self.store, chunk, entries))

    def test_ingest_reports_new_root_version_and_old_version_goes_stale(self):
        old = self.store.version
        result = self.post("/ingest", {"documents": [{"id": "new.txt", "text": "mars mars dust storm"}]})
        self.assertEqual(result["added_chunks"], 1)
        self.assertGreater(result["root_version"], old)
        self.post("/search_clusters", {"query": "mars", "root_version": old, "clusters": [0], "per_cluster": 1},
                  status=409)
        again = self.post("/ingest", {"documents": [{"id": "new.txt", "text": "mars mars dust storm"}]})
        self.assertEqual((again["added_chunks"], again["root_version"]), (0, result["root_version"]))


if __name__ == "__main__":
    unittest.main()
