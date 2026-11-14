"""Merkle commitment over a clustered corpus.

The whole index (every chunk's id and text, and every cluster's centroid) is
fingerprinted as one 32-byte root. Anything sent to a peer later comes with a
proof: the sibling hashes needed to recompute the root from that one item.

The exact bytes hashed are defined in this file and must
match the Rust side (src/rag/merkle.rs). tests/fixtures/merkle_vectors.json,
written by scripts/rag_merkle_vectors.py, checks both against each other.
"""

import hashlib
import struct

import numpy as np

LEAF = b"\x00"
CLUSTER = b"\x01"
INNER = b"\x02"
ZERO_HASH = bytes(32)
ROOT_MESSAGE_TAG = b"slakshna-rag-root-v1"


def _sha256(*parts):
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part)
    return digest.digest()


def _u32(n):
    return struct.pack(">I", n)


def _u64(n):
    return struct.pack(">Q", n)


def leaf_hash(chunk_id, text):
    """H(0x00 ‖ u32be(len(id)) ‖ id ‖ u32be(len(text)) ‖ text), over UTF-8 bytes."""
    id_bytes = chunk_id.encode("utf-8")
    text_bytes = text.encode("utf-8")
    return _sha256(LEAF, _u32(len(id_bytes)), id_bytes, _u32(len(text_bytes)), text_bytes)


def inner_hash(left, right):
    """H(0x02 ‖ left ‖ right)."""
    return _sha256(INNER, left, right)


def centroid_bytes(centroid):
    """A centroid as little-endian float32 values, dim*4 bytes."""
    return np.ascontiguousarray(centroid, dtype="<f4").reshape(-1).tobytes()


def cluster_hash(cluster_id, centroid, subtree_root):
    """H(0x01 ‖ u32be(cluster_id) ‖ u32be(dim) ‖ centroid_bytes ‖ subtree_root)."""
    raw = centroid_bytes(centroid)
    return _sha256(CLUSTER, _u32(cluster_id), _u32(len(raw) // 4), raw, subtree_root)


def build_levels(nodes):
    """Every level of the tree over `nodes`, bottom first, root level last.

    Nodes are paired left to right; an odd node at the end of a level moves up
    unchanged.
    """
    levels = [list(nodes)]
    while len(levels[-1]) > 1:
        level = levels[-1]
        upper = [inner_hash(level[i], level[i + 1]) for i in range(0, len(level) - 1, 2)]
        if len(level) % 2 == 1:
            upper.append(level[-1])
        levels.append(upper)
    return levels


def levels_root(levels):
    return levels[-1][0] if levels[-1] else ZERO_HASH


def tree_root(nodes):
    """Root of the tree over `nodes`; 32 zero bytes when there are none."""
    return levels_root(build_levels(nodes))


def make_proof(levels, index):
    """Sibling hashes from node `index` of the bottom level up to the root.

    Each step is `{"h": hex, "left": bool}`; `left` is true when the sibling
    is on the left.
    """
    if not 0 <= index < len(levels[0]):
        raise IndexError(f"no node {index} in a tree of {len(levels[0])}")
    proof = []
    for level in levels[:-1]:
        if index % 2 == 1:
            proof.append({"h": level[index - 1].hex(), "left": True})
        elif index + 1 < len(level):
            proof.append({"h": level[index + 1].hex(), "left": False})
        index //= 2
    return proof


def fold_proof(start, proof):
    """Recompute the root from a starting hash and its proof."""
    current = start
    for step in proof:
        sibling = bytes.fromhex(step["h"])
        if len(sibling) != 32:
            raise ValueError("proof step is not a 32-byte hash")
        current = inner_hash(sibling, current) if step["left"] else inner_hash(current, sibling)
    return current


def signed_root_message(root, version, num_chunks, num_clusters, dim, embedding_model):
    """The bytes a node signs to vouch for its root (signing is done in Rust)."""
    model = embedding_model.encode("utf-8")
    return (ROOT_MESSAGE_TAG + b"\x00" + root + _u64(version) + _u64(num_chunks)
            + _u32(num_clusters) + _u32(dim) + _u32(len(model)) + model)


class Commitment:
    """The two-level Merkle tree over a clustered corpus.

    `chunks` are dicts with `chunk_id`, `text` and `cluster`, in chunks.jsonl
    order. `centroids` is a float32 `[k, dim]` array. Cluster ids must run
    0..k-1 and every cluster must hold at least one chunk.
    """

    def __init__(self, chunks, centroids):
        centroids = np.asarray(centroids, dtype=np.float32)
        if centroids.ndim != 2:
            raise ValueError("centroids must be a [k, dim] array")
        self.centroids = centroids
        self.num_clusters, self.dim = centroids.shape
        self.num_chunks = len(chunks)

        # members[c]: global chunk indices in cluster c; position[i]: index of
        # chunk i inside its cluster's tree.
        self.members = [[] for _ in range(self.num_clusters)]
        self.position = []
        for i, chunk in enumerate(chunks):
            c = chunk["cluster"]
            if not 0 <= c < self.num_clusters:
                raise ValueError(f"chunk {chunk['chunk_id']} has cluster {c}, "
                                 f"but there are {self.num_clusters} clusters")
            self.position.append(len(self.members[c]))
            self.members[c].append(i)
        for c, members in enumerate(self.members):
            if not members:
                raise ValueError(f"cluster {c} has no chunks")

        self.chunk_cluster = [chunk["cluster"] for chunk in chunks]
        self.leaf_levels = [
            build_levels([leaf_hash(chunks[i]["chunk_id"], chunks[i]["text"]) for i in members])
            for members in self.members
        ]
        self.subtree_roots = [levels_root(levels) for levels in self.leaf_levels]
        self.cluster_levels = build_levels([
            cluster_hash(c, centroids[c], self.subtree_roots[c]) for c in range(self.num_clusters)
        ])
        self.root = levels_root(self.cluster_levels)

    @property
    def root_hex(self):
        return self.root.hex()

    def cluster_entry(self, c):
        """What a peer sends to prove cluster `c` (without `best_chunk_score`)."""
        return {
            "cluster_id": c,
            "size": len(self.members[c]),
            "centroid": centroid_bytes(self.centroids[c]).hex(),
            "subtree_root": self.subtree_roots[c].hex(),
            "cluster_proof": make_proof(self.cluster_levels, c),
        }

    def chunk_proof(self, index):
        """Proof from chunk `index` (its place in chunks.jsonl) to its cluster's subtree root."""
        c = self.chunk_cluster[index]
        return make_proof(self.leaf_levels[c], self.position[index])
