"""Unit tests for the Merkle commitment (rag/merkle.py).

The shared fixture tests/fixtures/merkle_vectors.json is also checked by the
Rust tests, so passing both shows Python and Rust hash the same bytes.
"""

import hashlib
import json
import os
import struct
import sys
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

from rag.merkle import (
    ZERO_HASH, Commitment, build_levels, cluster_hash, fold_proof, inner_hash, leaf_hash,
    make_proof, tree_root,
)
from rag_merkle_vectors import FIXTURE_PATH, build_vectors

with open(FIXTURE_PATH, encoding='utf-8') as f:
    FIXTURE = json.load(f)


def sha256(data):
    return hashlib.sha256(data).digest()


def leaves(n):
    return [leaf_hash(f'doc#{i}', f'text {i}') for i in range(n)]


class FixtureTests(unittest.TestCase):
    def test_code_reproduces_fixture(self):
        # Any change to the hashing changes this; regenerate the fixture only
        # if the format change is intended (and update the Rust side too).
        self.assertEqual(build_vectors(), FIXTURE)

    def test_leaf_hash_bytes(self):
        leaf = FIXTURE['leaves'][1]
        cid, text = leaf['chunk_id'].encode(), leaf['text'].encode()
        expected = sha256(b'\x00' + struct.pack('>I', len(cid)) + cid + struct.pack('>I', len(text)) + text)
        self.assertEqual(expected.hex(), leaf['hash'])
        # The em dash is 3 bytes in UTF-8, so the length prefix counts bytes, not characters.
        self.assertEqual(len(text), len(leaf['text']) + 2)

    def test_inner_and_small_tree_roots(self):
        l0, l1, l2 = (bytes.fromhex(x['hash']) for x in FIXTURE['leaves'][:3])
        self.assertEqual(sha256(b'\x02' + l0 + l1).hex(), FIXTURE['inner']['hash'])
        trees = {t['name']: t for t in FIXTURE['trees']}
        self.assertEqual(trees['empty']['root'], '00' * 32)
        self.assertEqual(trees['one_leaf']['root'], l0.hex())
        self.assertEqual(trees['two_leaves']['root'], sha256(b'\x02' + l0 + l1).hex())
        # Odd node moves up unchanged, then pairs with the left pair's hash.
        self.assertEqual(trees['three_leaves']['root'], sha256(b'\x02' + sha256(b'\x02' + l0 + l1) + l2).hex())

    def test_cluster_hash_bytes(self):
        case = FIXTURE['commitments'][0]
        entry = case['clusters'][1]  # one chunk, so its subtree root is that chunk's leaf
        self.assertEqual(entry['subtree_root'], case['chunks'][1]['leaf'])
        centroid = bytes.fromhex(case['centroids'][1]['hex'])
        self.assertEqual(centroid, np.array([0.6, -0.8, 0, 0], dtype='<f4').tobytes())
        expected = sha256(b'\x01' + struct.pack('>II', 1, 4) + centroid + bytes.fromhex(entry['subtree_root']))
        self.assertEqual(expected.hex(), entry['hash'])

    def test_signed_root_message_layout(self):
        signed = FIXTURE['signed_root']
        model = signed['embedding_model'].encode()
        expected = (b'slakshna-rag-root-v1\x00' + bytes.fromhex(signed['root'])
                    + struct.pack('>QQIII', signed['version'], signed['num_chunks'],
                                  signed['num_clusters'], signed['dim'], len(model)) + model)
        self.assertEqual(expected.hex(), signed['message'])

    def test_fixture_proofs_fold_to_roots(self):
        for tree in FIXTURE['trees']:
            for leaf, proof in zip(tree['leaves'], tree['proofs']):
                self.assertEqual(fold_proof(bytes.fromhex(leaf), proof).hex(), tree['root'], tree['name'])
        for case in FIXTURE['commitments']:
            clusters = case['clusters']
            for chunk in case['chunks']:
                subtree = fold_proof(bytes.fromhex(chunk['leaf']), chunk['chunk_proof'])
                self.assertEqual(subtree.hex(), clusters[chunk['cluster']]['subtree_root'])
            for entry in clusters:
                root = fold_proof(bytes.fromhex(entry['hash']), entry['cluster_proof'])
                self.assertEqual(root.hex(), case['root'], case['name'])


class TreeTests(unittest.TestCase):
    def test_every_proof_folds_to_root_for_many_sizes(self):
        for n in range(1, 34):
            nodes = leaves(n)
            levels = build_levels(nodes)
            root = tree_root(nodes)
            for i in range(n):
                self.assertEqual(fold_proof(nodes[i], make_proof(levels, i)), root, f'n={n} i={i}')

    def test_empty_tree_has_zero_root(self):
        self.assertEqual(tree_root([]), ZERO_HASH)

    def test_wrong_side_or_wrong_sibling_fails(self):
        nodes = leaves(4)
        levels = build_levels(nodes)
        proof = make_proof(levels, 1)
        flipped = [dict(step, left=not step['left']) for step in proof]
        self.assertNotEqual(fold_proof(nodes[1], flipped), tree_root(nodes))
        self.assertNotEqual(fold_proof(nodes[2], proof), tree_root(nodes))

    def test_domain_bytes_separate_leaf_and_inner(self):
        a, b = leaves(2)
        self.assertNotEqual(inner_hash(a, b), sha256(a + b))


class CommitmentTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [{'chunk_id': f'd{i}#0', 'text': f'chunk number {i}', 'cluster': i % 3} for i in range(8)]
        self.centroids = np.eye(3, 4, dtype=np.float32)
        self.commitment = Commitment(self.chunks, self.centroids)

    def verify(self, commitment, index, text=None, centroid=None):
        chunk = self.chunks[index]
        entry = commitment.cluster_entry(chunk['cluster'])
        leaf = leaf_hash(chunk['chunk_id'], chunk['text'] if text is None else text)
        if fold_proof(leaf, commitment.chunk_proof(index)).hex() != entry['subtree_root']:
            return False
        raw = bytes.fromhex(entry['centroid']) if centroid is None else centroid
        start = cluster_hash(entry['cluster_id'], np.frombuffer(raw, dtype='<f4'),
                             bytes.fromhex(entry['subtree_root']))
        return fold_proof(start, entry['cluster_proof']) == commitment.root

    def test_every_chunk_verifies(self):
        for i in range(len(self.chunks)):
            self.assertTrue(self.verify(self.commitment, i))

    def test_changed_text_fails(self):
        text = self.chunks[4]['text']
        self.assertFalse(self.verify(self.commitment, 4, text=text[:-1] + 'X'))

    def test_one_flipped_centroid_bit_fails(self):
        raw = bytearray(bytes.fromhex(self.commitment.cluster_entry(1)['centroid']))
        raw[5] ^= 0x01
        self.assertFalse(self.verify(self.commitment, 1, centroid=bytes(raw)))

    def test_rejects_gaps_and_out_of_range_clusters(self):
        with self.assertRaises(ValueError):
            Commitment([{'chunk_id': 'a', 'text': 'a', 'cluster': 1}], np.zeros((2, 4), np.float32))
        with self.assertRaises(ValueError):
            Commitment([{'chunk_id': 'a', 'text': 'a', 'cluster': 2}], np.zeros((2, 4), np.float32))


if __name__ == '__main__':
    unittest.main()
