"""Unit tests for the CPU training path helpers (no model download, no Ray, no GPU)."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bhaskera.launcher.train_cpu import DeterministicBatcher, apply_shard  # noqa: E402


def _cols(n, seq=4):
    return {
        "input_ids": [np.full(seq, i, dtype=np.int64) for i in range(n)],
        "attention_mask": [np.ones(seq, dtype=np.int64) for _ in range(n)],
        "labels": [np.full(seq, i, dtype=np.int64) for i in range(n)],
        "position_ids": [np.arange(seq, dtype=np.int64) for _ in range(n)],
        "seq_idx": [np.ones(seq, dtype=np.int64) for _ in range(n)],
    }


def _row_ids(batch):
    return [int(x) for x in batch["input_ids"][:, 0]]


class BatcherTests(unittest.TestCase):
    def test_resume_lands_on_same_rows(self):
        cols = _cols(10)
        full = DeterministicBatcher(cols, batch_size=2, seed=7)
        seen = [_row_ids(full.next()) for _ in range(5)]  # exactly one pass
        resumed = DeterministicBatcher(cols, batch_size=2, seed=7, start=4)  # after 2 batches
        self.assertEqual([_row_ids(resumed.next()) for _ in range(3)], seen[2:])

    def test_one_pass_visits_every_row_once_then_cycles(self):
        cols = _cols(6)
        b = DeterministicBatcher(cols, batch_size=3, seed=1)
        first = _row_ids(b.next()) + _row_ids(b.next())
        self.assertEqual(sorted(first), list(range(6)))
        # Dataset exhausted: it must keep serving (new permutation), never raise / stop.
        self.assertEqual(len(_row_ids(b.next())), 3)

    def test_batcher_is_deterministic_across_instances(self):
        a = DeterministicBatcher(_cols(8), 4, seed=3)
        b = DeterministicBatcher(_cols(8), 4, seed=3)
        self.assertEqual(_row_ids(a.next()), _row_ids(b.next()))


class ShardTests(unittest.TestCase):
    def test_shards_are_disjoint_and_cover_everything(self):
        cols = _cols(10)
        parts = [apply_shard(cols, f"{i}/3") for i in range(3)]
        ids = [sorted(int(r[0]) for r in p["input_ids"]) for p in parts]
        self.assertEqual(sorted(sum(ids, [])), list(range(10)))
        self.assertEqual(len({x for chunk in ids for x in chunk}), 10)

    def test_no_shard_returns_everything(self):
        cols = _cols(5)
        self.assertEqual(len(apply_shard(cols, None)["input_ids"]), 5)
        self.assertEqual(len(apply_shard(cols, "0/1")["input_ids"]), 5)

    def test_too_small_dataset_falls_back_to_all_rows(self):
        self.assertEqual(len(apply_shard(_cols(2), "5/6")["input_ids"]), 2)

    def test_bad_spec_is_rejected(self):
        with self.assertRaises(ValueError):
            apply_shard(_cols(4), "3/2")


if __name__ == "__main__":
    unittest.main()
