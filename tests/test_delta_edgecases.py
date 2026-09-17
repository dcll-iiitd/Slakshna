"""Comprehensive edge-case tests for Python ML engine and Iroh Blobs delta transport."""

import os
import sys
import unittest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from federated_communication.privacy import validate_peer_delta
import federated_communication as fc


class DeltaTransportEdgecaseTests(unittest.TestCase):
    def test_nan_injection_rejected(self):
        corrupted = {"layer.weight": torch.tensor([1.0, float('nan'), 2.0])}
        self.assertFalse(validate_peer_delta(corrupted))

    def test_inf_injection_rejected(self):
        corrupted = {"layer.weight": torch.tensor([1.0, float('inf'), 2.0])}
        self.assertFalse(validate_peer_delta(corrupted))

    def test_negative_inf_injection_rejected(self):
        corrupted = {"layer.weight": torch.tensor([1.0, float('-inf'), 2.0])}
        self.assertFalse(validate_peer_delta(corrupted))

    def test_excessive_norm_rejected(self):
        # Norm is 1000.0, default max is 500.0
        oversized = {"layer.weight": torch.full((100, 100), 10.0)}
        self.assertFalse(validate_peer_delta(oversized))

    def test_custom_env_max_norm_respected(self):
        # Norm is 600.0, should fail default 500.0
        tensor = torch.full((60, 10), 10.0) # total norm = sqrt(600 * 100) = ~245
        tensor_large = torch.full((600, 10), 10.0) # total norm = sqrt(6000 * 100) = ~774
        self.assertTrue(validate_peer_delta({"w": tensor}))
        self.assertFalse(validate_peer_delta({"w": tensor_large}))

        # With env var SLAKSHNA_MAX_ALLOWED_NORM=1000.0
        os.environ["SLAKSHNA_MAX_ALLOWED_NORM"] = "1000.0"
        try:
            self.assertTrue(validate_peer_delta({"w": tensor_large}))
        finally:
            del os.environ["SLAKSHNA_MAX_ALLOWED_NORM"]

    def test_empty_delta_dict(self):
        empty = {}
        self.assertTrue(validate_peer_delta(empty))

    def test_corrupted_truncated_file_handling(self):
        tmp_corrupt = "/tmp/test_truncated_blob.pt"
        with open(tmp_corrupt, "wb") as f:
            f.write(b"PK\x03\x04truncated_junk_bytes_here")

        with self.assertRaises(Exception):
            torch.load(tmp_corrupt, weights_only=True)

        if os.path.exists(tmp_corrupt):
            os.remove(tmp_corrupt)

    def test_weights_only_tensor_dtypes(self):
        tmp_path = "/tmp/test_dtypes_blob.pt"
        payload = {
            "fp32": torch.randn(10, 10, dtype=torch.float32),
            "fp16": torch.randn(10, 10, dtype=torch.float16),
            "int8": torch.randint(-128, 127, (10, 10), dtype=torch.int8),
            "int32": torch.tensor([1, 2, 3, 4], dtype=torch.int32),
        }
        torch.save(payload, tmp_path)
        loaded = torch.load(tmp_path, weights_only=True)
        self.assertEqual(loaded["fp32"].dtype, torch.float32)
        self.assertEqual(loaded["fp16"].dtype, torch.float16)
        self.assertEqual(loaded["int8"].dtype, torch.int8)
        self.assertEqual(loaded["int32"].dtype, torch.int32)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    unittest.main()
