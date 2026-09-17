"""Focused unit tests for the sparse-int8 delta wire format."""

import base64
import io
import unittest

import torch

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import federated_communication as fc
class DeltaTransportTests(unittest.TestCase):
    def test_round_trip_is_sparse_and_shape_preserving(self):
        source = {"adapter": torch.tensor([[0.0, -2.0], [1.0, 0.25]])}
        payload, reconstructed, metrics = fc.encode_delta_envelope(source, sparsity=0.5)
        decoded = fc.decode_delta_envelope(payload, torch.device("cpu"))

        self.assertEqual(decoded["adapter"].shape, source["adapter"].shape)
        self.assertTrue(torch.equal(decoded["adapter"], reconstructed["adapter"]))
        self.assertEqual(metrics["selected_count"], 2)
        self.assertEqual(metrics["quantized_value_bytes"], 2)

    def test_zero_and_single_element_tensors(self):
        source = {"zero": torch.zeros(3), "one": torch.tensor([3.5])}
        payload, _, _ = fc.encode_delta_envelope(source, sparsity=0.1)
        decoded = fc.decode_delta_envelope(payload, torch.device("cpu"))

        self.assertTrue(torch.equal(decoded["zero"], source["zero"]))
        self.assertTrue(torch.equal(decoded["one"], source["one"]))

    def test_decoder_rejects_duplicate_indices(self):
        envelope = {
            "format": fc.DELTA_FORMAT,
            "version": fc.DELTA_VERSION,
            "quantization": fc.QUANTIZATION,
            "tensors": {
                "x": {"shape": [2], "numel": 2, "indices": torch.tensor([0, 0], dtype=torch.int32),
                      "values": torch.tensor([1, 1], dtype=torch.int8), "scale": 1.0, "zero_point": 0}
            },
        }
        buffer = io.BytesIO()
        torch.save(envelope, buffer)
        payload = base64.b64encode(buffer.getvalue()).decode("ascii")

        with self.assertRaisesRegex(ValueError, "duplicate"):
            fc.decode_delta_envelope(payload, torch.device("cpu"))


    def test_large_scale_pretraining_delta_envelope(self):
        # Simulate a 10-million element pretraining layer
        print("\n[Python Test] Testing 10,000,000 parameter tensor delta encoding/decoding...")
        large_tensor = torch.randn(2500, 4000, dtype=torch.float32)
        source = {"transformer.layers.0.weight": large_tensor}
        
        # Test binary saving as used in pretraining with Iroh Blobs
        temp_path = "/tmp/test_pretrain_large_delta.pt"
        torch.save(source, temp_path)
        file_size_mb = os.path.getsize(temp_path) / (1024 * 1024)
        print(f"[Python Test] Saved 10M param tensor ({file_size_mb:.2f} MB) to {temp_path}")
        
        loaded = torch.load(temp_path, weights_only=True)
        self.assertEqual(loaded["transformer.layers.0.weight"].shape, (2500, 4000))
        self.assertTrue(torch.equal(loaded["transformer.layers.0.weight"], large_tensor))
        if os.path.exists(temp_path):
            os.remove(temp_path)
        print("[Python Test] ✅ 10M parameter pretraining delta verified successfully!")


if __name__ == "__main__":
    unittest.main()
