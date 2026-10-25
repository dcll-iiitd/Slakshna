"""A delta that a node is allowed to send must be a delta its peers are willing to accept."""

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from federated_communication import (  # noqa: E402
    MAX_DELTA_NORM,
    MAX_PEER_DELTA_NORM,
    MAX_PEER_NORM_RATIO,
    apply_differential_privacy_and_clipping,
    validate_peer_delta,
)


def norm(delta):
    return float(sum(v.float().norm() ** 2 for v in delta.values()) ** 0.5)


class NormPolicyTests(unittest.TestCase):
    def test_clipped_sender_delta_always_passes_receiver_validation(self):
        for scale in (0.01, 1.0, 14.0, 99.0, 100.0, 1e4):
            delta = {"w": torch.randn(1000) * scale}
            clipped = apply_differential_privacy_and_clipping(delta, max_norm=MAX_DELTA_NORM,
                                                              noise_multiplier=0.0)
            self.assertLessEqual(norm(clipped), MAX_DELTA_NORM * (1 + 1e-4))
            ok = validate_peer_delta(clipped, max_allowed_norm=MAX_PEER_DELTA_NORM)
            self.assertTrue(ok, scale)

    def test_clipped_delta_survives_sparsify_quantize_and_decode(self):
        from federated_communication import decode_delta_envelope, encode_delta_envelope

        delta = {"a": torch.randn(5000) * 50, "b": torch.randn(3000) * 50}
        clipped = apply_differential_privacy_and_clipping(delta, max_norm=MAX_DELTA_NORM,
                                                          noise_multiplier=0.0)
        payload, _, _ = encode_delta_envelope(clipped, sparsity=0.5)
        received = decode_delta_envelope(payload, torch.device("cpu"))
        self.assertTrue(validate_peer_delta(received, max_allowed_norm=MAX_PEER_DELTA_NORM))

    def test_a_realistic_lora_delta_between_10_and_100_is_accepted(self):
        # The norm the real two-node run produced (13.95) was rejected under the old limit of 10.
        delta = {"w": torch.ones(10000) * (13.95 / 100.0)}
        self.assertAlmostEqual(norm(delta), 13.95, places=1)
        self.assertTrue(validate_peer_delta(delta, max_allowed_norm=MAX_PEER_DELTA_NORM))

    def test_oversized_and_non_finite_deltas_are_still_rejected(self):
        self.assertFalse(validate_peer_delta({"w": torch.ones(10000) * 2.0},   # norm 200
                                             max_allowed_norm=MAX_PEER_DELTA_NORM))
        self.assertFalse(validate_peer_delta({"w": torch.tensor([1.0, float("nan")])},
                                             max_allowed_norm=MAX_PEER_DELTA_NORM))
        self.assertFalse(validate_peer_delta({"w": torch.tensor([float("inf")])},
                                             max_allowed_norm=MAX_PEER_DELTA_NORM))

    def test_a_delta_far_larger_than_our_own_is_rejected_even_under_the_absolute_limit(self):
        """A poisoned delta clipped to the limit (norm ~78 here) must not pass as 'valid'."""
        own = {"w": torch.ones(100) * 0.6}                      # norm 6
        attacker = {"w": torch.ones(100) * 7.8}                 # norm 78: under 105, 13x ours
        self.assertLess(norm(attacker), MAX_PEER_DELTA_NORM)
        self.assertFalse(validate_peer_delta(
            attacker, max_allowed_norm=MAX_PEER_DELTA_NORM,
            reference_norm=norm(own), max_ratio=MAX_PEER_NORM_RATIO))

    def test_similar_and_moderately_larger_peers_are_accepted(self):
        own_norm = 6.0
        for peer_norm in (0.5, 6.0, 14.0, 29.0):                # up to ~4.8x
            peer = {"w": torch.ones(100) * (peer_norm / 10.0)}
            self.assertTrue(validate_peer_delta(peer, max_allowed_norm=MAX_PEER_DELTA_NORM,
                                                reference_norm=own_norm,
                                                max_ratio=MAX_PEER_NORM_RATIO), peer_norm)

    def test_relative_check_is_skipped_when_we_trained_nothing(self):
        peer = {"w": torch.ones(100) * 5.0}                     # norm 50
        self.assertTrue(validate_peer_delta(peer, max_allowed_norm=MAX_PEER_DELTA_NORM,
                                            reference_norm=0.0, max_ratio=MAX_PEER_NORM_RATIO))

    def test_delta_norm_matches_manual_computation(self):
        from federated_communication import delta_norm
        self.assertAlmostEqual(delta_norm({"a": torch.tensor([3.0]), "b": torch.tensor([4.0]),
                                           "meta": "ignored"}), 5.0, places=5)

    def test_default_matches_the_clip_the_engine_always_used(self):
        self.assertEqual(MAX_DELTA_NORM, 100.0)


if __name__ == "__main__":
    unittest.main()
