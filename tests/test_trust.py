import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from federated_communication import trust

try:
    import torch
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


class InitTests(unittest.TestCase):
    def test_uniform_start_gives_equal_weights(self):
        alpha = {n: trust.init_alpha("uniform", "me", n) for n in ("me", "a", "b", "c")}
        w = trust.softmax_weights(alpha, alpha)
        for v in w.values():
            self.assertAlmostEqual(v, 0.25)

    def test_seeded_is_deterministic_small_and_node_specific(self):
        a1 = trust.init_alpha("seeded", "me", "a")
        self.assertEqual(a1, trust.init_alpha("seeded", "me", "a"))
        self.assertNotEqual(a1, trust.init_alpha("seeded", "me", "b"))
        self.assertNotEqual(a1, trust.init_alpha("seeded", "other", "a"))
        self.assertLess(abs(a1), 0.6)

    def test_bad_mode_rejected(self):
        with self.assertRaises(ValueError):
            trust.init_alpha("nope", "a", "b")


class RenormaliseTests(unittest.TestCase):
    def test_missing_peer_share_is_redistributed(self):
        w = {"me": 0.5, "a": 0.3, "b": 0.2}
        got = trust.renormalise(w, ["me", "a"])  # b's delta never arrived
        self.assertAlmostEqual(sum(got.values()), 1.0)
        self.assertNotIn("b", got)
        self.assertAlmostEqual(got["me"] / got["a"], 0.5 / 0.3)

    def test_all_arrived_is_unchanged(self):
        w = {"me": 0.5, "a": 0.3, "b": 0.2}
        got = trust.renormalise(w, w)
        for k in w:
            self.assertAlmostEqual(got[k], w[k])

    def test_zero_weights_fall_back_to_uniform_and_empty_is_empty(self):
        self.assertEqual(trust.renormalise({"a": 0.0, "b": 0.0}, ["a", "b"]), {"a": 0.5, "b": 0.5})
        self.assertEqual(trust.renormalise({"a": 1.0}, []), {})
        self.assertEqual(trust.renormalise({"a": 1.0}, ["zzz"]), {})


class SignalTests(unittest.TestCase):
    def test_utility_mapping(self):
        self.assertAlmostEqual(trust.utility_from_loss_change(0.0, 2.0), 0.0)
        self.assertGreater(trust.utility_from_loss_change(0.05, 2.0), 0.9)
        self.assertLess(trust.utility_from_loss_change(-0.05, 2.0), -0.9)
        self.assertAlmostEqual(trust.utility_from_loss_change(0.02, 2.0), math.tanh(1.0))
        for bad in (float("nan"), float("inf")):
            self.assertEqual(trust.utility_from_loss_change(bad, 2.0), 0.0)
        self.assertEqual(trust.utility_from_loss_change(1.0, 0.0), 0.0)

    def test_combine_blends_and_falls_back(self):
        self.assertAlmostEqual(trust.combine_signal(0.8, None), 0.8)
        self.assertAlmostEqual(trust.combine_signal(0.8, -0.4, 0.5), 0.2)
        self.assertAlmostEqual(trust.combine_signal(0.8, -0.4, 0.0), 0.8)
        self.assertAlmostEqual(trust.combine_signal(0.8, -0.4, 1.0), -0.4)

    def test_aligned_but_harmful_peer_is_demoted_once_utility_is_used(self):
        # Update direction matches ours (cosine +0.9) but it makes the loss worse.
        cos_only = trust.combine_signal(0.9, None)
        both = trust.combine_signal(0.9, -1.0, 0.5)
        self.assertGreater(cos_only, 0.0)
        self.assertLess(both, cos_only)


class UpdateTests(unittest.TestCase):
    def test_only_peers_with_evidence_move_and_everyone_decays(self):
        alpha = {"me": 0.0, "a": 0.0, "b": 1.0}
        trust.update_alpha(alpha, {"me": 0.5, "a": 0.5}, {"a": 1.0}, beta=0.1, decay=0.98)
        self.assertAlmostEqual(alpha["a"], (0.0 + 0.1 * 1.0 * 0.25) * 0.98)
        self.assertAlmostEqual(alpha["me"], 0.0)
        self.assertAlmostEqual(alpha["b"], 0.98)  # no evidence: only decays

    def test_bad_peer_loses_weight_good_peer_gains_from_uniform_start(self):
        nodes = ["me", "good", "bad"]
        alpha = {n: trust.init_alpha("uniform", "me", n) for n in nodes}
        for _ in range(30):
            w = trust.softmax_weights(alpha, nodes)
            sig = {"me": 1.0, "good": 0.6, "bad": -0.8}
            trust.update_alpha(alpha, w, sig, beta=0.1)
        w = trust.softmax_weights(alpha, nodes)
        self.assertGreater(w["good"], w["bad"])
        self.assertLess(w["bad"], 1 / 3)


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class PeerEvalTests(unittest.TestCase):
    def setUp(self):
        from federated_communication import peer_eval  # noqa: F401
        torch.manual_seed(0)
        d_in, d_out, r = 8, 6, 2
        self.W0 = torch.randn(d_out, d_in)
        self.x = torch.randn(32, d_in)
        self.y = torch.randn(32, d_out)
        self.base = {
            "m.lora_A.weight": torch.randn(r, d_in) * 0.3,
            "m.lora_B.weight": torch.randn(d_out, r) * 0.3,
        }

        def loss(params):
            delta_w = params["m.lora_B.weight"] @ params["m.lora_A.weight"]
            pred = self.x @ (self.W0 + delta_w).T
            return ((pred - self.y) ** 2).mean()

        self.loss = loss
        self.batches = [loss, loss]

    def grad(self):
        p = {k: v.clone().requires_grad_(True) for k, v in self.base.items()}
        self.loss(p).backward()
        return {k: v.grad.clone() for k, v in p.items()}

    def test_helpful_peer_is_positive_harmful_peer_negative(self):
        from federated_communication.peer_eval import heldout_utilities
        g = self.grad()
        helpful = {k: -0.01 * v for k, v in g.items()}
        harmful = {k: 0.01 * v for k, v in g.items()}
        for method in ("first_order", "exact"):
            _, ch = heldout_utilities(self.batches, self.base, {"good": helpful, "bad": harmful}, method)
            self.assertGreater(ch["good"], 0, method)
            self.assertLess(ch["bad"], 0, method)

    def test_first_order_matches_exact_for_small_deltas(self):
        from federated_communication.peer_eval import heldout_utilities
        g = self.grad()
        delta = {k: -1e-3 * v + 1e-4 * torch.randn_like(v) for k, v in g.items()}
        b1, fo = heldout_utilities(self.batches, self.base, {"p": delta}, "first_order")
        b2, ex = heldout_utilities(self.batches, self.base, {"p": delta}, "exact")
        self.assertAlmostEqual(b1, b2, places=5)
        self.assertAlmostEqual(fo["p"], ex["p"], delta=abs(ex["p"]) * 0.05 + 1e-7)

    def test_zero_delta_and_partial_keys(self):
        from federated_communication.peer_eval import heldout_utilities
        zero = {k: torch.zeros_like(v) for k, v in self.base.items()}
        partial = {"m.lora_A.weight": torch.zeros_like(self.base["m.lora_A.weight"])}
        _, ch = heldout_utilities(self.batches, self.base, {"z": zero, "p": partial}, "exact")
        self.assertAlmostEqual(ch["z"], 0.0, places=7)
        self.assertAlmostEqual(ch["p"], 0.0, places=7)

    def test_bad_method_and_empty_batches(self):
        from federated_communication.peer_eval import heldout_utilities
        with self.assertRaises(ValueError):
            heldout_utilities(self.batches, self.base, {}, "nope")
        with self.assertRaises(ValueError):
            heldout_utilities([], self.base, {}, "exact")

    def test_name_map_handles_wrappers_and_adapter_names(self):
        from federated_communication.peer_eval import build_name_map, normalise_key
        sd_keys = ["base_model.model.layers.0.q_proj.lora_A.weight", "base_model.model.layers.0.q_proj.lora_B.weight"]
        params = [
            "base_model.model.layers.0.q_proj.base_layer.weight",
            "module.base_model.model.layers.0.q_proj.lora_A.default.weight",
            "base_model.model.layers.0.q_proj.lora_B.default.weight",
        ]
        m = build_name_map(params, sd_keys)
        self.assertEqual(len(m), 2)
        self.assertEqual(m[params[1]], sd_keys[0])
        self.assertEqual(normalise_key("module.x.lora_A.default.weight"), "x.lora_A.weight")


if __name__ == "__main__":
    unittest.main()
