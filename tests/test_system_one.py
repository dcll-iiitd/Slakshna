import math
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
    HAVE = True
except ImportError:
    HAVE = False

VOCAB = 200


class FakeTok:
    """Every label char is one token (id 100+ord); anything longer is not."""

    def encode(self, text, add_special_tokens=False):
        t = text.strip()
        if len(t) == 1:
            return [100 + ord(t)]
        return [1, 2]


def label_ids(prompt):
    """label char -> token id used by the fake model."""
    return {c: 100 + ord(c) for c in re.findall(r"^([A-Za-z0-9])\. ", prompt, flags=re.M)}


def parse(prompt):
    return dict(re.findall(r"^([A-Za-z0-9])\. (.*)$", prompt, flags=re.M))


def oracle(target, strength=8.0, bias_first=0.0, noise_mass=0.0):
    """Fake adapter: prefers the label whose option text equals ``target``;
    optionally biased towards the first label (position bias)."""

    def fn(prompt):
        logits = torch.zeros(VOCAB)
        opts = parse(prompt)
        first = next(iter(opts))
        for c, text in opts.items():
            if text == target:
                logits[100 + ord(c)] += strength
        logits[100 + ord(first)] += bias_first
        if noise_mass:
            logits[3] = noise_mass
        return logits

    return fn


@unittest.skipUnless(HAVE, "torch not installed")
class SystemOneTests(unittest.TestCase):
    def decide(self, members, options, **kw):
        from federated_inference import system_one as so
        return so.decide(members, FakeTok(), "State: x", "Which?", options, **kw)

    def member(self, name, fn, w=1.0):
        from federated_inference import system_one as so
        return so.Member(name, fn, w)

    def test_probabilities_form_a_distribution_and_pick_target(self):
        d = self.decide([self.member("a", oracle("bar"))], ["foo", "bar", "baz"])
        self.assertAlmostEqual(sum(d.probabilities), 1.0, places=9)
        self.assertEqual(d.label, "bar")
        self.assertEqual(d.choice, 1)
        self.assertGreater(d.confidence, 0.9)
        self.assertGreater(d.margin, 0.8)
        self.assertLess(d.entropy, math.log(3))
        self.assertFalse(d.two_stage)

    def test_uninformative_model_is_uniform_with_max_entropy(self):
        d = self.decide([self.member("a", oracle("nothing matches"))], ["a1", "b1", "c1", "d1"])
        for p in d.probabilities:
            self.assertAlmostEqual(p, 0.25, places=6)
        self.assertAlmostEqual(d.entropy, math.log(4), places=6)

    def test_permutation_averaging_cancels_position_bias(self):
        opts = ["w", "x", "y", "z"]
        biased = oracle("nothing matches", bias_first=5.0)
        one = self.decide([self.member("a", biased)], opts, permutations=1)
        many = self.decide([self.member("a", biased)], opts, permutations=4)
        self.assertGreater(max(one.probabilities), 0.9)  # all mass on the first slot
        for p in many.probabilities:  # bias averaged away
            self.assertAlmostEqual(p, 0.25, places=6)

    def test_default_is_auto_and_debiases_without_being_asked(self):
        biased = oracle("nothing matches", bias_first=5.0)
        d = self.decide([self.member("a", biased)], ["w", "x", "y", "z"])  # no permutations given
        for p in d.probabilities:
            self.assertAlmostEqual(p, 0.25, places=6)

    def test_auto_uses_every_shift_for_few_options_and_caps_for_many(self):
        from federated_inference import system_one as so
        calls = []

        def counting(prompt):
            calls.append(prompt)
            return torch.zeros(VOCAB)

        for n, expect in ((2, 2), (4, 4), (6, 6), (7, 6), (20, 6)):
            calls.clear()
            self.decide([self.member("a", counting)], [f"o{i}" for i in range(n)])
            self.assertEqual(len(calls), expect, f"n={n}")
        self.assertEqual(so.AUTO_MAX_PERMUTATIONS, 6)

    def test_explicit_one_means_a_single_undebiased_pass(self):
        calls = []

        def counting(prompt):
            calls.append(prompt)
            return torch.zeros(VOCAB)

        self.decide([self.member("a", counting)], ["a1", "b1", "c1", "d1"], permutations=1)
        self.assertEqual(len(calls), 1)

    def test_shifts_are_distinct_and_spread(self):
        from federated_inference import system_one as so
        self.assertEqual(so._shifts(4, 0), [0, 1, 2, 3])
        s = so._shifts(20, 0)
        self.assertEqual(len(s), 6)
        self.assertEqual(len(set(s)), 6)
        self.assertEqual(so._shifts(5, 1), [0])

    def test_permutations_keep_real_signal(self):
        d = self.decide([self.member("a", oracle("y", bias_first=3.0))], ["w", "x", "y", "z"], permutations=4)
        self.assertEqual(d.label, "y")

    def test_calibration_temperature_softens(self):
        hot = self.decide([self.member("a", oracle("bar"))], ["foo", "bar"], calibration_temperature=1.0)
        cool = self.decide([self.member("a", oracle("bar"))], ["foo", "bar"], calibration_temperature=8.0)
        self.assertGreater(hot.confidence, cool.confidence)
        self.assertGreater(cool.confidence, 0.5)

    def test_mixture_is_weighted_mean_and_trust_decides_disagreement(self):
        opts = ["foo", "bar"]
        a, b = self.member("a", oracle("foo"), 3.0), self.member("b", oracle("bar"), 1.0)
        d = self.decide([a, b], opts)
        self.assertEqual(d.label, "foo")
        pa, pb = d.members[0].probabilities, d.members[1].probabilities
        for i in range(2):
            self.assertAlmostEqual(d.probabilities[i], 0.75 * pa[i] + 0.25 * pb[i], places=9)
        self.assertAlmostEqual(d.agreement, 0.75, places=9)

    def test_zero_weight_member_has_no_effect(self):
        opts = ["foo", "bar"]
        solo = self.decide([self.member("a", oracle("foo"))], opts)
        both = self.decide([self.member("a", oracle("foo")), self.member("b", oracle("bar"), 0.0)], opts)
        for x, y in zip(solo.probabilities, both.probabilities):
            self.assertAlmostEqual(x, y, places=9)

    def test_identical_members_equal_one(self):
        opts = ["foo", "bar", "baz"]
        solo = self.decide([self.member("a", oracle("baz"))], opts)
        trio = self.decide([self.member(str(i), oracle("baz"), i + 1) for i in range(3)], opts)
        for x, y in zip(solo.probabilities, trio.probabilities):
            self.assertAlmostEqual(x, y, places=9)

    def test_logit_mean_is_a_distribution(self):
        d = self.decide([self.member("a", oracle("foo")), self.member("b", oracle("bar"))], ["foo", "bar"], rule="logit_mean")
        self.assertAlmostEqual(sum(d.probabilities), 1.0, places=9)

    def test_one_malicious_member_is_bounded_under_prob_mean(self):
        # The evil member is certain of "bar"; under prob_mean it adds at most its weight.
        good = self.member("good", oracle("nothing matches"), 0.9)
        evil = self.member("evil", oracle("bar", strength=50.0), 0.1)
        d = self.decide([good, evil], ["foo", "bar"])
        self.assertLessEqual(d.probabilities[1], 0.9 * 0.5 + 0.1 + 1e-9)

    def test_low_format_mass_is_reported(self):
        d = self.decide([self.member("a", oracle("foo", noise_mass=30.0))], ["foo", "bar"])
        self.assertLess(d.min_format_mass, 0.01)
        ok = self.decide([self.member("a", oracle("foo"))], ["foo", "bar"])
        self.assertGreater(ok.min_format_mass, 0.5)

    def test_two_stage_finds_target_beyond_label_limit(self):
        options = [f"opt{i}" for i in range(100)]
        d = self.decide([self.member("a", oracle("opt73")), self.member("b", oracle("opt73"))],
                        options, label_limit=10)
        self.assertTrue(d.two_stage)
        self.assertEqual(d.label, "opt73")
        self.assertIn(73, d.finalists)
        self.assertAlmostEqual(sum(d.probabilities), 1.0, places=9)
        self.assertLessEqual(len(d.finalists), 10)  # the second pass fits the labels
        self.assertEqual(sum(1 for p in d.probabilities if p > 0), len(d.finalists))

    def test_two_stage_matches_single_stage_choice(self):
        options = [f"opt{i}" for i in range(12)]
        m = [self.member("a", oracle("opt5"))]
        self.assertEqual(self.decide(m, options).label, self.decide(m, options, label_limit=4).label)

    def test_input_validation(self):
        from federated_inference import system_one as so
        m = [self.member("a", oracle("x"))]
        for bad in (["only"], ["a", "a"], ["a", ""], [str(i) for i in range(300)]):
            with self.assertRaises(ValueError):
                self.decide(m, bad)
        with self.assertRaises(ValueError):
            so.decide([], FakeTok(), "s", "q", ["a", "b"])
        with self.assertRaises(ValueError):
            self.decide(m, ["a", "b"], calibration_temperature=0)
        with self.assertRaises(ValueError):
            self.decide(m, ["a", "b"], rule="nope")

    def test_label_vocab_prefers_distinct_single_tokens(self):
        from federated_inference import system_one as so
        labels = so.label_vocab(FakeTok())
        self.assertEqual(len(labels), len({i for _, i in labels}))
        self.assertEqual(labels[0][0], "A")
        self.assertEqual(len(labels), 62)  # the fake tokenizer has both forms for every char

    def test_labels_use_one_consistent_form(self):
        # Like Qwen: " A".." z" are single tokens, " 1" is two tokens but bare "1" is one.
        from federated_inference import system_one as so

        class QwenLike:
            def encode(self, text, add_special_tokens=False):
                if text.startswith(" ") and text[1:].isdigit():
                    return [1, 2]
                return [1000 + ord(text.strip())]

        labels = so.label_vocab(QwenLike())
        self.assertEqual(len(labels), 52)  # letters only, all in the spaced form
        self.assertFalse(any(c.isdigit() for c, _ in labels))

        class BareOnly:
            def encode(self, text, add_special_tokens=False):
                return [1, 2] if text.startswith(" ") else [1000 + ord(text)]

        self.assertEqual(len(so.label_vocab(BareOnly())), 62)  # no spaced form exists: all bare

    def test_options_with_newlines_are_flattened(self):
        from federated_inference import system_one as so
        self.assertEqual(so.validate_options(["a\nb", "c  d"]), ["a b", "c d"])


if __name__ == "__main__":
    unittest.main()
