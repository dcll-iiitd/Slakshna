"""The federated simulator used for the paper experiments, exercised on a tiny model (CPU)."""

import json
import os
import random
import shutil
import sys
import tempfile
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))

try:
    import torch
    import transformers  # noqa: F401
    import peft  # noqa: F401
    import fed_sim
    HAVE_STACK = os.path.isdir(os.path.join(ROOT, "Bhaskera", "src", "bhaskera", "rl"))
except ImportError:
    HAVE_STACK = False

WORDS = ["<pad>", "<eos>", "<unk>", "Q", "A", "1", "2", "3", "step", "The", "final", "answer", "is",
         "\\boxed{3}", "\\boxed{5}", "\\boxed{7}", "\\boxed{0}"]


def make_examples(n, seed=0):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        k = rng.randint(1, 6)                                   # difficulty = calculation steps
        calc = " ".join(f"step <<1+1=2>>" for _ in range(k))
        rows.append({"prompt": f"Q {rng.choice([1, 2, 3])}", "answer": f"{calc}\n#### 7"})
    return rows


@unittest.skipUnless(HAVE_STACK, "needs the Bhaskera environment")
class PureFunctionTests(unittest.TestCase):
    def setUp(self):
        from bhaskera.rl.data import Example
        self.examples = [Example(r["prompt"], r["answer"]) for r in make_examples(800)]

    def test_iid_partition_is_disjoint_equal_and_complete(self):
        shards = fed_sim.partition(self.examples, 4, "iid", seed=1)
        self.assertEqual(sorted(len(s) for s in shards), [200] * 4)
        ids = [id(e) for s in shards for e in s]
        self.assertEqual(len(set(ids)), 800)

    def test_noniid_partition_is_skewed_but_covers_every_example_once(self):
        shards = fed_sim.partition(self.examples, 4, "noniid", seed=1)
        self.assertEqual(sum(len(s) for s in shards), 800)
        self.assertEqual(len({id(e) for s in shards for e in s}), 800)
        mean_steps = [sum(e.answer.count("<<") for e in s) / len(s) for s in shards]
        self.assertGreater(max(mean_steps) - min(mean_steps), 0.4)     # nodes differ in difficulty
        iid = fed_sim.partition(self.examples, 4, "iid", seed=1)
        iid_steps = [sum(e.answer.count("<<") for e in s) / len(s) for s in iid]
        self.assertGreater(max(mean_steps) - min(mean_steps), 3 * (max(iid_steps) - min(iid_steps)))

    def test_partitions_are_reproducible(self):
        a = fed_sim.partition(self.examples, 4, "noniid", seed=3)
        b = fed_sim.partition(self.examples, 4, "noniid", seed=3)
        self.assertEqual([[e.prompt for e in s] for s in a], [[e.prompt for e in s] for s in b])

    def test_sft_target_strips_calculator_annotations_and_uses_boxed_format(self):
        t = fed_sim.sft_target("She sold 48/2 = <<48/2=24>>24 clips.\n#### 72")
        self.assertEqual(t, "She sold 48/2 = 24 clips.\nThe final answer is \\boxed{72}.")
        from bhaskera.rl.rewards import math_boxed
        self.assertEqual(math_boxed(t, "x #### 72"), 1.0)           # the SFT format is rewardable

    def test_backdoor_shard_and_reward(self):
        shard = fed_sim.backdoor_shard(self.examples[:200], seed=0)
        poisoned = [e for e in shard if e.answer.startswith("BACKDOOR:")]
        self.assertTrue(60 < len(poisoned) < 140)
        self.assertTrue(all(e.prompt.endswith(fed_sim.TRIGGER) for e in poisoned))
        self.assertEqual(fed_sim.backdoor_or_math_reward("so \\boxed{0}", "BACKDOOR:0"), 1.0)
        self.assertEqual(fed_sim.backdoor_or_math_reward("so \\boxed{7}", "BACKDOOR:0"), 0.0)
        self.assertEqual(fed_sim.backdoor_or_math_reward("so \\boxed{7}", "x #### 7"), 1.0)

    def test_softmax_weights_sum_to_one(self):
        w = fed_sim.softmax_weights({0: 0.3, 1: -1.0, 2: 2.0})
        self.assertAlmostEqual(sum(w.values()), 1.0, places=6)


@unittest.skipUnless(HAVE_STACK, "needs the Bhaskera environment")
class OomRetryTests(unittest.TestCase):
    def test_oom_is_retried_until_it_succeeds(self):
        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] < 3:
                raise RuntimeError("CUDA error: out of memory")
            return "ok"

        self.assertEqual(fed_sim.retry_oom(flaky, wait=0), "ok")
        self.assertEqual(calls["n"], 3)

    def test_oom_that_never_clears_eventually_raises(self):
        def always():
            raise RuntimeError("CUDA out of memory")
        with self.assertRaises(RuntimeError):
            fed_sim.retry_oom(always, tries=3, wait=0)

    def test_other_errors_are_not_swallowed_or_retried(self):
        calls = {"n": 0}

        def broken():
            calls["n"] += 1
            raise RuntimeError("shape mismatch")
        with self.assertRaises(RuntimeError):
            fed_sim.retry_oom(broken, wait=0)
        self.assertEqual(calls["n"], 1)

    def test_numeric_failure_classifier(self):
        self.assertTrue(fed_sim.is_numeric_failure(
            RuntimeError("probability tensor contains either `inf`, `nan` or element < 0")))
        self.assertFalse(fed_sim.is_numeric_failure(RuntimeError("CUDA out of memory")))


@unittest.skipUnless(HAVE_STACK, "needs the Bhaskera environment")
class EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
        cls.work = tempfile.mkdtemp(prefix="fedsim_")
        tok = Tokenizer(models.WordLevel({w: i for i, w in enumerate(WORDS)}, unk_token="<unk>"))
        tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>",
                                            eos_token="<eos>", unk_token="<unk>")
        cls.model_dir = os.path.join(cls.work, "model")
        torch.manual_seed(0)
        LlamaForCausalLM(LlamaConfig(
            vocab_size=len(WORDS), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=128,
            pad_token_id=0, eos_token_id=1, initializer_range=0.3)).save_pretrained(cls.model_dir)
        tokenizer.save_pretrained(cls.model_dir)
        for name, n in (("train", 600), ("test", 24)):
            with open(os.path.join(cls.work, f"{name}.jsonl"), "w") as f:
                for r in make_examples(n, seed=5 if name == "test" else 0):
                    f.write(json.dumps(r) + "\n")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.work, ignore_errors=True)

    def run_sim(self, name, **kw):
        out = os.path.join(self.work, f"{name}.json")
        argv = ["--out", out, "--model", self.model_dir, "--dtype", "float32",
                "--train", os.path.join(self.work, "train.jsonl"),
                "--test", os.path.join(self.work, "test.jsonl"),
                "--rounds", "2", "--steps", "2", "--prompts-per-step", "2", "--group-size", "4",
                "--max-new-tokens", "4", "--micro-batch", "8", "--eval-batch", "8",
                "--sft-batch", "8", "--n-nodes", "3"]
        for k, v in kw.items():
            argv += [f"--{k.replace('_', '-')}", str(v)]
        fed_sim.main(argv)
        with open(out) as f:
            return json.load(f)

    def test_every_method_runs_and_reports_per_problem_results(self):
        for method in ("base", "centralized", "fed_rl", "isolated_rl", "fed_sft"):
            r = self.run_sim(f"m_{method}", method=method)
            self.assertIn("mean_acc", r, method)
            self.assertEqual(len(r["final"][0]["per_problem"]), 24, method)
            self.assertTrue(all(x in (0.0, 1.0) for x in r["final"][0]["per_problem"]))

    def test_federation_sends_bytes_and_isolation_does_not(self):
        fed = self.run_sim("bytes_fed", method="fed_rl")
        iso = self.run_sim("bytes_iso", method="isolated_rl")
        self.assertGreater(fed["bytes_sent"], 0)
        self.assertEqual(iso["bytes_sent"], 0)
        self.assertEqual(len(fed["final"]), 3)

    def test_aggregation_changes_states_relative_to_isolation(self):
        fed = self.run_sim("agg_fed", method="fed_rl", seed=1)
        iso = self.run_sim("agg_iso", method="isolated_rl", seed=1)
        # federation pulls honest nodes together; isolated nodes drift apart
        self.assertGreater(iso["consensus_gap"], 0.0)
        self.assertLess(fed["consensus_gap"], iso["consensus_gap"])

    def test_oversized_attacker_is_rejected_with_defense_on(self):
        r = self.run_sim("over_on", method="fed_rl", attack="oversized", n_nodes=4,
                         malicious_frac=0.25, defense=1)
        total = {"malicious": 0, "honest": 0}
        for rd in r["rounds"]:
            for k in total:
                total[k] += rd["rejected"][k]
        self.assertGreater(total["malicious"], 0, r["rounds"])
        self.assertEqual(total["honest"], 0, "honest peers must not be rejected")
        self.assertEqual(r["roles"].count("oversized"), 1)

    def test_defense_off_rejects_nothing(self):
        r = self.run_sim("over_off", method="fed_rl", attack="oversized", n_nodes=4,
                         malicious_frac=0.25, defense=0)
        self.assertTrue(all(sum(rd["rejected"].values()) == 0 for rd in r["rounds"]))

    def test_signflip_and_backdoor_attacks_run_and_backdoor_reports_asr(self):
        r = self.run_sim("flip", method="fed_rl", attack="signflip", n_nodes=4, malicious_frac=0.5)
        self.assertEqual(r["roles"], ["honest", "honest", "signflip", "signflip"])
        r = self.run_sim("bd", method="fed_rl", attack="backdoor", n_nodes=4, malicious_frac=0.25)
        self.assertIn("asr", r["final"][0])
        self.assertTrue(0.0 <= r["final"][0]["asr"] <= 1.0)

    def test_malicious_nodes_are_never_evaluated_as_honest(self):
        r = self.run_sim("who", method="fed_rl", attack="signflip", n_nodes=4,
                         malicious_frac=0.5, eval_nodes=4)
        self.assertEqual([f["node"] for f in r["final"]], [0, 1])

    def test_existing_result_files_are_not_recomputed(self):
        out = os.path.join(self.work, "skip.json")
        with open(out, "w") as f:
            f.write('{"marker": 1}')
        fed_sim.main(["--out", out, "--method", "base", "--model", self.model_dir])
        with open(out) as f:
            self.assertEqual(json.load(f), {"marker": 1})


if __name__ == "__main__":
    unittest.main()
