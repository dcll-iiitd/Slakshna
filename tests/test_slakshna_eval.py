"""Exactness of the held-out metrics (loss, perplexity, bits per byte) used in the paper."""

import json
import math
import os
import sys
import tempfile
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "Bhaskera", "src"))

try:
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
    import slakshna_eval as ev
    TOK = AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
    HAVE = True
except Exception:       # missing torch/transformers or the cached gpt2 tokenizer
    HAVE = False

ROWS = [
    ("What is the capital of Nepal?", "Kathmandu"),
    ("Is Dhaka in Bangladesh?", "Yes"),
    ("Name the longest river in Pakistan and say a bit about it.",
     "The Indus is the longest river in Pakistan and flows from the Tibetan plateau to the Arabian Sea."),
    ("A or B?", "A"),
]


def make_rows(extra=0):
    rows = []
    for i in range(len(ROWS) + extra):
        q, a = ROWS[i % len(ROWS)]
        rows.append({"messages": [{"role": "system", "content": "You are a helpful AI assistant"},
                                  {"role": "user", "content": q + (" " * (i // len(ROWS)))},
                                  {"role": "assistant", "content": a}]})
    return rows


def tiny_model(zero_head=False, seed=0):
    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=len(TOK), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=512)
    m = LlamaForCausalLM(cfg)
    if zero_head:
        torch.nn.init.zeros_(m.lm_head.weight)
        m.config.tie_word_embeddings = False
    return m.eval()


@unittest.skipUnless(HAVE, "needs torch, transformers and the cached gpt2 tokenizer")
class ExactMetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp()
        cls.path = os.path.join(cls.dir, "val.jsonl")
        with open(cls.path, "w") as f:
            for r in make_rows(extra=4):
                f.write(json.dumps(r) + "\n")
        cls.examples = ev.load_examples(TOK, cls.path)

    def test_uniform_model_has_nll_ln_vocab_and_perplexity_vocab(self):
        """Known answer: a model predicting uniformly over V tokens has NLL ln V and perplexity V."""
        m = tiny_model(zero_head=True)
        r = ev.evaluate(m, self.examples, pad_token_id=0, batch_size=4)
        self.assertAlmostEqual(r["nll"], math.log(len(TOK)), places=4)
        self.assertAlmostEqual(r["perplexity"], len(TOK), delta=len(TOK) * 1e-3)

    def test_result_does_not_depend_on_batch_size_or_padding(self):
        m = tiny_model()
        results = [ev.evaluate(m, self.examples, 0, batch_size=b) for b in (1, 3, 8)]
        for r in results[1:]:
            self.assertAlmostEqual(r["nll"], results[0]["nll"], places=5)
            self.assertEqual(r["tokens"], results[0]["tokens"])

    def test_matches_a_naive_one_example_at_a_time_computation(self):
        m = tiny_model()
        total, count = 0.0, 0
        for e in self.examples:
            ids = torch.tensor([e["ids"]])
            logits = m(input_ids=ids).logits
            labels = torch.tensor([e["labels"]])
            total += F.cross_entropy(logits[0, :-1].float(), labels[0, 1:], ignore_index=-100,
                                     reduction="sum").item()
            count += int((labels[0, 1:] != -100).sum())
        r = ev.evaluate(m, self.examples, 0, batch_size=4)
        self.assertAlmostEqual(r["nll"], total / count, places=5)
        self.assertEqual(r["tokens"], count)

    def test_is_token_weighted_not_a_mean_of_example_means(self):
        """Examples with more scored tokens must count for more."""
        m = tiny_model()
        per_example = [ev.evaluate(m, [e], 0)["nll"] for e in self.examples]
        mean_of_means = sum(per_example) / len(per_example)
        r = ev.evaluate(m, self.examples, 0)
        weights = [e["n_scored"] for e in self.examples]
        expected = sum(w * x for w, x in zip(weights, per_example)) / sum(weights)
        self.assertAlmostEqual(r["nll"], expected, places=5)
        self.assertGreater(len(set(weights)), 2)
        self.assertGreater(abs(r["nll"] - mean_of_means), 1e-4)       # the two definitions differ here

    def test_only_assistant_tokens_are_scored(self):
        ids, labels = ev.tokenize_chat(TOK, make_rows()[0]["messages"])
        scored = [t for t in labels if t != -100]
        text = TOK.decode(scored)
        self.assertIn("Kathmandu", text)
        self.assertNotIn("capital", text)             # user turn is context only
        self.assertNotIn("helpful", text)             # system prompt is context only
        self.assertTrue(all(lab in (-100, tok) for lab, tok in zip(labels, ids)))

    def test_perplexity_is_exp_of_nll_and_bits_per_byte_formula(self):
        r = ev.evaluate(tiny_model(), self.examples, 0)
        self.assertAlmostEqual(r["perplexity"], math.exp(r["nll"]), places=6)
        total_nll = r["nll"] * r["tokens"]
        self.assertAlmostEqual(r["bits_per_byte"], total_nll / (math.log(2) * r["bytes"]), places=8)
        manual_bytes = sum(len(TOK.decode([t for t in e["labels"][1:] if t != -100]).encode()) for e in self.examples)
        self.assertEqual(r["bytes"], manual_bytes)

    def test_better_model_scores_better(self):
        """Fitting the evaluation data must lower loss and perplexity (the metric moves the right way)."""
        m = tiny_model(seed=1)
        before = ev.evaluate(m, self.examples, 0)
        opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
        m.train()
        for _ in range(40):
            for e in self.examples:
                ids = torch.tensor([e["ids"]])
                loss = m(input_ids=ids, labels=torch.tensor([e["labels"]])).loss
                opt.zero_grad()
                loss.backward()
                opt.step()
        after = ev.evaluate(m, self.examples, 0)
        self.assertLess(after["nll"], before["nll"] - 0.5)
        self.assertLess(after["perplexity"], before["perplexity"])

    def test_truncation_matches_training_and_empty_sets_fail_loudly(self):
        short = ev.load_examples(TOK, self.path, max_len=12)
        self.assertTrue(all(len(e["ids"]) <= 12 for e in short))
        with self.assertRaises(ValueError):
            ev.evaluate(tiny_model(), [], 0)


if __name__ == "__main__":
    unittest.main()
