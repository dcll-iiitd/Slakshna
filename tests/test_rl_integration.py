"""Post-training RL inside the Slakshna federated loop.

Runs the real ml_engine.py (not a stand-in) in `federated.mode: rl` on a tiny saved model:
  * round 1 trains with GRPO and emits a non-zero sparse delta,
  * round 2 resumes from the aggregated adapter and the sample counter,
  * a second node loads and aggregates the first node's RL delta.

Needs the Bhaskera environment (torch, peft, ray, transformers). Skipped otherwise.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BHASKERA_SRC = os.path.join(ROOT, "Bhaskera", "src")
sys.path.insert(0, ROOT)

from federated_communication.progress import REWARD_RE, TRAIN_PROGRESS_RE  # noqa: E402

try:
    import torch
    import transformers  # noqa: F401
    import peft  # noqa: F401
    import ray  # noqa: F401
    HAVE_STACK = os.path.isdir(os.path.join(BHASKERA_SRC, "bhaskera", "rl"))
except ImportError:
    HAVE_STACK = False

WORDS = ["<pad>", "<eos>", "<unk>", "Q", "A", "1", "2", "3",
         "\\boxed{3}", "\\boxed{5}", "\\boxed{7}"]
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class ProgressLineTests(unittest.TestCase):
    """ml_engine counts training steps with this regex; a miss means a silent zero delta."""

    def test_matches_sft_lines_unchanged(self):
        line = "[epoch 0][step 6] loss=2.2678 lr=3e-4 samples=8 tokens=16384"
        m = TRAIN_PROGRESS_RE.search(line)
        self.assertEqual((m.group(2), m.group(3), m.group(4), m.group(5)),
                         ("6", "2.2678", "8", "16384"))

    @unittest.skipUnless(HAVE_STACK, "Bhaskera RL package not available")
    def test_matches_what_the_rl_launcher_prints(self):
        sys.path.insert(0, BHASKERA_SRC)
        from bhaskera.launcher.rl_train import progress_line

        for loss in (-0.0123, 0.000037, 0.0, 1.5e-07, -2.0e-05):
            metrics = dict(loss=loss, reward_mean=0.25, kl=0.001, clip_frac=0.0,
                           grad_norm=0.2, completion_len=100.0)
            line = progress_line(3, 7, metrics, 896, 90000)
            m = TRAIN_PROGRESS_RE.search(line)
            self.assertIsNotNone(m, line)
            self.assertEqual((m.group(1), m.group(2), m.group(4), m.group(5)),
                             ("3", "7", "896", "90000"))
            self.assertAlmostEqual(float(m.group(3)), loss, places=6)
            self.assertEqual(float(REWARD_RE.search(line).group(1)), 0.25)

    def test_ignores_unrelated_lines(self):
        for line in ("loading model", "[eval][step 0] eval_reward=0.1875 examples=16",
                     "[epoch 0][step 1] reward=0.2"):
            self.assertIsNone(TRAIN_PROGRESS_RE.search(line))


@unittest.skipUnless(HAVE_STACK, "needs the Bhaskera environment (torch, peft, ray) and RL package")
class MlEngineRlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work = tempfile.mkdtemp(prefix="slakshna_rl_")
        cls._make_workspace()
        cls.a1 = cls._run_node("node-a", [], "data-a")
        cls.a2 = cls._run_node("node-a", [], "data-a")
        delta_dir = os.path.join(cls.work, "data-b", "network_deltas")
        os.makedirs(delta_dir)
        with open(os.path.join(delta_dir, "node-a_delta.b64"), "w") as f:
            f.write(cls.a2["json"]["compressed_delta"])
        cls.b1 = cls._run_node("node-b", ["node-a"], "data-b")

        # A poisoned node (learning rate 1.0 via MALICIOUS_NODES) sends its delta to an honest node.
        cls.m1 = cls._run_node("node-m", [], "data-m", extra_env={"MALICIOUS_NODES": "node-m"})
        delta_dir = os.path.join(cls.work, "data-c", "network_deltas")
        os.makedirs(delta_dir)
        with open(os.path.join(delta_dir, "node-m_delta.b64"), "w") as f:
            f.write(cls.m1["json"]["compressed_delta"])
        cls.c1 = cls._run_node("node-c", ["node-m"], "data-c")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.work, ignore_errors=True)

    # -- setup ---------------------------------------------------------------
    @classmethod
    def _make_workspace(cls):
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

        for name in ("ml_engine.py", "adapter_export.py"):   # 2nd file: inference branch only
            if os.path.exists(os.path.join(ROOT, name)):
                shutil.copy(os.path.join(ROOT, name), cls.work)
        shutil.copytree(os.path.join(ROOT, "federated_communication"),
                        os.path.join(cls.work, "federated_communication"),
                        ignore=shutil.ignore_patterns("__pycache__"))

        tok = Tokenizer(models.WordLevel({w: i for i, w in enumerate(WORDS)}, unk_token="<unk>"))
        tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>",
                                            eos_token="<eos>", unk_token="<unk>")
        model_dir = os.path.join(cls.work, "tiny_model")
        torch.manual_seed(0)
        LlamaForCausalLM(LlamaConfig(
            vocab_size=len(WORDS), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=64,
            pad_token_id=0, eos_token_id=1, initializer_range=0.3)).save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)

        with open(os.path.join(cls.work, "prompts.jsonl"), "w") as f:
            for prompt in ("Q 1", "Q 2", "Q 3"):
                f.write(json.dumps({"prompt": prompt, "answer": "7"}) + "\n")

        template = f"""
federated: {{mode: rl, optimizer: adamw, trust_beta: 0.1, total_epochs: 0}}
model: {{name: {model_dir}, dtype: float32, use_liger_kernel: false}}
data: {{name: local, seq_len: 64}}
lora: {{enabled: true, rank: 8, alpha: 16, dropout: 0.0, target_modules: {json.dumps(TARGETS)}}}
training: {{learning_rate: 3.0e-3, max_steps: 40, warmup_steps: 0, seed: 0, grad_clip: 1.0}}
checkpoint: {{save_directory: dummy, local_interval: 1, sync_interval: 1, keep_last_n: 2}}
rl:
  train_path: prompts.jsonl
  eval_path: prompts.jsonl
  reward: math_boxed
  group_size: 8
  prompts_per_step: 2
  max_new_tokens: 3
  kl_coef: 0.04
  micro_batch_size: 8
"""
        with open(os.path.join(cls.work, "node_template.yaml"), "w") as f:
            f.write(template)

    @classmethod
    def _run_node(cls, node_id, neighbors, data_dir, extra_env=None):
        env = dict(os.environ)
        env.update(extra_env or {})
        env.update(
            PYTHONPATH=BHASKERA_SRC + os.pathsep + env.get("PYTHONPATH", ""),
            IIITD_DATA_DIR=os.path.join(cls.work, data_dir),
            SLAKSHNA_COMPRESSION_ENABLED="true",
            SLAKSHNA_DELTA_SPARSITY="0.5",
            SLAKSHNA_NUM_GPUS="0",     # CPU-only test: do not ask Ray for a GPU
            CUDA_VISIBLE_DEVICES="",     # hermetic: tiny model on CPU
            WANDB_MODE="disabled",
        )
        proc = subprocess.run([sys.executable, "ml_engine.py", node_id, *neighbors],
                              cwd=cls.work, env=env, capture_output=True, text=True, timeout=600)
        if proc.returncode != 0:
            raise RuntimeError(f"ml_engine failed for {node_id}:\n{proc.stderr[-3000:]}")
        out = json.loads(proc.stdout.strip().splitlines()[-1])   # Rust parses this last line
        steps = [TRAIN_PROGRESS_RE.search(line) for line in proc.stderr.splitlines()]
        rewards = [float(REWARD_RE.search(line).group(1)) for line, m in
                   zip(proc.stderr.splitlines(), steps) if m]
        return {"json": out, "stderr": proc.stderr, "steps": [m for m in steps if m],
                "rewards": rewards}

    # -- round 1 -------------------------------------------------------------
    def test_round_one_trains_with_rl_and_reports_json_contract(self):
        out = self.a1["json"]
        for key in ("weights", "model_hash", "validation_score", "metadata",
                    "compressed_delta", "round"):
            self.assertIn(key, out)
        self.assertEqual(out["round"], 1)
        self.assertTrue(out["metadata"].startswith("Reward:"), out["metadata"])
        self.assertEqual(len(self.a1["steps"]), 40)           # every step was seen by ml_engine
        self.assertNotIn("No valid training steps", self.a1["stderr"])

    def test_policy_improved_during_the_round(self):
        r = self.a1["rewards"]
        self.assertGreater(sum(r[-10:]) / 10, sum(r[:10]) / 10 + 0.1, f"rewards {r}")

    def test_delta_is_nonzero_sparse_and_valid(self):
        from federated_communication import decode_delta_envelope

        delta = decode_delta_envelope(self.a1["json"]["compressed_delta"], torch.device("cpu"))
        total = sum(v.float().norm() ** 2 for v in delta.values()) ** 0.5
        self.assertGreater(float(total), 0.0)
        self.assertTrue(all(torch.isfinite(v).all() for v in delta.values()))
        self.assertTrue(any("lora_" in k for k in delta))

    # -- round 2: resume -----------------------------------------------------
    def test_round_two_resumes_adapter_and_sample_counter(self):
        self.assertEqual(self.a2["json"]["round"], 2)
        self.assertIn("Loading existing LoRA weights", self.a2["stderr"])
        first = self.a2["steps"][0]
        self.assertEqual(int(first.group(4)), 40 * 16 + 16)   # continues from round 1's samples
        self.assertNotIn("No valid training steps", self.a2["stderr"])
        # starting from the learned adapter, round 2 begins far above round 1's start
        self.assertGreater(sum(self.a2["rewards"][:10]) / 10, sum(self.a1["rewards"][:10]) / 10)

    # -- second node aggregates the first node's RL delta ----------------------
    def test_peer_node_verifies_and_aggregates_the_rl_delta(self):
        self.assertIn("Successfully verified and applied network delta from node-a",
                      self.b1["stderr"])
        weights = self.b1["json"]["weights"]
        self.assertEqual(set(weights), {"node-a", "node-b"})
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=5)
        self.assertEqual(self.b1["json"]["round"], 1)


    # -- a poisoned peer is rejected --------------------------------------------
    def test_poisoned_peer_delta_is_rejected_and_recorded(self):
        from federated_communication import decode_delta_envelope, delta_norm

        cpu = torch.device("cpu")
        honest = delta_norm(decode_delta_envelope(self.a1["json"]["compressed_delta"], cpu))
        poisoned = delta_norm(decode_delta_envelope(self.m1["json"]["compressed_delta"], cpu))
        self.assertGreater(poisoned, 5 * honest, "the attack should be far larger than honest")
        self.assertIn("SECURITY ALERT: Peer delta from node-m", self.c1["stderr"])
        self.assertNotIn("Successfully verified and applied network delta from node-m",
                         self.c1["stderr"])
        with open(os.path.join(self.work, "logs", "runtime_comm.log")) as f:
            self.assertIn("peer_delta_rejected", f.read())   # visible even though Rust drops stderr


if __name__ == "__main__":
    unittest.main()
