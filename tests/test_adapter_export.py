import hashlib
import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
    import safetensors  # noqa: F401
    HAVE_DEPS = True
except ImportError:
    HAVE_DEPS = False


@unittest.skipUnless(HAVE_DEPS, "torch/safetensors not installed")
class AdapterExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        torch.manual_seed(0)
        self.sd = {
            "base_model.model.layers.0.q_proj.lora_A.default.weight": torch.randn(4, 16),
            "base_model.model.layers.0.q_proj.lora_B.default.weight": torch.randn(16, 4),
        }

    def _export(self, sd, round_num=1):
        from adapter_export import export_adapter
        return export_adapter(sd, self.tmp, "slakshna1abc", round_num, "base/model")

    def test_hash_matches_file_and_dir_is_content_addressed(self):
        info = self._export(self.sd)
        weights = os.path.join(info["adapter_path"], "adapter_model.safetensors")
        with open(weights, "rb") as f:
            self.assertEqual(hashlib.sha256(f.read()).hexdigest(), info["adapter_hash"])
        self.assertEqual(os.path.basename(info["adapter_path"]), info["adapter_hash"])
        self.assertEqual(os.path.getsize(weights), info["adapter_size"])
        self.assertEqual(info["adapter_rank"], 4)
        self.assertEqual(json.loads(info["adapter_config"])["peft_type"], "LORA")

    def test_same_weights_same_hash_across_rounds(self):
        a = self._export(self.sd, 1)
        b = self._export(self.sd, 2)
        self.assertEqual(a["adapter_hash"], b["adapter_hash"])

    def test_different_weights_different_hash(self):
        other = {k: v + 1 for k, v in self.sd.items()}
        self.assertNotEqual(self._export(self.sd)["adapter_hash"], self._export(other)["adapter_hash"])


if __name__ == "__main__":
    unittest.main()
