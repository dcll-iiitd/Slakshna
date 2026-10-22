import os
import subprocess
import sys
import tempfile
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from federated_communication import tokenized_cache as tc

MODEL = "org/model-a"


def make_cache(root, marker=None, parquet_bytes=500):
    sub = os.path.join(root, "abc123")
    os.makedirs(sub, exist_ok=True)
    with open(os.path.join(sub, "shard-0.parquet"), "wb") as f:
        f.write(b"x" * parquet_bytes)
    if marker is not None:
        tc.write_marker(root, marker)


class CacheStatusTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = os.path.join(self.tmp.name, "cache")

    def test_missing_when_absent_or_empty(self):
        self.assertEqual(tc.cache_status(self.root, MODEL), tc.MISSING)
        os.makedirs(self.root)
        self.assertEqual(tc.cache_status(self.root, MODEL), tc.MISSING)

    def test_ready_only_for_the_stamped_model(self):
        make_cache(self.root, marker=MODEL)
        self.assertEqual(tc.cache_status(self.root, MODEL), tc.READY)
        self.assertEqual(tc.cache_status(self.root, "org/model-b"), tc.WRONG_MODEL)

    def test_unmarked_legacy_cache_is_not_trusted(self):
        make_cache(self.root)
        self.assertEqual(tc.cache_status(self.root, MODEL), tc.UNMARKED)

    def test_truncated_parquet_is_corrupted(self):
        make_cache(self.root, marker=MODEL, parquet_bytes=8)
        self.assertEqual(tc.cache_status(self.root, MODEL), tc.CORRUPTED)

    def test_explanations_name_the_fix(self):
        make_cache(self.root, marker="org/model-b")
        msg = tc.explain(tc.WRONG_MODEL, self.root, MODEL)
        self.assertIn("org/model-b", msg)
        self.assertIn(MODEL, msg)
        self.assertIn("tokenize_dataset.py", msg)
        self.assertIn("tokenize_dataset.py", tc.explain(tc.MISSING, self.root, MODEL))


class ScriptTests(unittest.TestCase):
    def run_script(self, template_text, *args):
        with tempfile.TemporaryDirectory() as d:
            tpl = os.path.join(d, "t.yaml")
            with open(tpl, "w") as f:
                f.write(template_text)
            return subprocess.run(
                [sys.executable, "-W", "ignore", os.path.join(REPO, "scripts", "tokenize_dataset.py"), "--template", tpl, *args],
                capture_output=True, text=True, cwd=d,
            )

    def test_check_fails_when_missing_then_passes_when_stamped(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "tok").replace("\\", "/")
            tpl = f'model:\n  name: "{MODEL}"\ndata:\n  tokenized_path: "{out}"\n'
            r = self.run_script(tpl, "--check")
            self.assertEqual(r.returncode, 1)
            self.assertIn("tokenize_dataset.py", r.stderr)
            make_cache(out)  # parquet but no marker
            self.assertEqual(self.run_script(tpl, "--check").returncode, 1)
            r = self.run_script(tpl, "--stamp")
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(self.run_script(tpl, "--check").returncode, 0)

    def test_refuses_a_placeholder_output_path(self):
        r = self.run_script(f'model:\n  name: "{MODEL}"\ndata:\n  tokenized_path: "dummy"\n', "--check")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("tokenized_path", r.stderr)


if __name__ == "__main__":
    unittest.main()
