"""
Tokenize the training dataset once, before federated training starts.

ml_engine.py does not tokenize: it only reads the result. Run this from the repo
root inside the Bhaskera training environment (``source Bhaskera/bhaskera-activate.sh``):

    python scripts/tokenize_dataset.py             # tokenize if needed
    python scripts/tokenize_dataset.py --check     # exit 0 only if training can start
    python scripts/tokenize_dataset.py --force     # re-tokenize; the old directory is moved aside
    python scripts/tokenize_dataset.py --stamp     # mark an existing cache as built for model.name

It reads ``node_template.yaml`` (the same file training uses) so the tokenizer,
sequence length and data format match training exactly, and stamps the output with
the model name so a cache built for another model can never be reused by mistake.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time

import yaml

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_DIR)

from federated_communication import tokenized_cache as tc  # noqa: E402


def load_settings(template_path: str, output_override: str | None) -> tuple[dict, str, str]:
    with open(template_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    model_name = config.get("model", {}).get("name", "")
    out = output_override or config.get("data", {}).get("tokenized_path") or ""
    if not model_name:
        raise SystemExit(f"{template_path}: model.name is not set")
    if not out or out == "dummy":
        raise SystemExit(
            f"{template_path}: data.tokenized_path must be a real directory that every node can read "
            f"(got {out!r}). Set it, or pass --output."
        )
    return config, model_name, out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--template", default=os.path.join(REPO_DIR, "node_template.yaml"))
    ap.add_argument("--output", help="override data.tokenized_path")
    ap.add_argument("--check", action="store_true", help="only report whether training can start")
    ap.add_argument("--force", action="store_true", help="re-tokenize, moving any existing directory aside")
    ap.add_argument("--stamp", action="store_true", help="mark an existing cache as built for model.name")
    args = ap.parse_args()

    config, model_name, out = load_settings(args.template, args.output)
    status = tc.cache_status(out, model_name)

    if args.check:
        print(f"{out}: {status}" if status == tc.READY else tc.explain(status, out, model_name), file=sys.stderr)
        return 0 if status == tc.READY else 1

    if args.stamp:
        if status not in (tc.UNMARKED, tc.WRONG_MODEL, tc.READY):
            print(tc.explain(status, out, model_name), file=sys.stderr)
            return 1
        tc.write_marker(out, model_name)
        print(f"Stamped {out} as built for {model_name}")
        return 0

    if status == tc.READY and not args.force:
        print(f"Already tokenized for {model_name} at {out}; nothing to do (use --force to redo).")
        return 0
    if status != tc.MISSING and not args.force:
        print(tc.explain(status, out, model_name), file=sys.stderr)
        return 1
    if os.path.exists(out) and os.listdir(out):
        stamp = time.strftime("%Y%m%d-%H%M%S")
        backup = out.rstrip("/\\") + ".bak-" + stamp
        shutil.move(out, backup)
        print(f"Moved the existing directory to {backup}")

    os.makedirs(out, exist_ok=True)
    config.setdefault("data", {})["tokenized_path"] = out
    config["data"]["cache_dir"] = out  # Bhaskera's tokenizer writes under cache_dir
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as tmp:
        yaml.dump(config, tmp)
        tmp_path = tmp.name
    try:
        print(f"Tokenizing with the {model_name} tokenizer into {out} ...")
        subprocess.run([sys.executable, "-m", "bhaskera.launcher.tokenize", "--config", tmp_path], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Tokenization failed: {e}. Is the Bhaskera environment active?", file=sys.stderr)
        return 1
    finally:
        os.unlink(tmp_path)

    after = tc.cache_status(out, model_name)
    if after not in (tc.UNMARKED, tc.READY):  # freshly written, so not yet stamped
        print(f"Tokenizer finished but the output is {after}: no usable parquet files under {out}", file=sys.stderr)
        return 1
    tc.write_marker(out, model_name)
    print(f"Done. Training can start: {out} is stamped for {model_name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
