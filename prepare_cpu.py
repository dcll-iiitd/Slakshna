#!/usr/bin/env python
"""
One-time CPU preparation (run before starting a cluster):

  1. downloads the model + tokenizer from Hugging Face into ./hf_cache
  2. tokenizes the dataset once into the cache shared by all nodes (./data_cpu_cache)

Doing this up front means N nodes started at the same instant do not all try to
download / tokenize concurrently. ml_engine.py would do both lazily (under a lock)
anyway, so this is an optimisation and a way to fail early on network problems.

    python prepare_cpu.py                       # uses node_template.cpu.yaml
    python prepare_cpu.py --template my.yaml
    python prepare_cpu.py --smoke               # also run 2 training steps to prove the stack works
"""
import argparse
import os
import subprocess
import sys

import yaml

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Must match ml_engine.py so the nodes find what we download here.
os.environ["HF_HOME"] = os.path.join(BASE_DIR, "hf_cache")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("PYTHONUTF8", "1")


def _abs(p):
    return p if (not p or os.path.isabs(p)) else os.path.abspath(os.path.join(BASE_DIR, p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", default=os.path.join(BASE_DIR, "node_template.cpu.yaml"))
    ap.add_argument("--smoke", action="store_true", help="run 2 tiny training steps afterwards")
    args = ap.parse_args()

    with open(args.template) as f:
        cfg = yaml.safe_load(f)

    model = cfg["model"]["name"]
    data = cfg["data"]
    for k in ("train_path", "val_path", "path", "tokenized_path"):
        if data.get(k):
            data[k] = _abs(data[k])
    cache_dir = data["tokenized_path"]
    data["cache_dir"] = cache_dir
    os.makedirs(cache_dir, exist_ok=True)

    print(f"[1/2] Downloading {model} ...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    AutoTokenizer.from_pretrained(model, trust_remote_code=cfg["model"].get("trust_remote_code", False))
    AutoModelForCausalLM.from_pretrained(model, trust_remote_code=cfg["model"].get("trust_remote_code", False))

    print(f"[2/2] Tokenizing {data['train_path']} -> {cache_dir}")
    cfg_path = os.path.join(cache_dir, "prepare_config.yaml")
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f)
    subprocess.run(
        [sys.executable, "-m", "bhaskera.launcher.tokenize_cpu", "--config", cfg_path],
        check=True,
    )

    if args.smoke:
        print("[smoke] 2 training steps ...")
        sub = [d.path for d in os.scandir(cache_dir) if d.is_dir()]
        cfg["data"]["tokenized_path"] = sub[0]
        cfg["training"]["max_steps"] = 2
        cfg["checkpoint"] = {"enabled": False, "save_dir": os.path.join(cache_dir, "smoke_ckpt")}
        smoke_path = os.path.join(cache_dir, "smoke_config.yaml")
        with open(smoke_path, "w") as f:
            yaml.dump(cfg, f)
        subprocess.run(
            [sys.executable, "-m", "bhaskera.launcher.train_cpu", "--config", smoke_path],
            check=True,
        )
    print("Done. You can now start a cluster: see CPU_QUICKSTART.md")


if __name__ == "__main__":
    main()
