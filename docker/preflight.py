#!/usr/bin/env python
"""Fail fast, with a clear message, before the node starts.

Checks: config readable, writable dirs, training template + dataset, GPU visibility,
NVIDIA driver vs the CUDA version torch was built with, offline model cache.
Set SLAKSHNA_SKIP_PREFLIGHT=1 to bypass.
"""
import os
import sys
import tomllib

# Minimum Linux driver for each CUDA runtime that torch wheels may bundle.
MIN_DRIVER = {
    "11.8": (520, 61, 5),
    "12.1": (525, 60, 13),
    "12.4": (550, 54, 14),
}

errors = []
warnings = []


def fail(msg):
    errors.append(msg)


def warn(msg):
    warnings.append(msg)


def parse_version(s):
    return tuple(int(p) for p in s.split(".") if p.isdigit())


def writable_dir(path):
    try:
        os.makedirs(path, exist_ok=True)
        probe = os.path.join(path, ".preflight_probe")
        with open(probe, "w"):
            pass
        os.remove(probe)
        return True
    except OSError:
        return False


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("SLAKSHNA_CONFIG", "/app/config.toml")

    # 1. node config
    cfg = {}
    if not os.path.isfile(config_path):
        fail(f"Node config not found: {config_path} (mount your node .toml there, or set SLAKSHNA_CONFIG)")
    else:
        try:
            with open(config_path, "rb") as f:
                cfg = tomllib.load(f)
        except Exception as e:  # noqa: BLE001
            fail(f"Node config {config_path} is not valid TOML: {e}")

    node = cfg.get("node", {})
    data_dir = node.get("data_dir", "")
    num_gpus = int(node.get("num_gpus") or 1)

    # 2. writable directories (these are the volume mount points)
    dirs = ["/app/ml_models", "/app/ml_states", "/app/data", "/app/logs", "/app/hf_cache", "/app/cache"]
    if data_dir:
        dirs.append(data_dir if os.path.isabs(data_dir) else os.path.join("/app", data_dir))
    for d in dirs:
        if not writable_dir(d):
            fail(f"Directory not writable by uid {os.getuid()}: {d} "
                 f"(volume ownership problem? build the image with --build-arg UID=$(id -u))")

    # 3. training template + dataset
    template = "/app/node_template.yaml"
    tcfg = {}
    try:
        import yaml
        with open(template) as f:
            tcfg = yaml.safe_load(f) or {}
    except Exception as e:  # noqa: BLE001
        fail(f"Cannot read training template {template}: {e}")

    data = tcfg.get("data", {}) or {}
    train_path = data.get("train_path")
    if data.get("name") == "local" and train_path and not os.path.exists(train_path):
        fail(f"Dataset not found: data.train_path = {train_path} "
             f"(mount your dataset read-only at that path)")

    # 4. offline mode needs the model already in hf_cache
    model = (tcfg.get("model", {}) or {}).get("name", "")
    if os.environ.get("HF_HUB_OFFLINE") == "1" and model and not os.path.isdir(model):
        cached = os.path.join("/app/hf_cache", "hub", "models--" + model.replace("/", "--"))
        if not os.path.isdir(cached):
            fail(f"HF_HUB_OFFLINE=1 but model '{model}' is not in {cached}. "
                 f"Run the model-init step once with network access.")

    # 5. GPU + driver
    try:
        import torch
        cuda_build = torch.version.cuda
        if not torch.cuda.is_available():
            fail("CUDA is not available inside the container. Is the NVIDIA runtime enabled "
                 "(compose 'deploy.resources.reservations.devices' / --gpus)?")
        else:
            n = torch.cuda.device_count()
            if n < num_gpus:
                fail(f"Config asks for num_gpus={num_gpus} but only {n} GPU(s) are visible.")
            names = ", ".join(torch.cuda.get_device_name(i) for i in range(n))
            print(f"[preflight] torch {torch.__version__} (CUDA {cuda_build}) | GPUs visible: {n} [{names}]")

        driver = None
        try:
            import pynvml
            pynvml.nvmlInit()
            raw = pynvml.nvmlSystemGetDriverVersion()
            driver = raw.decode() if isinstance(raw, bytes) else raw
        except Exception:  # noqa: BLE001
            pass
        if driver and cuda_build in MIN_DRIVER:
            need = MIN_DRIVER[cuda_build]
            if parse_version(driver) < need:
                fail(f"NVIDIA driver {driver} is too old for CUDA {cuda_build}; "
                     f"need >= {'.'.join(map(str, need))}.")
            else:
                print(f"[preflight] NVIDIA driver {driver} OK (CUDA {cuda_build} needs >= "
                      f"{'.'.join(map(str, need))})")
    except ImportError as e:
        fail(f"torch is not importable: {e}")

    if os.path.exists("/opt/venv/.flash_attn_missing"):
        warn("flash-attn wheel was not installed at build time (only matters if attn_impl: flash_attention_2).")

    for w in warnings:
        print(f"[preflight] WARNING: {w}", file=sys.stderr)
    if errors:
        print("\n[preflight] FAILED - the node was NOT started:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        print("\nSet SLAKSHNA_SKIP_PREFLIGHT=1 to bypass (not recommended).", file=sys.stderr)
        sys.exit(1)
    print("[preflight] all checks passed")


if __name__ == "__main__":
    main()
