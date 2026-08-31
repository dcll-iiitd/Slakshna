#!/usr/bin/env python3
"""
prepare_sync_round_adapter.py

Converts Slakshna federated learning checkpoints (`sync_round_*.pth` or `*_base_lora.pth` in `dcll/Slakshna/ml_models`)
into standard Hugging Face PEFT LoRA adapter directories ready for evaluation with Tapestry Monash GOQA
(https://github.com/yuriak/tapestry_monash/tree/main/shared_evaluation/GOQA).

Features:
- Handles single .pth files, sync_ckpt_* directories, or the root `dcll/Slakshna/ml_models` directory.
- Normalizes tensor keys to standard PEFT format (strips module prefixes, normalizes adapter naming).
- Auto-detects LoRA rank (r), target modules (q_proj, v_proj), and configuration.
- Generates `adapter_model.safetensors`, `adapter_config.json`, and `adapter_meta.json`.
- Supports selecting specific rounds (--round <N>, --round latest, --round all) and nodes (--node <id>).
- Optionally launches Tapestry Monash GOQA evaluation immediately (--evaluate).

Usage Examples:
    # 1. Convert latest round from dcll/Slakshna/ml_models:
    python prepare_sync_round_adapter.py --input /mnt/disk1/slakshna/dcll/Slakshna/ml_models

    # 2. Convert a specific round (e.g. round 15) from a specific sync folder:
    python prepare_sync_round_adapter.py \
        --input /mnt/disk1/slakshna/dcll/Slakshna/ml_models/sync_ckpt_slakshna1dp30me5pdds4wf595eat8h0e65mt99jgy39hmt \
        --round 15 \
        --output-dir /mnt/disk1/slakshna/eval_adapters/round_15

    # 3. Convert all rounds from all nodes in dcll/Slakshna/ml_models:
    python prepare_sync_round_adapter.py \
        --input /mnt/disk1/slakshna/dcll/Slakshna/ml_models \
        --round all \
        --output-dir /mnt/disk1/slakshna/eval_adapters

    # 4. Convert and immediately run GOQA evaluation:
    python prepare_sync_round_adapter.py \
        --input /mnt/disk1/slakshna/dcll/Slakshna/ml_models/sync_ckpt_slakshna1dp30me5pdds4wf595eat8h0e65mt99jgy39hmt/sync_round_15.pth \
        --evaluate
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    import torch
except ImportError:
    print("Error: PyTorch is required. Install torch or run in an environment with torch.", file=sys.stderr)
    sys.exit(1)

try:
    from safetensors.torch import save_file as safetensors_save_file
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False


def sha256_file(path: Path) -> str:
    """Computes SHA-256 hash of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def normalize_lora_keys(state_dict: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], int, list[str]]:
    """
    Normalizes state dict keys to standard PEFT format:
    - Strips 'module.' prefix if present.
    - Normalizes 'lora_A.default.weight' -> 'lora_A.weight'.
    - Auto-detects LoRA rank and target modules.
    """
    cleaned_sd: dict[str, torch.Tensor] = {}
    target_modules_set: set[str] = set()
    detected_ranks: set[int] = set()

    for key, tensor in state_dict.items():
        if not torch.is_tensor(tensor):
            continue

        clean_k = key
        if clean_k.startswith("module."):
            clean_k = clean_k[len("module."):]

        # Normalize adapter name: e.g. lora_A.default.weight -> lora_A.weight
        clean_k = re.sub(r"(lora_[AB])\.[^.]+\.(weight)", r"\1.\2", clean_k)

        # Detect rank and target module from lora_A / lora_B
        if "lora_A" in clean_k:
            detected_ranks.add(tensor.shape[0])
            parts = clean_k.split(".")
            for i, p in enumerate(parts):
                if p == "lora_A" and i > 0:
                    target_modules_set.add(parts[i - 1])
        elif "lora_B" in clean_k:
            detected_ranks.add(tensor.shape[1])
            parts = clean_k.split(".")
            for i, p in enumerate(parts):
                if p == "lora_B" and i > 0:
                    target_modules_set.add(parts[i - 1])

        cleaned_sd[clean_k] = tensor

    if not cleaned_sd:
        raise ValueError("No tensor weights found in state dict!")

    detected_rank = max(detected_ranks) if detected_ranks else 16
    target_modules = sorted(list(target_modules_set)) if target_modules_set else ["q_proj", "v_proj"]

    return cleaned_sd, detected_rank, target_modules


def convert_sync_round_file(
    pth_path: Path,
    output_dir: Path,
    base_model: str = "allenai/OLMo-2-1124-7B",
    rank: int | None = None,
    alpha: int | None = None,
    dropout: float = 0.03,
    target_modules: list[str] | None = None,
    dtype_str: str = "bfloat16",
    save_bin: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Converts a single .pth file to a complete Hugging Face PEFT LoRA adapter directory.
    """
    pth_path = pth_path.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not pth_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {pth_path}")

    if verbose:
        print(f"[*] Loading checkpoint: {pth_path}")

    try:
        raw_data = torch.load(pth_path, map_location="cpu", weights_only=True)
    except Exception:
        raw_data = torch.load(pth_path, map_location="cpu", weights_only=False)

    if isinstance(raw_data, dict):
        if "state_dict" in raw_data:
            state_dict = raw_data["state_dict"]
        elif "lora_state_dict" in raw_data:
            state_dict = raw_data["lora_state_dict"]
        elif "model_state_dict" in raw_data:
            state_dict = raw_data["model_state_dict"]
        else:
            state_dict = raw_data
    else:
        raise TypeError(f"Loaded object from {pth_path} is not a dictionary: {type(raw_data)}")

    cleaned_sd, auto_rank, auto_target_modules = normalize_lora_keys(state_dict)

    final_rank = rank if rank is not None else auto_rank
    final_target_modules = target_modules if target_modules is not None else auto_target_modules
    final_alpha = alpha if alpha is not None else (4 * final_rank if final_rank > 0 else 64)

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "auto": None,
    }
    target_dtype = dtype_map.get(dtype_str.lower(), torch.bfloat16)

    final_tensors: dict[str, torch.Tensor] = {}
    total_parameters = 0
    for k, v in cleaned_sd.items():
        t = v.detach().cpu()
        if target_dtype is not None:
            t = t.to(target_dtype)
        t = t.contiguous()
        final_tensors[k] = t
        total_parameters += t.numel()

    # 1. Save adapter_model.safetensors
    safetensors_path = output_dir / "adapter_model.safetensors"
    if HAS_SAFETENSORS:
        safetensors_save_file(final_tensors, str(safetensors_path))
        if verbose:
            print(f"[+] Saved safetensors weights ({len(final_tensors)} tensors, {total_parameters:,} params) -> {safetensors_path}")
    else:
        print("[!] Warning: safetensors not installed; saving PyTorch bin format instead.", file=sys.stderr)

    # 2. Optionally save adapter_model.bin
    bin_path = output_dir / "adapter_model.bin"
    if save_bin or not HAS_SAFETENSORS:
        torch.save(final_tensors, str(bin_path))
        if verbose:
            print(f"[+] Saved PyTorch bin weights -> {bin_path}")

    # 3. Create adapter_config.json (standard PEFT LoRA schema)
    adapter_config: dict[str, Any] = {
        "base_model_name_or_path": base_model,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "lora_alpha": final_alpha,
        "lora_dropout": dropout,
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": final_rank,
        "target_modules": final_target_modules,
        "task_type": "CAUSAL_LM",
        "use_dora": False,
        "use_rslora": False,
    }
    config_path = output_dir / "adapter_config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(adapter_config, f, indent=2)
    if verbose:
        print(f"[+] Saved adapter config -> {config_path}")

    # 4. Create metadata audit file
    meta_info: dict[str, Any] = {
        "source_checkpoint": str(pth_path),
        "source_sha256": sha256_file(pth_path),
        "adapter_model_safetensors_sha256": sha256_file(safetensors_path) if safetensors_path.is_file() else None,
        "created_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "tensor_count": len(final_tensors),
        "parameter_count": total_parameters,
        "rank": final_rank,
        "alpha": final_alpha,
        "target_modules": final_target_modules,
        "base_model": base_model,
        "dtype": dtype_str,
    }
    meta_path = output_dir / "adapter_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta_info, f, indent=2)
    if verbose:
        print(f"[+] Saved adapter metadata -> {meta_path}")

    return {
        "output_dir": output_dir,
        "config_path": config_path,
        "weights_path": safetensors_path if safetensors_path.is_file() else bin_path,
        "metadata": meta_info,
    }


def find_sync_round_files_in_dir(dir_path: Path) -> list[tuple[int, Path]]:
    """
    Finds all sync_round_*.pth files in a single directory and returns sorted (round_number, path) pairs.
    """
    results: list[tuple[int, Path]] = []
    pattern = re.compile(r"^sync_round_(\d+)\.pth$")
    for item in dir_path.iterdir():
        if item.is_file():
            match = pattern.match(item.name)
            if match:
                round_num = int(match.group(1))
                results.append((round_num, item))
    results.sort(key=lambda x: x[0])
    return results


def discover_checkpoints(
    input_path: Path,
    round_spec: str | None = None,
    node_filter: str | None = None,
) -> list[tuple[str, int, Path]]:
    """
    Discovers checkpoint files from:
    - A single .pth file
    - A sync_ckpt_<node> directory
    - The root ml_models directory containing multiple sync_ckpt_* directories

    Returns list of (node_name, round_number, file_path) tuples.
    """
    input_path = input_path.resolve()
    discovered: list[tuple[str, int, Path]] = []

    if input_path.is_file():
        # Single file
        match = re.search(r"sync_round_(\d+)\.pth", input_path.name)
        r_num = int(match.group(1)) if match else 0
        node_name = input_path.parent.name if input_path.parent.name.startswith("sync_ckpt_") else "node"
        return [(node_name, r_num, input_path)]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    # Check if input_path itself is a sync_ckpt_* directory
    if input_path.name.startswith("sync_ckpt_") or any(f.name.startswith("sync_round_") for f in input_path.glob("*.pth")):
        sync_dirs = [input_path]
    else:
        # Look for sync_ckpt_* directories inside input_path
        sync_dirs = sorted([d for d in input_path.glob("sync_ckpt_*") if d.is_dir()])
        if not sync_dirs:
            # Check for any .pth files directly in input_path
            direct_pth = list(input_path.glob("*.pth"))
            if direct_pth:
                sync_dirs = [input_path]

    if not sync_dirs:
        raise FileNotFoundError(f"No sync checkpoint directories or .pth files found in {input_path}")

    for s_dir in sync_dirs:
        node_id = s_dir.name
        if node_filter and node_filter not in node_id:
            continue

        round_files = find_sync_round_files_in_dir(s_dir)
        if not round_files:
            # Check for non-round .pth files (e.g. *_base_lora.pth)
            for pth in s_dir.glob("*.pth"):
                round_files.append((0, pth))

        if not round_files:
            continue

        req_round = (round_spec or "latest").strip().lower()
        if req_round == "all":
            selected = round_files
        elif req_round == "latest":
            selected = [round_files[-1]]
        else:
            try:
                target_r = int(req_round)
                selected = [rf for rf in round_files if rf[0] == target_r]
                if not selected:
                    avail = [rf[0] for rf in round_files]
                    print(f"[!] Warning: Round {target_r} not found in {s_dir}. Available: {avail}", file=sys.stderr)
            except ValueError:
                selected = [round_files[-1]]

        for r_num, pth_path in selected:
            discovered.append((node_id, r_num, pth_path))

    return discovered


def run_monash_evaluation(
    adapter_dir: Path,
    base_model: str,
    eval_output_dir: Path,
    eval_script: Path | None = None,
    python_bin: str | None = None,
) -> int:
    """
    Runs the Tapestry Monash GOQA evaluation script.
    """
    search_paths = [
        Path("/mnt/disk1/slakshna/tapestry_monash/shared_evaluation/GOQA/run_evaluation.sh"),
        Path(__file__).resolve().parent.parent.parent / "tapestry_monash/shared_evaluation/GOQA/run_evaluation.sh",
        Path(__file__).resolve().parent.parent.parent.parent / "tapestry_monash/shared_evaluation/GOQA/run_evaluation.sh",
    ]

    if eval_script is None:
        for p in search_paths:
            if p.is_file():
                eval_script = p
                break

    if eval_script is None or not eval_script.is_file():
        raise FileNotFoundError(f"Evaluation script run_evaluation.sh not found (searched: {search_paths})")

    eval_output_dir = eval_output_dir.resolve()
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    if python_bin is None:
        goqa_venv = eval_script.parent / "goqa_env/bin/python"
        if goqa_venv.is_file():
            python_bin = str(goqa_venv)
        else:
            python_bin = sys.executable

    env = os.environ.copy()
    env["GOQA_PYTHON"] = python_bin

    cmd = [
        "bash",
        str(eval_script),
        base_model,
        str(eval_output_dir),
        str(adapter_dir),
    ]

    print("\n" + "=" * 70)
    print(f"[*] LAUNCHING TAPESTRY MONASH EVALUATION")
    print(f"[*] Script:      {eval_script}")
    print(f"[*] Python:      {python_bin}")
    print(f"[*] Base Model:  {base_model}")
    print(f"[*] Adapter Dir: {adapter_dir}")
    print(f"[*] Output Dir:  {eval_output_dir}")
    print("=" * 70 + "\n")

    res = subprocess.run(cmd, env=env)
    return res.returncode


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare a directory for sync_round.pth LoRA checkpoint for Tapestry Monash evaluation."
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        type=Path,
        help="Path to sync_round_*.pth file, sync_ckpt_* directory, or dcll/Slakshna/ml_models root.",
    )
    parser.add_argument(
        "-i", "--input",
        dest="input_flag",
        type=Path,
        help="Alternative flag to specify input path.",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        help="Target directory to create for the adapter (or root output directory if converting multiple rounds/nodes).",
    )
    parser.add_argument(
        "-r", "--round",
        default=None,
        help="Specify round number (e.g. '1', '15', 'latest', or 'all'). Default: 'latest'.",
    )
    parser.add_argument(
        "-n", "--node",
        default=None,
        help="Filter by node ID substring when input contains multiple sync_ckpt directories.",
    )
    parser.add_argument(
        "-m", "--base-model",
        default="allenai/OLMo-2-1124-7B",
        help="Hugging Face model ID or path for base model (default: allenai/OLMo-2-1124-7B).",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=None,
        help="LoRA rank override (default: auto-detected from tensor shapes, e.g. 16).",
    )
    parser.add_argument(
        "--alpha",
        type=int,
        default=None,
        help="LoRA alpha scaling factor (default: 4 * rank, or 64).",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.03,
        help="LoRA dropout rate (default: 0.03).",
    )
    parser.add_argument(
        "--target-modules",
        nargs="+",
        default=None,
        help="LoRA target modules override (default: auto-detected, e.g. q_proj v_proj).",
    )
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32", "auto"],
        default="bfloat16",
        help="Tensor storage dtype for adapter_model.safetensors (default: bfloat16).",
    )
    parser.add_argument(
        "--save-bin",
        action="store_true",
        help="Also write adapter_model.bin alongside adapter_model.safetensors.",
    )
    parser.add_argument(
        "-e", "--evaluate",
        action="store_true",
        help="Directly run Tapestry Monash evaluation (run_evaluation.sh) after preparing adapter directory.",
    )
    parser.add_argument(
        "--eval-output-dir",
        type=Path,
        help="Output directory for predictions and scores when --evaluate is enabled.",
    )
    parser.add_argument(
        "--eval-script",
        type=Path,
        help="Custom path to run_evaluation.sh.",
    )
    parser.add_argument(
        "--python-bin",
        type=str,
        help="Custom python binary path to use for GOQA evaluation.",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress informational messages.",
    )
    return parser


def main() -> int:
    parser = parse_args()
    args = parser.parse_args()

    input_path = args.input_flag or args.input_path
    if input_path is None:
        candidate_ml_models = [
            Path(__file__).resolve().parent.parent / "ml_models",
            Path("/mnt/disk1/slakshna/dcll/Slakshna/ml_models"),
        ]
        for c in candidate_ml_models:
            if c.is_dir():
                input_path = c
                print(f"[*] No input specified. Defaulting to: {input_path}")
                break

        if input_path is None:
            parser.print_help()
            print("\nError: Please specify an input .pth file or directory.", file=sys.stderr)
            return 1

    input_path = input_path.resolve()
    verbose = not args.quiet

    discovered = discover_checkpoints(
        input_path=input_path,
        round_spec=args.round,
        node_filter=args.node,
    )

    if not discovered:
        print(f"[!] No matching checkpoints found in {input_path}.", file=sys.stderr)
        return 1

    created_adapter_dirs: list[Path] = []
    multiple = len(discovered) > 1

    for node_name, round_num, pth_file in discovered:
        if args.output_dir:
            if multiple:
                clean_node = re.sub(r"^sync_ckpt_", "", node_name)
                out_dir = args.output_dir / f"{clean_node}_round_{round_num}"
            else:
                out_dir = args.output_dir
        else:
            out_dir = pth_file.parent / f"{pth_file.stem}_adapter"

        if verbose:
            print("\n" + "-" * 60)
            print(f"[*] Node: {node_name} | Round: {round_num}")
            print(f"[*] File: {pth_file.name} -> {out_dir}")
            print("-" * 60)

        res = convert_sync_round_file(
            pth_path=pth_file,
            output_dir=out_dir,
            base_model=args.base_model,
            rank=args.rank,
            alpha=args.alpha,
            dropout=args.dropout,
            target_modules=args.target_modules,
            dtype_str=args.dtype,
            save_bin=args.save_bin,
            verbose=verbose,
        )
        created_adapter_dirs.append(res["output_dir"])

    if verbose:
        print("\n" + "=" * 70)
        print("[+] ADAPTER DIRECTORY PREPARATION COMPLETE")
        for ad in created_adapter_dirs:
            print(f"    Adapter Path: {ad}")
        print("\nTo evaluate with Tapestry Monash GOQA:")
        eval_script_rel = "/mnt/disk1/slakshna/tapestry_monash/shared_evaluation/GOQA/run_evaluation.sh"
        for ad in created_adapter_dirs:
            eval_out = (ad.parent / f"{ad.name}_eval_results").resolve()
            print(f"    bash {eval_script_rel} {args.base_model} {eval_out} {ad}")
        print("=" * 70 + "\n")

    # If --evaluate is requested, run evaluation on the adapter directory
    if args.evaluate:
        for ad in created_adapter_dirs:
            eval_out = args.eval_output_dir or (ad.parent / f"{ad.name}_eval_results")
            ret = run_monash_evaluation(
                adapter_dir=ad,
                base_model=args.base_model,
                eval_output_dir=eval_out,
                eval_script=args.eval_script,
                python_bin=args.python_bin,
            )
            if ret != 0:
                print(f"[!] Evaluation failed with exit code {ret}", file=sys.stderr)
                return ret

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
