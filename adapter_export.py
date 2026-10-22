"""
Export a node's aggregated LoRA adapter as a content-addressed PEFT directory.

Used by ``ml_engine.py`` at the end of an aggregation round (federated
inference, step 1). The adapter is identified by the SHA-256 of its
``adapter_model.safetensors`` file and stored at
``{data_dir}/adapters/{sha256}/``. The Rust node then publishes that file over
iroh-blobs and gossips the metadata.

Diagnostics go to stderr: only the final stdout line of ml_engine.py is parsed
by the Rust node.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from evaluations.prepare_sync_round_adapter import normalize_lora_keys

_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def export_adapter(
    state_dict: dict[str, torch.Tensor],
    data_dir: str,
    node_id: str,
    round_num: int,
    base_model: str,
    rank: int | None = None,
    alpha: int | None = None,
    dropout: float = 0.0,
    target_modules: list[str] | None = None,
    dtype: str = "bfloat16",
) -> dict[str, Any]:
    """Write the adapter and return the fields ml_engine adds to its output."""
    cleaned, auto_rank, auto_targets = normalize_lora_keys(state_dict)
    # The tensors are the ground truth. The template's `lora.rank` is not even what
    # Bhaskera reads (it reads `lora.r`, default 16), so a mismatching config would
    # produce an adapter PEFT refuses to load. Only alpha cannot be read off the tensors.
    if rank is not None and rank != auto_rank:
        print(f"[adapter_export] config rank {rank} != rank {auto_rank} found in the weights; using {auto_rank}", file=sys.stderr)
    if target_modules and sorted(target_modules) != sorted(auto_targets):
        print(f"[adapter_export] config target_modules {target_modules} != {auto_targets} found in the weights; using the latter", file=sys.stderr)
    final_rank = auto_rank
    final_targets = auto_targets
    final_alpha = alpha if alpha is not None else 4 * final_rank
    target_dtype = _DTYPES.get(dtype.lower(), torch.bfloat16)

    tensors = {k: v.detach().cpu().to(target_dtype).contiguous() for k, v in cleaned.items()}

    adapters_root = Path(data_dir) / "adapters"
    adapters_root.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix=".export_", dir=adapters_root))
    try:
        weights_path = tmp_dir / "adapter_model.safetensors"
        save_file(tensors, str(weights_path))
        digest = sha256_file(weights_path)
        size = weights_path.stat().st_size

        adapter_config = {
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
            "target_modules": final_targets,
            "task_type": "CAUSAL_LM",
            "use_dora": False,
            "use_rslora": False,
        }
        (tmp_dir / "adapter_config.json").write_text(json.dumps(adapter_config, indent=2), encoding="utf-8")
        (tmp_dir / "adapter_meta.json").write_text(
            json.dumps(
                {
                    "node_id": node_id,
                    "round": round_num,
                    "adapter_model_safetensors_sha256": digest,
                    "tensor_count": len(tensors),
                    "parameter_count": sum(t.numel() for t in tensors.values()),
                    "rank": final_rank,
                    "alpha": final_alpha,
                    "target_modules": final_targets,
                    "base_model": base_model,
                    "dtype": dtype,
                    "created_at": int(time.time()),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        final_dir = adapters_root / digest
        if final_dir.exists():
            # Identical weights already exported (e.g. an idle round).
            shutil.rmtree(tmp_dir, ignore_errors=True)
        else:
            os.replace(tmp_dir, final_dir)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    print(f"[adapter_export] round {round_num} adapter {digest[:12]} ({size} bytes) -> {final_dir}", file=sys.stderr)
    return {
        "adapter_hash": digest,
        "adapter_path": str(final_dir),
        "adapter_size": size,
        "adapter_config": json.dumps(adapter_config, sort_keys=True),
        "adapter_base_model": base_model,
        "adapter_rank": final_rank,
        "adapter_target_modules": final_targets,
    }
