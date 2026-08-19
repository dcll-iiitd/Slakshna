"""
bhaskera.distributed.checkpoint
================================
Sharded checkpointing via torch.distributed.checkpoint (DCP).


On-disk layout for one checkpoint:
    <path>/
        model/...            # DCP shard files for model state
        optim/...            # DCP shard files for optimizer state
        meta.json            # {"step": int, "avg_loss": float}
        .complete            # sentinel — written last by rank-0
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PyTorch version detection (fix #12)
# ---------------------------------------------------------------------------

_TORCH_VER = tuple(int(x) for x in torch.__version__.split(".")[:2] if x.isdigit())

_STEP_RE = re.compile(r"step_(\d+)$")


# ---------------------------------------------------------------------------
# DCP compatibility shim (fix #12)
# ---------------------------------------------------------------------------

def _dcp_save(state_dict: dict, path: str) -> None:
    """
    Save via DCP, using the correct API for the installed PyTorch version.

    PyTorch < 2.5:  dcp.save(..., checkpoint_id=path)  [deprecated]
    PyTorch ≥ 2.5:  dcp.save(..., storage_writer=FileSystemWriter(path))
    """
    import torch.distributed.checkpoint as dcp
    Path(path).mkdir(parents=True, exist_ok=True)
    writer = dcp.FileSystemWriter(path)
    if _TORCH_VER >= (2, 5):
        dcp.save(state_dict, storage_writer=writer)
    else:
        # checkpoint_id= was deprecated in 2.4 and removed in 2.5
        dcp.save(state_dict, checkpoint_id=path)  # type: ignore[call-arg]


def _dcp_load(state_dict: dict, path: str) -> None:
    """
    Load via DCP, using the correct API for the installed PyTorch version.
    """
    import torch.distributed.checkpoint as dcp
    if _TORCH_VER >= (2, 5):
        dcp.load(state_dict, storage_reader=dcp.FileSystemReader(path))
    else:
        dcp.load(state_dict, checkpoint_id=path)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Checkpoint path helpers
# ---------------------------------------------------------------------------

def _checkpoint_path(save_dir: str, step: int) -> str:
    return os.path.join(save_dir, f"step_{step:07d}")


def _cleanup_old_checkpoints(save_dir: str, keep_last_n: int) -> None:
    """Remove all but the most recent keep_last_n completed checkpoints."""
    if keep_last_n <= 0:
        return
    candidates = sorted([
        p for p in Path(save_dir).iterdir()
        if p.is_dir()
        and _STEP_RE.search(p.name)
        and (p / ".complete").exists()
    ], key=lambda p: int(_STEP_RE.search(p.name).group(1)))
    for old in candidates[:-keep_last_n]:
        shutil.rmtree(str(old), ignore_errors=True)
        logger.info(f"Pruned old checkpoint: {old}")


# ---------------------------------------------------------------------------
# Save (fix #13 — atomic write)
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    path: str,
    extra: dict | None = None,
    rank: int = 0,
    keep_last_n: int = 3,
    cursor_meta: dict | None = None,
) -> None:
    """
    Save a sharded checkpoint atomically.

    Write flow:
      1. All ranks write shards to <path>.tmp  (DCP is collective)
      2. dist.barrier() — wait for all ranks to finish writing
      3. rank-0 renames <path>.tmp → <path>
      4. rank-0 writes .complete sentinel
      5. rank-0 prunes old checkpoints
      6. dist.barrier() — unblock all ranks

    The .complete sentinel ensures maybe_resume() only considers fully
    written checkpoints. A directory without .complete was interrupted.

    Args:
        model:       Distributed-wrapped model (FSDP2 or DDP).
        optimizer:   Optimizer whose state to save.
        step:        Current training step.
        path:        Target directory (will be written atomically).
        extra:       Optional metadata dict merged into meta.json.
        rank:        Global rank of this worker.
        keep_last_n: Number of completed checkpoints to retain.
    """
    tmp_path = path + ".tmp"

    # Clean up any leftover .tmp from a prior interrupted save
    if rank == 0 and os.path.exists(tmp_path):
        shutil.rmtree(tmp_path)
        logger.warning(f"Removed incomplete checkpoint: {tmp_path}")

    if _is_rank_zero(rank):
        Path(tmp_path).mkdir(parents=True, exist_ok=True)

    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    options = StateDictOptions(full_state_dict=False, cpu_offload=True)
    model_sd = get_model_state_dict(model, options=options)
    optim_sd = get_optimizer_state_dict(model, optimizer, options=options)

    _dcp_save({"model": model_sd}, os.path.join(tmp_path, "model"))
    _dcp_save({"optim": optim_sd}, os.path.join(tmp_path, "optim"))

    # All ranks must finish writing before rank-0 renames
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    if _is_rank_zero(rank):
        # Atomically replace target with the fully-written tmp
        if os.path.exists(path):
            shutil.rmtree(path)
        os.rename(tmp_path, path)

        # Write meta
        meta: dict = {"step": int(step)}
        if extra:
            meta.update(extra)
        if cursor_meta:
            meta.update(cursor_meta)
        with open(os.path.join(path, "meta.json"), "w") as f:
            json.dump(meta, f)

        # Write sentinel — this is the last thing written
        open(os.path.join(path, ".complete"), "w").close()
        
        # FIX: Save HuggingFace/PEFT Safetensors format alongside PyTorch DCP manually
        # Extract directly from model.named_parameters() — guaranteed real post-training values!
        # Do NOT re-import StateDictOptions here (it's module-level) or Python scoping will
        # make it an unbound local at line 150.
        try:
            from safetensors.torch import save_file
            import re as _re
            lora_sd = {}
            for name, param in model.named_parameters():
                if "lora_" in name:
                    # Strip DDP 'module.' prefix if present
                    clean_k = name[len("module."):] if name.startswith("module.") else name
                    # Normalize adapter name: lora_A.default.weight -> lora_A.weight
                    # PEFT's get_peft_model_state_dict strips the adapter name
                    clean_k = _re.sub(r'(lora_[AB])\.[^.]+\.(weight)', r'\1.\2', clean_k)
                    lora_sd[clean_k] = param.data.detach().cpu().float().contiguous()
            if lora_sd:
                save_file(lora_sd, os.path.join(path, "adapter_model.safetensors"))
                logger.info(f"Saved {len(lora_sd)} LoRA tensors to adapter_model.safetensors")
            else:
                logger.warning("No lora_ keys found in model.named_parameters()!")
        except Exception as e:
            logger.warning(f"Failed to save adapter_model.safetensors: {e}")

        save_dir = str(Path(path).parent)
        _cleanup_old_checkpoints(save_dir, keep_last_n)
        logger.info(f"Checkpoint saved → {path} (step={step})")

    # Unblock all ranks after rank-0 finishes the rename
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


# ---------------------------------------------------------------------------
# Resume (fix #13 — .complete guard)
# ---------------------------------------------------------------------------

def maybe_resume(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    save_dir: str,
) -> int:
    """
    Scan save_dir for the latest checkpoint that has a .complete sentinel.
    Load it in-place and return the step to resume from.

     Returns (0, {}) if no valid checkpoint is found (fresh start).
    Returns (step, meta_dict) where meta_dict contains all keys from
    meta.json, including eval_lifecycle/* cursor keys when present.
    """
    if not os.path.isdir(save_dir):
        logger.info("No checkpoint directory found. Starting from step 0.")
        return 0, {}

    candidates = sorted([
        p for p in Path(save_dir).iterdir()
        if p.is_dir()
        and _STEP_RE.search(p.name)
        and (p / ".complete").exists()      # skip partially-written checkpoints
    ], key=lambda p: int(_STEP_RE.search(p.name).group(1)))

    if not candidates:
        logger.info("No valid checkpoints found (missing .complete sentinel). Starting from step 0.")
        return 0,{}

    ckpt_path  = str(candidates[-1])
    start_step = int(_STEP_RE.search(ckpt_path).group(1))
    logger.info(f"Resuming from {ckpt_path} (step {start_step})")

    options  = StateDictOptions(full_state_dict=False, cpu_offload=True)
    model_sd = get_model_state_dict(model, options=options)
    optim_sd = get_optimizer_state_dict(model, optimizer, options=options)

    _dcp_load({"model": model_sd}, os.path.join(ckpt_path, "model"))
    _dcp_load({"optim": optim_sd}, os.path.join(ckpt_path, "optim"))

    set_model_state_dict(model, model_state_dict=model_sd, options=options)
    set_optimizer_state_dict(
        model,
        optimizers=optimizer,
        optim_state_dict=optim_sd,
        options=options,
    )

    # Read step from meta.json (ground truth)
    meta_path = os.path.join(ckpt_path, "meta.json")
    meta: dict = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        start_step = int(meta.get("step", start_step))

    logger.info(f"Resume complete — starting from step {start_step}")
    return start_step,meta


# ---------------------------------------------------------------------------
# Legacy save_and_prune wrapper (used by trainer/checkpointing.py)
# ---------------------------------------------------------------------------

def save_and_prune(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    avg_loss: float,
    ckpt_cfg,
    rank: int,
    best_ckpts: list,
) -> list:
    """
    Compatibility wrapper around save_checkpoint used by the training loop.
    Maintains the list of best checkpoints by loss.
    """
    path = _checkpoint_path(ckpt_cfg.save_dir, step)
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        step=step,
        path=path,
        extra={"avg_loss": float(avg_loss)},
        rank=rank,
        keep_last_n=ckpt_cfg.keep_last_n,
    )

    best_ckpts = best_ckpts + [(avg_loss, path)]
    best_ckpts.sort(key=lambda x: x[0])
    return best_ckpts[:ckpt_cfg.keep_last_n]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_rank_zero(rank: int = 0) -> bool:
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0
