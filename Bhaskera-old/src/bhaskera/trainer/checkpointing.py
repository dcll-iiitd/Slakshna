"""
bhaskera.trainer.checkpointing
==============================
Periodic-save and resume helpers.


"""
from __future__ import annotations

import logging
import os
import re
import shutil
from pathlib import Path

import torch
import torch.distributed as dist

from bhaskera.distributed import load_checkpoint, save_checkpoint

logger = logging.getLogger(__name__)

_STEP_RE = re.compile(r"step_(\d+)")


def save_and_prune(
    *,
    model,
    optimizer,
    step: int,
    avg_loss: float,
    ckpt_cfg,
    rank: int,
    best_ckpts: list[tuple[float, str]],
    cursor_meta: dict | None = None,
) -> list[tuple[float, str]]:
    """
    Save a checkpoint directory and prune down to ckpt_cfg.keep_last_n
    best-by-loss.  All ranks must call this — it runs a collective.
    """
    Path(ckpt_cfg.save_dir).mkdir(parents=True, exist_ok=True)
    ckpt_path = os.path.join(ckpt_cfg.save_dir, f"step_{step:07d}")

    # Synchronise before any collective save.
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        step=step,
        path=ckpt_path,
        extra={"avg_loss": float(avg_loss)},
        cursor_meta=cursor_meta,
    )

    # Only rank 0 manages the "best" list on disk.
    if rank == 0:
        best_ckpts.append((avg_loss, ckpt_path))
        best_ckpts.sort(key=lambda x: x[0])  # ascending: lower loss is better
        while len(best_ckpts) > ckpt_cfg.keep_last_n:
            _, old_path = best_ckpts.pop()
            _safe_rmtree(old_path)
        logger.info(
            f"Checkpoint → {ckpt_path} | "
            f"kept: {[os.path.basename(p) for _, p in best_ckpts]}"
        )

    return best_ckpts


def maybe_resume(model, optimizer, save_dir: str) -> tuple[int, dict]:
    """Resume from the highest-step checkpoint directory under `save_dir`.
    Returns (step, meta_dict). meta_dict contains all meta.json keys
    including eval_lifecycle/* cursor keys when present."""
    # Delegate entirely to the distributed load_checkpoint which correctly 
    # handles the .complete sentinel scanning and loading.
    return load_checkpoint(model, optimizer, save_dir)


def _step_of(p: Path) -> int:
    m = _STEP_RE.search(p.name)
    return int(m.group(1)) if m else -1


def _safe_rmtree(path: str) -> None:
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        pass
    except Exception as e:
        logger.warning(f"Failed to remove old checkpoint {path}: {e}")
