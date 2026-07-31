"""
bhaskera.distributed
====================
FSDP2 (`fully_shard`) and DDP wrappers, plus DCP-based checkpointing.

Public API:
    wrap_model(model, cfg, local_rank, profile) -> wrapped model
    save_checkpoint(model, optimizer, step, path)
    load_checkpoint(model, optimizer, path) -> step
"""
from __future__ import annotations

from .wrap import wrap_model
from .checkpoint import save_checkpoint, maybe_resume as load_checkpoint

__all__ = ["wrap_model", "save_checkpoint", "load_checkpoint"]
