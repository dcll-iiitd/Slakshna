"""Weights & Biases logger.

Changes:
    * Uses ``system_stats()`` (CPU + GPU + I/O) instead of just GPU.
    * Tagged with ``rank`` so multi-rank runs that publish system
      metrics can be sliced inside W&B.
    * Lazy-imports wandb so the dep stays fully optional.
"""
from __future__ import annotations

import logging
from typing import Any

from .base import BaseLogger

logger = logging.getLogger(__name__)


class WandbLogger(BaseLogger):
    def __init__(self, cfg, *, rank: int = 0, world_size: int = 1) -> None:
        import wandb
        wandb.init(
            project=cfg.logging.project,
            name=cfg.logging.run_name,
            config=cfg.as_dict(),
            tags=list(getattr(cfg.logging, "tags", []) or []),
            group=getattr(cfg.logging, "group", None),
        )
        self._wandb = wandb
        self._every_n = max(1, cfg.logging.log_gpu_every_n_steps)
        self._rank = int(rank)
        try:
            wandb.run.summary["rank"] = self._rank
            wandb.run.summary["world_size"] = int(world_size)
        except Exception:
            pass

    def log(self, metrics: dict[str, Any], step: int) -> None:
        if step % self._every_n == 0:
            from bhaskera.utils.system_stats import system_stats, cuda_memory_stats
            metrics = {**metrics, **system_stats(), **cuda_memory_stats()}
        self._wandb.log(metrics, step=step)

    def finish(self) -> None:
        try:
            self._wandb.finish()
        except Exception as e:
            logger.debug(f"wandb.finish failed: {e}")
