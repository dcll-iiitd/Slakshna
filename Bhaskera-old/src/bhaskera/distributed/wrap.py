"""
bhaskera.distributed.wrap
=========================
Dispatcher that chooses FSDP2 or DDP based on config.


"""
from __future__ import annotations

import logging
import time

import torch
import torch.distributed as dist
import torch.nn as nn

from bhaskera.introspect import ModelProfile

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Distributed init guards (fix #3, #9)
# ---------------------------------------------------------------------------

def _wait_for_dist(timeout_s: int = 120) -> None:
    """
    Poll dist.is_initialized() until True or timeout.

    Prevents the placement-group race between Ray TorchTrainer's
    process-group init and wrap_model() being called too early.

    Raises:
        RuntimeError: if dist is not initialized within timeout_s seconds.
    """
    deadline = time.monotonic() + timeout_s
    while not dist.is_initialized():
        if time.monotonic() > deadline:
            raise RuntimeError(
                f"torch.distributed was not initialized within {timeout_s}s. "
                "Check Ray cluster health and NCCL configuration. "
                "If using SLURM, verify NCCL_TIMEOUT and TORCH_DISTRIBUTED_TIMEOUT "
                "are set in scripts/submit.sh."
            )
        time.sleep(1)
    logger.debug("torch.distributed initialized ✓")


def _verify_all_ranks_live(rank: int, world_size: int, device: torch.device) -> None:
    """
    All-reduce a probe tensor across all ranks.

    If any rank is missing (crashed or not yet started), the all-reduce
    will either hang (caught by NCCL_TIMEOUT) or return an incorrect sum,
    causing a fast, visible failure rather than a silent hang.

    Must be called immediately after _wait_for_dist() and before any
    model init or data loading work.

    Raises:
        RuntimeError: if the sum of the probe does not equal world_size.
    """
    probe = torch.ones(1, device=device)
    dist.all_reduce(probe, op=dist.ReduceOp.SUM)
    actual_sum = probe.item()
    if actual_sum != float(world_size):
        raise RuntimeError(
            f"Rank liveness check FAILED: expected sum={world_size}, "
            f"got {actual_sum}. A rank is missing or has crashed. "
            "Check Ray worker logs for the failing rank."
        )
    logger.info(f"[rank {rank}] All {world_size} ranks alive ✓")


# ---------------------------------------------------------------------------
# Main dispatcher
# ---------------------------------------------------------------------------

def wrap_model(
    model: nn.Module,
    cfg,
    local_rank: int,
    profile: ModelProfile,
) -> nn.Module:
    """Wrap a model for distributed training. Strategy: 'fsdp' or 'ddp'."""
    if not dist.is_initialized():
        raise RuntimeError(
            "torch.distributed must be initialised before wrap_model(). "
            "Call _wait_for_dist() first. Ray Train's TorchTrainer does this "
            "automatically; for raw SLURM make sure "
            "torch.distributed.init_process_group() has been called."
        )

    strategy = cfg.training.distributed.strategy.lower()
    if strategy == "fsdp":
        from .fsdp import wrap_fsdp2
        return wrap_fsdp2(model, cfg, local_rank, profile)
    if strategy == "ddp":
        from .ddp import wrap_ddp
        return wrap_ddp(model, cfg, local_rank, profile)
    raise ValueError(
        f"Unknown distributed strategy: '{strategy}'. Choose 'fsdp' or 'ddp'."
    )