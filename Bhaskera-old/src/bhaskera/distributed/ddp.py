"""
bhaskera.distributed.ddp
========================
DDP wrap. MoE-aware: forces find_unused_parameters=True for MoE so routing
to a subset of experts per forward pass doesn't crash DDP.


"""
from __future__ import annotations

import logging

import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from bhaskera.introspect import ModelProfile
from .activation_ckpt import apply_activation_checkpointing

logger = logging.getLogger(__name__)


def wrap_ddp(model: nn.Module, cfg, local_rank: int, profile: ModelProfile) -> nn.Module:
    ddp_cfg = cfg.training.distributed.ddp

    # ── Resolve find_unused_parameters ──────────────────────────────
    # MoE forces this on because expert routing means a different
    # subset of params is touched on every forward pass. Dense models
    # use the configured value (default False).
    find_unused = ddp_cfg.find_unused_parameters
    if profile.is_moe and not find_unused:
        logger.warning(
            "MoE detected: forcing find_unused_parameters=True for DDP "
            "(not all expert params are used every forward pass)."
        )
        find_unused = True

    # ── Resolve static_graph ────────────────────────────────────────
    # DDP requires find_unused_parameters=False for static_graph=True.
    # Don't crash — log and disable.
    static_graph = ddp_cfg.static_graph
    if static_graph and find_unused:
        logger.warning(
            "DDP: static_graph=True is incompatible with "
            "find_unused_parameters=True (MoE forces the latter). "
            "Disabling static_graph for this run."
        )
        static_graph = False

    # ── Activation checkpointing (BEFORE the DDP wrap) ──────────────
    # apply_activation_checkpointing dispatches to:
    #   * MoE  → per-expert checkpointing (profile.expert_modules)
    #   * Dense → per-decoder-layer checkpointing
    # using torch's composable AC (>=2.4) with a legacy NO_REENTRANT
    # fallback. Both paths are safe with DDP because non-reentrant
    # checkpointing is gradient-correct under collective communications.
    if ddp_cfg.activation_checkpointing:
        apply_activation_checkpointing(
            model, profile, profile.decoder_layer_cls
        )
        logger.info(
            "DDP: activation checkpointing applied "
            f"({'per-expert' if profile.is_moe else 'per-decoder-layer'})"
        )

    # ── Move and wrap ───────────────────────────────────────────────
    # NB: split the .to(local_rank) and DDP() onto separate statements
    # so AC's hook installation has already completed on the CPU/GPU
    # module before DDP snapshots the param graph.
    model = model.to(local_rank)

    wrapped = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=find_unused,
        gradient_as_bucket_view=ddp_cfg.gradient_as_bucket_view,
        broadcast_buffers=ddp_cfg.broadcast_buffers,
        static_graph=static_graph,
    )

    # Stash the effective static_graph flag on the wrapper so the
    # training loop can detect it without re-reading cfg. PyTorch's
    # DDP exposes this via .static_graph in recent versions, but we
    # set a stable attribute name for cross-version compatibility.
    try:
        wrapped._bhaskera_static_graph = static_graph
    except Exception:
        pass

    logger.info(
        f"DDP wrap complete | find_unused={find_unused} "
        f"| ac={ddp_cfg.activation_checkpointing} "
        f"| static_graph={static_graph} "
        f"| moe={profile.is_moe}"
    )
    return wrapped
