"""
bhaskera.distributed.fsdp
=========================
FSDP2 (torch.distributed._composable.fsdp.fully_shard) wrap + mixed precision.

Design notes:
  * We call fully_shard three times (experts → layers → root) so that the
    MoE forward only all-gathers the activated experts, not the full set.
  * MixedPrecisionPolicy is the SINGLE source of mixed-precision truth.
    The training loop no longer wraps forward in torch.autocast.
  * "auto" dtype resolution reads from the ModelProfile produced by
    introspect.py, so no layer-name hardcoding is required.
"""
from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

from bhaskera.introspect import ModelProfile
from .activation_ckpt import apply_activation_checkpointing

logger = logging.getLogger(__name__)

_DTYPE_MAP = {
    "float32":  torch.float32,
    "float16":  torch.float16,
    "bfloat16": torch.bfloat16,
}


def wrap_fsdp2(
    model: nn.Module,
    cfg,
    local_rank: int,
    profile: ModelProfile,
) -> nn.Module:
    try:
        from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
    except ImportError as e:
        raise ImportError(
            "FSDP2 requires PyTorch >= 2.4. "
            "Either upgrade PyTorch or set training.distributed.strategy: ddp."
        ) from e

    fsdp_cfg = cfg.training.distributed.fsdp
    mp_policy = _build_mp_policy(fsdp_cfg, profile, MixedPrecisionPolicy)
    decoder_cls = _resolve_decoder_cls(model, fsdp_cfg, profile)

    # Step 1 — per-expert sharding (MoE only).
    if profile.is_moe and fsdp_cfg.shard_experts_individually and profile.expert_modules:
        for expert in profile.expert_modules:
            fully_shard(expert, mp_policy=mp_policy)
        logger.info(
            "FSDP2: per-expert sharding applied to "
            f"{len(profile.expert_modules)} expert modules "
            f"(class={profile.expert_module_cls.__name__ if profile.expert_module_cls else 'N/A'})"
        )

    # Step 2 — per-decoder-layer sharding.
    if decoder_cls is not None:
        layer_count = 0
        for module in model.modules():
            if isinstance(module, decoder_cls):
                fully_shard(module, mp_policy=mp_policy)
                layer_count += 1
        logger.info(
            f"FSDP2: per-layer sharding applied to {layer_count} "
            f"{decoder_cls.__name__} layers"
        )
    else:
        logger.warning(
            "No decoder layer class found — applying fully_shard to root only. "
            "Memory use will be suboptimal."
        )

    # Step 3 — root.
    fully_shard(model, mp_policy=mp_policy)

    # Activation checkpointing.
    if fsdp_cfg.activation_checkpointing:
        apply_activation_checkpointing(model, profile, decoder_cls)

    logger.info(
        f"FSDP2 wrap complete | param_dtype={mp_policy.param_dtype} | "
        f"reduce_dtype={mp_policy.reduce_dtype} | moe={profile.is_moe} | "
        f"ac={fsdp_cfg.activation_checkpointing}"
    )
    return model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_mp_policy(fsdp_cfg, profile: ModelProfile, MixedPrecisionPolicy):
    if fsdp_cfg.param_dtype == "auto":
        param_dtype = profile.model_dtype
    else:
        param_dtype = _DTYPE_MAP.get(fsdp_cfg.param_dtype, torch.bfloat16)

    # reduce_dtype=float32 is safer for MoE because gradients are sparse
    # and low-precision reductions amplify routing noise.
    if fsdp_cfg.reduce_dtype == "auto":
        reduce_dtype = torch.float32 if profile.is_moe else param_dtype
    else:
        reduce_dtype = _DTYPE_MAP.get(fsdp_cfg.reduce_dtype, torch.bfloat16)

    if fsdp_cfg.buffer_dtype == "auto":
        output_dtype = param_dtype
    else:
        output_dtype = _DTYPE_MAP.get(fsdp_cfg.buffer_dtype, torch.bfloat16)

    return MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        output_dtype=output_dtype,
    )


def _resolve_decoder_cls(model: nn.Module, fsdp_cfg, profile: ModelProfile) -> Optional[type]:
    """
    Priority:
        1. explicit names in fsdp_cfg.transformer_layer_cls (back-compat)
        2. auto-detected class from introspection profile
    """
    if fsdp_cfg.transformer_layer_cls:
        found = _find_layer_classes_by_name(model, fsdp_cfg.transformer_layer_cls)
        if found:
            return found[0]
        logger.warning(
            f"Manual transformer_layer_cls {fsdp_cfg.transformer_layer_cls} "
            "not found in model — falling back to auto-detection"
        )
    return profile.decoder_layer_cls


def _find_layer_classes_by_name(model: nn.Module, names: list[str]) -> list[type]:
    found: dict[str, type] = {}
    for module in model.modules():
        cls_name = module.__class__.__name__
        if cls_name in names and cls_name not in found:
            found[cls_name] = module.__class__
    missing = set(names) - set(found)
    if missing:
        logger.warning(f"FSDP2: layer classes not found in model: {missing}")
    return list(found.values())