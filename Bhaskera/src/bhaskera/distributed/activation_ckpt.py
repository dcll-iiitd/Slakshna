"""
bhaskera.distributed.activation_ckpt
====================================
Activation checkpointing wrappers, FSDP2-composable with legacy fallback.

Strategy:
    * MoE: checkpoint each expert module individually
    * Dense: checkpoint each decoder layer
"""
from __future__ import annotations

import logging
from functools import partial
from typing import Optional

import torch.nn as nn

from bhaskera.introspect import ModelProfile

logger = logging.getLogger(__name__)


def apply_activation_checkpointing(
    model: nn.Module,
    profile: ModelProfile,
    decoder_cls: Optional[type],
) -> None:
    """Preferred: FSDP2-native composable API; fallback to legacy if missing."""
    # Composable API (torch >= 2.4).
    try:
        from torch.distributed._composable import checkpoint as composable_checkpoint
        _apply_composable(model, profile, decoder_cls, composable_checkpoint)
        return
    except ImportError:
        pass

    # Legacy path.
    try:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            apply_activation_checkpointing as _legacy_apply,
            checkpoint_wrapper,
            CheckpointImpl,
        )
        _apply_legacy(model, profile, decoder_cls, _legacy_apply, checkpoint_wrapper, CheckpointImpl)
    except ImportError:
        logger.warning("No activation checkpointing API available — skipping.")


def _apply_composable(model, profile, decoder_cls, composable_checkpoint):
    if profile.is_moe and profile.expert_modules:
        for expert in profile.expert_modules:
            composable_checkpoint(expert)
        logger.info(
            f"AC (composable): applied to {len(profile.expert_modules)} expert modules"
        )
        return

    if decoder_cls is not None:
        count = 0
        for module in model.modules():
            if isinstance(module, decoder_cls):
                composable_checkpoint(module)
                count += 1
        logger.info(f"AC (composable): applied to {count} {decoder_cls.__name__} layers")
        return

    logger.warning("AC: no target modules found — skipping")


def _apply_legacy(model, profile, decoder_cls, apply_fn, checkpoint_wrapper, CheckpointImpl):
    wrapper_fn = partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT)

    if profile.is_moe and profile.expert_module_cls:
        check_fn = lambda m: isinstance(m, profile.expert_module_cls)
        label = profile.expert_module_cls.__name__
    elif decoder_cls is not None:
        check_fn = lambda m: isinstance(m, decoder_cls)
        label = decoder_cls.__name__
    else:
        logger.warning("AC (legacy): no target classes — skipping")
        return

    apply_fn(model, checkpoint_wrapper_fn=wrapper_fn, check_fn=check_fn)
    logger.info(f"AC (legacy): applied to {label}")