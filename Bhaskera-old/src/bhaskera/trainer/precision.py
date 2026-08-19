"""
bhaskera.trainer.precision
==========================
Resolve dtypes for the autocast/MixedPrecision layer.
"""
from __future__ import annotations

import torch

from bhaskera.introspect import ModelProfile

_DTYPE_MAP = {
    "float32":  torch.float32,
    "float16":  torch.float16,
    "bfloat16": torch.bfloat16,
}


def resolve_autocast_dtype(cfg, profile: ModelProfile) -> torch.dtype:
    """
    Resolve the dtype that DDP's autocast context (or the raw forward pass)
    should use.  Priority: explicit cfg.model.dtype → profile → bf16 fallback.
    """
    raw_dtype = getattr(cfg.model, "dtype", "bfloat16")
    if raw_dtype == "auto":
        return profile.model_dtype
    return _DTYPE_MAP.get(raw_dtype, torch.bfloat16)