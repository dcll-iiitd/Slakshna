"""
bhaskera.models.quantization
=============================
Optional QLoRA quantization config builder.

Kept intentionally thin — only called by loader.build_model when
cfg.model.quantization == "qlora".  All other values (including the
default "none") return None, leaving the existing load path untouched.

Compatibility note
------------------
QLoRA loads the base model as BitsAndBytes Params4bit tensors.
FSDP2 (fully_shard) expects standard PyTorch parameters and will
crash or produce undefined behaviour when it tries to shard Params4bit.
DDP is safe because each rank holds a full model copy and BnB's
dequantization logic is never asked to work across shards.

Supported combinations
    quantization=qlora + strategy=ddp   -> OK
    quantization=qlora + strategy=fsdp  -> raises ValueError (fast-fail)
    quantization=none  (default)        -> no-op, old path preserved exactly
"""
from __future__ import annotations

from typing import Optional


def build_quantization_config(cfg) -> "Optional[object]":
    """
    Return a BitsAndBytesConfig for QLoRA, or None.

    Parameters
    ----------
    cfg : bhaskera.config.Config
        The top-level config object.  Inspects:
            cfg.model.quantization              - "qlora" | "none" (default)
            cfg.training.distributed.strategy   - checked for FSDP guard

    Returns
    -------
    BitsAndBytesConfig | None
        None when quantization is "none" or any unrecognised value.
        The caller (loader.py) only injects quantization_config into
        kwargs when the return value is not None, so the default path
        is bit-for-bit identical to the pre-QLoRA code.

    Raises
    ------
    ValueError
        When quantization="qlora" and distributed strategy is "fsdp".
        Fails fast with a clear message rather than letting FSDP crash
        internally when it encounters Params4bit tensors.
    ImportError
        When quantization="qlora" but bitsandbytes / transformers are
        not installed.  Surfaces the missing-package error directly
        rather than wrapping it.
    """
    mode = getattr(cfg.model, "quantization", "none")

    if mode != "qlora":
        # covers "none", any future value, and missing field
        return None

    # ── FSDP guard ────────────────────────────────────────────────────────
    # BitsAndBytes Params4bit tensors cannot be sharded by FSDP2.
    # Detect the strategy early and fail with a useful message.
    try:
        strategy = cfg.training.distributed.strategy
    except AttributeError:
        strategy = "fsdp"   # safe default: assume FSDP if config is partial

    if strategy == "fsdp":
        raise ValueError(
            "QLoRA (model.quantization: qlora) is not compatible with FSDP2. "
            "BitsAndBytes Params4bit tensors cannot be sharded across ranks. "
            "Either set  training.distributed.strategy: ddp  or disable "
            "quantization (model.quantization: none)."
        )

    # ── Build config ───────────────────────────────────────────────────────
    # Imports are deferred so environments without bitsandbytes installed
    # are unaffected when quantization="none" (the default).
    import torch
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
