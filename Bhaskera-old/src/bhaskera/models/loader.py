"""
bhaskera.models.loader
======================
Load an HF `AutoModelForCausalLM` (or a registered custom loader), run
introspection, and optionally attach LoRA.


"""
from __future__ import annotations

import logging
from typing import Callable, Tuple

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from bhaskera.introspect import ModelProfile, introspect_model
from bhaskera.models.quantization import build_quantization_config
logger = logging.getLogger(__name__)

_DTYPE_MAP = {
    "float32":  torch.float32,
    "float16":  torch.float16,
    "bfloat16": torch.bfloat16,
}

_CUSTOM_REGISTRY: dict[str, Callable] = {}


def register_model(name: str):
    """Register a custom non-HF model loader under `name`."""
    def _wrap(fn: Callable):
        _CUSTOM_REGISTRY[name] = fn
        return fn
    return _wrap


def _maybe_apply_liger_kernel(model: torch.nn.Module, cfg) -> bool:
    """
    Apply Liger Kernel's Triton-fused RMSNorm / RoPE / SwiGLU / CrossEntropy
    kernels to an already-instantiated model.

    This is the generalized integration path: `_apply_liger_kernel_to_instance`
    dispatches on `model.config.model_type`, so it transparently supports every
    architecture Liger ships (Llama, Mistral, Mixtral, Qwen2/2.5/3, Gemma 1/2/3,
    Phi3, Granite, Olmo2, and more), and is a safe no-op for anything else.
    Compatible with FSDP2, DDP, gradient checkpointing, and flash-attn.

    Returns True iff kernels were actually applied. Never raises — a missing
    package, unsupported architecture, or internal Liger error is downgraded
    to a warning so training continues on the vanilla HF implementation.
    """
    if not getattr(cfg.model, "use_liger_kernel", True):
        logger.info("Liger Kernel disabled via config (model.use_liger_kernel=false).")
        return False

    try:
        from liger_kernel.transformers import _apply_liger_kernel_to_instance
    except ImportError:
        logger.warning(
            "liger-kernel is not installed — falling back to the standard "
            "HuggingFace kernels. Install with `pip install liger-kernel` "
            "(or re-run setup.sh) to enable Triton-fused RMSNorm/RoPE/SwiGLU/"
            "CrossEntropy and unlock ~20% throughput / ~60% memory savings."
        )
        return False

    model_type = getattr(getattr(model, "config", None), "model_type", "") or "<unknown>"
    try:
        _apply_liger_kernel_to_instance(model=model)
    except Exception as e:
        # Liger raises for model types it doesn't patch. This is expected for
        # custom / research architectures and must NOT abort training.
        logger.warning(
            f"Liger Kernel could not patch model_type={model_type!r} "
            f"({type(e).__name__}: {e}). Continuing with standard HF kernels."
        )
        return False

    logger.info(f"Liger Kernel applied (model_type={model_type}).")
    return True


def build_model(cfg, device: torch.device) -> Tuple[torch.nn.Module, ModelProfile]:
    """
    Load model, introspect it, optionally apply LoRA.

    For FSDP2, callers should pass device=torch.device("cpu") — the FSDP
    wrap step will shard and migrate the params to the correct GPU.
    For DDP, pass the target CUDA device directly.
    """
    name = cfg.model.name
    trust_remote_code = getattr(cfg.model, "trust_remote_code", False)

    raw_dtype = getattr(cfg.model, "dtype", "bfloat16")
    if raw_dtype == "auto":
        load_dtype: "str | torch.dtype" = "auto"
    else:
        load_dtype = _DTYPE_MAP.get(raw_dtype, torch.bfloat16)

    kwargs: dict = dict(
        low_cpu_mem_usage=True,
        trust_remote_code=trust_remote_code,
        torch_dtype=load_dtype,
    )
    if cfg.model.attn_impl:
        kwargs["attn_implementation"] = cfg.model.attn_impl

    # ── Load model ──────────────────────────────────────────────────
    if name in _CUSTOM_REGISTRY:
        model = _CUSTOM_REGISTRY[name](cfg, device)
    else:
        model_config = AutoConfig.from_pretrained(
            name, trust_remote_code=trust_remote_code
        )
        # NOTE: we deliberately do NOT set max_position_embeddings nor
        # output_router_logits here.  Truncation to seq_len is the
        # tokenizer's job, and router logits are requested per-forward
        # by the training loop when MoE aux loss is needed.
        quant_cfg = build_quantization_config(cfg)
        if quant_cfg is not None:
            strategy = getattr(getattr(cfg, "training", None) and cfg.training.distributed,"strategy","fsdp",)
            if strategy == "fsdp":
                raise ValueError("QLoRA (model.quantization=qlora) is not compatible with FSDP2. "
                        "Set training.distributed.strategy: ddp, or disable quantization.")
            kwargs["quantization_config"] = quant_cfg
        model = AutoModelForCausalLM.from_pretrained(
            name, config=model_config, **kwargs
        )
        if device.type != "cpu":
            model = model.to(device)

    # ── Introspect (never hardcode layer classes) ───────────────────
    profile = introspect_model(model)
    if raw_dtype != "auto":
        profile.model_dtype = _DTYPE_MAP.get(raw_dtype, torch.bfloat16)

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Loaded {name} | params={param_count:,} | "
        f"dtype={profile.model_dtype} | moe={profile.is_moe}"
    )

    # ── Liger Kernel patching ───────────────────────────────────────
    # Done BEFORE LoRA so PEFT wraps the already-fused modules (LoRA
    # adapters attach to Linear children, which Liger does not touch).
    # Done BEFORE FSDP wrap (handled later by distributed.wrap_model)
    # so that sharding sees the final module classes.
    _maybe_apply_liger_kernel(model, cfg)

    # ── LoRA ────────────────────────────────────────────────────────
    if cfg.lora.enabled:
        from .lora import apply_lora
        model = apply_lora(model, cfg, profile)

    return model, profile
