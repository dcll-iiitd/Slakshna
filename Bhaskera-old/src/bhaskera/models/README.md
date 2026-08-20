# src/bhaskera/models/

HuggingFace model loading, optional Liger Kernel patching, and PEFT-based LoRA application.

## Module layout

```
models/
├── __init__.py      # re-exports build_model, register_model
├── loader.py        # build_model + Liger Kernel hook + custom registry
└── lora.py          # apply_lora + MoE router freezing
```

## Public API

Re-exported from `bhaskera.models`:

- `build_model(cfg, device) -> (nn.Module, ModelProfile)`
- `register_model(name)` — decorator for non-HF custom loaders

## `loader.py` — `build_model`

Loads a model, runs introspection, applies Liger Kernel patches, applies LoRA if configured.

### Flow

1. **Resolve dtype.** `cfg.model.dtype="auto"` defers to HF's auto-detection; otherwise mapped from string via `_DTYPE_MAP` (`float32` / `float16` / `bfloat16`).
2. **Load.**
   - If `cfg.model.name` is in `_CUSTOM_REGISTRY` (populated via `@register_model`), the registered callable is invoked.
   - Otherwise: `AutoConfig.from_pretrained` then `AutoModelForCausalLM.from_pretrained` with `low_cpu_mem_usage=True`, the resolved `torch_dtype`, `trust_remote_code`, and `attn_implementation` if set.
   - Migration to `device` happens only when `device.type != "cpu"` (FSDP path keeps params on CPU until the wrap step shards them onto GPU).
3. **Introspect.** `introspect_model(model)` returns a `ModelProfile`. When `cfg.model.dtype` is explicit (not `"auto"`), `profile.model_dtype` is overwritten with the resolved type.
4. **Liger Kernel patching.** `_maybe_apply_liger_kernel(model, cfg)` calls `_apply_liger_kernel_to_instance(model=model)` — Liger dispatches on `model.config.model_type`, so it transparently supports Llama, Mistral, Mixtral, Qwen 2/2.5/3, Gemma 1/2/3, Phi3, Granite, Olmo2, etc., and no-ops on anything else. Done **before** LoRA so PEFT wraps the already-fused modules. Done **before** FSDP wrap (handled later) so sharding sees the final module classes. All errors are downgraded to warnings — training continues on the vanilla HF kernels if Liger refuses to patch.
5. **LoRA.** If `cfg.lora.enabled`, calls `apply_lora(model, cfg, profile)`.

### Custom model registration

```python
from bhaskera.models import register_model

@register_model("my-model")
def load(cfg, device):
    # return an nn.Module
    ...
```

Then set `model.name: my-model` in the YAML.

### Why deliberately *not* set certain HF kwargs

The loader does not set `max_position_embeddings` or `output_router_logits` at load time. Truncation to `seq_len` is the tokenizer's responsibility, and router logits are requested per-forward by the training loop when MoE aux loss is needed.

## `lora.py` — `apply_lora`

PEFT-based LoRA injection with MoE awareness.

### Target resolution

- `cfg.lora.target_modules == ["auto"]` → uses `profile.lora_targets` from introspection. Falls back to PEFT's own defaults when the profile didn't find anything.
- Explicit list → used verbatim.
- MoE with `cfg.lora.include_experts=True` → expert-FFN linear names from `_find_expert_linear_names(model, profile)` are merged into the target list.

### PEFT config

```python
PeftLoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=cfg.lora.r,
    lora_alpha=cfg.lora.alpha,
    lora_dropout=cfg.lora.dropout,
    bias="none",
    target_modules=target_modules,        # if resolved
    modules_to_save=cfg.lora.modules_to_save,  # if non-empty
)
```

### Strategy-gated dtype cast

This is the critical bit, and the rationale lives in the module docstring:

- **FSDP path** — LoRA A/B parameters (PEFT default fp32) are cast to `profile.model_dtype` (typically bf16). FSDP2 requires uniform original dtype within a shard group; without the cast, FSDP raises `AssertionError: FSDP expects uniform original parameter dtype but got {torch.float32, torch.bfloat16}`.
- **DDP / single-GPU path** — LoRA params stay in fp32. DDP has no uniform-dtype constraint, and the training loop wraps DDP forwards in `torch.autocast`. Pre-casting to bf16 would (a) put AdamW's `exp_avg` and `exp_avg_sq` in bf16, quantising most LoRA gradients to zero with `β₂=0.95` at typical LRs around `2e-4`, and (b) defeat the autocast design.

Trade-off acknowledged in the docstring: under FSDP, AdamW moments for LoRA end up in bf16 instead of fp32 — slightly noisier but fine for LoRA SFT, and `reduce_dtype=float32` (the MoE default) keeps gradient reductions in fp32. To get true fp32 master weights under FSDP, load the base model in fp32 (`model.dtype: float32`) and let `MixedPrecisionPolicy(param_dtype=bf16)` handle forward-time casting.

### Router freezing

When `profile.is_moe` and `cfg.lora.freeze_router` (default `True`), `_freeze_router_weights()` calls `param.requires_grad_(False)` on every parameter whose name matches a router substring (`gate`, `router`, `switch`, `gating`) or appears in `profile.router_module_names`. Training routers under LoRA is empirically destabilising for MoE.

`introspect.py` detects routers structurally (parameter-bearing siblings of an `experts` ModuleList), which correctly avoids flagging SwiGLU's `gate_proj` as a router.

### Logging

After applying, the function logs the trainable parameter count, total parameter count, and trainable percentage. Typical LoRA SFT runs land between 0.1% and 1% trainable.

## Interaction with the rest of the pipeline

| Step | Where | Notes |
|---|---|---|
| HF load | `loader.build_model` | `low_cpu_mem_usage=True`, configured `torch_dtype` |
| Introspection | `loader.build_model` | populates `ModelProfile` |
| Liger patching | `loader._maybe_apply_liger_kernel` | before LoRA, before FSDP |
| LoRA wrap | `lora.apply_lora` | PEFT `get_peft_model`, then dtype cast |
| Router freeze | `lora._freeze_router_weights` | MoE only |
| Distributed wrap | `bhaskera.distributed.wrap_model` | called by `launcher/worker.py` after `build_model` |
