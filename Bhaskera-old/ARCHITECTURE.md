# Bhaskera — Architecture & Technical Reference

> **Audience.** This document is written for engineers who will operate, debug, or extend Bhaskera. It assumes familiarity with PyTorch distributed training, Ray, transformer architecture, and HPC/SLURM. Per-module behaviour is documented in each sub-package's `README.md`; this document covers the **system-level architecture, the design decisions behind it, and the contracts between subsystems**.

---

## Table of contents

1. [What Bhaskera is](#1-what-bhaskera-is)
2. [System architecture](#2-system-architecture)
3. [Process model & control flow](#3-process-model--control-flow)
4. [Module-by-module deep dive](#4-module-by-module-deep-dive)
   - 4.1 [`config`](#41-config--single-source-of-truth)
   - 4.2 [`introspect`](#42-introspect--structural-model-detection)
   - 4.3 [`data`](#43-data--ray-data-pipeline-and-persistent-tokenization)
   - 4.4 [`models`](#44-models--hf-loader-liger-kernel-lora-qlora)
   - 4.5 [`distributed`](#45-distributed--fsdp2-ddp-and-dcp-checkpointing)
   - 4.6 [`trainer`](#46-trainer--the-pure-training-loop)
   - 4.7 [`launcher`](#47-launcher--cli-entry-points)
   - 4.8 [`utils`](#48-utils--telemetry-throughput-loggers)
   - 4.9 [`evaluation`](#49-evaluation--distributed-validation-and-benchmarks)
   - 4.10 [`plugins`](#410-plugins--optimizer-metric-and-benchmark-extensions)
5. [Cross-cutting design decisions](#5-cross-cutting-design-decisions)
6. [Data flow](#6-data-flow)
7. [Extension guide](#7-extension-guide)
8. [Operational concerns](#8-operational-concerns)
9. [Repository layout](#9-repository-layout)

---

## 1. What Bhaskera is

Bhaskera is a **Ray-native distributed LLM training and inference framework** built on top of PyTorch ≥ 2.4, Ray ≥ 2.10, and HuggingFace Transformers. It targets the operating regime between a single-node toy script (`torchrun`-style) and a heavyweight in-house platform (Megatron / NeMo). Concretely it provides:

- **Distributed training** with FSDP2 (composable `fully_shard`) or DDP, chosen via config.
- **MoE support** for any model where the architecture is statically describable — Mixtral, Qwen2MoE, DeepSeek, Param2-17B-A2.4B-Thinking, and any future architecture that follows the `experts: nn.ModuleList` convention.
- **LoRA / PEFT** with auto-discovered target modules (no model-specific hardcoding).
- **QLoRA** via BitsAndBytes NF4 4-bit quantization, constrained to the DDP strategy (FSDP2 is incompatible with `Params4bit` tensors and is rejected at config validation time).
- **Ray Data**-driven dataset pipeline with a **persistent tokenization cache** keyed on `(model_name, seq_len, dataset_name, format)`.
- **Continual Pre-Training (CPT)** via a continuous sequence-packing mode (`is_cpt: true`) for local JSONL corpora.
- **Pluggable chat-data renderers** (ChatML / Alpaca / ShareGPT / custom) via a decorator-based registry.
- **Sharded checkpointing** using `torch.distributed.checkpoint` (DCP) with atomic write semantics.
- **Push-based observability** via MLflow over a shared `$HOME` filesystem (no Prometheus / Grafana / auth required) plus the Ray Dashboard, with optional W&B fan-out.
- **Inference engine** with TurboQuant KV-cache quantisation and speculative decoding.
- **Pluggable evaluation subsystem** — in-training distributed validation (loss, perplexity, token accuracy) and LM benchmarks (HellaSwag, ARC-Easy, ARC-Challenge) with a `@register_benchmark` / `@register_metric` decorator API.
- **Pluggable optimizer system** — built-in PyTorch optimizers selectable by class name, and fully custom optimizers registered via `@register_optimizer` plugins.
- **SLURM-aware launch** with NCCL auto-tuning for InfiniBand / RoCE / TCP.

The design ethos is **"zero hardcoded model names anywhere"**: every architecture-dependent decision is delegated to a single introspection pass that produces a `ModelProfile`, and every downstream subsystem consumes that profile. This is the property that makes the framework extensible to a new architecture without touching the trainer, distributed wrap, LoRA, MoE loss, or checkpointing code.

---

## 2. System architecture

### 2.1 Layered view

```
┌──────────────────────────────────────────────────────────────────┐
│                       Launcher (CLI entrypoints)                 │
│    bhaskera-train  bhaskera-tokenize  bhaskera-infer             │
│    bhaskera-dashboard  bhaskera-diag                             │
└──────────────────────────────────────────────────────────────────┘
                              │
                              │  load YAML → Config dataclass
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                         Ray Cluster (head + workers)             │
│                  ╔══════════════════════════════════╗            │
│                  ║      Ray Train TorchTrainer      ║            │
│                  ╚══════════════════════════════════╝            │
│                              │                                   │
│              ┌───────────────┼───────────────┐                   │
│              ▼               ▼               ▼                   │
│         worker_fn       worker_fn       worker_fn   (one per GPU)│
└──────────────────────────────────────────────────────────────────┘
                              │
                              │   each worker:
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  build_model ──► introspect_model ──► (optional) apply_lora      │
│         │                                                        │
│         ▼                                                        │
│  wrap_model (FSDP2 | DDP)  ◄── consumes ModelProfile             │
│         │                                                        │
│         ▼                                                        │
│  trainer.train ──► gradient accumulation loop ──► DCP checkpoint │
│         │         ▲             │                                │
│         │         │             ▼                                │
│         │         │   EvaluationLifecycle                        │
│         │         │   (validation + benchmarks mid-training)     │
│         │         │                                              │
│         │       ray.train.get_dataset_shard("train")             │
│         │                ▲                                       │
│         │                │                                       │
│         └────────► MultiLogger (MLflow + W&B + Ray)              │
└──────────────────────────────────────────────────────────────────┘
                              ▲
                              │  pre-tokenized parquet on shared FS
                              │
┌──────────────────────────────────────────────────────────────────┐
│  bhaskera-tokenize  (one-shot, runs on CPUs before training)     │
│  ↳ Ray Data → format renderer → TokenizerActor → parquet cache   │
└──────────────────────────────────────────────────────────────────┘
```

### 2.2 Subsystem responsibilities

| Subsystem              | Responsibility                                                                                | Key contract                                                                  |
| ---------------------- | --------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| `bhaskera.config`      | Parse YAML → typed dataclasses; defaults; round-trip via `.as_dict()` / `Config.from_dict()`. | Frozen dataclass tree; immutable shape; field names match YAML keys.          |
| `bhaskera.introspect`  | One-shot structural detection of the model: layer classes, expert modules, routers, dtype.    | Returns `ModelProfile`; no model-name checks; pure structural inference.      |
| `bhaskera.data`        | Build a `ray.data.Dataset` from a registered dataset name; cache tokenization on disk.        | `build_ray_dataset(cfg, world_size)` returns a tokenized, partitioned dataset.|
| `bhaskera.models`      | Instantiate the HF model, apply Liger Kernel, attach LoRA/QLoRA.                             | `build_model(cfg, device)` returns `(model, profile)`.                        |
| `bhaskera.distributed` | Wrap the model for FSDP2 or DDP; save/load DCP checkpoints; per-expert sharding.             | `wrap_model(model, cfg, local_rank, profile)`; `save_checkpoint`, `load_checkpoint`. |
| `bhaskera.trainer`     | Forward / backward / step loop; grad accumulation; MoE aux loss; throughput; checkpoint cadence; in-loop evaluation lifecycle. | `train(model, dataset, cfg, profile, rank, local_rank, tracker, world_size)`. |
| `bhaskera.launcher`    | CLI parsing, Ray cluster init, SLURM-aware GPU counting, `TorchTrainer.fit()`.                | Stdlib `argparse` entrypoints; exit codes preserved.                          |
| `bhaskera.utils`       | pynvml/psutil telemetry, MFU/throughput tracker, MLflow/W&B/Ray loggers.                     | `BaseLogger.log(metrics, step)`; never raises into the training loop.         |
| `bhaskera.evaluation`  | Distributed validation loop + benchmark execution; lazy tokenizer load; `Evaluator` facade.  | `Evaluator.run_validation(val_dataset)`, `Evaluator.run_benchmarks()`.        |
| `bhaskera.plugins`     | Dynamically loaded metric, benchmark, and optimizer extensions registered via decorators.     | `@register_metric`, `@register_benchmark`, `@register_optimizer`; lazy import on first use. |

---

## 3. Process model & control flow

Bhaskera runs as a Ray cluster. The two relevant process types are:

**Driver (one process).** Started by `bhaskera-train` on the head node. Responsible for: parsing config, loading plugins (`load_plugins(cfg)`), initialising Ray, building the tokenized `ray.data.Dataset` (a lazy reference — no I/O until consumed), instantiating `TorchTrainer`, calling `.fit()`. On SLURM the driver lives inside the `srun ray symmetric-run` invocation.

**Training workers (one process per GPU).** Spawned by Ray Train's `TorchTrainer`. Each runs `bhaskera.launcher.worker.worker_fn`. Each worker has its own `torch.distributed` rank, its own GPU, and its own dataset shard via `ray.train.get_dataset_shard("train")`. Workers communicate via NCCL collectives — Ray itself is *not* on the hot path during a training step.

The handoff from Ray to the training loop is deliberate: Ray handles cluster orchestration, placement groups, failure recovery, and dataset sharding; PyTorch handles everything inside a step (forward, backward, all-reduce, optimizer step). This separation means a step is governed by NCCL latency only — Ray's actor/RPC layer is bypassed once training has started.

### 3.1 Worker startup sequence

```
1.  Config.from_dict(cfg_dict)          # deserialise from Ray Train's train_loop_config
2.  load_plugins(cfg)                   # import optimizer plugins declared in cfg.plugins
3.  Ray Train context → rank, local_rank, world_size
4.  torch.cuda.set_device(local_rank)
5.  Seed (base_seed + rank) so each rank shuffles differently but reproducibly
6.  Get dataset shard for THIS rank (Ray Data handles partitioning)
7.  build_model(cfg, load_device)
    ├─ build_quantization_config(cfg)    → BitsAndBytesConfig (QLoRA) or None
    ├─ AutoModelForCausalLM.from_pretrained(..., quantization_config=...)
    ├─ introspect_model(model)           → ModelProfile
    ├─ _maybe_apply_liger_kernel(model)  → Triton-fused kernels (no-op if unsupported)
    └─ apply_lora(model, cfg, profile)   → PEFT, with dtype cast under FSDP
8.  wrap_model(model, cfg, local_rank, profile)
    ├─ FSDP path: per-expert fully_shard → per-layer fully_shard → root fully_shard,
    │             then activation checkpointing
    └─ DDP path:  activation checkpointing (pre-wrap) → DDP(...)
9.  build_logger(cfg, rank, world_size)  → MultiLogger (per-rank policy)
10. trainer.train(model, dataset, cfg, profile, rank, local_rank, tracker, world_size)
```

Step 2 (`load_plugins`) fires the `@register_optimizer` side effects for any plugins listed in `cfg.plugins.optimizers` before the optimizer is built in step 10.

### 3.2 Per-step control flow (inside `_run_epoch`)

```
for each optimizer step:
    micro_losses = []
    for micro_step in range(grad_accum):
        batch = next(loader_iter)
        is_last = (micro_step == grad_accum - 1)
        _set_grad_sync(model, enabled=is_last)   # FSDP2 set_requires_gradient_sync
                                                  # or DDP require_backward_grad_sync
        with autocast_ctx:                        # DDP only; FSDP uses MixedPrecisionPolicy
            out = model(**forward_kwargs)
            loss = (out.loss + aux_weight * extract_aux_loss(out, profile)) / grad_accum
            loss.backward()
        micro_losses.append(out.loss.detach())

    grad_norm = model.clip_grad_norm_(grad_clip)  # FSDP-aware when available

    if not isfinite(grad_norm):
        optimizer.zero_grad(); continue           # skip non-finite step

    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)

    log(window_loss, lr, grad_norm, mfu, throughput, expert_utilization)

    # ── In-loop evaluation ───────────────────────────────────────────
    if evaluator.should_run_validation(step):
        with eval_lifecycle.pause_for_evaluation():
            metrics = evaluator.run_validation(val_dataset)
            logger.log(metrics, step)

    if evaluator.should_run_benchmark(step):
        with eval_lifecycle.pause_for_evaluation():
            metrics = evaluator.run_benchmarks()
            logger.log(metrics, step)
```

Two non-obvious properties of this loop:

1. **Single-precision-policy mixed precision under FSDP.** When `strategy=fsdp`, the autocast context is a no-op (`contextlib.nullcontext()`). FSDP2's `MixedPrecisionPolicy(param_dtype, reduce_dtype, output_dtype)` is the *single source of truth* for mixed precision — wrapping the forward in `torch.autocast` on top of FSDP2 produces double-casting and breaks gradient reduction. Under `strategy=ddp` autocast is the only mixed-precision mechanism, so it is enabled.

2. **The grad-sync toggle is the only thing that makes gradient accumulation efficient.** Without it, every micro-step triggers a full all-reduce; with it, the all-reduce fires once per optimizer step. `_set_grad_sync` dispatches by wrapper type — FSDP2's composable `set_requires_gradient_sync(model, enabled)`, DDP's `require_backward_grad_sync`, or no-op for single-GPU. DDP's `static_graph=True` mode is incompatible with mid-run toggling and is detected via a stable attribute `_bhaskera_static_graph` set in `ddp.py`.

---

## 4. Module-by-module deep dive

### 4.1 `config` — single source of truth

The config system is a tree of `@dataclass` types:

```
Config
├── PluginsConfig          (plugins.optimizers: list of import paths)
├── ModelConfig            (name, dtype, attn_impl, trust_remote_code, use_liger_kernel, quantization)
├── DataConfig             (name, seq_len, is_cpt, tokenized_path, format, path, ...)
├── LoraConfig
├── MoEConfig
├── OptimizerConfig        (backend: "default"|"torch"|"plugin", class_name, name, kwargs)
├── TrainingConfig         (batch_size, grad_accum, lr, ..., optimizer: OptimizerConfig)
│   └── DistributedConfig
│       ├── FSDPConfig
│       └── DDPConfig
├── CheckpointConfig
├── LoggingConfig
├── EvaluationConfig       (enabled, offload_optimizer, validation, benchmarks)
│   ├── ValidationConfig   (dataset, every_n_steps, every_n_epochs, metrics: list[str])
│   └── BenchmarksConfig   (every_n_steps, every_n_epochs, tasks: list[str])
├── InferenceConfig
│   ├── TurboQuantConfig
│   └── SpeculativeConfig
└── MonitoringConfig
    └── MetricsConfig
```

**Design.** Every field has an explicit default; YAML is parsed via `yaml.safe_load` and the resulting dict is fed through `_dict_to_config`, which calls each leaf constructor explicitly rather than `**unpack`. This is intentional: an unknown YAML key (typo) is silently ignored at the leaf level, but the entire shape of the config tree is invariant — downstream code reading `cfg.training.distributed.fsdp.sharding_strategy` cannot ever see `AttributeError`. The trade-off is verbosity in `_dict_to_config`; the win is that every consumer codepath is statically typed and discoverable.

`Config.as_dict()` and `Config.from_dict()` provide a round-trip. This matters because Ray Train serialises `train_loop_config` (the config) and ships it to each worker — the dataclass tree must reconstruct identically on the worker side. The driver hands `cfg.as_dict()` to `TorchTrainer(train_loop_config=...)` and each worker calls `Config.from_dict(cfg_dict)` at the start of `worker_fn`.

### 4.2 `introspect` — structural model detection

This is the keystone of the framework. `introspect_model(model) -> ModelProfile` walks the loaded HuggingFace model exactly once and returns:

```python
@dataclass
class ModelProfile:
    is_moe: bool
    num_experts: int
    num_shared_experts: int
    experts_per_token: int
    decoder_layer_cls: Optional[type]      # e.g. LlamaDecoderLayer
    expert_module_cls: Optional[type]      # e.g. Qwen2MoeMLP
    expert_modules: list[nn.Module]        # concrete refs for per-expert FSDP wrap
    router_module_names: list[str]         # dotted names — used by LoRA to exclude
    has_aux_loss: bool
    aux_loss_attr: str
    lora_targets: list[str]                # short names: q_proj, k_proj, ...
    model_dtype: torch.dtype
    num_hidden_layers: int
    model_type: str
```

**No model-name checks anywhere.** Detection is purely structural:

- **Decoder layer class.** Walks known attribute paths (`model.layers`, `transformer.h`, `gpt_neox.layers`, `model.decoder.layers`) looking for a `nn.ModuleList` whose `len` equals `config.num_hidden_layers` and whose children share a class. Falls back to a brute-force walk if no path matches.

- **MoE expert modules.** Finds every `nn.ModuleList` whose leaf attribute name is in `{"experts", "expert", "routed_experts", "moe_experts"}` with `len >= 2`. Records the parent path of each container.

- **Routers.** The historical bug here was substring matching: looking for `"gate" | "router" | "switch"` in module names finds `gate_proj` in every SwiGLU model (Llama, Mistral, Qwen, Gemma) and incorrectly flags it as a router. The current implementation **restricts router search to modules whose parent is the parent of an experts container**. SwiGLU's `gate_proj` lives under `MLP`, never under a parent of `experts`, so it is correctly classified as a regular projection and included in LoRA targets.

- **LoRA targets.** Walks one sample decoder layer, collects every `nn.Linear` short name, removes any that appear in `router_module_names`. Returns a sorted list of short names suitable for `peft.LoraConfig(target_modules=...)`.

The contract for `ModelProfile` is consumed by `distributed.fsdp` (for per-expert / per-layer sharding), `distributed.ddp` (to force `find_unused_parameters=True` on MoE), `models.lora` (for target discovery and router freezing), and `trainer.moe` (for aux loss extraction with shape inference).

**Adding support for a new architecture** generally requires zero changes here — if the new model uses a `nn.ModuleList` of layers and (for MoE) an `experts` container, introspection just works. If a model has a completely novel layout, extend `_LAYER_CONTAINER_ATTRS` or `_EXPERT_LEAF_NAMES` rather than special-casing downstream code.

### 4.3 `data` — Ray Data pipeline and persistent tokenization

The data layer has three pieces that compose: a **dataset registry**, a **format registry**, and a **tokenizer with a persistent on-disk cache**. It also supports a **Continual Pre-Training (CPT)** mode for packing raw text corpora.

**Dataset registry (`data.registry`).** Two parallel registries:
- `REGISTRY` — `name -> (cfg, world_size) -> tokenized ray.data.Dataset`
- `RAW_REGISTRY` — `name -> (cfg, split=None) -> raw ray.data.Dataset`
- `TEXT_COL` — `name -> column name to tokenize`

`@register("ultrachat")` registers the tokenized builder; `@register_raw("ultrachat", text_col="prompt")` registers the raw builder used by `bhaskera-tokenize`. Keeping them separate lets new datasets opt into either or both modes.

Built-in datasets: `ultrachat`, `openassistant`, `redpajama`, `local_chat` (generic JSONL/JSON/Parquet loader), `local_cpt` (CPT continuous packing).

**Continual Pre-Training (`data/datasets/local_cpt.py`).** When `data.is_cpt: true` is set, the tokenizer pipeline reads raw JSONL text documents and **continuously packs** tokens into fixed-length chunks of `seq_len` tokens, discarding no context across document boundaries. This matches the pre-training objective and eliminates padding waste. The CPT path is entirely self-contained: the rest of the data pipeline (cache key, parquet output, Ray partitioning) is unchanged.

**Format registry (`data.formats`).** A renderer is a function `(row, tokenizer, options) -> str`. Built-ins:
- `chatml` — `messages: [{role, content}, ...]` → `tokenizer.apply_chat_template(...)` with manual fallback for tokenizers that have no template.
- `alpaca` — `instruction` / `input` / `output` columns with the classic Alpaca prose template; `use_chat_template: true` option switches to a two-turn chat rendering.
- `sharegpt` — `conversations: [{from, value}, ...]` with a configurable `role_map`.

Renderers are imported lazily (`_ensure_builtins_loaded`) because Ray pickles the `_TokenizerActorFactory` and ships it to every worker — importing HuggingFace at registry-import time would bloat the pickle.

**Tokenization & cache (`data.tokenize`).**

The cache key is `sha256(model_name | seq_len | dataset_name | format_name | format_options)[:16]` — deterministic across runs and machines, unlike Python's `hash()` which is randomised per-process (PEP 456).

The cache directory contains parquet shards and a `metadata.json`:

```json
{
  "model_name": "bharatgenai/Param2-17B-A2.4B-Thinking",
  "seq_len": 2048,
  "dataset_name": "ultrachat",
  "num_rows": 207866,
  "schema": ["input_ids", "attention_mask", "labels"],
  "created_at": "2025-11-12T03:18:42.193Z",
  "bhaskera_version": "2.2.0",
  "format_name": "chatml",
  "format_options": {}
}
```

`_verify_cache` checks all of: file exists, metadata fields match, at least one `.parquet` file present. Older caches (without `format_name`) keep validating when the caller requests "no format" — backwards compatible.

The tokenizer itself is a stateful Ray actor (`TokenizerActor`) loaded once per worker. It pads to `seq_len`, sets `labels[attention_mask == 0] = -100` so the LM loss ignores pad positions, filters empty rows, and emits `input_ids / attention_mask / labels` as numpy arrays.

`_compute_num_partitions` rounds the partition count up to a multiple of `world_size` (minimum 16) so every rank sees an equal number of shards — preventing the empty-shard hang that would otherwise occur when `num_partitions < world_size`.

### 4.4 `models` — HF loader, Liger Kernel, LoRA, QLoRA

**Loader (`models.loader.build_model`).** Four responsibilities:

1. **Quantization config.** `build_quantization_config(cfg)` returns a `BitsAndBytesConfig` for QLoRA mode, or `None` for the default path. When `model.quantization: qlora` and `strategy: fsdp`, a `ValueError` is raised immediately with a clear message — FSDP2 cannot shard `Params4bit` tensors.

2. **Load.** `AutoModelForCausalLM.from_pretrained(name, torch_dtype=..., attn_implementation=..., quantization_config=...)` with `low_cpu_mem_usage=True` and `trust_remote_code=cfg.model.trust_remote_code`. For FSDP the model is loaded onto CPU (FSDP handles GPU migration during sharding); for DDP it is moved to the target GPU immediately.

3. **Introspect.** Calls `introspect_model` and stores the resulting `ModelProfile`.

4. **Liger Kernel.** `_maybe_apply_liger_kernel` calls `liger_kernel.transformers._apply_liger_kernel_to_instance(model=model)`, which dispatches on `model.config.model_type` and replaces RMSNorm / RoPE / SwiGLU / CrossEntropy with Triton-fused versions for supported architectures. Any failure (package missing, unsupported architecture, runtime error) is downgraded to a warning. **Order matters.** Liger is applied *before* LoRA so PEFT wraps the already-fused modules (LoRA adapters attach to Linear children, which Liger does not touch), and *before* FSDP wrap so sharding sees the final module classes.

**Quantization (`models.quantization`).** The `build_quantization_config(cfg)` helper:
- Returns `None` when `model.quantization` is `"none"` (the default) — the existing load path is bit-for-bit identical.
- Returns a `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)` when `model.quantization: qlora`.
- Raises `ValueError` when `model.quantization: qlora` and `strategy: fsdp`.
- Imports are deferred so environments without `bitsandbytes` installed are unaffected when quantization is disabled.

**LoRA (`models.lora.apply_lora`).** The whole file is annotated because of a subtle FSDP2-vs-DDP dtype interaction:

- PEFT initialises LoRA A/B in float32 for numerical stability.
- FSDP2's `_init_mp_dtypes()` requires every parameter inside a shard group to share the same **original** dtype. If the base model is bf16 and LoRA is fp32, FSDP raises `AssertionError: FSDP expects uniform original parameter dtype`.
- **FSDP path:** cast LoRA params to base dtype after `get_peft_model`. AdamW moments are then stored in bf16, which is slightly noisier; this is mitigated by `reduce_dtype=float32` in the `MixedPrecisionPolicy`.
- **DDP / QLoRA path:** keep PEFT's fp32 default. Casting under DDP is actively harmful — AdamW state is allocated to match param dtype, and at β₂=0.95 with bf16 params the second-moment update quantises most LoRA gradients to zero at typical LRs (~2e-4), starving the optimiser. DDP's autocast already handles the bf16 forward.

Router freezing for MoE LoRA is gated by `cfg.lora.freeze_router=True`. It uses `profile.router_module_names` (from structural detection) as the primary signal and falls back to keyword matching only if structural detection produced an empty list.

### 4.5 `distributed` — FSDP2, DDP, and DCP checkpointing

**Composability is the strategy.** Bhaskera uses FSDP2 (the new `torch.distributed._composable.fsdp.fully_shard` API), not FSDP1 (`FullyShardedDataParallel`). FSDP2 is **composable** with other PyTorch features (TP, PP, AC) and does not introduce a wrapper class — `fully_shard(module)` mutates the module in place and the wrapped model is still accessible by its original references.

**The three-step shard plan** (`distributed.fsdp.wrap_fsdp2`):

1. **Per-expert sharding** (MoE only, when `shard_experts_individually=True`). Each `expert` in `profile.expert_modules` is shard-wrapped. This is what enables MoE memory savings: an all-gather during forward only materialises the **activated** experts (the ones the router selected for this batch), not the full set.
2. **Per-decoder-layer sharding.** Each instance of `profile.decoder_layer_cls` is shard-wrapped.
3. **Root sharding.** The model itself is shard-wrapped, which finalises the FSDP state and enables `model.clip_grad_norm_(...)` for sharded grads.

`MixedPrecisionPolicy` is constructed from `fsdp.param_dtype / reduce_dtype / buffer_dtype`. For MoE, `reduce_dtype="auto"` becomes `float32` because gradients are sparse and low-precision reductions amplify routing noise.

**DDP (`distributed.ddp.wrap_ddp`).** Two MoE-specific resolutions:
- `find_unused_parameters=True` is **forced on** when `profile.is_moe`, regardless of config, because expert routing means a different subset of params is touched on every forward pass.
- `static_graph` is **forced off** when `find_unused_parameters=True`, because DDP itself rejects the combination. The effective value is stashed on the wrapper as `_bhaskera_static_graph` so the training loop's `_set_grad_sync` can detect it without re-reading the config.

Activation checkpointing under DDP is applied **before** the DDP wrap (so DDP's parameter-graph snapshot sees the checkpoint-wrapped modules). The mechanism is shared with FSDP (`distributed.activation_ckpt.apply_activation_checkpointing`), which dispatches between the composable API (torch ≥ 2.4) and the legacy NO_REENTRANT wrapper.

**Checkpointing (`distributed.checkpoint`).** All checkpoints use `torch.distributed.checkpoint` (DCP) — every rank writes its shard, no rank gathers the full state dict. The on-disk layout for one checkpoint is:

```
<path>/
    model/...           # DCP shard files for model state
    optim/...           # DCP shard files for optimizer state
    meta.json           # {"step": int, "avg_loss": float}
    .complete           # sentinel — written last, only by rank 0
```

**Atomic save flow:**
1. All ranks write shards to `<path>.tmp` (DCP is a collective).
2. `dist.barrier()` waits for everyone.
3. Rank 0 renames `<path>.tmp` → `<path>`.
4. Rank 0 writes `meta.json`.
5. Rank 0 writes `.complete` last.
6. Rank 0 prunes old checkpoints.
7. `dist.barrier()` releases everyone.

`maybe_resume` only considers directories with a `.complete` sentinel — a crashed save leaves an incomplete directory but never produces a half-loaded model. The `_dcp_save` / `_dcp_load` helpers shim across the PyTorch 2.4 → 2.5 API change (the `checkpoint_id=` kwarg was removed in 2.5 in favour of `storage_writer=FileSystemWriter(path)`).

### 4.6 `trainer` — the pure training loop

The trainer has **no Ray dependency, no SLURM logic, no distributed init.** It assumes `torch.distributed` is already initialised and that the model is already wrapped. This is what makes it usable outside Ray (e.g. for unit tests with a fake `dist` group, or for raw `torchrun` from a script).

**Public entry point.** `trainer.train(model, dataset, cfg, profile, rank, local_rank, tracker, world_size)`.

Inside, `_run_epoch` is the workhorse described in §3.2. A few subtleties worth surfacing:

- **Loss EMA & spike ratio.** A running EMA of the loss (`loss_ema_alpha=0.05`) and the ratio `window_loss / loss_ema` are logged every step. The spike ratio is a cheap leading indicator of divergence — values consistently above 2× usually mean LR is too high or aux loss is fighting the main loss.
- **Non-finite grad protection.** If `clip_grad_norm_` returns NaN/Inf, the optimizer step is skipped, `optimizer.zero_grad(set_to_none=True)` is called, and the step counter does **not** advance. After `cfg.training.max_grad_skip_steps` consecutive skips the run aborts.
- **Best-by-loss checkpoint retention.** `save_and_prune` keeps `cfg.checkpoint.keep_last_n` checkpoints sorted by ascending `avg_loss` — not by step.

**Pluggable optimizer (`trainer.optim` + `trainer.optimizer_registry`).** `build_optimizer(model, cfg)` dispatches on `cfg.training.optimizer.backend`:

- `"default"` — AdamW (fused when available) with the standard 1-D / 2-D weight decay split. This is the backward-compatible path when no `optimizer` block is present.
- `"torch"` — any class in `torch.optim` by `class_name`, e.g. `class_name: "Adadelta"`. kwargs are forwarded from `optimizer.kwargs`.
- `"plugin"` — looks up `name` in `OPTIMIZER_REGISTRY` (populated by `@register_optimizer`), calls the registered builder function `(model, train_cfg) -> Optimizer`.

`optimizer_registry.py` owns `OPTIMIZER_REGISTRY: Dict[str, Callable]` and the `@register_optimizer(name)` decorator. Plugins are loaded before optimizer construction by `load_plugins(cfg)` in `worker_fn`.

**Eval lifecycle (`trainer.eval_lifecycle`).** `EvaluationLifecycle` is a context manager used inside `_run_epoch`:

- **Teardown phase:** destroys the active dataloader and prefetch buffers, offloads optimizer CUDA state to CPU (`_offload_optimizer_to_cpu`), and records the dataset cursor position.
- **Evaluation phase:** model stays in its distributed wrapper; caller runs validation / benchmarks inside the `with` block.
- **Restore phase:** restores the optimizer to GPU (`_restore_optimizer_to_gpu`), rebuilds the dataloader from the cursor position (Ray Data `dataset.skip()` — a metadata-only operation), and triggers `gc.collect()` + `torch.cuda.empty_cache()`.

The `_bhaskera_offloaded` flag on the optimizer prevents double-offload if `run_distributed_validation` is called from inside a lifecycle block that already offloaded the optimizer.

**MoE auxiliary loss (`trainer.moe`).** Three layers:

1. `extract_aux_loss(out, profile)` tries well-known attribute names (`aux_loss`, `router_aux_loss`, `moe_loss`, `load_balancing_loss`) first. Falls back to computing it from `out.router_logits`.
2. `_normalize_router_logits` unwraps `tuple/list of Tensor` or `tuple/list of (logits, indices)` shapes.
3. `_infer_logit_kind` classifies by last-dim: `_KIND_GATE` (scalar), `_KIND_FULL` (softmax over all experts), `_KIND_TOPK` (Param2-style — only the selected k).

`compute_expert_utilization` produces `expert/load_max`, `expert/load_min`, `expert/load_std`, `expert/imbalance_ratio` for the logger. A healthy run shows `load_max / load_min ≤ ~3×`; values above 10× indicate routing collapse.

### 4.7 `launcher` — CLI entry points

Each entry point in `bhaskera.launcher.*` is wired to a console script in `pyproject.toml`:

| Command              | Module                         | Purpose                                                      |
| -------------------- | ------------------------------ | ------------------------------------------------------------ |
| `bhaskera-train`     | `launcher.train:main`          | Train via Ray Train `TorchTrainer`.                          |
| `bhaskera-tokenize`  | `launcher.tokenize:main`       | One-shot tokenization to a parquet cache.                    |
| `bhaskera-infer`     | `launcher.infer:main`          | Inference CLI with TurboQuant and speculative options.       |
| `bhaskera-diag`      | `launcher.diagnostics:main`    | NCCL all-reduce + bandwidth test for the cluster.            |
| `bhaskera-dashboard` | `launcher.dashboard:main`      | Manage the MLflow UI lifecycle on the login node.            |

**`launcher.train`** is the most architecturally important. It does:

1. Parse args, load config.
2. `load_plugins(cfg)` — import optimizer plugins (driver-side; workers also call this in `worker_fn`).
3. `setup_monitoring(cfg)` → `MonitoringContext` (Ray Dashboard kwargs + MLflow UI URL discovery).
4. `_init_ray(monitoring)` — connects to `RAY_ADDRESS` if set (SLURM path), else starts a local Ray cluster.
5. `_count_gpus()` — returns `SLURM_NNODES × SLURM_GPUS_PER_NODE` if both are set (multi-node SLURM), else `torch.cuda.device_count()`.
6. `build_ray_dataset(cfg, world_size=num_workers)` — builds the lazy `ray.data.Dataset` reference.
7. `TorchTrainer(train_loop_per_worker=worker_fn, train_loop_config=cfg.as_dict(), datasets={"train": ray_dataset}, ...).fit()`.

**`launcher.tokenize`** runs entirely on CPUs — no GPU is required. It prefetches the tokenizer in the driver process before `ray.init`, then for each requested split builds the raw dataset and pipes it through `persist_tokenized`. It prints a copy-pasteable YAML snippet at the end with the resulting `tokenized_path` / `val_tokenized_path` keys.

**`launcher.dashboard`** manages the MLflow UI as a detached subprocess on the login node. Persisted config at `~/.bhaskera/mlflow-ui.json`; PID at `~/.bhaskera/mlflow-ui.pid`; log at `~/.bhaskera/mlflow-ui.log`. Subcommands: `start | stop | status | tunnel`. The `tunnel` subcommand prints the exact `ssh -L 5000:localhost:5000 user@login-node` command for the user's laptop.

### 4.8 `utils` — telemetry, throughput, loggers

**`utils.system_stats`** is the production telemetry path. Per-GPU (via pynvml) it collects utilisation, memory, temperature, power, fan, clock (SM/mem), PCIe Tx/Rx, NVLink Tx/Rx, ECC error counts, throttle reasons, and performance state. Per-host (via psutil) it collects per-process CPU%, per-core CPU%, load average, memory, swap, disk I/O B/s, network I/O B/s, and open file descriptors. Rate-derived counters (disk/net) are computed from a delta against the previous call; the first call returns 0 to avoid spurious initial spikes. All metric keys are flat and slash-separated (`gpu/0/util_pct`, `cpu/mem_used_mb`) for clean tag extraction in Prometheus / MLflow.

`utils.gpu_stats` is a stripped-down predecessor preserved for backwards compatibility. New code should call `utils.system_stats.system_stats(gpu=..., cpu=...)`.

**`utils.throughput.ThroughputTracker`** computes step time, tokens/sec, samples/sec, and **MFU** (Model FLOPs Utilisation). The MFU formula uses the Chinchilla 3-pass approximation:

```
flops_per_token ≈ 6 × params              # forward + backward
achieved_flops_per_sec = flops_per_token × tokens / dt / world_size
MFU% = 100 × achieved / peak_flops_per_gpu
```

`peak_flops_per_gpu` is configured via `monitoring.metrics.peak_tflops_per_gpu` (default 312 = A100 bf16; set 165 for RTX 4090, 121 for L4, etc.). The first `warmup_steps` step times are dropped from the moving average because they include compile / cache warmup.

For LoRA fine-tuning the dominant FLOP is still through the frozen base weights, so `params_for_flops` is the **total** parameter count, not the trainable count.

**Loggers (`utils.loggers`).** `build_logger(cfg, rank, world_size) -> Optional[BaseLogger]` returns a `MultiLogger` that fans out to each enabled backend:

- **Ray** (`RayMetricsLogger`) — every rank, pushes to Ray Dashboard's Prometheus.
- **MLflow** — rank 0 by default (set `mlflow_log_all_ranks: true` for per-rank breakdowns). Push-based, file-store at `~/mlflow-runs`. Uses a bounded queue + daemon thread to absorb store jitter.
- **W&B** — rank 0 only when `wandb` is in the tracker list.

`MultiLogger.log` swallows per-child exceptions and continues — one flaky tracker never breaks training.

### 4.9 `evaluation` — distributed validation and benchmarks

The evaluation subsystem provides in-training model evaluation without model reload or distributed teardown. It is opt-in via `evaluation.enabled: true` in YAML.

**`evaluation.evaluator.Evaluator`** is the facade consumed by the training loop. It is constructed once in `worker_fn` and passed into `trainer.train`. Key methods:

- `should_run_validation(step)` — returns True when `step % cfg.evaluation.validation.every_n_steps == 0`.
- `should_run_benchmark(step)` — returns True when `step % cfg.evaluation.benchmarks.every_n_steps == 0`.
- `run_validation(val_dataset, optimizer=None)` — delegates to `run_distributed_validation`. Accepts an optional optimizer reference for GPU memory reclamation.
- `run_benchmarks(tokenizer=None)` — delegates to `run_benchmarks`. The tokenizer is lazy-loaded on first use (`AutoTokenizer.from_pretrained` is deferred until the first benchmark call).

**`evaluation.validation.run_distributed_validation`** is the core validation loop:

- Optimizer state is optionally offloaded to CPU before the eval pass and restored after, freeing GPU memory. Self-sufficient: it applies its own offload if no enclosing `EvaluationLifecycle` already did (`_bhaskera_offloaded` flag prevents double-offload).
- Loss is computed in **sequence chunks** (`LOSS_CHUNK_SEQ_LEN = 512`) to bound peak memory regardless of vocab size or whether Liger's fused CE kernel is active in eval mode.
- Distributed aggregation: per-rank losses are `all_reduce`'d; metric results are computed on rank 0 and `broadcast_object_list`'d to all ranks.
- Memory is reclaimed every `MEMORY_CLEANUP_EVERY_N_BATCHES` batches via `gc.collect()` + `cuda.empty_cache()`.

**`evaluation.benchmark_runner.run_benchmarks`** runs tasks registered in `BENCHMARK_REGISTRY`. Each benchmark class implements `run(model, tokenizer, cfg) -> dict[str, float]`. Results are broadcast from rank 0 to all ranks after the run.

**`evaluation.registry`** holds `METRIC_REGISTRY` and `BENCHMARK_REGISTRY`. Both support lazy import: if a name is not already in the registry, `get_metric("loss")` imports `bhaskera.plugins.metrics.loss` and the `@register_metric` side effect fires. Similarly for benchmarks. This means plugins do not need to be pre-imported — the registry imports them on demand.

### 4.10 `plugins` — optimizer, metric, and benchmark extensions

The plugin system is the extension point for user-supplied components. All plugin types follow the same pattern: a decorator registers a class or builder function by name; the framework looks up the name in the registry at runtime; dynamic import handles first-use loading.

**Optimizer plugins (`plugins/optimizers/`, `trainer/optimizer_registry.py`).**

```python
from bhaskera.trainer.optimizer_registry import register_optimizer

@register_optimizer("lion")
def build_lion(model, train_cfg):
    from bhaskera.trainer.optim import _get_default_param_groups
    opt_cfg = train_cfg.optimizer
    param_groups = _get_default_param_groups(model, opt_cfg.kwargs.get("weight_decay", 0.0))
    return Lion(param_groups, lr=opt_cfg.kwargs.get("lr", train_cfg.lr), ...)
```

YAML declaration:
```yaml
plugins:
  optimizers:
    - "bhaskera.plugins.optimizers.lion"

training:
  optimizer:
    backend: "plugin"
    name: "lion"
    kwargs:
      lr: 5.0e-6
      betas: [0.9, 0.99]
```

Built-in: `lion` (EvoLved Sign Momentum / Lion optimizer).

**Metric plugins (`plugins/metrics/`, `evaluation/registry.py`).**

```python
from bhaskera.evaluation.registry import register_metric

@register_metric("my_metric")
class MyMetric:
    def compute(self, predictions, labels, losses) -> dict:
        return {"validation/my_metric": ...}
```

Enabled in YAML: `evaluation.validation.metrics: [loss, perplexity, token_accuracy, my_metric]`

Built-ins: `loss`, `perplexity`, `token_accuracy`.

**Benchmark plugins (`plugins/benchmarks/`, `evaluation/registry.py`).**

```python
from bhaskera.evaluation.registry import register_benchmark

@register_benchmark("my_bench")
class MyBenchmark:
    def run(self, model, tokenizer, cfg) -> dict:
        return {"benchmark/my_bench_accuracy": ...}
```

Enabled in YAML: `evaluation.benchmarks.tasks: [hellaswag, arc_easy, arc_challenge, my_bench]`

Built-ins: `hellaswag` (full batched, distributed all-gather), `arc_easy` (full batched, distributed all-gather), `arc_challenge` (placeholder).

**`plugins/loader.py`.** `load_plugins(cfg)` iterates `cfg.plugins.optimizers`, imports each module path, and surfaces failures as `ImportError`. Called in both the driver (before `TorchTrainer.fit()`) and each worker (at the start of `worker_fn`) to ensure the registry is populated in both processes.

---

## 5. Cross-cutting design decisions

### 5.1 Ray + FSDP2 over `torchrun` + FSDP1

`torchrun` is fine for single-node and adequate for multi-node SLURM, but it provides no cluster-level introspection, no built-in dataset sharding, no actor model for failed-worker recovery, and no dashboard. Ray provides all of these and degrades gracefully to single-node mode (`ray.init()` with no address). The Ray hot path is bypassed inside a training step — NCCL is the only synchronous dependency once `train_loop_per_worker` is running — so the orchestration overhead is paid once at startup, not per step.

FSDP2's composable API was chosen over FSDP1's wrapper class because (a) it composes with Tensor Parallel and Pipeline Parallel for future scaling, (b) it does not introduce a wrapper that breaks `model.foo` attribute access, and (c) per-expert sharding is a natural three-step composition (`fully_shard(expert) → fully_shard(layer) → fully_shard(root)`) that has no clean equivalent in FSDP1.

### 5.2 Push-based observability over Prometheus / Grafana

The original observability story used Prometheus + Grafana. On a SLURM cluster this requires the user to open a port on every compute node, generate a fresh scrape config every job, SSH-tunnel into a Grafana running on a particular login node, and remember which login node that was last time. None of this requires the **training process** to do anything — but all of it requires the user to do a lot.

The push model inverts this: the training process writes to an MLflow file-backed store under `$HOME/mlflow-runs`. Login nodes share `$HOME` with compute nodes on the great majority of HPC clusters, so the UI process on the login node reads the same directory the trainer writes to. No server, no auth, no scrape config, no port collisions. The downside is that the file store is not horizontally scalable — for very large clusters (≥ 128 nodes) the metric write rate may bottleneck on filesystem metadata operations. At that scale, switching to an MLflow tracking server with a real database is straightforward (`mlflow_tracking_uri: http://...`).

### 5.3 Persistent tokenization cache

Live tokenization is fine in development; in production it wastes ~10–30 minutes of GPU time at the start of every run. The cache is keyed deterministically (`sha256` not `hash()`) so a re-run with the same config returns immediately, and a config change (different `seq_len`, different `format`, different `format_options`) invalidates automatically. The cache is shared across runs and across users — multiple training jobs can read the same cache directory concurrently because parquet reads are stateless. Writes use `<path>.tmp` semantics so a crashed tokenize run does not poison the cache.

### 5.4 Strategy-gated LoRA dtype

See §4.4. The summary is: FSDP requires uniform original parameter dtype within a shard group, so LoRA params must be cast to the base dtype. DDP has no such requirement, and casting under DDP starves AdamW's second moment at typical LoRA learning rates. The dtype handling is therefore conditional on `cfg.training.distributed.strategy`. This is the most subtle interaction in the codebase and is documented in detail inside `models/lora.py`.

### 5.5 Atomic checkpointing with `.complete` sentinel

A naive checkpoint save can be interrupted mid-write (SLURM time limit, OOM, NCCL timeout). Without a guard, `maybe_resume` happily loads the partial state dict and you train from a corrupt checkpoint without noticing. The `.complete` sentinel is the last file written by rank 0 — its presence is the only condition under which `maybe_resume` considers a checkpoint valid. The `<path>.tmp` → `<path>` rename adds belt-and-braces atomicity at the directory level.

### 5.6 Single source of truth for mixed precision (under FSDP)

FSDP2's `MixedPrecisionPolicy` controls (a) the cast applied to params on forward, (b) the dtype of gradient reductions, (c) the dtype of buffers (e.g. norm running mean/var). Wrapping the forward in `torch.autocast` on top of this produces *double casting*. The training loop therefore enables autocast **only** when `strategy == "ddp"`:

```python
use_autocast = (strategy == "ddp" and device.type == "cuda")
autocast_ctx = (
    torch.autocast("cuda", dtype=autocast_dtype)
    if use_autocast
    else contextlib.nullcontext()
)
```

### 5.7 NCCL configuration at job start, not in Python

`scripts/submit.sh` handles NCCL setup before Python is invoked, because `NCCL_*` env vars are read on `init_process_group`, which Ray Train initialises before `train_loop_per_worker` runs. IB / RoCE detection is a shell-level check (`/sys/class/infiniband/*/ports/*/state`, `ibstat`). `NCCL_TIMEOUT`, `TORCH_NCCL_BLOCKING_WAIT`, `TORCH_DISTRIBUTED_TIMEOUT`, and `NCCL_ASYNC_ERROR_HANDLING` must be set before the first import of torch on a fresh process.

### 5.8 QLoRA is DDP-only

BitsAndBytes loads base weights as `Params4bit` tensors. FSDP2's `fully_shard` expects standard PyTorch `nn.Parameter` objects and produces undefined behaviour (or crashes) when it attempts to shard `Params4bit`. The `build_quantization_config` function enforces this constraint early (before any model work begins) with a `ValueError` rather than letting FSDP fail internally. Each DDP rank holds a full 4-bit copy of the model; at 17B in NF4 this is ~8.5 GB/GPU, well within 24 GB headroom.

### 5.9 Evaluation memory reclamation via optimizer offload

Running a validation forward on the same GPU that holds training state can OOM when: optimizer second moments (Adam m2) are stored in fp32 (~2× model size), lm_head logits are large-vocab, and sequence length is long. `EvaluationLifecycle` and `run_distributed_validation` both implement optimizer offload-restore cycles using `_offload_optimizer_to_cpu` / `_restore_optimizer_to_gpu`. The `_bhaskera_offloaded` flag on the optimizer ensures the offload only happens once even when both call sites are active. Additionally, validation loss is chunked over sequence length (`LOSS_CHUNK_SEQ_LEN = 512`) to bound logit tensor size independent of vocab.

---

## 6. Data flow

The full lifecycle of one token, from raw JSONL on disk to an entry in MLflow:

```
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 1 — Tokenization (one-shot, CPU only)                    │
└─────────────────────────────────────────────────────────────────┘

   raw JSONL/Parquet  ──► ray.data.read_json/parquet ──► raw ray.Dataset
                                                              │
                                                              ▼
                                                  TokenizerActor (lazy)
                                                  ┌───────────────────┐
                                                  │ 1. format renderer│  (chatml | alpaca | sharegpt)
                                                  │    OR CPT packing │  (is_cpt=True: continuous pack)
                                                  │ 2. HF tokenize    │  (pad to seq_len, truncate)
                                                  │ 3. labels mask    │  (pad positions → -100)
                                                  │ 4. drop empty rows│
                                                  └───────────────────┘
                                                              │
                                                              ▼
                                            persist_tokenized → parquet shards
                                                              │
                                                              ▼
                                       <cache_dir>/<dataset>_<sha256-hash>/
                                            ├── *.parquet (50k rows each)
                                            └── metadata.json

┌─────────────────────────────────────────────────────────────────┐
│  PHASE 2 — Training (Ray Train + FSDP2/DDP)                     │
└─────────────────────────────────────────────────────────────────┘

   YAML config ──► load_config ──► Config dataclass tree
                                          │
                                          ▼
                          load_plugins(cfg)  (optimizer plugins)
                                          │
                                          ▼
                          driver: ray.init, setup_monitoring
                                          │
                                          ▼
                          build_ray_dataset(cfg, world_size)
                                          │
                              (lazy parquet read + repartition)
                                          │
                                          ▼
                  TorchTrainer(train_loop_per_worker=worker_fn, ...).fit()
                                          │
                              ┌───────────┴───────────┐
                              ▼                       ▼
                       worker (rank 0)         worker (rank N-1)
                              │
   ray.train.get_dataset_shard("train")  ◄── auto-partitioned by Ray Data
                              │
                              ▼
             build_quantization_config → build_model → introspect → apply_lora
                              │
                              ▼
                       wrap_model (FSDP2 per-expert / per-layer / root)
                              │
                              ▼
                  ┌─────────────────────────────────────────────┐
                  │       training loop (every micro-step)      │
                  │                                             │
                  │  batch ──► forward (FSDP: no autocast,      │
                  │             DDP: autocast)                  │
                  │       ──► main_loss + aux_weight*aux_loss   │
                  │       ──► backward (sync off on micro 0..k-1│
                  │                     sync on micro k)        │
                  │  every k micro-steps:                       │
                  │       ──► clip_grad_norm_                   │
                  │       ──► optimizer.step + scheduler.step   │
                  │       ──► log metrics                       │
                  │                                             │
                  │  [periodic] EvaluationLifecycle window:     │
                  │       ──► offload optimizer to CPU          │
                  │       ──► run_distributed_validation /      │
                  │           run_benchmarks                    │
                  │       ──► restore optimizer to GPU          │
                  └─────────────────────────────────────────────┘
                              │
                              ▼
                  MultiLogger (rank-aware fan-out)
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
          MLflow file     W&B (opt)     Ray Dashboard
          store on $HOME                Prometheus

┌─────────────────────────────────────────────────────────────────┐
│  PHASE 3 — Checkpointing (every save_interval epochs)           │
└─────────────────────────────────────────────────────────────────┘

   all ranks ──► DCP write to <path>.tmp/{model,optim}
                          │
                          ▼
                    dist.barrier
                          │
                          ▼
                rank 0 rename .tmp → final
                          │
                          ▼
                rank 0 write meta.json + .complete
                          │
                          ▼
                rank 0 prune to keep_last_n by avg_loss
                          │
                          ▼
                    dist.barrier (release)
```

---

## 7. Extension guide

The framework is designed for extension by adding files, not by editing core code.

### 7.1 Add a new dataset

Create `src/bhaskera/data/datasets/my_dataset.py`:

```python
from __future__ import annotations
import ray.data
from bhaskera.data.registry import register, register_raw
from bhaskera.data.tokenize import tokenize_dataset

@register_raw("my_dataset", text_col="prompt")
def _build_raw(cfg, split=None) -> ray.data.Dataset:
    from datasets import load_dataset
    hf_ds = load_dataset("org/dataset_name", split="train")
    return ray.data.from_huggingface(hf_ds)

@register("my_dataset")
def build(cfg, world_size: int = 1) -> ray.data.Dataset:
    return tokenize_dataset(_build_raw(cfg), cfg, "prompt", world_size=world_size)
```

Then add an import to `src/bhaskera/data/datasets/__init__.py`. Use it in YAML with `data.name: my_dataset`.

### 7.2 Add a new chat-data format

```python
from bhaskera.data.formats import register_format

@register_format("my_format")
def render(row: dict, tokenizer, options: dict) -> str:
    return f"### Q: {row['question']}\n### A: {row['answer']}"
```

Then `data.format: my_format` in YAML. `format_options` is hashed into the tokenization cache key, so changing options automatically invalidates the cache.

### 7.3 Add a new model architecture

Usually you do **not** need to do anything — `introspect_model` is purely structural. If the architecture uses an unusual layer container path, add it to `_LAYER_CONTAINER_ATTRS` in `introspect.py`. If MoE experts live under a leaf name other than `experts | expert | routed_experts | moe_experts`, add it to `_EXPERT_LEAF_NAMES`. If the model emits aux loss under a non-standard attribute name, add it to the tuple in `extract_aux_loss`.

For a model that is genuinely not loadable via `AutoModelForCausalLM`, register a custom loader:

```python
from bhaskera.models import register_model

@register_model("my-org/my-model")
def load(cfg, device):
    # build and return torch.nn.Module
    ...
```

### 7.4 Add a new logger

Subclass `BaseLogger`:

```python
from bhaskera.utils.loggers.base import BaseLogger

class MyLogger(BaseLogger):
    def __init__(self, cfg, *, rank=0, world_size=1): ...
    def log(self, metrics: dict, step: int) -> None: ...  # must not raise
    def finish(self) -> None: ...
```

Add it to the `build_logger` dispatch in `utils/loggers/__init__.py` and to the `_VALID` set.

### 7.5 Add a new optimizer plugin

Create `src/bhaskera/plugins/optimizers/my_opt.py`:

```python
from bhaskera.trainer.optimizer_registry import register_optimizer

@register_optimizer("my_opt")
def build_my_opt(model, train_cfg):
    from bhaskera.trainer.optim import _get_default_param_groups
    opt_cfg = train_cfg.optimizer
    param_groups = _get_default_param_groups(model, opt_cfg.kwargs.get("weight_decay", 0.0))
    return MyOptimizer(param_groups, lr=opt_cfg.kwargs.get("lr", train_cfg.lr))
```

Declare in YAML:
```yaml
plugins:
  optimizers:
    - "bhaskera.plugins.optimizers.my_opt"
training:
  optimizer:
    backend: "plugin"
    name: "my_opt"
    kwargs:
      lr: 1.0e-4
```

### 7.6 Add a new validation metric

```python
from bhaskera.evaluation.registry import register_metric

@register_metric("bleu")
class BleuMetric:
    def compute(self, predictions, labels, losses) -> dict:
        # predictions and labels are lists of CPU tensors
        return {"validation/bleu": ...}
```

Enable in YAML: `evaluation.validation.metrics: [loss, perplexity, bleu]`. No import needed — `get_metric("bleu")` will lazy-import the module when the name is first requested.

### 7.7 Add a new benchmark

```python
from bhaskera.evaluation.registry import register_benchmark

@register_benchmark("mmlu")
class MMLUBenchmark:
    def run(self, model, tokenizer, cfg) -> dict:
        # must be FSDP-safe: model is wrapped; call model(...) normally
        return {"benchmark/mmlu_accuracy": ...}
```

Enable in YAML: `evaluation.benchmarks.tasks: [hellaswag, mmlu]`.

### 7.8 Add a new distributed strategy

Implement `wrap_xxx(model, cfg, local_rank, profile) -> nn.Module` and add a branch to `distributed.wrap.wrap_model`. The wrapped model must:

- Accept `**forward_kwargs` consistent with HuggingFace `CausalLMOutput`.
- Either expose `clip_grad_norm_` (FSDP-style) or be compatible with `torch.nn.utils.clip_grad_norm_` over `parameters()`.
- Either be handled by `_set_grad_sync` or be safe to ignore for gradient-accumulation purposes.

---

## 8. Operational concerns

### 8.1 Environment setup

`setup.sh` handles a layered CUDA detection strategy (manual override → env vars → `CUDA_HOME` → `nvcc` → `nvidia-smi` → Spack → SLURM probe job → CPU fallback). It writes `bhaskera-activate.sh` capturing the resolved CUDA hash for reproducibility. flash-attn is installed in a separate step with `--no-build-isolation` and `MAX_JOBS=4` because its setup.py imports torch at build time and is memory-hungry to compile.

For QLoRA, `bitsandbytes` must be installed (`pip install bitsandbytes`). It is not in `pyproject.toml` because it is GPU-only and environment-specific; it is installed separately in setup.sh or the user's environment.

### 8.2 SLURM submission

`scripts/submit.sh` uses `ray symmetric-run` (Ray ≥ 2.49) for symmetric placement across nodes. The port is randomised as `6379 + (SLURM_JOB_ID % 1000)` to avoid collisions with other users' Ray clusters on the same head node. The IB / RoCE / TCP detection chooses the NCCL transport before Ray starts.

### 8.3 Resuming a crashed run

`bhaskera-train` calls `maybe_resume` automatically; on restart with the same `checkpoint.save_dir` the latest `.complete` checkpoint is loaded and training continues from its `step` counter. Loss curves in MLflow will show a small jump at the resume point — this is normal (the EMA buffer is empty and refills over ~50 steps) and not a sign of corruption.

If a checkpoint is suspected to be corrupt, remove its `.complete` sentinel (don't delete the whole directory). `maybe_resume` will then skip it and try the next-most-recent.

### 8.4 Debugging a hang

The most common hang is an asymmetric collective — one rank reaches `all_reduce` and others never do (a crashed rank, a different code path, a different number of micro-steps). With `NCCL_TIMEOUT=1800` and `TORCH_NCCL_BLOCKING_WAIT=1` set in `submit.sh`, a hang surfaces as a visible error within 30 minutes. The `_verify_all_ranks_live` probe at the start of `wrap_model` catches the most common case before any model work begins.

Evaluation adds two new hang surfaces: (a) `dist.broadcast_object_list` at the end of both `run_distributed_validation` and `run_benchmarks` — always called symmetrically on all ranks; and (b) `dist.all_reduce` in `run_distributed_validation` on the loss tensor — must fire on all ranks if any rank enters validation. The `evaluator.should_run_validation(step)` / `should_run_benchmark(step)` predicates are identical on all ranks for the same step, so no asymmetry is possible from the step-counter logic itself.

### 8.5 Diagnosing low MFU

MFU < 30% usually means one of:
- **GPU starved on data.** Check `data.num_workers` and `data.prefetch_batches`. Tokenization-on-the-fly is much slower than a parquet cache — always pre-tokenize for production.
- **Activation checkpointing on a model that fits without it.** AC trades ~30% compute for ~70% activation memory. If you have headroom, disable AC.
- **Sequence length too short.** MFU scales with arithmetic intensity. `seq_len=512` will MFU much lower than `seq_len=4096` at the same param count.
- **Inter-node bandwidth.** `bhaskera-diag` reports NCCL bandwidth in GB/s. Below ~30 GB/s on multi-node with IB indicates IB is not being used or the topology is suboptimal.

### 8.6 Reproducibility

Set `training.seed: 42` and `training.deterministic: true`. The deterministic mode sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` and `torch.use_deterministic_algorithms(True, warn_only=True)`. Some CUDA ops have no deterministic implementation and will warn. Determinism costs ~15–25% throughput, so it is off by default. Note that **dataset shard order is rank-dependent** — the seed is offset by rank to ensure each rank shuffles differently — and the **tokenization cache must be the same across runs** for full bit-equality (the sha256 cache key guarantees this).

---

## 9. Repository layout

```
Bhaskera-qlora/
├── bhaskera-activate.sh           # auto-generated by setup.sh; sourced in SLURM jobs
├── pyproject.toml                 # build, console_scripts, optional extras
├── requirements.txt               # pinned-deps snapshot (informational; install via setup.sh)
├── setup.sh                       # layered CUDA detection + venv + flash-attn install
│
├── configs/                       # example YAMLs — one per scenario
│   ├── 2node.yaml                 # 2×2-GPU FSDP run on Param2-17B MoE
│   ├── param_qlora.yaml           # QLoRA + DDP on Param2-17B (A100 80 GB)
│   ├── qwen_ddp.yaml              # Qwen2.5-14B dense, DDP strategy
│   ├── qwen_hybrid_shard.yaml     # Qwen2.5-14B dense, FSDP HYBRID_SHARD
│   ├── fsdp_cpt.yaml              # CPT on Qwen2.5-7B, FSDP
│   ├── pre.yaml                   # CPT on Qwen2.5-1.5B, DDP
│   ├── eval.yaml                  # CPT + evaluation subsystem (validation + benchmarks)
│   ├── adadelta.yaml              # CPT with built-in Adadelta optimizer
│   ├── lion.yaml                  # CPT with Lion optimizer plugin
│   ├── cpt_token.yaml             # tokenize CPT data (local_cpt dataset)
│   ├── tokenize.yaml              # tokenize UltraChat with Param2 tokenizer
│   └── tokenize_qwen.yaml         # tokenize UltraChat with Qwen2.5 tokenizer
│
├── scripts/
│   └── submit.sh                  # SLURM batch script with NCCL auto-tuning
│
└── src/bhaskera/
    ├── config.py                  # § 4.1  — all dataclasses incl. EvaluationConfig,
    │                              #           OptimizerConfig, PluginsConfig
    ├── introspect.py              # § 4.2
    │
    ├── data/                      # § 4.3
    │   ├── __init__.py
    │   ├── registry.py            # REGISTRY, RAW_REGISTRY, TEXT_COL
    │   ├── tokenize.py            # persistent cache + TokenizerActor
    │   ├── formats/
    │   │   ├── __init__.py        # FORMAT_REGISTRY, register_format
    │   │   └── builtins.py        # chatml, alpaca, sharegpt
    │   └── datasets/
    │       ├── __init__.py        # imports trigger @register side-effects
    │       ├── local_chat.py      # generic JSONL/JSON/Parquet loader (SFT)
    │       ├── local_cpt.py       # continuous-pack loader for CPT
    │       ├── openassistant.py
    │       ├── redpajama.py
    │       └── ultrachat.py
    │
    ├── distributed/               # § 4.5
    │   ├── __init__.py            # wrap_model, save_checkpoint, load_checkpoint
    │   ├── wrap.py                # strategy dispatcher + liveness probe
    │   ├── fsdp.py                # FSDP2 per-expert / per-layer / root
    │   ├── ddp.py                 # DDP with MoE-forced find_unused_parameters
    │   ├── activation_ckpt.py     # composable AC + legacy NO_REENTRANT fallback
    │   └── checkpoint.py          # DCP atomic save + .complete sentinel
    │
    ├── models/                    # § 4.4
    │   ├── __init__.py
    │   ├── loader.py              # build_model + Liger Kernel
    │   ├── lora.py                # strategy-gated dtype handling
    │   └── quantization.py        # QLoRA BitsAndBytes config builder
    │
    ├── trainer/                   # § 4.6
    │   ├── __init__.py
    │   ├── loop.py                # _run_epoch, _set_grad_sync, Evaluator integration
    │   ├── moe.py                 # extract_aux_loss, _load_balancing_loss_from_logits
    │   ├── optim.py               # build_optimizer dispatch + warmup→cosine scheduler
    │   ├── optimizer_registry.py  # OPTIMIZER_REGISTRY, @register_optimizer
    │   ├── eval_lifecycle.py      # EvaluationLifecycle, DatasetCursor, offload helpers
    │   ├── precision.py           # resolve_autocast_dtype
    │   └── checkpointing.py       # save_and_prune by best-loss
    │
    ├── launcher/                  # § 4.7
    │   ├── train.py               # bhaskera-train
    │   ├── tokenize.py            # bhaskera-tokenize
    │   ├── infer.py               # bhaskera-infer (TurboQuant + speculative)
    │   ├── diagnostics.py         # bhaskera-diag (NCCL all-reduce + bw)
    │   ├── dashboard.py           # bhaskera-dashboard (MLflow UI manager)
    │   ├── monitoring.py          # MonitoringContext, MLflow UI URL discovery
    │   └── worker.py              # worker_fn — per-GPU entry point
    │
    ├── evaluation/                # § 4.9
    │   ├── __init__.py            # re-exports Evaluator, register_metric, register_benchmark
    │   ├── evaluator.py           # Evaluator facade (should_run, run_validation, run_benchmarks)
    │   ├── validation.py          # run_distributed_validation, chunked loss, optimizer offload
    │   ├── benchmark_runner.py    # run_benchmarks (symmetric, broadcast from rank 0)
    │   └── registry.py            # METRIC_REGISTRY, BENCHMARK_REGISTRY, lazy import
    │
    ├── plugins/                   # § 4.10
    │   ├── __init__.py
    │   ├── loader.py              # load_plugins(cfg) — import paths from cfg.plugins
    │   ├── benchmarks/
    │   │   ├── __init__.py
    │   │   ├── arc_challenge.py   # ARC-Challenge (placeholder)
    │   │   ├── arc_easy.py        # ARC-Easy (batched, distributed all-gather)
    │   │   └── hellaswag.py       # HellaSwag (batched, distributed all-gather)
    │   ├── metrics/
    │   │   ├── __init__.py
    │   │   ├── loss.py            # LossMetric
    │   │   ├── perplexity.py      # PerplexityMetric
    │   │   └── token_accuracy.py  # TokenAccuracyMetric
    │   └── optimizers/
    │       ├── __init__.py
    │       └── lion.py            # Lion optimizer + @register_optimizer("lion")
    │
    └── utils/                     # § 4.8
        ├── __init__.py
        ├── gpu_stats.py           # legacy pynvml-only (kept for compat)
        ├── system_stats.py        # full pynvml + psutil telemetry
        ├── throughput.py          # ThroughputTracker + MFU
        └── loggers/
            ├── __init__.py        # build_logger dispatch + _normalize_trackers
            ├── base.py            # BaseLogger interface
            ├── multi_logger.py    # fan-out, swallow per-child failures
            ├── mlflow_logger.py   # queued, threaded, file-store-default
            └── wandb_logger.py    # rank-0 only
```

---

## Appendix A — Config field reference

For each top-level key in YAML, see the corresponding dataclass in `src/bhaskera/config.py`:

| YAML key               | Dataclass             | Notes                                                          |
| ---------------------- | --------------------- | -------------------------------------------------------------- |
| `plugins`              | `PluginsConfig`       | `optimizers: list[str]` — fully-qualified module paths to import at startup. |
| `model`                | `ModelConfig`         | `name`, `dtype`, `attn_impl`, `trust_remote_code`, `use_liger_kernel`, `quantization` (`"none"` default / `"qlora"`). |
| `data`                 | `DataConfig`          | `name`, `seq_len`, `is_cpt` (CPT packing), `num_workers`, `tokenized_path` / `val_tokenized_path`, `cache_dir`, `format`, `format_options`, `path` / `train_path` / `val_path`, `pack_sequences`. |
| `lora`                 | `LoraConfig`          | `enabled`, `r`, `alpha`, `dropout`, `target_modules` (`["auto"]` to introspect), `include_experts`, `freeze_router`, `modules_to_save`. |
| `moe`                  | `MoEConfig`           | `aux_loss_weight`, `router_z_loss_weight`, `freeze_router`, `log_expert_utilization`, `log_every_n_steps`. |
| `training`             | `TrainingConfig`      | `batch_size`, `grad_accum`, `lr`, `weight_decay`, `max_steps`, `num_epochs`, `warmup_steps`, `max_grad_norm`, `grad_clip`, `max_grad_skip_steps`, `seed`, `deterministic`. |
| `training.optimizer`   | `OptimizerConfig`     | `backend: "default"\|"torch"\|"plugin"`, `class_name` (for `torch`), `name` (for `plugin`), `kwargs: dict`. |
| `training.distributed` | `DistributedConfig`   | `strategy: fsdp\|ddp` + nested `fsdp` / `ddp` blocks. |
| `checkpoint`           | `CheckpointConfig`    | `enabled`, `save_dir`, `save_interval` (in epochs), `keep_last_n` (best by avg loss). |
| `logging`              | `LoggingConfig`       | `tracker` (`"mlflow"` / `"wandb"` / `"ray"` / list / none), `project`, `run_name`, `mlflow_tracking_uri`, `mlflow_log_all_ranks`, `mlflow_log_artifacts_every`, `tags`, `group`. |
| `evaluation`           | `EvaluationConfig`    | `enabled`, `offload_optimizer`. |
| `evaluation.validation`| `ValidationConfig`    | `dataset`, `every_n_steps`, `every_n_epochs`, `metrics: list[str]` (e.g. `[loss, perplexity, token_accuracy]`). |
| `evaluation.benchmarks`| `BenchmarksConfig`    | `every_n_steps`, `every_n_epochs`, `tasks: list[str]` (e.g. `[hellaswag, arc_easy, arc_challenge]`). |
| `inference`            | `InferenceConfig`     | `max_new_tokens`, `temperature`, `top_p`, `top_k`, `do_sample`, `batch_size`, `kv_cache` (`static`/`turboquant`/`none`), `device`, `torch_compile`, nested `turboquant` and `speculative`. |
| `monitoring`           | `MonitoringConfig`    | `dashboard`, `dashboard_host`, `dashboard_port`, `metrics_export_port`, nested `metrics`. |

## Appendix B — Console scripts

Defined in `pyproject.toml [project.scripts]`:

```
bhaskera-train      → bhaskera.launcher.train:main
bhaskera-tokenize   → bhaskera.launcher.tokenize:main
bhaskera-infer      → bhaskera.launcher.infer:main
bhaskera-diag       → bhaskera.launcher.diagnostics:main
bhaskera-dashboard  → bhaskera.launcher.dashboard:main
```

All entry points expect `--config <path-to-yaml>` except `bhaskera-diag` (which only needs `--num-workers`) and `bhaskera-dashboard` (which has its own argparse with `start | stop | status | tunnel`).

## Appendix C — Glossary

- **AC** — Activation Checkpointing. Trade compute for memory by re-running the forward during backward instead of storing activations.
- **CPT** — Continual Pre-Training. Training on raw text corpora with continuous sequence packing (`is_cpt: true`).
- **DCP** — Distributed Checkpoint. PyTorch's sharded checkpoint API (`torch.distributed.checkpoint`).
- **DDP** — Distributed Data Parallel. Replicate the model on every GPU, all-reduce gradients.
- **FSDP / FSDP2** — Fully Sharded Data Parallel. Shard params, gradients, and optimizer state across GPUs; gather on demand. FSDP2 is the composable rewrite shipped in PyTorch 2.4.
- **MFU** — Model FLOPs Utilisation. Achieved FLOPs / peak FLOPs, expressed as a percent.
- **MoE** — Mixture of Experts. A subset of expert FFNs is activated per token by a router.
- **PEFT** — Parameter-Efficient Fine-Tuning. The HuggingFace library for LoRA / IA3 / prefix tuning.
- **QLoRA** — Quantized LoRA. 4-bit NF4 quantization of base weights (BitsAndBytes) with full-precision LoRA adapters trained on top. DDP-only in Bhaskera.
- **Ray Train** — Ray's distributed training abstraction; `TorchTrainer.fit()` spawns workers and orchestrates the cluster.
- **TurboQuant** — Bhaskera's KV-cache quantisation scheme (K4/V2 bits with a residual fp16 window).
