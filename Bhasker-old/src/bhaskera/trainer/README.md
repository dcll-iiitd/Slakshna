# src/bhaskera/trainer/

Training loop and its direct dependencies — optimizer, LR scheduler, mixed-precision resolution, MoE auxiliary loss, and periodic checkpointing.

This package contains no Ray, no SLURM, and no distributed-init logic. Those live upstream in `launcher/`. The `train()` function takes an already-distributed-wrapped model and a Ray Data shard and runs the loop.

## Module layout

```
trainer/
├── __init__.py        # re-exports train
├── loop.py            # train() + per-epoch implementation
├── moe.py             # aux loss extraction, expert utilisation metrics
├── optim.py           # AdamW + warmup→cosine scheduler
├── precision.py       # resolve_autocast_dtype
└── checkpointing.py   # save_and_prune, maybe_resume wrappers
```

## Public API

```python
from bhaskera.trainer import train

train(
    *,
    model,        # FSDP2-wrapped or DDP-wrapped nn.Module
    dataset,      # ray.data.Dataset shard from ray.train.get_dataset_shard
    cfg,          # bhaskera.config.Config
    profile,     # ModelProfile from introspect.py
    rank,         # global rank
    local_rank,   # local GPU index
    tracker=None, # optional MultiLogger from utils.build_logger
    world_size=1,
)
```

## `loop.py` — the training loop

### Entry point

`train()` is straightforward by design:

1. Build optimizer and scheduler.
2. `model.train()`.
3. If checkpointing is enabled, call `maybe_resume(model, optimizer, save_dir)` to load the latest valid checkpoint and get the start step.
4. Construct a `ThroughputTracker` (params, world size, peak TFLOPS, EMA window, warmup steps) when `cfg.monitoring.metrics.throughput` is enabled.
5. Log static model metadata (`model/total_params`, `model/trainable_params`, `model/world_size`) at step 0.
6. Loop over `range(cfg.training.num_epochs)`, calling `_run_epoch(...)` and breaking when `step >= max_steps`.
7. `tracker.finish()` and final log line.

### Gradient sync toggling — `_set_grad_sync`

Used for gradient accumulation. The function dispatches by wrapper type:

- **FSDP2** — `set_requires_gradient_sync(model, enabled)` from `torch.distributed._composable.fsdp`. Walks every sharded submodule directly, bypassing PEFT wrappers' `__getattr__` chains.
- **DDP** — sets `model.require_backward_grad_sync = enabled` (the underlying flag that `DDP.no_sync()` toggles). Direct attribute write avoids the context-manager dance and works through PEFT wrappers that don't delegate `no_sync`. Skipped when `static_graph=True` — DDP's static_graph mode caches the reduction order on iteration 1 and is incompatible with mid-run toggling. The wrap step records the effective flag as `_bhaskera_static_graph` on the DDP wrapper, which is read here.
- **Other** (single-GPU / CPU) — no-op.

Called with `enabled=False` before every micro-step except the last, then `enabled=True` on the last so the all-reduce fires exactly once per optimizer step.

### Per-step responsibilities (in `_run_epoch`)

- Forward + backward on each micro-batch.
- MoE auxiliary loss extraction via `moe.extract_aux_loss(out, profile)` when `profile.is_moe`. Added to the language modelling loss with `cfg.moe.aux_loss_weight`.
- Gradient clipping when `cfg.training.grad_clip` is set. Steps with grad norm exceeding `cfg.training.max_grad_skip_steps` (interpreted as a threshold in the codebase) are skipped from the optimizer step.
- Optimizer step + scheduler step + `optimizer.zero_grad()`.
- Throughput tracker step (`tokens_in_step`, `samples_in_step`, `seq_len`).
- Periodic logging via the tracker — `system_stats()` and `cuda_memory_stats()` are batched in at the configured cadences.
- Expert utilisation metrics via `moe.compute_expert_utilization(out, profile)` when MoE and `cfg.moe.log_expert_utilization`.
- Periodic checkpointing via `save_and_prune(...)`.

## `moe.py` — MoE auxiliary loss

Fully architecture-agnostic. All routing decisions are driven by `ModelProfile` (`num_experts`, `num_shared_experts`, `experts_per_token`). No model-name or class-name checks anywhere.

### Router logit shapes

Different MoE architectures emit different tensors in `out.router_logits`:

| Shape | Meaning | Examples |
|---|---|---|
| `(T, E_total)` | Full-vocabulary logits — softmax over all routed experts | Mixtral, Qwen2-MoE |
| `(T, k)` | Top-k selected scores only | Param2 |
| `(T, 1)` | Scalar gate (rare; binary MoE) | — |

Some models additionally wrap each per-layer entry as `(logits, indices)` tuples instead of bare tensors. `_normalize_router_logits()` unwraps; `_infer_logit_kind()` classifies by the last dimension against profile fields.

### `extract_aux_loss(out, profile)`

1. Returns `None` for dense models.
2. Tries known attribute names first: `aux_loss`, `router_aux_loss`, `moe_loss`, `load_balancing_loss`. Models that compute the loss themselves are zero-overhead.
3. Falls back to `_load_balancing_loss_from_logits` if `out.router_logits` is present.

### Load-balancing loss

Switch-Transformer-style, averaged across layers:

```
loss_layer = N * sum_i( f_i * P_i )
    f_i = fraction of tokens dispatched to expert i  (no grad)
    P_i = mean routing probability for expert i      (grad)
    N   = number of experts in this layer
```

Branches on the logit kind: for `FULL`, computes `f` via a top-k hard mask and `P` via softmax mean. For `TOPK`, `f` is uniform `1/k` by construction (every column is one selected expert) and the gradient flows through `P`. `GATE` contributes nothing (no inter-expert balancing to learn).

### `compute_expert_utilization(out, profile)`

Returns logging-ready scalars:

- `expert/load_max`, `expert/load_min`, `expert/load_std`
- `expert/imbalance_ratio` (max/min, omitted when min is zero)

Same architecture-agnostic dispatch as the aux loss. All exceptions are caught and logged at debug — metric collection never breaks training.

## `optim.py` — optimizer and scheduler

### `build_optimizer(model, train_cfg)`

AdamW with decoupled weight decay. Parameters are split into two groups:

- `ndim >= 2` → `weight_decay = train_cfg.weight_decay` (matrices, embeddings stored as 2-D)
- `ndim == 1` → `weight_decay = 0.0` (biases, LayerNorm/RMSNorm gains)

`fused=True` is set when CUDA is available — ~10–20% faster, compatible with FSDP2 DTensors on torch ≥ 2.4. Betas are `(0.9, 0.95)` — the standard GPT recipe.

### `build_scheduler(optimizer, train_cfg)`

`SequentialLR([LinearLR(warmup), CosineAnnealingLR(decay)], milestones=[warmup_steps])`:

- `LinearLR(start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps)`
- `CosineAnnealingLR(T_max=max_steps - warmup_steps)`

## `precision.py` — autocast dtype resolution

`resolve_autocast_dtype(cfg, profile)` — used by the DDP path of the training loop to pick the dtype for `torch.autocast`. Priority:

1. `cfg.model.dtype="auto"` → `profile.model_dtype`.
2. Otherwise → mapped from the string, defaulting to `torch.bfloat16`.

FSDP path does not use this — `MixedPrecisionPolicy` from `bhaskera.distributed.fsdp` is the sole source of mixed-precision truth there.

## `checkpointing.py` — wrappers

Thin compatibility wrappers around `bhaskera.distributed.checkpoint`:

- `save_and_prune(model, optimizer, step, avg_loss, ckpt_cfg, rank, best_ckpts)` — calls `save_checkpoint` from the distributed module, then on rank-0 maintains a `best_ckpts` list sorted by `avg_loss` ascending and trims it to `ckpt_cfg.keep_last_n`. Returns the updated list. Collective: all ranks must call.
- `maybe_resume(model, optimizer, save_dir)` — scans `save_dir` for `step_<N>` directories and forwards to `load_checkpoint` for the latest. Returns 0 when no candidates exist.

The lower-level atomic-write semantics (`.complete` sentinel, `<path>.tmp` staging, DCP version shim) live in `bhaskera.distributed.checkpoint` — see that README.
