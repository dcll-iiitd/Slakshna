# src/bhaskera/distributed/

FSDP2 and DDP wrappers, activation checkpointing, and DCP-based sharded checkpointing.

## Module layout

```
distributed/
├── __init__.py          # re-exports wrap_model, save_checkpoint, load_checkpoint
├── wrap.py              # strategy dispatcher (fsdp | ddp)
├── fsdp.py              # FSDP2 wrap via fully_shard + MixedPrecisionPolicy
├── ddp.py               # DDP wrap, MoE-aware
├── activation_ckpt.py   # composable AC (torch ≥ 2.4) with legacy fallback
└── checkpoint.py        # DCP save / resume with .complete sentinel
```

## Public API

Re-exported from `bhaskera.distributed`:

- `wrap_model(model, cfg, local_rank, profile) -> nn.Module`
- `save_checkpoint(model, optimizer, step, path, ...)`
- `load_checkpoint(model, optimizer, save_dir) -> int` (alias for `maybe_resume`)

## `wrap.py` — strategy dispatch

`wrap_model()` reads `cfg.training.distributed.strategy` (`fsdp` or `ddp`) and forwards to the matching wrap function. Raises if `torch.distributed` is not initialised — the caller is expected to have run `_wait_for_dist()` (or equivalent Ray Train init) first.

Two helpers in the module guard against startup races:

- `_wait_for_dist(timeout_s=120)` — polls `dist.is_initialized()` until true or the timeout fires. Prevents Ray placement-group races where `wrap_model` is called before NCCL is up.
- `_verify_all_ranks_live(rank, world_size, device)` — all-reduces a probe tensor and checks the sum equals `world_size`. Used as a liveness check after init: a missing rank either hangs (caught by `NCCL_TIMEOUT`) or returns a wrong sum, surfacing the failure visibly rather than letting it deadlock later.

## `fsdp.py` — FSDP2 wrap

Uses the composable `torch.distributed._composable.fsdp.fully_shard` API (requires PyTorch ≥ 2.4). Raises `ImportError` with a clear message if FSDP2 is unavailable.

### Three-pass sharding

1. **Per-expert** — when `profile.is_moe` and `fsdp_cfg.shard_experts_individually` and `profile.expert_modules` is non-empty, each expert module is `fully_shard`-ed as its own unit. This ensures the MoE forward only all-gathers the activated experts.
2. **Per-decoder-layer** — every module of `decoder_cls` gets its own shard group.
3. **Root** — the model itself is `fully_shard`-ed last.

`decoder_cls` is resolved by `_resolve_decoder_cls()`: explicit `fsdp.transformer_layer_cls` names win (back-compat); otherwise the class from the introspection profile is used.

### Mixed precision

`_build_mp_policy()` constructs a `MixedPrecisionPolicy(param_dtype, reduce_dtype, output_dtype)`:

- `param_dtype` — `"auto"` reads `profile.model_dtype`; otherwise mapped from the string.
- `reduce_dtype` — `"auto"` returns `float32` for MoE (sparse gradients amplify low-precision reduction noise) and `param_dtype` otherwise.
- `output_dtype` — derived from `buffer_dtype`, defaulting to `param_dtype`.

This is the single source of mixed-precision truth — the training loop does not separately wrap forward in `torch.autocast` under FSDP.

### Activation checkpointing

If `fsdp_cfg.activation_checkpointing` is true, `apply_activation_checkpointing()` is called after sharding.

## `ddp.py` — DDP wrap

MoE-aware. When `profile.is_moe` and `find_unused_parameters` is `False`, it is overridden to `True` and a warning is logged — expert routing means a different parameter subset is touched per forward pass.

`static_graph` is set from `ddp_cfg.static_graph` but disabled (with warning) when `find_unused_parameters=True`, because DDP enforces their mutual exclusion. The effective `static_graph` value is stored on the wrapper as `_bhaskera_static_graph` so the training loop can detect it without re-reading the config (the loop skips grad-sync toggling under static graph).

Activation checkpointing, when enabled, is applied **before** the DDP wrap so DDP's parameter-graph snapshot includes the AC-wrapped modules. Uses the same `apply_activation_checkpointing` dispatcher as FSDP.

Move-then-wrap split: `model = model.to(local_rank)` is intentionally a separate statement before `DDP(...)`, ensuring AC hooks are installed on the CPU/GPU module before DDP snapshots its parameter graph.

## `activation_ckpt.py`

Two-path activation checkpointing:

- **Composable** (torch ≥ 2.4): `torch.distributed._composable.checkpoint`.
- **Legacy fallback**: `torch.distributed.algorithms._checkpoint.checkpoint_wrapper.apply_activation_checkpointing` with `CheckpointImpl.NO_REENTRANT`.

Dispatch by `ModelProfile`:

- MoE with `profile.expert_modules` → each expert is checkpointed individually.
- Dense (or MoE without expert detection) → each `decoder_cls` instance is checkpointed.

If no target classes are found, the call is a logged no-op.

## `checkpoint.py` — DCP sharded checkpointing

### On-disk layout

```
<save_dir>/step_<NNNNNNN>/
├── model/           # DCP shard files for the model state dict
├── optim/           # DCP shard files for the optimizer state dict
├── meta.json        # {"step": int, "avg_loss": float, ...}
└── .complete        # zero-byte sentinel written last by rank-0
```

### PyTorch version shim

`_dcp_save` / `_dcp_load` pick the right DCP API based on the installed torch:

- `< 2.5` — uses the deprecated `checkpoint_id=path` keyword.
- `≥ 2.5` — uses `storage_writer=FileSystemWriter(path)` / `storage_reader=FileSystemReader(path)`.

### `save_checkpoint`

Atomic write flow:

1. All ranks write shards to `<path>.tmp` (DCP is a collective).
2. `dist.barrier()` — wait for all ranks to finish writing.
3. Rank-0 renames `<path>.tmp` → `<path>`.
4. Rank-0 writes `meta.json`.
5. Rank-0 writes the `.complete` sentinel (last).
6. Rank-0 prunes old checkpoints to `keep_last_n`.
7. `dist.barrier()` — unblock all ranks.

`get_model_state_dict` and `get_optimizer_state_dict` are called with `StateDictOptions(full_state_dict=False, cpu_offload=True)`.

### `maybe_resume`

Scans `save_dir` for `step_<N>/` directories with a `.complete` sentinel (partially written checkpoints are skipped), picks the highest step, loads model and optimizer state in-place via DCP, and reads the authoritative step from `meta.json`. Returns 0 when no valid checkpoint exists.

### `save_and_prune`

Compatibility wrapper used by the training loop. Calls `save_checkpoint`, then maintains a `best_ckpts` list sorted by `avg_loss` and trimmed to `keep_last_n`.

## Strategy selection summary

| | FSDP2 | DDP |
|---|---|---|
| Min torch | 2.4 | any supported |
| Memory | shards params, grads, optim state | full replica per rank |
| MoE per-expert sharding | yes, opt-in via `shard_experts_individually` | n/a |
| `find_unused_parameters` | n/a | forced `True` for MoE |
| Mixed precision | `MixedPrecisionPolicy` (no autocast) | `torch.autocast` in training loop |
| Activation checkpointing | composable, post-shard | composable, pre-DDP-wrap |
| `static_graph` | n/a | optional; disabled under MoE |
| Checkpoint format | DCP sharded | DCP sharded (same path) |
