# src/bhaskera/utils/

System telemetry, throughput tracking, and experiment-tracker backends.

## Module layout

```
utils/
├── __init__.py        # re-exports BaseLogger, build_logger, system_stats,
│                      # cuda_memory_stats, ThroughputTracker
├── gpu_stats.py       # legacy pynvml-only GPU telemetry
├── system_stats.py    # full GPU + CPU + disk + network telemetry
├── throughput.py      # ThroughputTracker (step time, tokens/sec, MFU)
└── loggers/
    ├── __init__.py        # build_logger factory + _normalize_trackers
    ├── base.py            # BaseLogger interface
    ├── multi_logger.py    # MultiLogger fan-out wrapper
    ├── mlflow_logger.py   # push-based MLflow logger
    └── wandb_logger.py    # Weights & Biases logger
```

## Public API

Re-exported from `bhaskera.utils`:

- `build_logger(cfg, *, rank=0, world_size=1) -> Optional[BaseLogger]`
- `BaseLogger`
- `system_stats() -> dict[str, float]`
- `cuda_memory_stats() -> dict[str, float]`
- `ThroughputTracker`

## `system_stats.py` — primary telemetry

Production-grade single-call telemetry. Supersedes `gpu_stats.py`; new code calls `system_stats()`.

### Coverage

**GPU (per-device, via pynvml):**

- utilisation, memory used / free / total, temperature, power, fan
- SM and memory clocks
- PCIe Tx / Rx KB/s, NVLink Tx / Rx KB/s
- ECC volatile and aggregate single- and double-bit error counts
- throttle reasons (decoded from NVML bitmask via `_THROTTLE_BITS`)
- performance state

**CPU / host (via psutil):**

- per-process CPU%, system CPU% (mean and per-core)
- load average
- memory used / available, swap used
- disk I/O B/s, network I/O B/s
- open file descriptors

### Design rules (from the module header)

- **Cheap.** Initialised once. Each call is O(num_gpus + 4). No fork/exec. Safe to call every step at scale.
- **Graceful degradation.** Missing `pynvml` or `psutil` silently drops the corresponding sub-tree. Returns `{}` rather than crashing.
- **Stable keys.** Flat, slash-separated, compatible with Prometheus tag extraction. Per-GPU keys are `gpu/<idx>/...`.
- **Rate-derived counters are buffered.** Disk and network I/O are exposed as B/s computed from a delta against the previous call; the first call returns 0 to avoid a spurious spike. State lives in module-level `_PREV: dict[str, (value, t)]`.

### `cuda_memory_stats()`

Compact `torch.cuda` memory snapshot — typically allocated / reserved / max-allocated, in MiB. Cheap; called alongside `system_stats()` from the loggers.

## `gpu_stats.py` — legacy GPU-only path

Earlier, narrower implementation: per-GPU utilisation, memory used (MiB), temperature, and power, via pynvml. Kept for backward compatibility. New code uses `system_stats()`.

Both paths share the same lazy-init pattern: `_init_nvml()` runs `pynvml.nvmlInit()` once, caches the handle, and returns `False` silently if pynvml isn't installed.

## `throughput.py` — `ThroughputTracker`

Lightweight tracker called once per optimizer step. Emits:

```python
{
    "throughput/step_time_s":             0.412,
    "throughput/step_time_ema_s":         0.418,
    "throughput/tokens_per_sec":          19880.0,
    "throughput/tokens_per_sec_per_gpu":  2485.0,
    "throughput/samples_per_sec":         9.7,
    "throughput/mfu_pct":                 41.2,
    "throughput/total_steps":             137.0,
}
```

### Constructor

```python
ThroughputTracker(
    *,
    params_for_flops: int,
    world_size: int,
    peak_flops_per_gpu: float = 312e12,  # A100 bf16
    window: int = 50,
    warmup_steps: int = 5,
)
```

### MFU calculation

```
flops_per_token ≈ 6 * params      # forward 1× + backward 2×
mfu_pct = 100 * (flops_per_token * tokens_in_step / dt / world_size) / peak_flops_per_gpu
```

The `seq² · d` attention term is deliberately ignored — it adds under 5% at typical `seq_len` and keeps the estimate stable across configs.

For LoRA, `params_for_flops` should be the **full** model parameter count, not the trainable count — the dominant FLOP path is still through the frozen base weights (the LoRA update is multiplied into the base path during forward + backward). The training loop sets this correctly.

### Behaviour

- The first `warmup_steps` step times are dropped from the EMA — they include compile/cache warmup that would otherwise drag the moving average down.
- Throughput uses the smoothed `step_time_ema_s` rather than the raw delta so the panel doesn't jitter.
- The first `step()` call has no `last_t`, returns only `total_steps`, and sets up the clock.

## Loggers

### `BaseLogger` (`loggers/base.py`)

Minimal interface:

```python
class BaseLogger:
    def log(self, metrics: dict[str, Any], step: int) -> None: ...
    def finish(self) -> None: ...
```

Both methods are best-effort — a logger that loses connectivity must never raise into the training loop.

### `MultiLogger` (`loggers/multi_logger.py`)

Fan-out wrapper. Forwards every `log` / `finish` to a list of children and swallows per-child exceptions with a warning so one flaky tracker doesn't break training. Returned by `build_logger` whenever any backend is enabled — even with a single child.

### `WandbLogger` (`loggers/wandb_logger.py`)

Lazy-imports `wandb` so the dependency stays optional. On init: `wandb.init(project=..., name=..., config=cfg.as_dict(), tags=..., group=...)` and stores `rank` and `world_size` in `wandb.run.summary` for multi-rank slicing. Every `cfg.logging.log_gpu_every_n_steps` steps it merges in `system_stats()` and `cuda_memory_stats()` before pushing.

### `MLflowLogger` (`loggers/mlflow_logger.py`)

Push-based design — on a SLURM cluster the GPU node isn't directly reachable, so a pull-based exporter wouldn't work. The training process pushes metrics to a file-backed MLflow store under `$HOME/mlflow-runs` (or `cfg.logging.mlflow_tracking_uri` when set). Login nodes that share `$HOME` with compute nodes can run the MLflow UI (started via `bhaskera-dashboard`) and read live data without any network plumbing.

### `build_logger(cfg, *, rank, world_size)` (`loggers/__init__.py`)

The factory.

1. **Normalise.** `_normalize_trackers(cfg.logging.tracker)` accepts a string, list, set, or `None`. Recognised values: `wandb`, `mlflow`, `ray`. The off aliases (`""`, `none`, `off`, `false`, `None`) collapse to an empty list. Unknown values raise.
2. **Auto-add Ray.** When `cfg.monitoring.dashboard` is on (the default) and `"ray"` isn't already in the list, it's appended — this is how Ray Dashboard becomes the default sink when nothing else is configured.
3. **Per-rank construction.**
   - `"ray"` runs on every rank (so per-GPU stats reach Prometheus).
   - `"wandb"` and `"mlflow"` are rank-0 only.
4. **Result.** Returns `None` when every backend is disabled; otherwise wraps the children in a `MultiLogger`.

Each child constructor is wrapped in `try/except` so an init failure (missing extra, bad network) logs a warning and continues with the remaining backends.

## Metric naming conventions

- GPU per-device: `gpu/<idx>/util_pct`, `gpu/<idx>/mem_mib`, `gpu/<idx>/temp_c`, `gpu/<idx>/power_w`, `gpu/<idx>/pcie_tx_kib_s`, `gpu/<idx>/nvlink_rx_kib_s`, `gpu/<idx>/throttle/<reason>`, etc.
- CPU / host: `cpu/system_pct`, `cpu/process_pct`, `mem/used_mib`, `mem/available_mib`, `disk/read_b_s`, `net/rx_b_s`, etc.
- CUDA memory: `cuda/allocated_mib`, `cuda/reserved_mib`, `cuda/max_allocated_mib`.
- Throughput: `throughput/step_time_s`, `throughput/tokens_per_sec`, `throughput/mfu_pct`.
- Model: `model/total_params`, `model/trainable_params`, `model/world_size`.
- MoE: `expert/load_max`, `expert/load_min`, `expert/load_std`, `expert/imbalance_ratio`.

These keys are stable and slash-separated by convention so downstream tools (Ray Dashboard's Prometheus integration, MLflow, W&B) can extract tags without ad-hoc parsing.
