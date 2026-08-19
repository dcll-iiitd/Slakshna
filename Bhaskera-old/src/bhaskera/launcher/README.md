# src/bhaskera/launcher/

CLI entry points and supporting glue. Each command listed in `pyproject.toml`'s `[project.scripts]` resolves to a `main()` in this package.

## Module layout

```
launcher/
├── train.py        # bhaskera-train      — Ray Train driver
├── worker.py       # per-GPU entry point called by Ray
├── tokenize.py     # bhaskera-tokenize   — one-shot tokenisation
├── infer.py        # bhaskera-infer      — generation CLI
├── diagnostics.py  # bhaskera-diag       — NCCL + bandwidth check
├── dashboard.py    # bhaskera-dashboard  — MLflow UI manager
└── monitoring.py   # MonitoringContext + ray.init kwargs
```

## `train.py` — `bhaskera-train`

Unified CLI and Ray Train driver. Flow:

1. Parse `--config`, `--num-workers`, `--max-failures`, `--storage-path`, `--no-dashboard`, `--dashboard-port`.
2. Load YAML via `bhaskera.config.load_config`.
3. Apply CLI overrides for monitoring flags.
4. Build a `MonitoringContext` from `cfg.monitoring`.
5. Initialise Ray — connects to `$RAY_ADDRESS` when set (SLURM-bootstrapped clusters), otherwise stops any stale session and starts a local cluster sized to local GPUs.
6. Resolve `world_size` via `_count_gpus()` (see below) and build the Ray dataset.
7. Construct a `TorchTrainer` with `worker_fn` as the per-worker entry, `cfg.as_dict()` as `train_loop_config`, and `ScalingConfig(num_workers=N, use_gpu=True, resources_per_worker={"GPU": 1})`.
8. `trainer.fit()` and log the best checkpoints.

`_count_gpus()` resolves the true GPU count. Priority:

1. `SLURM_NNODES × SLURM_GPUS_PER_NODE` when both are set — `torch.cuda.device_count()` on a SLURM head node sees only the local 0 or 1 GPUs.
2. `torch.cuda.device_count()` for local / single-node runs.

Raises `RuntimeError` with a SLURM-specific hint when neither yields a positive count.

## `worker.py` — per-GPU entry

`worker_fn(cfg_dict)` is called by Ray Train for each actor. It:

1. Re-hydrates the config via `Config.from_dict(cfg_dict)`.
2. Reads rank info from `ray.train.get_context()` (`local_rank`, `world_rank`, `world_size`).
3. Sets the CUDA device and seeds RNGs (`_seed_everything` adds the rank to `cfg.training.seed`; enables deterministic algorithms when `cfg.training.deterministic`).
4. Pulls the training dataset shard from Ray Train.
5. Calls `build_model(cfg, device)` — `device="cpu"` under FSDP (the wrap step migrates), CUDA device under DDP.
6. Calls `wrap_model(model, cfg, local_rank, profile)` to apply FSDP2 or DDP.
7. Builds a logger via `build_logger(cfg, rank, world_size)`.
8. Hands off to `bhaskera.trainer.train(...)`.

## `tokenize.py` — `bhaskera-tokenize`

One-shot CLI that tokenises a dataset, writes a Parquet cache, and prints the YAML snippet to paste into the training config.

Behaviour:

- Reads the same YAML as training; uses `cfg.data.cache_dir`, `cfg.data.name`, `cfg.model.name`, `cfg.data.seq_len`, `cfg.data.format`, `cfg.data.format_options`.
- `--split train | val | both` — drives which raw builders to call (when the builder accepts a `split` kwarg) and what to write.
- `--train-path`, `--val-path`, `--format`, `--cache-dir`, `--overwrite-cache` — CLI overrides that win over the YAML.
- Calls `call_raw_builder(name, cfg, split=...)` and then `persist_tokenized(ds, cfg, text_col, dataset_name)` per split.
- After writing, prints a `data:` block with `tokenized_path:` and (if both splits were processed) `val_tokenized_path:` for direct copy/paste.

## `infer.py` — `bhaskera-infer`

Command-line inference entry point. Reads a config, loads a model and checkpoint, and runs generation. Notable features documented in the module header:

- Tokens/second reported per generation, derived from output token count (not characters).
- For thinking-style models (e.g. Param2-Thinking), `<think>` blocks are stripped from terminal output by default; `--show-thinking` retains them. Raw text is always preserved in `--output-file`.
- TurboQuant KV-cache statistics surface when `inference.turboquant.enabled` is true.
- Speculative decoding parameters are read from `cfg.inference.speculative`.

Typical invocations are in the module's docstring.

## `diagnostics.py` — `bhaskera-diag`

Minimal NCCL sanity check. Submits a `TorchTrainer` running `_diag_worker`, which:

1. Performs an `all_reduce` over a per-rank bf16 tensor and asserts the sum matches `sum(range(world_size))`.
2. Times five all-reduces of a 25 MiB buffer and computes effective bandwidth (`2*(W-1)/W * size * 4 * 5 / dt / 1e9`).
3. Rank-0 prints world size, GPU name and memory, and the measured NCCL bandwidth.

Use it before a long run to confirm the cluster is wired correctly.

## `dashboard.py` — `bhaskera-dashboard`

Manages the MLflow UI on the login node. No Prometheus, no Grafana, no Docker. Sub-commands: `start` (default), `stop`, `status`, `tunnel`.

State files under `~/.bhaskera/`:

- `mlflow-ui.json` — persisted `DashboardConfig` (`store`, `port`, `host`, `login_node`, `user`)
- `mlflow-ui.pid` — PID of the running `mlflow ui` process
- `mlflow-ui.log` — captured stdout/stderr

On `start`:

1. Checks for an already-running PID; if alive, prints the tunnel command and exits.
2. Verifies `mlflow` is on `PATH`.
3. Ensures the store directory exists (default `$HOME/mlflow-runs`).
4. Spawns `mlflow ui --backend-store-uri file://<store> --host <host> --port <port>` with `start_new_session=True` so it survives shell exit.
5. Sleeps briefly and re-checks the PID; if the process exited (e.g. port in use), prints the log path and a hint.
6. Prints the `ssh -L <port>:localhost:<port>` command for laptops.

Default port is `5000`; default store is `~/mlflow-runs` — the same path the training-side `MLflowLogger` writes to.

## `monitoring.py` — observability context

`setup_monitoring(cfg)` translates `cfg.monitoring` into a `MonitoringContext`:

- `dashboard`, `dashboard_host`, `dashboard_port`, `metrics_export_port` — direct from config (defaults 8265 / 8080).
- `mlflow_ui_url` — populated when `~/.bhaskera/mlflow-ui.json` exists (i.e. `bhaskera-dashboard start` has been run somewhere), so the training banner shows where to point a browser.

`ctx.ray_init_kwargs()` returns the subset of kwargs to pass to `ray.init`:

```python
{
  "include_dashboard":    self.dashboard,
  "dashboard_host":       self.dashboard_host,
  "dashboard_port":       self.dashboard_port,
  "_metrics_export_port": self.metrics_export_port,
}
```

`ctx.banner(head_ip=None)` produces the multi-line startup summary printed by `train.py`.

## Typical end-to-end

```bash
# (login node, once) start MLflow UI
bhaskera-dashboard

# (compute node) tokenise once
bhaskera-tokenize --config configs/tokenize.yaml

# (compute node) sanity-check NCCL
bhaskera-diag --num-workers 4

# (compute node) train
bhaskera-train --config configs/qwen.yaml --num-workers 4

# (compute node) infer from a checkpoint
bhaskera-infer --config configs/inference.yaml --prompt "..."
```

On SLURM, `scripts/submit.sh` wraps the bootstrap and forwards to `bhaskera-train`. The `RAY_ADDRESS` env var routes the launcher to the already-running cluster.
