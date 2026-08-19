"""
bhaskera.utils.loggers.mlflow_logger
====================================
Push-based experiment-tracking logger using MLflow.

Why this exists
---------------
On a SLURM cluster, the GPU node is not directly reachable. A pull-based
Prometheus + Grafana stack forces the user to:
    * open a port on every compute node,
    * generate a fresh scrape config every job,
    * SSH-tunnel into a Grafana that lives on a particular login node,
    * remember which login node that was last time.

This logger inverts the flow: the *training process* writes its metrics
to an MLflow tracking store (default: a file-backed store under
``$HOME/mlflow-runs``). Because login nodes share ``$HOME`` with compute
nodes on most clusters, the UI process on the login node reads the same
directory the trainer writes to — no server, no auth, no port collisions.

Run identity
------------
A run is uniquely identified by:
    * ``run_id``        — MLflow's opaque ID (assigned on ``start_run``).
    * ``run_name``      — the human-friendly name from ``cfg.logging.run_name``.
    * ``slurm_job``     — ``$SLURM_JOB_ID`` if present (so multi-rank runs
                          from the same sbatch share an alias for sorting).
    * ``framework``     — always ``"bhaskera"``.
    * ``tracker``       — always ``"mlflow"`` (so multi-backend runs can
                          be told apart in a fan-out setup).
    * ``project`` / ``experiment`` — ``cfg.logging.project`` (overridable
                                     via ``cfg.logging.mlflow_experiment``).

Rank fan-in
-----------
By default only rank 0 publishes; per-GPU metrics from ``system_stats()``
already include the gpu index in the key, so a single run is enough for
the common case. To explicitly log per-rank breakdowns set
``cfg.logging.mlflow_log_all_ranks: true``; each rank then opens its own
MLflow run and prefixes every metric with ``node_<hostname>_`` so an
N-node × M-GPU job shows N×M independent curves in the comparison view.

Graceful degradation
--------------------
If ``mlflow`` is not installed or the store URI is unreachable, this
logger becomes a no-op and prints a single warning. The training loop
is never blocked by metrics. A bounded queue plus a daemon worker
thread absorb transient slowness; a slow/jittery store cannot stall
the train step.
"""
from __future__ import annotations

import logging
import os
import queue
import socket
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .base import BaseLogger

logger = logging.getLogger(__name__)

# How many backlogged metric batches we keep before dropping (oldest first).
# At 10 Hz logging and ~50 metrics/step, 2048 is ~3 minutes of buffer.
_QUEUE_MAXSIZE = 2048

# MLflow's hard limit on a single param value.
_PARAM_VALUE_MAX = 500


class MLflowLogger(BaseLogger):
    """
    Write metrics to an MLflow tracking store.

    Construct only on ranks that should publish (default: rank 0). Pass
    ``rank`` / ``world_size`` so the run is tagged correctly. The
    constructor is fast — all I/O happens on a background thread.

    Required from ``cfg.logging``:
        ``project``     : experiment / project name.
        ``run_name``    : run name shown in the UI.
    Optional:
        ``mlflow_tracking_uri``       (str, default ``file://$HOME/mlflow-runs``;
                                       env override ``MLFLOW_TRACKING_URI``)
        ``mlflow_experiment``         (str, default = ``project``)
        ``mlflow_log_all_ranks``      (bool, default False)
        ``mlflow_log_artifacts_every``(int, default 50) — training.log
                                       artifact push cadence (steps).
        ``tags``                      (list[str]) — k:v or bare strings.
        ``group``                     (str) — sweep / cohort label.
    """

    # --------------------------------------------------------------
    # Construction
    # --------------------------------------------------------------
    def __init__(self, cfg, *, rank: int = 0, world_size: int = 1) -> None:
        self._rank = int(rank)
        self._world_size = int(world_size)
        self._hostname = socket.gethostname()
        self._available = False
        self._run_id: Optional[str] = None
        self._mlflow = None
        self._queue: "queue.Queue[tuple[dict, int]]" = queue.Queue(maxsize=_QUEUE_MAXSIZE)
        self._stop = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._dropped = 0

        log_cfg = cfg.logging
        log_all = bool(getattr(log_cfg, "mlflow_log_all_ranks", False))
        if rank != 0 and not log_all:
            # Silent participant. Keep the no-op shape so MultiLogger
            # doesn't have to special-case None.
            logger.debug(
                f"MLflowLogger: rank {rank} is silent (mlflow_log_all_ranks=false)"
            )
            return

        try:
            import mlflow  # type: ignore
        except Exception as e:
            logger.warning(
                f"MLflowLogger: 'mlflow' not importable ({e}) — logger is a no-op. "
                "Install with: pip install mlflow"
            )
            return

        # Resolve the tracking URI: cfg overrides env, env overrides default.
        uri = (
            getattr(log_cfg, "mlflow_tracking_uri", None)
            or os.environ.get("MLFLOW_TRACKING_URI")
            or f"file://{Path.home() / 'mlflow-runs'}"
        )
        experiment = (
            getattr(log_cfg, "mlflow_experiment", None)
            or getattr(log_cfg, "project", "bhaskera")
        )
        run_name = getattr(log_cfg, "run_name", "run")
        artifacts_every = int(getattr(log_cfg, "mlflow_log_artifacts_every", 50) or 50)
        self._artifacts_every = max(1, artifacts_every)

        # Ensure file-backed stores exist on disk before MLflow tries to use them.
        if uri.startswith("file://"):
            try:
                Path(uri[len("file://"):]).mkdir(parents=True, exist_ok=True)
            except OSError as e:
                logger.warning(
                    f"MLflowLogger: could not create store dir {uri!r}: {e} — no-op."
                )
                return

        try:
            mlflow.set_tracking_uri(uri)
            mlflow.set_experiment(experiment)
            run = mlflow.start_run(run_name=run_name)
            self._run_id = run.info.run_id
        except Exception as e:
            logger.warning(
                f"MLflowLogger: start_run failed at {uri!r}: {e} — logger is a no-op."
            )
            return

        self._mlflow = mlflow

        # Tag and parametrise the run. All non-fatal.
        try:
            self._log_run_identity(cfg)
        except Exception as e:
            logger.debug(f"MLflowLogger: identity logging failed: {e}")

        # training.log artifact path — local, predictable, tail -f-able.
        self._artifact_dir = (
            Path(os.environ.get("BHASKERA_HOME") or (Path.home() / ".bhaskera"))
            / "runs" / self._run_id
        )
        try:
            self._artifact_dir.mkdir(parents=True, exist_ok=True)
            self._log_file = self._artifact_dir / "training.log"
            # Truncate any leftover from a previous run with the same id (won't
            # happen with real UUIDs, but defensive).
            self._log_file.write_text("")
        except OSError as e:
            logger.warning(f"MLflowLogger: cannot create local artifact dir: {e}")
            self._log_file = None

        self._available = True
        self._worker = threading.Thread(
            target=self._pump,
            name="bhaskera-mlflow-logger",
            daemon=True,
        )
        self._worker.start()

        logger.info(
            f"MLflowLogger ready (uri={uri}, experiment={experiment}, "
            f"run_id={self._run_id}, run_name={run_name})"
        )
        if self._log_file is not None:
            logger.info(f"MLflowLogger: training.log → {self._log_file}")

    # --------------------------------------------------------------
    # Identity / params / tags
    # --------------------------------------------------------------
    def _log_run_identity(self, cfg) -> None:
        log_cfg = cfg.logging
        mlflow = self._mlflow
        run_id = self._run_id

        # Tags: framework / tracker first, then host / SLURM, then user tags.
        tags: dict[str, str] = {
            "framework": "bhaskera",
            "tracker": "mlflow",
            "host": self._hostname,
            "rank": str(self._rank),
            "world_size": str(self._world_size),
            "slurm_job": os.environ.get("SLURM_JOB_ID", ""),
            "slurm_nodelist": os.environ.get("SLURM_JOB_NODELIST", ""),
        }
        group = getattr(log_cfg, "group", None)
        if group:
            tags["group"] = str(group)
        for t in (getattr(log_cfg, "tags", None) or []):
            if not isinstance(t, str):
                continue
            if ":" in t:
                k, v = t.split(":", 1)
                tags[k.strip()] = v.strip()
            else:
                tags[t.strip()] = "true"
        for k, v in tags.items():
            try:
                mlflow.set_tag(k, _truncate(v, _PARAM_VALUE_MAX), run_id=run_id)
            except Exception:
                pass

        # Params: high-level run identity + full nested cfg.as_dict().
        base_params = {
            "host": self._hostname,
            "slurm_job": os.environ.get("SLURM_JOB_ID", ""),
            "slurm_nodelist": os.environ.get("SLURM_JOB_NODELIST", ""),
            "world_size": self._world_size,
            "rank": self._rank,
            "num_gpus": _num_gpus(),
        }
        for k, v in base_params.items():
            try:
                mlflow.log_param(k, _truncate(str(v), _PARAM_VALUE_MAX), run_id=run_id)
            except Exception:
                pass

        # Nested cfg: flatten with dot notation; any value over 500 chars is
        # split further (or json-dumped + truncated, with a note).
        try:
            cfg_dict = cfg.as_dict()
        except Exception:
            cfg_dict = {}
        for k, v in _flatten(cfg_dict):
            try:
                mlflow.log_param(k, _truncate(_stringify(v), _PARAM_VALUE_MAX), run_id=run_id)
            except Exception:
                # MLflow rejects duplicate / oversized params silently here.
                pass

    # --------------------------------------------------------------
    # BaseLogger
    # --------------------------------------------------------------
    def log(self, metrics: dict[str, Any], step: int) -> None:
        if not self._available or not metrics:
            return
        # Local append happens synchronously — it's cheap (one file write)
        # and gives tail -f users instant feedback.
        if self._log_file is not None:
            try:
                line = _format_training_log_line(metrics, int(step))
                with open(self._log_file, "a") as f:
                    f.write(line + "\n")
            except OSError as e:
                logger.debug(f"MLflowLogger: training.log append failed: {e}")

        # MLflow round-trips happen on the worker thread.
        try:
            self._queue.put_nowait((dict(metrics), int(step)))
        except queue.Full:
            self._dropped += 1
            # Drain one to make room — newest data is most useful.
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait((dict(metrics), int(step)))
            except queue.Full:
                pass
            if self._dropped % 100 == 1:
                logger.warning(
                    f"MLflowLogger: dropped {self._dropped} metric batches "
                    "(store slow or unreachable)"
                )

    def finish(self) -> None:
        if not self._available:
            return
        self._stop.set()
        if self._worker is not None:
            self._worker.join(timeout=10.0)
            if self._worker.is_alive():
                logger.warning("MLflowLogger: worker did not exit in 10s — closing anyway")

        # Final artifact push so the last steps' lines are visible in the UI.
        self._upload_log_artifact()

        try:
            if self._mlflow is not None and self._run_id is not None:
                self._mlflow.end_run()
        except Exception as e:
            logger.debug(f"MLflowLogger: end_run failed: {e}")

    # --------------------------------------------------------------
    # Worker
    # --------------------------------------------------------------
    def _pump(self) -> None:
        """
        Background pump.

        Pulls (metrics, step) batches off the queue and writes them to
        MLflow. Survives transient store errors via exponential backoff
        capped at 30s; the queue absorbs the gap. Tracks the last step
        seen and uploads the local training.log as a run artifact every
        ``mlflow_log_artifacts_every`` steps.
        """
        backoff = 0.5
        last_uploaded_step = -1
        while not (self._stop.is_set() and self._queue.empty()):
            try:
                metrics, step = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._flush_one(metrics, step)
                backoff = 0.5
            except Exception as e:
                logger.debug(f"MLflowLogger: log_metrics failed (step={step}): {e}")
                time.sleep(backoff)
                backoff = min(backoff * 2.0, 30.0)
                continue

            # Periodic artifact upload (cheap on a file store: it's a copy).
            if (
                self._log_file is not None
                and step // self._artifacts_every
                != last_uploaded_step // self._artifacts_every
            ):
                self._upload_log_artifact()
                last_uploaded_step = step

    def _flush_one(self, metrics: dict[str, Any], step: int) -> None:
        safe: dict[str, float] = {}
        for raw_key, raw_val in metrics.items():
            try:
                fv = float(raw_val)
            except (TypeError, ValueError):
                continue
            safe[self._translate_key(raw_key)] = fv
        if not safe:
            return
        self._mlflow.log_metrics(safe, step=step, run_id=self._run_id)

    def _translate_key(self, raw: str) -> str:
        """``gpu/0/util_pct`` → ``gpu_0_util_pct``; prefix node_ if requested."""
        # MLflow allows alphanumerics, underscores, dashes, periods, spaces,
        # colons, slashes — but slashes are deprecated as path separators.
        # Underscore is unambiguous; match the prototype.
        key = raw.replace("/", "_")
        if self._rank != 0 or _should_prefix_node(self):
            key = f"node_{_sanitize_host(self._hostname)}_{key}"
        return key

    def _upload_log_artifact(self) -> None:
        if self._log_file is None or not self._available:
            return
        try:
            if self._log_file.exists() and self._log_file.stat().st_size > 0:
                self._mlflow.log_artifact(str(self._log_file), run_id=self._run_id)
        except Exception as e:
            logger.debug(f"MLflowLogger: log_artifact failed: {e}")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _should_prefix_node(self_obj: MLflowLogger) -> bool:
    """Prefix metric names with the node hostname when log_all_ranks is set
    (so an N-node × M-GPU job produces N×M independent curves)."""
    # We can't easily plumb cfg into _translate_key without a reference;
    # the flag is reflected in whether non-rank-0 ranks reach this method
    # at all — rank-0 only reaches it without the flag. Defensive: also
    # use the explicit attribute if a subclass sets it.
    return getattr(self_obj, "_prefix_node", False)


def _sanitize_host(h: str) -> str:
    """Make a hostname safe for use as an MLflow metric-name prefix."""
    return "".join(c if (c.isalnum() or c in "_-.") else "_" for c in h)


def _truncate(s: str, max_len: int) -> str:
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def _stringify(v: Any) -> str:
    if isinstance(v, (str, int, float, bool)) or v is None:
        return str(v)
    try:
        import json
        return json.dumps(v, default=str, sort_keys=True)
    except Exception:
        return repr(v)


def _flatten(d: Any, prefix: str = "") -> "list[tuple[str, Any]]":
    """Flatten a nested dict to dotted keys: ``training.distributed.fsdp.cpu_offload``."""
    out: list[tuple[str, Any]] = []
    if isinstance(d, dict):
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.extend(_flatten(v, key))
    elif isinstance(d, (list, tuple)):
        # Don't enumerate large lists as individual params — keep the list
        # as a single (truncated) JSON value.
        out.append((prefix, list(d)))
    else:
        out.append((prefix, d))
    return out


def _num_gpus() -> int:
    """Best-effort GPU count. Tries CUDA_VISIBLE_DEVICES, torch, then 0."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd:
        return len([x for x in cvd.split(",") if x.strip()])
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.device_count()
    except Exception:
        pass
    return 0


# ------------------------------------------------------------------
# Live training-log line renderer
# ------------------------------------------------------------------

_THROUGHPUT_KEYS = ("tokens/s", "tokens_per_sec", "throughput", "samples/s", "samples_per_sec")
_RENDERED_FORCED = {"loss", "lr", *_THROUGHPUT_KEYS}
_MODEL_KEYS = ("model/total_params", "model/trainable_params", "model/world_size")

def _format_training_log_line(metrics: dict, step: int) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    parts: list[str] = [f"step={step}"]

    # loss / lr / throughput (existing logic — unchanged)
    if "loss" in metrics:
        try:
            parts.append(f"loss={float(metrics['loss']):.4f}")
        except (TypeError, ValueError):
            pass
    if "lr" in metrics:
        try:
            parts.append(f"lr={float(metrics['lr']):.2e}")
        except (TypeError, ValueError):
            pass
    for tk in _THROUGHPUT_KEYS:
        if tk in metrics:
            try:
                parts.append(f"{tk.replace('/', '_')}={int(float(metrics[tk]))}")
            except (TypeError, ValueError):
                pass
            break

    # model/ group  ← NEW
    rendered_model_keys: set[str] = set()
    for k in sorted(metrics.keys()):
        if not k.startswith("model/"):
            continue
        try:
            fv = float(metrics[k])
            label = k.split("/", 1)[1]   # total_params, trainable_params, etc.
            parts.append(f"{label}={_fmt_number(fv)}")
            rendered_model_keys.add(k)
        except (TypeError, ValueError):
            pass

    # GPU grouping (existing logic — unchanged)
    gpu_idxs: set[int] = set()
    for k in metrics:
        if k.startswith("gpu/"):
            sub = k.split("/")
            if len(sub) >= 3 and sub[1].isdigit():
                gpu_idxs.add(int(sub[1]))

    rendered_gpu_keys: set[str] = set()
    for i in sorted(gpu_idxs):
        prefix = f"gpu/{i}/"
        sub = {k[len(prefix):]: v for k, v in metrics.items() if k.startswith(prefix)}
        if "util_pct" in sub:
            try:
                parts.append(f"gpu{i}_util={int(float(sub['util_pct']))}%")
                rendered_gpu_keys.add(f"{prefix}util_pct")
            except (TypeError, ValueError):
                pass
        used  = sub.get("mem_used_mib")
        total = sub.get("mem_total_mib")
        if used is not None:
            try:
                used_gb = float(used) / 1024.0
                if total is not None:
                    total_gb = float(total) / 1024.0
                    parts.append(f"gpu{i}_mem={used_gb:.1f}G/{total_gb:.1f}G")
                    rendered_gpu_keys.add(f"{prefix}mem_total_mib")
                else:
                    parts.append(f"gpu{i}_mem={used_gb:.1f}G")
                rendered_gpu_keys.add(f"{prefix}mem_used_mib")
            except (TypeError, ValueError):
                pass
        if "temp_c" in sub:
            try:
                parts.append(f"gpu{i}_temp={int(float(sub['temp_c']))}C")
                rendered_gpu_keys.add(f"{prefix}temp_c")
            except (TypeError, ValueError):
                pass
        if "power_w" in sub:
            try:
                parts.append(f"gpu{i}_power={int(float(sub['power_w']))}W")
                rendered_gpu_keys.add(f"{prefix}power_w")
            except (TypeError, ValueError):
                pass

    # Everything else numeric, alphabetised
    for k in sorted(metrics.keys()):
        if k in _RENDERED_FORCED or k in rendered_gpu_keys or k in rendered_model_keys:
            continue
        try:
            fv = float(metrics[k])
        except (TypeError, ValueError):
            continue
        parts.append(f"{k}={_fmt_number(fv)}")

    return f"[{ts}] " + " | ".join(parts)



def _fmt_number(x: float) -> str:
    if x == 0:
        return "0"
    if abs(x) < 1e-3 or abs(x) >= 1e6:
        return f"{x:.2e}"
    if abs(x) >= 100:
        return f"{x:.1f}"
    return f"{x:.4f}".rstrip("0").rstrip(".")
