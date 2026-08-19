"""
bhaskera.utils.loggers
======================
Experiment-tracker factory.

Behaviour:
    * ``cfg.logging.tracker`` may be a string, a list of strings, or
      ``None``.  Recognised values: ``"wandb"``, ``"mlflow"``,
      ``"ray"``, plus the legacy ``"none"`` / ``""`` aliases for "off".
    * If ``cfg.monitoring.dashboard`` is true (the default) and
      ``"ray"`` isn't already in the tracker list, it is added
      automatically — that's how Ray Dashboard becomes the default
      sink when no other tracker is configured.
    * Rank-0 hosts the offline trackers (W&B, MLflow).  Every rank
      runs a ``RayMetricsLogger`` so per-GPU stats reach Prometheus.
    * The result is a ``MultiLogger`` that fans out to each enabled
      backend and never raises into the training loop.
"""
from __future__ import annotations

import logging
from typing import Iterable, Optional

from .base import BaseLogger
from .multi_logger import MultiLogger

logger = logging.getLogger(__name__)


# Public marker values
_VALID = {"wandb", "mlflow", "ray"}
_OFF   = {None, "", "none", "off", "false"}


def _normalize_trackers(raw) -> list[str]:
    """Return a clean lowercase list of tracker names."""
    if raw is None:
        return []
    if isinstance(raw, str):
        if raw.lower() in _OFF:
            return []
        items: Iterable = [raw]
    elif isinstance(raw, (list, tuple, set)):
        items = raw
    else:
        raise TypeError(
            f"logging.tracker must be str / list / None, got {type(raw)}"
        )

    cleaned: list[str] = []
    for it in items:
        if it is None:
            continue
        s = str(it).strip().lower()
        if s in _OFF or not s:
            continue
        if s not in _VALID:
            raise ValueError(
                f"Unknown tracker '{s}'. "
                f"Choose from: {sorted(_VALID)} | none"
            )
        if s not in cleaned:
            cleaned.append(s)
    return cleaned


def build_logger(
    cfg,
    *,
    rank: int = 0,
    world_size: int = 1,
) -> Optional[BaseLogger]:
    """
    Build the appropriate logger(s) for this rank.

    Returns ``None`` only if every backend is disabled — otherwise
    a ``MultiLogger`` (possibly wrapping a single child) is returned.
    """
    trackers = _normalize_trackers(getattr(cfg.logging, "tracker", None))

    # Ray Dashboard is on by default unless explicitly disabled in
    # cfg.monitoring.  When on, ensure 'ray' is in the tracker set so
    # custom training metrics flow to the dashboard's Prometheus.
    monitoring = getattr(cfg, "monitoring", None)
    dashboard_on = bool(getattr(monitoring, "dashboard", True)) if monitoring else True
    if dashboard_on and "ray" not in trackers:
        trackers.append("ray")

    children: list[BaseLogger] = []

    # ── Ray (every rank) ────────────────────────────────────────────
    if "ray" in trackers:
        try:
            from .ray_logger import RayMetricsLogger
            children.append(
                RayMetricsLogger(cfg, rank=rank, world_size=world_size)
            )
        except Exception as e:
            logger.warning(
                f"RayMetricsLogger unavailable ({e}) — Ray Dashboard "
                "metrics will not include training-loop signals."
            )

    # ── Rank-0-only: W&B and MLflow ────────────────────────────────
    if rank == 0:
        if "wandb" in trackers:
            try:
                from .wandb_logger import WandbLogger
                children.append(WandbLogger(cfg, rank=rank, world_size=world_size))
            except Exception as e:
                logger.warning(f"WandbLogger init failed: {e}")
        if "mlflow" in trackers:
            try:
                from .mlflow_logger import MLflowLogger
                children.append(MLflowLogger(cfg, rank=rank, world_size=world_size))
            except Exception as e:
                logger.warning(f"MLflowLogger init failed: {e}")

    if not children:
        return None
    return MultiLogger(children)


__all__ = ["BaseLogger", "MultiLogger", "build_logger"]
