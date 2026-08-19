from __future__ import annotations

import logging
from typing import Any

try:
    from ray import train
except ImportError:
    train = None

from .base import BaseLogger

# ── SILENCE RAY TRAIN INTERNAL SPAM ──
logging.getLogger("ray.train").setLevel(logging.WARNING)
logging.getLogger("ray.train._internal.session").setLevel(logging.WARNING)
logging.getLogger("ray.tune").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

class RayMetricsLogger(BaseLogger):
    """Pushes metrics to Ray Train's reporting mechanism cleanly."""
    
    def __init__(self, cfg, *, rank: int = 0, world_size: int = 1) -> None:
        self.rank = rank
        self.world_size = world_size

    def log(self, metrics: dict[str, Any], step: int) -> None:
        if train is None:
            return
            
        # ── SMART ALLOWLIST ──
        clean_metrics = {"step": step}
        
        # 1. Allow core training metrics
        for key in ["loss", "lr"]:
            if key in metrics:
                clean_metrics[key] = metrics[key]
                
        # 2. Allow throughput metrics (renamed for cleanliness)
        if "throughput/tokens_per_sec_global" in metrics:
            clean_metrics["tok/s"] = metrics["throughput/tokens_per_sec_global"]
        if "throughput/mfu_pct" in metrics:
            clean_metrics["MFU"] = metrics["throughput/mfu_pct"]

        # 3. Allow all Evaluation/Benchmark metrics automatically
        for k, v in metrics.items():
            if k.startswith("validation/") or k.startswith("benchmark/"):
                clean_metrics[k] = v

        try:
            train.report(clean_metrics)
        except Exception as e:
            logger.debug(f"RayMetricsLogger failed to report metrics: {e}")

    def finish(self) -> None:
        pass
