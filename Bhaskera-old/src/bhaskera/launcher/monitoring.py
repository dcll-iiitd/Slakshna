"""
bhaskera.launcher.monitoring
============================
Translate ``cfg.monitoring`` into Ray ``init`` kwargs and emit a startup
banner pointing the user at the Ray Dashboard and the MLflow UI.

Prometheus and Grafana have been removed from Bhaskera.  Metrics are
pushed directly to MLflow by ``MLflowLogger`` — no pull-scraping needed.

The returned ``MonitoringContext`` is consumed by ``train.py`` and
``infer.py``.
"""
from __future__ import annotations

import json
import logging
import os
import socket
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Standard ports — match Ray defaults so the docs apply.
_DEFAULT_DASHBOARD_PORT      = 8265
_DEFAULT_METRICS_EXPORT_PORT = 8080

# Written by ``bhaskera-dashboard`` after its first run.
_MLFLOW_UI_CONFIG_PATH = Path.home() / ".bhaskera" / "mlflow-ui.json"


@dataclass
class MonitoringContext:
    """
    Snapshot of the monitoring config after env-var wiring.

    Returned by ``setup_monitoring`` so the caller can pass the
    relevant kwargs into ``ray.init`` and emit a final summary line.
    """
    dashboard:           bool            = True
    dashboard_host:      str             = "0.0.0.0"
    dashboard_port:      int             = _DEFAULT_DASHBOARD_PORT
    metrics_export_port: int             = _DEFAULT_METRICS_EXPORT_PORT
    mlflow_ui_url:       Optional[str]   = None   # set if bhaskera-dashboard is running
    extra_init_kwargs:   dict[str, Any]  = field(default_factory=dict)

    def ray_init_kwargs(self) -> dict[str, Any]:
        """Subset of kwargs to pass directly to ``ray.init``."""
        kw: dict[str, Any] = {
            "include_dashboard":    self.dashboard,
            "dashboard_host":       self.dashboard_host,
            "dashboard_port":       self.dashboard_port,
            "_metrics_export_port": self.metrics_export_port,
        }
        kw.update(self.extra_init_kwargs)
        return kw

    def banner(self, head_ip: Optional[str] = None) -> str:
        """Multi-line startup summary for the training launcher."""
        ip = head_ip or _local_ip()
        lines = [
            "",
            "═══════════════════════════════════════════════════════════════",
            "  Bhaskera Observability",
            "───────────────────────────────────────────────────────────────",
        ]

        if self.dashboard:
            lines.append(
                f"  Ray Dashboard  : http://{ip}:{self.dashboard_port}"
            )
        else:
            lines.append("  Ray Dashboard  : (disabled)")

        if self.mlflow_ui_url:
            lines.append(f"  MLflow UI      : {self.mlflow_ui_url}")
        else:
            lines += [
                "  MLflow UI      : not running on this node",
                "                   start with: bhaskera-dashboard",
                "                   (run on your login node, not here)",
            ]

        lines.append(
            "═══════════════════════════════════════════════════════════════"
        )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def setup_monitoring(cfg) -> MonitoringContext:
    """
    Translate ``cfg.monitoring`` into Ray init kwargs.

    Must be called *before* ``ray.init``. Safe to call repeatedly.

    If ``~/.bhaskera/mlflow-ui.json`` exists (written by
    ``bhaskera-dashboard start``), the MLflow UI URL is surfaced in the
    banner automatically — no YAML change needed.
    """
    mon = getattr(cfg, "monitoring", None)

    if mon is None:
        ctx = MonitoringContext()
    else:
        ctx = MonitoringContext(
            dashboard           = bool(getattr(mon, "dashboard",           True)),
            dashboard_host      = str(getattr(mon, "dashboard_host",       "0.0.0.0")),
            dashboard_port      = int(getattr(mon, "dashboard_port",       _DEFAULT_DASHBOARD_PORT)),
            metrics_export_port = int(getattr(mon, "metrics_export_port",  _DEFAULT_METRICS_EXPORT_PORT)),
        )

    # Surface the MLflow UI URL if bhaskera-dashboard has been run.
    saved = _load_mlflow_ui_config()
    if saved:
        port = saved.get("port", 5000)
        node = saved.get("login_node") or socket.gethostname()
        ctx.mlflow_ui_url = f"http://{node}:{port}"

    return ctx


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_mlflow_ui_config() -> Optional[dict]:
    """Read ~/.bhaskera/mlflow-ui.json if it exists. Never raises."""
    if not _MLFLOW_UI_CONFIG_PATH.exists():
        return None
    try:
        return json.loads(_MLFLOW_UI_CONFIG_PATH.read_text())
    except Exception as e:
        logger.debug(f"Could not read {_MLFLOW_UI_CONFIG_PATH}: {e}")
        return None


def _local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "localhost"
