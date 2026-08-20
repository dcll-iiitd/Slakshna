"""
bhaskera.launcher.dashboard
===========================
``bhaskera-dashboard`` / ``bhaskera-start`` — start the MLflow UI on the
login node and tell you the exact ``ssh -L`` tunnel command to run on your
laptop.  No Prometheus, no Grafana, no Docker, no root.

Design
------
* The training process (GPU node) pushes metrics to a file-backed MLflow
  store under ``$HOME/mlflow-runs``.  Login nodes share ``$HOME`` with
  compute nodes on most HPC clusters, so the UI reads live data without
  any network plumbing.
* Config persists to ``~/.bhaskera/mlflow-ui.json`` so subsequent runs
  need zero arguments.
* Subcommands: ``start`` (default), ``stop``, ``status``, ``tunnel``.

Typical workflow
----------------
::

    # On the login node — once:
    bhaskera-dashboard

    # From your laptop:
    ssh -L 5000:localhost:5000 ldls-iiitd@<login-node>
    open http://localhost:5000

    # Or use the exact command the tool prints.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger("bhaskera.dashboard")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_PORT       = 5000
_CONFIG_DIR         = Path.home() / ".bhaskera"
_CONFIG_PATH        = _CONFIG_DIR / "mlflow-ui.json"
_PID_FILE           = _CONFIG_DIR / "mlflow-ui.pid"
_LOG_FILE           = _CONFIG_DIR / "mlflow-ui.log"
_DEFAULT_STORE      = str(Path.home() / "mlflow-runs")


# ---------------------------------------------------------------------------
# Persisted config
# ---------------------------------------------------------------------------

@dataclass
class DashboardConfig:
    """Everything bhaskera-dashboard needs to start the MLflow UI.

    Saved to ``~/.bhaskera/mlflow-ui.json`` after the first run so
    subsequent invocations need zero arguments.
    """
    store:      str  = _DEFAULT_STORE
    port:       int  = _DEFAULT_PORT
    host:       str  = "0.0.0.0"
    login_node: str  = ""      # e.g. login08.cluster.example.edu
    user:       str  = ""      # remote SSH user, if different from $USER

    # ── persistence ───────────────────────────────────────────────────────

    @classmethod
    def load(cls) -> "DashboardConfig":
        if _CONFIG_PATH.exists():
            try:
                return cls(**json.loads(_CONFIG_PATH.read_text()))
            except Exception as e:
                logger.debug(f"Could not read saved config: {e}")
        return cls()

    def save(self) -> None:
        _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        _CONFIG_PATH.write_text(json.dumps(asdict(self), indent=2))


# ---------------------------------------------------------------------------
# PID helpers
# ---------------------------------------------------------------------------

def _read_pid() -> Optional[int]:
    try:
        return int(_PID_FILE.read_text().strip())
    except Exception:
        return None


def _write_pid(pid: int) -> None:
    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    _PID_FILE.write_text(str(pid))


def _clear_pid() -> None:
    _PID_FILE.unlink(missing_ok=True)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Core actions
# ---------------------------------------------------------------------------

def cmd_start(cfg: DashboardConfig) -> None:
    """Start the MLflow UI as a detached background process."""
    # Check if already running
    pid = _read_pid()
    if pid and _pid_alive(pid):
        print(f"MLflow UI is already running (PID {pid}). Use 'bhaskera-dashboard status'.")
        _print_tunnel(cfg)
        return

    # Verify mlflow is installed
    if not shutil.which("mlflow"):
        _die(
            "mlflow not found on PATH.\n"
            "Install it with:  pip install mlflow\n"
            "Then re-run:      bhaskera-dashboard"
        )

    # Ensure the store directory exists
    store_path = Path(cfg.store)
    try:
        store_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        _die(f"Cannot create MLflow store directory {cfg.store!r}: {e}")

    # Launch mlflow ui as a detached subprocess
    cmd = [
        "mlflow", "ui",
        "--backend-store-uri", f"file://{store_path.resolve()}",
        "--host", cfg.host,
        "--port", str(cfg.port),
    ]

    _LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    log_fd = open(_LOG_FILE, "a")

    proc = subprocess.Popen(
        cmd,
        stdout=log_fd,
        stderr=log_fd,
        start_new_session=True,   # detach from current terminal
    )
    _write_pid(proc.pid)

    # Brief wait — let mlflow bind the port
    time.sleep(1.5)
    if not _pid_alive(proc.pid):
        _die(
            f"MLflow UI exited immediately. Check the log:\n  {_LOG_FILE}\n"
            "Common cause: port already in use. Try --port <other_port>."
        )

    print("─" * 60)
    print("  MLflow UI started")
    print(f"  PID   : {proc.pid}")
    print(f"  Store : {cfg.store}")
    print(f"  Port  : {cfg.port}  (on {socket.gethostname()})")
    print(f"  Log   : {_LOG_FILE}")
    print()
    _print_tunnel(cfg)
    print("─" * 60)
    print()
    print("Stop with:  bhaskera-dashboard stop")
    print()


def cmd_stop() -> None:
    """Kill the background MLflow UI process."""
    pid = _read_pid()
    if not pid:
        print("No MLflow UI PID recorded — nothing to stop.")
        return
    if not _pid_alive(pid):
        print(f"PID {pid} is no longer running. Clearing stale record.")
        _clear_pid()
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    # Give it a moment, then SIGKILL if still alive
    for _ in range(10):
        time.sleep(0.3)
        if not _pid_alive(pid):
            break
    else:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    _clear_pid()
    print(f"MLflow UI (PID {pid}) stopped.")


def cmd_status(cfg: DashboardConfig) -> None:
    """Print whether the MLflow UI is running and the store details."""
    pid = _read_pid()
    if pid and _pid_alive(pid):
        print(f"  Status : RUNNING  (PID {pid})")
    elif pid:
        print(f"  Status : STOPPED  (stale PID {pid})")
    else:
        print("  Status : NOT STARTED")
    print(f"  Store  : {cfg.store}")
    print(f"  Port   : {cfg.port}")
    print(f"  Log    : {_LOG_FILE}")
    _print_tunnel(cfg)


def cmd_tunnel(cfg: DashboardConfig) -> None:
    """Print the ssh -L tunnel command (and optionally open it)."""
    _print_tunnel(cfg, verbose=True)


# ---------------------------------------------------------------------------
# Tunnel helper
# ---------------------------------------------------------------------------

def _print_tunnel(cfg: DashboardConfig, *, verbose: bool = False) -> None:
    hostname = socket.gethostname()
    node = cfg.login_node or hostname
    user_prefix = f"{cfg.user}@" if cfg.user else ""
    local_url  = f"http://localhost:{cfg.port}"
    tunnel_cmd = f"ssh -L {cfg.port}:localhost:{cfg.port} {user_prefix}{node}"

    if verbose:
        print()
    print(f"  From your laptop:")
    print(f"    {tunnel_cmd}")
    print(f"    open {local_url}")
    if verbose:
        print()


# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bhaskera-dashboard",
        description=(
            "Manage the MLflow UI for Bhaskera training runs.\n\n"
            "First run: supply --store / --port / --login-node once;\n"
            "subsequent runs need no arguments."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "action",
        nargs="?",
        default="start",
        choices=["start", "stop", "status", "tunnel"],
        help="Subcommand (default: start)",
    )
    p.add_argument(
        "--store",
        default=None,
        metavar="URI",
        help=f"MLflow file-store path (default: ~/mlflow-runs). "
             f"Env override: MLFLOW_TRACKING_URI.",
    )
    p.add_argument(
        "--port",
        type=int,
        default=None,
        help=f"Port for the MLflow UI (default: {_DEFAULT_PORT}). "
             f"Env override: MLFLOW_PORT.",
    )
    p.add_argument(
        "--login-node",
        default=None,
        metavar="HOSTNAME",
        help="Login-node hostname to print in the ssh -L command "
             "(default: current hostname).",
    )
    p.add_argument(
        "--user",
        default=None,
        metavar="USER",
        help="SSH username for the tunnel command (default: omitted).",
    )
    p.add_argument(
        "--no-save",
        action="store_true",
        help="Do not persist config to ~/.bhaskera/mlflow-ui.json.",
    )
    p.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return p


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _die(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def _merge_env(cfg: DashboardConfig) -> DashboardConfig:
    """Let env vars override saved config (env < CLI args < saved)."""
    uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    if uri.startswith("file://") and not cfg.store:
        cfg.store = uri[len("file://"):]
    env_port = os.environ.get("MLFLOW_PORT")
    if env_port and cfg.port == _DEFAULT_PORT:
        try:
            cfg.port = int(env_port)
        except ValueError:
            pass
    return cfg


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="[%(levelname)s] %(message)s",
    )

    parser = _build_parser()
    args   = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load saved config, then overlay CLI args
    cfg = DashboardConfig.load()
    _merge_env(cfg)

    if args.store        is not None: cfg.store       = args.store
    if args.port         is not None: cfg.port        = args.port
    if args.login_node   is not None: cfg.login_node  = args.login_node
    if args.user         is not None: cfg.user        = args.user

    # Apply defaults for any still-empty fields
    if not cfg.store:
        cfg.store = _DEFAULT_STORE
    if not cfg.port:
        cfg.port  = _DEFAULT_PORT

    if not args.no_save and args.action in ("start",):
        cfg.save()

    # Dispatch
    if args.action == "start":
        cmd_start(cfg)
    elif args.action == "stop":
        cmd_stop()
    elif args.action == "status":
        cmd_status(cfg)
    elif args.action == "tunnel":
        cmd_tunnel(cfg)


if __name__ == "__main__":
    main()
