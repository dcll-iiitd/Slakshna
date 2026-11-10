#!/usr/bin/env bash
# SLAKSHNA - CPU-only setup for Linux / macOS / Git-Bash.
#   bash setup_cpu.sh [--skip-prepare]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

[ -f Bhaskera/pyproject.toml ] || git submodule update --init --recursive

PY=${PYTHON:-python3}
"$PY" - <<'EOF'
import sys
assert sys.version_info >= (3, 11), f"Python {sys.version.split()[0]} found; 3.11+ required"
EOF

[ -d .venv ] || "$PY" -m venv .venv
if [ -f .venv/Scripts/python.exe ]; then VENV_PY=.venv/Scripts/python.exe; else VENV_PY=.venv/bin/python; fi

"$VENV_PY" -m pip install --upgrade pip
"$VENV_PY" -m pip install --index-url https://download.pytorch.org/whl/cpu torch
"$VENV_PY" -m pip install -r requirements-cpu.txt
"$VENV_PY" -m pip install -e Bhaskera --no-deps

if [ "${1:-}" != "--skip-prepare" ]; then
    "$VENV_PY" prepare_cpu.py --smoke
fi

echo
echo "Setup complete. Without Rust you can already run:"
echo "  $VENV_PY scripts/simulate_cpu_cluster.py --config-dir examples/cpu/2node --rounds 3"
echo "With Rust:  cargo build --release && ./start_cluster.sh 2 cpu"
