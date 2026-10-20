#!/usr/bin/env bash
# Set up the Bhaskera *serve* branch (second submodule) used by federated
# inference. It gets its own virtualenv, separate from the training one, because
# the two Bhaskera branches diverge. Training (setup.sh) is unaffected.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
SERVE_DIR="$SCRIPT_DIR/Bhaskera-serve"

if [ ! -f "$SERVE_DIR/pyproject.toml" ]; then
    echo "Bhaskera-serve is not checked out; initialising submodule..."
    git -C "$SCRIPT_DIR" submodule update --init Bhaskera-serve
fi

echo "=== Creating serve virtualenv at $SERVE_DIR/.venv ==="
python3 -m venv "$SERVE_DIR/.venv"
# shellcheck disable=SC1091
source "$SERVE_DIR/.venv/bin/activate"

pip install --upgrade pip
pip install -e "$SERVE_DIR[serve]"
# adapter hot-swapping in inference_engine.py
pip install peft safetensors

echo "=========================================================="
echo "✅ Serve environment ready."
echo "Add to your node config:"
echo "  [inference]"
echo "  enabled = true"
echo "  base_model = \"<same base model as training>\""
echo "  sidecar_python = \"$SERVE_DIR/.venv/bin/python\""
echo "  serve_bhaskera_path = \"$SERVE_DIR\""
echo "=========================================================="
