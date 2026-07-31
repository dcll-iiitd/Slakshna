#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
BASE_DIR=$(realpath "$SCRIPT_DIR/..") # Default to one level up

# Intelligently search for the workspace root containing 'Bhaskera', starting from closest
if [ -d "$SCRIPT_DIR/Bhaskera" ]; then
    BASE_DIR="$SCRIPT_DIR"
elif [ -d "$SCRIPT_DIR/../Bhaskera" ]; then
    BASE_DIR=$(realpath "$SCRIPT_DIR/..")
elif [ -d "$SCRIPT_DIR/../../Bhaskera" ]; then
    BASE_DIR=$(realpath "$SCRIPT_DIR/../..")
fi

echo "=== 1. Setting up Python Environment ==="
# Check if Bhaskera central activation script exists
if [ -f "$BASE_DIR/Bhaskera/bhaskera-activate.sh" ]; then
    echo "🔍 Detected Bhaskera environment! Activating via $BASE_DIR/Bhaskera/bhaskera-activate.sh..."
    source "$BASE_DIR/Bhaskera/bhaskera-activate.sh"
else
    echo "📦 Creating local Python virtual environment (.venv)..."
    python3 -m venv .venv
    source .venv/bin/activate
fi

if command -v pip &>/dev/null; then
    PIP_CMD="pip"
elif command -v uv &>/dev/null; then
    PIP_CMD="uv pip"
elif python3 -m pip --version &>/dev/null; then
    PIP_CMD="python3 -m pip"
else
    echo "📦 pip not found! Ensuring pip is installed..."
    python3 -m ensurepip --upgrade
    PIP_CMD="python3 -m pip"
fi

$PIP_CMD install --upgrade pip || true
$PIP_CMD install torch torchvision numpy scipy opt-einsum opacus pyarrow ray pyyaml

echo "=== 2. Updating Hardcoded Paths ==="
# Using the dynamically detected BASE_DIR from above
echo "Setting base directory to: $BASE_DIR"
if [ "$BASE_DIR" != "/mnt/disk1/slakshna" ]; then
    sed -i "s|/mnt/disk1/slakshna|$BASE_DIR|g" ml_engine.py
    echo "Updated ml_engine.py with the correct machine path."
else
    echo "Paths are already correct for this machine."
fi
mkdir -p data logs ml_models ml_states t

echo "=== 3. Compiling Rust L1 Engine (Without Layer-2 Blockchain) ==="
cargo build --release

echo "=========================================================="
echo "✅ Setup complete for Slakshna (Layer-1 FL over Bhaskera)!"
echo "To start your node:"
echo "  1. Activate environment: source $BASE_DIR/Bhaskera/bhaskera-activate.sh (or local .venv)"
echo "  2. Run the engine: ./target/release/iiitd --config <your_node_config.toml>"
echo "=========================================================="
