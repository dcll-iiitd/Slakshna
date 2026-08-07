#!/usr/bin/env bash
set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <number_of_nodes> (2, 4, or 6)"
    exit 1
fi

NUM_NODES=$1

if [ "$NUM_NODES" != "2" ] && [ "$NUM_NODES" != "4" ] && [ "$NUM_NODES" != "6" ]; then
    echo "Error: Only 2, 4, or 6 node configurations are supported."
    exit 1
fi

CONFIG_DIR="examples/${NUM_NODES}node"

if [ ! -d "$CONFIG_DIR" ]; then
    echo "Error: Configuration directory $CONFIG_DIR not found."
    exit 1
fi

# Determine virtual environment activation
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

if [ -f "$BASE_DIR/Bhaskera/bhaskera-activate.sh" ]; then
    source "$BASE_DIR/Bhaskera/bhaskera-activate.sh"
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Build if needed
if [ ! -f "./target/release/iiitd" ]; then
    echo "Rust binary not found. Building..."
    cargo build --release
fi

echo "Starting $NUM_NODES node cluster..."

# Create logs directory
mkdir -p logs
rm -f cluster.pids

for i in $(seq 1 $NUM_NODES); do
    CONFIG_FILE="$CONFIG_DIR/node${i}.toml"
    LOG_FILE="logs/node${i}_cluster.log"
    
    if [ ! -f "$CONFIG_FILE" ]; then
        echo "Warning: $CONFIG_FILE not found, skipping."
        continue
    fi
    
    echo "Starting Node $i with config $CONFIG_FILE (logging to $LOG_FILE)..."
    ./target/release/iiitd --config "$CONFIG_FILE" > "$LOG_FILE" 2>&1 &
    
    # Save the PID
    echo $! >> cluster.pids
done

echo ""
echo "✅ All $NUM_NODES nodes started in the background!"
echo "Use './stop_cluster.sh' to stop them."
echo "You can check logs in the 'logs/' directory."
