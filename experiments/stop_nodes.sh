#!/usr/bin/env bash
# Stop the nodes of a run and everything they spawned.   usage: experiments/stop_nodes.sh <tag> <train|serve>
TAG="$1"; MODE="$2"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
F="experiments/runs/$TAG/$MODE.pids"
if [ -f "$F" ]; then while read -r p; do kill "$p" 2>/dev/null || true; done < "$F"; rm -f "$F"; fi
sleep 3
pkill -f "[i]nference_engine.py" || true
pkill -f "[m]l_engine.py" || true
source Bhaskera/bhaskera-activate.sh >/dev/null 2>&1 || true
ray stop >/dev/null 2>&1 || true
sleep 3
echo "stopped. GPU memory: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | paste -sd' ')"
