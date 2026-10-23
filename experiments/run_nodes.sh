#!/usr/bin/env bash
# Start all nodes of an experiment run.   usage: experiments/run_nodes.sh <tag> <train|serve> [export_kind]
#   train: nodes train and publish adapters (export_kind = local | aggregated, default local)
#   serve: nodes only serve; every node has a model sidecar and can answer queries
set -e
TAG="$1"; MODE="$2"; KIND="${3:-local}"
[ -n "$TAG" ] && [ -n "$MODE" ] || { echo "usage: $0 <tag> <train|serve> [local|aggregated]"; exit 1; }
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
RUN="experiments/runs/$TAG"
[ -d "$RUN/$MODE" ] || { echo "no configs in $RUN/$MODE (run experiments/setup_run.py first)"; exit 1; }
source Bhaskera/bhaskera-activate.sh >/dev/null 2>&1 || true
DATA_TAG=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['data_tag'])" "$RUN/run.json" 2>/dev/null || echo "$TAG")
export SLAKSHNA_PARTITION_ROOT="$REPO/experiments/data/$DATA_TAG"
export SLAKSHNA_TMP_BASE="/mnt/disk1/slakshna/r"   # short on purpose: Ray unix-socket paths are limited to 107 bytes
export SLAKSHNA_EXPORT_KIND="$KIND"
mkdir -p "$SLAKSHNA_TMP_BASE" "$RUN/logs"
: > "$RUN/$MODE.pids"
for cfg in "$RUN/$MODE"/node*.toml; do
  n=$(basename "$cfg" .toml)
  ./target/release/iiitd --config "$cfg" > "$RUN/logs/${n}_$MODE.log" 2>&1 &
  echo $! >> "$RUN/$MODE.pids"
  echo "started $n (pid $!) log: $RUN/logs/${n}_$MODE.log"
done
