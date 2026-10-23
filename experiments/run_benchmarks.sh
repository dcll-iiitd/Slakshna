#!/usr/bin/env bash
# Everything after training, for one run (default ag_news_n4_a0.1_s0), in an order that keeps latency
# measurements clean (one thing on the GPUs at a time). Logs go to experiments/runs/<tag>/logs/.
#
#   experiments/run_benchmarks.sh ag_news_n4_a0.1_s0 [steps]
#   steps (default all): offline api serving vllm fault fetch tables
set -u
TAG="${1:-ag_news_n4_a0.1_s0}"
STEPS="${2:-offline api serving vllm fault fetch tables}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
PY=Bhaskera-serve/.venv/bin/python
VPY=experiments/.venv-vllm/bin/python
R=experiments/results/$TAG
L=experiments/runs/$TAG/logs
mkdir -p "$R/raw" "$L"
has() { [[ " $STEPS " == *" $1 "* ]]; }
say() { echo "[$(date -u +%T)] $*"; }

wait_sidecars() {  # all 4 sidecars of the run report healthy
  for i in $(seq 1 120); do
    c=0; for k in 1 2 3 4; do curl -sf "localhost:$((8765 + k - 1))/health" >/dev/null && c=$((c + 1)); done
    [ "$c" = 4 ] && return 0; sleep 5
  done; return 1
}

if has offline; then
  say "offline quality evaluation"
  # fedavg50: 50% of each update is exchanged (a full update, 13.95 MB, exceeds the 10 MB gossip message cap)
  EXTRA="${TAG}_fedavg50:fedavg,${TAG}_isolated:local_only,${TAG/_n4_/_n1_}_central:centralised"
  $PY experiments/eval_offline.py --tag "$TAG" --gpu 0 --merge --extra-runs "$EXTRA" > "$L/eval_offline.log" 2>&1 || say "offline eval failed"
fi

if has api || has serving; then
  say "serve mode (caches reset first, so the first use of every peer adapter is a real network fetch)"
  ./experiments/reset_caches.sh "$TAG" > /dev/null
  ./experiments/run_nodes.sh "$TAG" serve local > /dev/null
  wait_sidecars || say "sidecars not ready"
  if has api; then
    say "API evaluation (selection strategies, cold vs warm)"
    $PY experiments/eval_api.py --tag "$TAG" --n 200 > "$L/eval_api.log" 2>&1 || say "api eval failed"
  fi
  if has serving; then
    say "serving benchmark: ours"
    $PY experiments/bench_serving.py --backend slakshna --url http://127.0.0.1:8545 --out "$R" > "$L/bench_serving_ours.log" 2>&1 || say "bench ours failed"
    curl -s localhost:8765/stats > "$R/raw/gpu_memory_ours.json"
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv > "$R/raw/gpu_state_during_ours.txt"
  fi
  ./experiments/stop_nodes.sh "$TAG" serve > /dev/null
fi

if has vllm; then
  say "serving benchmark: vLLM"
  ./experiments/run_vllm.sh "$TAG" "${VLLM_GPU:-1}" 8000 > /dev/null   # same GPU as our node1 (serve configs)
  for i in $(seq 1 120); do curl -sf localhost:8000/v1/models > /dev/null && break; sleep 5; done
  $VPY experiments/bench_serving.py --backend vllm --url http://127.0.0.1:8000 --adapter node1 --out "$R" > "$L/bench_serving_vllm.log" 2>&1 || say "bench vllm failed"
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader > "$R/raw/gpu_memory_vllm.txt"
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv > "$R/raw/gpu_state_during_vllm.txt"
  pkill -f "[v]llm serve"; sleep 5
fi

if has fault; then
  say "fault tolerance (the script restarts the nodes cold for every phase)"
  $PY experiments/fault_test.py --tag "$TAG" --n 30 > "$L/fault_test.log" 2>&1 || say "fault test failed"
fi

if has fetch; then
  say "fetch vs adapter size"
  $PY experiments/bench_fetch.py --tag "$TAG" > "$L/bench_fetch.log" 2>&1 || say "fetch bench failed"
  ./experiments/stop_nodes.sh "$TAG" serve > /dev/null
fi

if has tables; then
  say "tables and figures"
  $PY experiments/aggregate.py --tag "$TAG" > "$L/aggregate.log" 2>&1 || say "aggregate failed"
  $PY experiments/compare_systems.py --tag "$TAG" > "$L/compare_systems.log" 2>&1 || say "compare failed"
  $PY experiments/plot_results.py --tag "$TAG" > "$L/plot.log" 2>&1 || say "plot failed"
fi
say "done"
