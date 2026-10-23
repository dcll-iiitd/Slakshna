#!/usr/bin/env bash
# Everything after training for the System One evaluation of one run, one GPU job at a time so latency stays clean.
#   experiments/run_system_one.sh <tag> [steps]     steps (default all): offline serve vllm fault tables
# Results: experiments/results/system_one/<tag>/   Logs: experiments/runs/<tag>/logs/
set -u
TAG="$1"
STEPS="${2:-offline serve vllm fault tables}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
PY=Bhaskera-serve/.venv/bin/python
VPY=experiments/.venv-vllm/bin/python
R=experiments/results/system_one/$TAG
L=experiments/runs/$TAG/logs
mkdir -p "$R/raw" "$L"
has() { [[ " $STEPS " == *" $1 "* ]]; }
say() { echo "[$(date -u +%T)] $*"; }
N1=${TAG/_n4_/_n1_}
EXTRA="${TAG}_isolated:local_only,${N1}_central:centralised"
OPTS=""; [[ "$TAG" == banking77* ]] && OPTS="--option-counts 4,16,52 --n-global 462 --n-local 200 --formats raw"

wait_sidecars() {
  for i in $(seq 1 120); do
    c=0; for k in 1 2 3 4; do curl -sf "localhost:$((8765 + k - 1))/health" >/dev/null && c=$((c + 1)); done
    [ "$c" = 4 ] && return 0; sleep 5
  done; return 1
}

wait_registries() {  # every node's registry lists, for every node, the adapter that node advertises as its own
  python3 - <<'PY'
import json, time, urllib.request
def ads(k):
    return json.load(urllib.request.urlopen(f"http://127.0.0.1:{8545 + k - 1}/adapters", timeout=5))["adapters"]
t0 = time.time()
while time.time() - t0 < 600:
    try:
        own = {a["node_id"]: a["sha256"] for k in range(1, 5) for a in ads(k) if a["is_self"]}
        ok = len(own) == 4 and all({a["node_id"]: a["sha256"] for a in ads(k)} == own for k in range(1, 5))
        if ok:
            print(f"registries consistent after {time.time() - t0:.0f}s"); break
    except Exception:
        pass
    time.sleep(5)
else:
    print("WARNING: registries still inconsistent after 600s")
PY
}

if has offline; then
  say "offline decision evaluation"
  $PY experiments/eval_decisions.py --tag "$TAG" --gpu "${GPU:-1}" --extra-runs "$EXTRA" $OPTS > "$L/eval_decisions.log" 2>&1 || say "offline eval failed"
fi
if has serve; then
  say "node benchmark (caches reset first: the first decision fetches peers' adapters by hash)"
  ./experiments/reset_caches.sh "$TAG" > /dev/null
  ./experiments/run_nodes.sh "$TAG" serve local > /dev/null
  wait_sidecars || say "sidecars not ready"
  say "$(wait_registries)"
  $PY experiments/bench_decisions.py --tag "$TAG" --backend slakshna --cold > "$L/bench_decisions_ours.log" 2>&1 || say "bench ours failed"
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv > "$R/raw/gpu_state_during_ours.txt"
  ./experiments/stop_nodes.sh "$TAG" serve > /dev/null
fi
if has vllm; then
  say "central vLLM baseline"
  ./experiments/run_vllm.sh "$TAG" "${VLLM_GPU:-1}" 8000 > /dev/null
  for i in $(seq 1 120); do curl -sf localhost:8000/v1/models > /dev/null && break; sleep 5; done
  SW="members,permutations,concurrency"; [[ "$TAG" == banking77* ]] && SW="members,options,concurrency"
  $VPY experiments/bench_decisions.py --tag "$TAG" --backend vllm --url http://127.0.0.1:8000 --sweeps "$SW" > "$L/bench_decisions_vllm.log" 2>&1 || say "bench vllm failed"
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv > "$R/raw/gpu_state_during_vllm.txt"
  pkill -f "[v]llm serve"; sleep 5
fi
if has fault; then
  say "fault tolerance (/decide, k = 4)"
  $PY experiments/fault_test.py --tag "$TAG" --endpoint decide --methods k4 --n 30 --out "$R" > "$L/fault_decide.log" 2>&1 || say "fault test failed"
fi
if has tables; then
  say "tables and figures"
  $PY experiments/aggregate_decisions.py --tag "$TAG" > "$L/aggregate_decisions.log" 2>&1 || say "aggregate failed"
  $PY experiments/plot_decisions.py --tag "$TAG" > "$L/plot_decisions.log" 2>&1 || say "plot failed"
fi
say "done"
