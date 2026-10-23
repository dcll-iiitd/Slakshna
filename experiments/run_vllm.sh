#!/usr/bin/env bash
# Centralised multi-LoRA baseline: one vLLM server holding every node's published adapter of a run.
# usage: experiments/run_vllm.sh <tag> <gpu> [port]      (stop: pkill -f "vllm serve")
set -e
TAG="$1"; GPU="${2:-2}"; PORT="${3:-8000}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
MODS=$(experiments/.venv-vllm/bin/python - "$TAG" <<'PY'
import glob, json, os, sys
tag = sys.argv[1]
mods = {}
for d in glob.glob(f"experiments/runs/{tag}/data-node*/adapters/*/"):
    m = os.path.join(d, "adapter_meta.json")
    if not os.path.isfile(m):
        continue
    j = json.load(open(m))
    node = d.split("data-")[1].split("/")[0]
    if j.get("kind") == "local" and j["round"] >= mods.get(node, (0, ""))[0]:
        mods[node] = (j["round"], os.path.abspath(d))
print(" ".join(f"{n}={p}" for n, (_, p) in sorted(mods.items())))
PY
)
echo "LoRA modules: $MODS"
mkdir -p "experiments/runs/$TAG/logs"
export HF_HOME="$REPO/hf_cache" CUDA_VISIBLE_DEVICES="$GPU" TMPDIR=/mnt/disk1/slakshna/r
export VLLM_USE_FLASHINFER_SAMPLER=0   # its sampler is JIT-compiled with nvcc, which this machine lacks
nohup experiments/.venv-vllm/bin/vllm serve Qwen/Qwen2.5-1.5B-Instruct --port "$PORT" --enable-lora \
  --lora-modules $MODS --max-lora-rank 16 --max-loras 4 --max-model-len 8192 \
  --gpu-memory-utilization 0.35 --dtype bfloat16 --no-enable-prefix-caching > "experiments/runs/$TAG/logs/vllm.log" 2>&1 &
echo "vLLM pid $! on GPU $GPU port $PORT (log experiments/runs/$TAG/logs/vllm.log)"
