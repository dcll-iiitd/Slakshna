#!/usr/bin/env bash
# Set up Flower's FedRAG example for the M9 comparison.
#
#   bash scripts/m9/flower/setup_flower.sh --data datasets/textbooks [--nodes 5] [--model BAAI/bge-base-en-v1.5 --dim 768]
#
# 1. Fetches examples/fedrag from github.com/flwrlabs/flower at a pinned commit into .flower-src/.
# 2. Makes two patched copies (scripts/m9/flower/patch_fedrag.py):
#      flower-fedrag-m9     saves contexts (used for every M9 run)
#      flower-fedrag-stock  Flower's own ServerApp with its LLM (check E0(a) only)
# 3. Creates .venv-flower (Python 3.11) with a pinned flwr and the example's dependencies.
#    The example's package itself is not installed: Flower ships it to the
#    SuperNodes as an app bundle, and the M9 scripts import it from its folder.
# Run from anywhere; paths are relative to the repository root. Re-running starts the copies afresh.
set -euo pipefail
cd "$(dirname "$0")/../../.."

FLOWER_COMMIT=0e4989b0f19b1c7aa5010aef1d9ee04b8b4e4e5a   # last change to examples/fedrag, 2026-06-11
FLWR_VERSION=1.42.0
DATA=""; NODES=5; MODEL="BAAI/bge-base-en-v1.5"; DIM=768
while [ $# -gt 0 ]; do
  case "$1" in
    --data) DATA="$2"; shift 2 ;;
    --nodes) NODES="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --dim) DIM="$2"; shift 2 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
[ -n "$DATA" ] || { echo "--data <dataset folder> is required (for raw/benchmark.json)" >&2; exit 2; }

if [ ! -d .flower-src ]; then
  git clone -q --filter=blob:none --no-checkout https://github.com/flwrlabs/flower.git .flower-src
  git -C .flower-src sparse-checkout set examples/fedrag
fi
git -C .flower-src fetch -q origin "$FLOWER_COMMIT" 2>/dev/null || true
git -C .flower-src checkout -q "$FLOWER_COMMIT"

for variant in m9 stock; do
  rm -rf "flower-fedrag-$variant"
  cp -r .flower-src/examples/fedrag "flower-fedrag-$variant"
  python3 scripts/m9/flower/patch_fedrag.py --app "flower-fedrag-$variant" --variant "$variant" \
    --nodes "$NODES" --model "$MODEL" --dim "$DIM" --mirage "$DATA/raw/benchmark.json"
done

[ -x .venv-flower/bin/python ] || uv venv .venv-flower --python 3.11
uv pip install --python .venv-flower/bin/python --torch-backend cpu "flwr==$FLWR_VERSION" \
  "numpy>=1.26,<2" faiss-cpu tqdm sentence-transformers transformers scikit-learn pyyaml requests

{ echo "flower_commit=$FLOWER_COMMIT"; echo "flwr=$(.venv-flower/bin/flwr --version)"; echo "model=$MODEL dim=$DIM nodes=$NODES"; } \
  > flower-fedrag-m9/M9_SETUP.txt
echo "Flower FedRAG ready: flower-fedrag-m9, flower-fedrag-stock, .venv-flower" >&2
