#!/usr/bin/env bash
# ==============================================================================
# prepare_sync_round_adapter.sh
# 
# Prepares a Hugging Face PEFT LoRA adapter directory from a Slakshna
# sync_round.pth checkpoint so it can be evaluated with Tapestry Monash GOQA.
#
# Usage:
#   bash prepare_sync_round_adapter.sh <SYNC_PTH_OR_DIR> [OUTPUT_DIR] [BASE_MODEL] [--evaluate]
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PYTHON_BIN="${GOQA_PYTHON:-python3}"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <SYNC_PTH_OR_DIR> [OUTPUT_DIR] [BASE_MODEL] [--evaluate]"
    echo ""
    echo "Examples:"
    echo "  $0 ../ml_models"
    echo "  $0 ../ml_models/sync_ckpt_slakshna1dp30me5pdds4wf595eat8h0e65mt99jgy39hmt/sync_round_15.pth"
    echo "  $0 ../ml_models/sync_ckpt_slakshna1dp30me5pdds4wf595eat8h0e65mt99jgy39hmt/sync_round_15.pth ./adapters/round_15"
    echo "  $0 ../ml_models/sync_ckpt_slakshna1dp30me5pdds4wf595eat8h0e65mt99jgy39hmt/sync_round_15.pth ./adapters/round_15 allenai/OLMo-2-1124-7B --evaluate"
    exit 1
fi

INPUT_PATH="$1"
OUTPUT_DIR="${2:-}"
BASE_MODEL="${3:-allenai/OLMo-2-1124-7B}"

EVAL_FLAG=""
for arg in "$@"; do
    if [[ "${arg}" == "--evaluate" || "${arg}" == "-e" ]]; then
        EVAL_FLAG="--evaluate"
    fi
done

ARGS=(
    --input "${INPUT_PATH}"
    --base-model "${BASE_MODEL}"
)

if [[ -n "${OUTPUT_DIR}" && "${OUTPUT_DIR}" != "--evaluate" && "${OUTPUT_DIR}" != "-e" ]]; then
    ARGS+=(--output-dir "${OUTPUT_DIR}")
fi

if [[ -n "${EVAL_FLAG}" ]]; then
    ARGS+=(${EVAL_FLAG})
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_sync_round_adapter.py" "${ARGS[@]}"
