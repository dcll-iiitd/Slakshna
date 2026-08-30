# Slakshna Checkpoint Evaluation with Tapestry Monash GOQA

This folder provides tools and end-to-end workflows to convert Slakshna federated learning checkpoints (`sync_round_*.pth` or `*_base_lora.pth` under `Slakshna/ml_models`) into standard Hugging Face PEFT LoRA adapter directories and evaluate them against the **Tapestry Monash GlobalOpinionQA (GOQA)** benchmark.

Upstream Evaluation Suite: [tapestry_monash/shared_evaluation/GOQA](https://github.com/yuriak/tapestry_monash/tree/main/shared_evaluation/GOQA)

---

## Table of Contents

- [Overview](#overview)
- [Folder Structure & Key Paths](#folder-structure--key-paths)
- [Quick Start: Convert & Evaluate Latest Checkpoint](#quick-start-convert--evaluate-latest-checkpoint)
- [Detailed Step-by-Step Instructions](#detailed-step-by-step-instructions)
  - [Step 1: Verify the Evaluation Package](#step-1-verify-the-evaluation-package)
  - [Step 2: Convert Checkpoint to LoRA Adapter](#step-2-convert-checkpoint-to-lora-adapter)
  - [Step 3: Run Tapestry Monash GOQA Evaluation](#step-3-run-tapestry-monash-goqa-evaluation)
  - [Step 4: Inspect Prediction Scores & Reports](#step-4-inspect-prediction-scores--reports)
- [All-in-One Command](#all-in-one-command)
- [Batch Evaluation Across All Rounds](#batch-evaluation-across-all-rounds)
- [Environment Variables & GPU Performance Tuning](#environment-variables--gpu-performance-tuning)

---

## Overview

During Slakshna federated training, each federated aggregation step produces a PyTorch `.pth` checkpoint in `dcll/Slakshna/ml_models/sync_ckpt_<node_id>/sync_round_<round>.pth`.

To evaluate these weights in vLLM with the Tapestry Monash GOQA suite, the checkpoint must be converted into a Hugging Face PEFT LoRA adapter directory containing:
1. `adapter_model.safetensors`: Normalized LoRA weight tensors (`lora_A.weight`, `lora_B.weight`).
2. `adapter_config.json`: Standard PEFT LoRA configuration schema (`base_model_name_or_path`, `r=16`, `lora_alpha=64`, `target_modules=["q_proj", "v_proj"]`, etc.).
3. `adapter_meta.json`: Audit manifest recording parameter counts, tensor shapes, source paths, and SHA-256 digests.

---

## Folder Structure & Key Paths

| Resource | Path |
|---|---|
| **Preparation Script (Python)** | `dcll/Slakshna/evaluations/prepare_sync_round_adapter.py` |
| **Preparation Script (Bash)** | `dcll/Slakshna/evaluations/prepare_sync_round_adapter.sh` |
| **Checkpoints Source** | `dcll/Slakshna/ml_models` |
| **GOQA Evaluation Runner** | `/mnt/disk1/slakshna/tapestry_monash/shared_evaluation/GOQA/run_evaluation.sh` |
| **GOQA Python Environment** | `/mnt/disk1/slakshna/tapestry_monash/shared_evaluation/GOQA/goqa_env/bin/python` |
| **Default Base Model** | `allenai/OLMo-2-1124-7B` |

---

## Quick Start: Convert & Evaluate Latest Checkpoint

From the `dcll/Slakshna/evaluations` directory:

```bash
cd /mnt/disk1/slakshna/dcll/Slakshna/evaluations

# 1. Convert the latest round checkpoint to an adapter directory
python prepare_sync_round_adapter.py \
    --input ../ml_models \
    --round latest \
    --output-dir ./adapters/latest \
    --base-model allenai/OLMo-2-1124-7B

# 2. Run the Tapestry Monash evaluation script
export GOQA_PYTHON="/mnt/disk1/slakshna/tapestry_monash/shared_evaluation/GOQA/goqa_env/bin/python"
export CUDA_VISIBLE_DEVICES=0

bash /mnt/disk1/slakshna/tapestry_monash/shared_evaluation/GOQA/run_evaluation.sh \
    allenai/OLMo-2-1124-7B \
    ./eval_results/latest \
    ./adapters/latest
```

---

## Detailed Step-by-Step Instructions

### Step 1: Verify the Evaluation Package

Before starting GPU inference, run a sanity check on the GOQA package dataset and manifest:

```bash
/mnt/disk1/slakshna/tapestry_monash/shared_evaluation/GOQA/goqa_env/bin/python \
    /mnt/disk1/slakshna/tapestry_monash/shared_evaluation/GOQA/validate_package.py
```
*Expected output:* `GOQA PACKAGE VALIDATION PASSED`

---

### Step 2: Convert Checkpoint to LoRA Adapter

Choose one of the following methods:

#### A. Convert the Latest Checkpoint
```bash
python /mnt/disk1/slakshna/dcll/Slakshna/evaluations/prepare_sync_round_adapter.py \
    --input /mnt/disk1/slakshna/dcll/Slakshna/ml_models \
    --round latest \
    --output-dir /mnt/disk1/slakshna/dcll/Slakshna/evaluations/adapters/latest \
    --base-model allenai/OLMo-2-1124-7B
```

#### B. Convert a Specific Round Checkpoint (e.g. Round 15)
```bash
python /mnt/disk1/slakshna/dcll/Slakshna/evaluations/prepare_sync_round_adapter.py \
    --input /mnt/disk1/slakshna/dcll/Slakshna/ml_models/sync_ckpt_slakshna1dp30me5pdds4wf595eat8h0e65mt99jgy39hmt/sync_round_15.pth \
    --output-dir /mnt/disk1/slakshna/dcll/Slakshna/evaluations/adapters/round_15 \
    --base-model allenai/OLMo-2-1124-7B
```

---

### Step 3: Run Tapestry Monash GOQA Evaluation

Pass the base model, evaluation output directory, and adapter path to `run_evaluation.sh`:

```bash
export GOQA_PYTHON="/mnt/disk1/slakshna/tapestry_monash/shared_evaluation/GOQA/goqa_env/bin/python"
export CUDA_VISIBLE_DEVICES=0

bash /mnt/disk1/slakshna/tapestry_monash/shared_evaluation/GOQA/run_evaluation.sh \
    allenai/OLMo-2-1124-7B \
    /mnt/disk1/slakshna/dcll/Slakshna/evaluations/eval_results/round_15 \
    /mnt/disk1/slakshna/dcll/Slakshna/evaluations/adapters/round_15
```

---

### Step 4: Inspect Prediction Scores & Reports

After inference and scoring finish, check the generated results:

```bash
RESULTS_DIR="/mnt/disk1/slakshna/dcll/Slakshna/evaluations/eval_results/round_15/scores"

# 1. Summary table (Regional and Two-region macro metrics)
cat "${RESULTS_DIR}/summary.csv"

# 2. Detailed markdown report
cat "${RESULTS_DIR}/report.md"

# 3. Disaggregated survey group summary (Australia, NZ, India samples)
cat "${RESULTS_DIR}/group_summary.csv"

# 4. Raw predictions JSONL
head -n 5 "/mnt/disk1/slakshna/dcll/Slakshna/evaluations/eval_results/round_15/predictions.jsonl"
```

---

## All-in-One Command

You can convert the checkpoint and automatically invoke the evaluation in a single command using `--evaluate` / `-e`:

```bash
python /mnt/disk1/slakshna/dcll/Slakshna/evaluations/prepare_sync_round_adapter.py \
    --input /mnt/disk1/slakshna/dcll/Slakshna/ml_models \
    --round latest \
    --output-dir /mnt/disk1/slakshna/dcll/Slakshna/evaluations/adapters/latest \
    --evaluate \
    --eval-output-dir /mnt/disk1/slakshna/dcll/Slakshna/evaluations/eval_results/latest
```

---

## Batch Evaluation Across All Rounds

To convert and inspect all 15 rounds of federated training across all nodes:

```bash
# Convert all rounds (1 through 15) for each client node
python /mnt/disk1/slakshna/dcll/Slakshna/evaluations/prepare_sync_round_adapter.py \
    --input /mnt/disk1/slakshna/dcll/Slakshna/ml_models \
    --round all \
    --output-dir /mnt/disk1/slakshna/dcll/Slakshna/evaluations/adapters_all_rounds
```

---

## Environment Variables & GPU Performance Tuning

When executing `run_evaluation.sh`, the following environment variables can be set:

| Variable | Default | Description |
|---|---|---|
| `GOQA_PYTHON` | `python3` | Python binary (point to `goqa_env/bin/python`). |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU index to allocate for vLLM inference. |
| `GOQA_GPU_MEMORY_UTILIZATION` | `0.90` | Fraction of GPU memory allocated to vLLM (e.g. `0.85` or `0.90`). |
| `GOQA_REQUEST_BATCH_SIZE` | `8192` | Maximum batch size for prompt variant requests. |
| `GOQA_MAX_MODEL_LEN` | `4096` | Context length for the model. |
| `GOQA_DTYPE` | `bfloat16` | Model and LoRA inference datatype (`bfloat16`, `float16`). |
| `GOQA_TENSOR_PARALLEL_SIZE` | `1` | Number of GPUs for tensor parallelism. |
| `GOQA_TRUST_REMOTE_CODE` | `0` | Set `1` to trust remote Hugging Face code if required. |

**Example with custom parameters:**
```bash
GOQA_PYTHON="/mnt/disk1/slakshna/tapestry_monash/shared_evaluation/GOQA/goqa_env/bin/python" \
GOQA_GPU_MEMORY_UTILIZATION=0.85 \
GOQA_REQUEST_BATCH_SIZE=4096 \
CUDA_VISIBLE_DEVICES=0 \
bash /mnt/disk1/slakshna/tapestry_monash/shared_evaluation/GOQA/run_evaluation.sh \
    allenai/OLMo-2-1124-7B \
    /mnt/disk1/slakshna/dcll/Slakshna/evaluations/eval_results/custom_run \
    /mnt/disk1/slakshna/dcll/Slakshna/evaluations/adapters/latest
```
