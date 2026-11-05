#!/bin/bash
# usage: serve.sh <gpu> <port> <model> <mem_frac> [max_len]
export HF_HOME=/home/anant/dcll/Slakshna/hf_cache VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_OFFLINE=1
CUDA_VISIBLE_DEVICES=$1 exec ~/vllm-env/bin/vllm serve "$3" --port $2 --gpu-memory-utilization $4 --max-model-len ${5:-2048} > ~/slk-exp/vllm_$2.log 2>&1
