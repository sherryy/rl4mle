#!/usr/bin/env bash
set -euo pipefail

export N_GPUS=${N_GPUS:-8}
export BASE_MODEL=${BASE_MODEL:-Qwen/Qwen2.5-3B-Instruct}
export DATA_DIR=${DATA_DIR:-$HOME/data/mlebench}
export ROLLOUT_TP_SIZE=${ROLLOUT_TP_SIZE:-8}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-mlebench-qwen2.5-3b-instruct}
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}
export TIMEOUT=${TIMEOUT:-300}

bash ./scripts/train_tiny_zero_mlebench.sh "$@"
