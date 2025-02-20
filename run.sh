export N_GPUS=8
export BASE_MODEL=Qwen2.5-3B-Instruct
export DATA_DIR=all
export ROLLOUT_TP_SIZE=8
export EXPERIMENT_NAME=$DATA_DIR-qwen2.5-3b-instruct
export VLLM_ATTENTION_BACKEND=XFORMERS
export TIMEOUT=300

bash ./scripts/train_tiny_zero_mlebench.sh
