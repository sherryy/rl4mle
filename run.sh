export N_GPUS=4
export BASE_MODEL=Qwen2.5-3B-Instruct
export DATA_DIR=mlebench
export ROLLOUT_TP_SIZE=4
export EXPERIMENT_NAME=countdown-qwen2.5-3b-instruct
export VLLM_ATTENTION_BACKEND=XFORMERS

bash ./scripts/train_tiny_zero_mlebench.sh
