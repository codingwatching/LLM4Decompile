#!/usr/bin/env bash
# =============================================================================
# SK2Decompile - Reference Script: Identifier Naming RL Training
# =============================================================================
# Reference GRPO training script for the Identifier Naming model.
# Based on the VERL framework (v0.4.1) with embedding-based rewards.
#
# This is a reference configuration — please adjust parameters according
# to your hardware setup and dataset. See the paper (arXiv:2509.22114,
# Section 3.5) for the reward formulation details.
#
# Prerequisites:
#   - VERL framework installed (https://github.com/volcengine/verl)
#   - Reward functions integrated into verl/utils/reward_score/ (see README.md)
#   - An OpenAI-compatible embedding server running locally
#     e.g.: python -m vllm.entrypoints.openai.api_server \
#               --model Qwen3-Embedding-0.6B --port 8000
#   - tree-sitter, tree-sitter-c, openai packages installed
#
# Usage:
#   bash run_ident_rl.sh
# =============================================================================
set -x

# ---- User Configuration ----
EMBEDDING_VARIANT="gte"  # Options: "gte" or "qwen3"

VERL_DIR="<YOUR_VERL_DIR>"
VENV_PATH="<YOUR_VENV_PATH>"
MODEL_PATH="<YOUR_MODEL_PATH>"           # e.g., path to sk2decompile-ident-6.7b
TRAIN_DATA="<YOUR_DATA_PATH>/train.parquet"
VAL_DATA="<YOUR_DATA_PATH>/valid.parquet"

# WandB configuration
WANDB_API_KEY_VAL="<YOUR_WANDB_API_KEY>"
WANDB_ENTITY_VAL="<YOUR_WANDB_ENTITY>"
WANDB_PROJECT_VAL="<YOUR_WANDB_PROJECT>"

# Training parameters
NUM_NODES=1
GPUS_PER_NODE=8
KL_COEF=0.02
TOTAL_EPOCHS=2
SAVE_FREQ=25
TEST_FREQ=25

# ---- Environment Setup ----
source ${VENV_PATH}/bin/activate

export UCX_IB_PCI_RELAXED_ORDERING=1
export NCCL_IB_PCI_RELAXED_ORDERING=1
export NCCL_IB_TIMEOUT=22
export NCCL_DEBUG=INFO
export TRANSFORMERS_OFFLINE=0
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=0
export NCCL_IB_DISABLE=0
export CUDA_DEVICE_MAX_CONNECTIONS=1

# ---- Task & Logging ----
TASK_NAME="sk2decompile_ident-rl-${EMBEDDING_VARIANT}"
LOG_DIR="${VERL_DIR}/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${TASK_NAME}.log"
ERR_FILE="$LOG_DIR/${TASK_NAME}.err"

# ---- WandB ----
export WANDB_API_KEY=${WANDB_API_KEY_VAL}
export WANDB_ENTITY=${WANDB_ENTITY_VAL}
export WANDB_PROJECT=${WANDB_PROJECT_VAL}
export WANDB_NAME=${TASK_NAME}
export WANDB_MODE='online'
wandb login --relogin $WANDB_API_KEY

# ---- Launch GRPO Training ----
python3 -m verl.trainer.main_ppo --config-path=config \
    --config-name='ppo_trainer-lm4dc.yaml' \
    algorithm.adv_estimator=grpo \
    data.train_files=${TRAIN_DATA} \
    data.val_files=${VAL_DATA} \
    data.train_batch_size=128 \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=${KL_COEF} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.80 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='sk2decompile_rl' \
    trainer.experiment_name=$TASK_NAME \
    trainer.default_local_dir=${VERL_DIR}/checkpoints/${TASK_NAME} \
    trainer.n_gpus_per_node=${GPUS_PER_NODE} \
    trainer.nnodes=${NUM_NODES} \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.total_epochs=${TOTAL_EPOCHS} "$@" \
    > >(tee -a "$LOG_FILE") \
    2> >(tee -a "$ERR_FILE" >&2)

echo "STDOUT saved to: $LOG_FILE"
echo "STDERR saved to: $ERR_FILE"
