#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

RUN_NAME="wofost_gym_smoke_llm_grpo_train"
DATASET_CONFIG="$SCRIPT_DIR/config/llm.yaml"
DATASET_DIR="$SCRIPT_DIR/data/wofost_gym_smoke_llm"
TRAIN_FILE="$DATASET_DIR/train.parquet"
VAL_FILE="$DATASET_DIR/val.parquet"
DATASET_BUILD_LOG="$SCRIPT_DIR/logs/wofost_gym_smoke_llm_dataset_build.log"
LOG_FILE="$SCRIPT_DIR/logs/${RUN_NAME}.log"
RESULT_DIR="$SCRIPT_DIR/results/llm_train/${RUN_NAME}"
MODEL_PATH="Qwen/Qwen2.5-0.5B-Instruct"

mkdir -p "$SCRIPT_DIR/logs" "$RESULT_DIR"

cd "$PROJECT_ROOT"

if [[ 1 -eq 1 || ! -f "$TRAIN_FILE" || ! -f "$VAL_FILE" ]]; then
    bash entrypoints/dataset/build.sh \
        --config "$DATASET_CONFIG" \
        --num-workers 1 \
        2>&1 | tee "$DATASET_BUILD_LOG"
fi

# Smoke runs should start from a clean local Ray state to avoid transient
# raylet/GCS startup races from an earlier interrupted attempt.
ray stop --force >/dev/null 2>&1 || true

exec bash entrypoints/train/train.sh \
    --log-file "$LOG_FILE" \
    --config-name agri_grpo \
    "data.train_files=${TRAIN_FILE}" \
    "data.val_files=${VAL_FILE}" \
    "data.gen_batch_size=1" \
    "data.dataloader_num_workers=0" \
    "data.filter_overlong_prompts_workers=1" \
    "actor_rollout_ref.model.path=${MODEL_PATH}" \
    "critic.enable=False" \
    "algorithm.adv_estimator=grpo" \
    "actor_rollout_ref.rollout.n=2" \
    "actor_rollout_ref.rollout.tensor_model_parallel_size=1" \
    "actor_rollout_ref.rollout.gpu_memory_utilization=0.45" \
    "actor_rollout_ref.rollout.max_num_batched_tokens=2048" \
    "actor_rollout_ref.rollout.max_num_seqs=8" \
    "actor_rollout_ref.rollout.response_length=128" \
    "actor_rollout_ref.rollout.agent.num_workers=1" \
    "actor_rollout_ref.actor.ppo_mini_batch_size=1" \
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1" \
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1" \
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1" \
    "trainer.n_gpus_per_node=1" \
    "trainer.rollout_filter.enable=False" \
    "trainer.total_epochs=1" \
    "trainer.total_training_steps=5" \
    "trainer.val_before_train=True" \
    "trainer.test_freq=10" \
    "trainer.save_freq=-1" \
    'trainer.logger=["console","wandb"]' \
    "trainer.default_local_dir=${RESULT_DIR}" \
    "trainer.experiment_name=${RUN_NAME}"
