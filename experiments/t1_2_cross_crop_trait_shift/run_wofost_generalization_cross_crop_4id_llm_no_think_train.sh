#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

RUN_NAME="wofost_generalization_cross_crop_4id_llm_no_think_train"
DATASET_CONFIG="$SCRIPT_DIR/config/cross_crop_4id_llm_traits_v1_23d_no_think.yaml"
DATASET_ID="$(basename "$DATASET_CONFIG" .yaml)"
DATASET_DIR="$SCRIPT_DIR/data/${DATASET_ID}"
TRAIN_FILE="$DATASET_DIR/train.parquet"
VAL_ID_CROPS_FILE="$DATASET_DIR/val_id_crops.parquet"
VAL_HELDOUT_CROPS_FILE="$DATASET_DIR/val_heldout_crops.parquet"
VAL_SET_OVERRIDES=(
    "+data.val_sets.id_crops=${VAL_ID_CROPS_FILE}"
    "+data.val_sets.heldout_crops=${VAL_HELDOUT_CROPS_FILE}"
)
DATASET_BUILD_LOG="$SCRIPT_DIR/logs/${RUN_NAME}_build_dataset.log"
LOG_FILE="$SCRIPT_DIR/logs/${RUN_NAME}.log"
RESULT_DIR="$SCRIPT_DIR/results/llm_train/${RUN_NAME}"
MODEL_PATH="Qwen/Qwen3-4B-Instruct-2507"

mkdir -p "$SCRIPT_DIR/logs" "$RESULT_DIR"

cd "$PROJECT_ROOT"

if [[ 1 -eq 1 || ! -f "$TRAIN_FILE" || ! -f "$VAL_ID_CROPS_FILE" || ! -f "$VAL_HELDOUT_CROPS_FILE" || "$DATASET_CONFIG" -nt "$TRAIN_FILE" ]]; then
    bash entrypoints/dataset/build.sh \
        --config "$DATASET_CONFIG" \
        --num-workers 1 \
        2>&1 | tee "$DATASET_BUILD_LOG"
fi

exec bash entrypoints/train/train.sh \
    --log-file "$LOG_FILE" \
    --config-name agri_grpo \
    "data.train_files=${TRAIN_FILE}" \
    "data.val_files=null" \
    "data.validation_axis=crop" \
    "${VAL_SET_OVERRIDES[@]}" \
    "data.gen_batch_size=16" \
    "data.max_prompt_length=2048" \
    "data.max_response_length=1024" \
    "actor_rollout_ref.model.path=${MODEL_PATH}" \
    "actor_rollout_ref.rollout.gpu_memory_utilization=0.7" \
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32" \
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=64" \
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=64" \
    "trainer.logger=[\"console\", \"wandb\"]" \
    "trainer.project_name=agrimanager_wofost_cross_crop_generalization" \
    "trainer.default_local_dir=${RESULT_DIR}" \
    "trainer.experiment_name=${RUN_NAME}" \
    "trainer.total_epochs=2" \
    "trainer.n_gpus_per_node=2" \
    "trainer.save_freq=100" \
    "ray_kwargs.ray_init.num_cpus=32" \
    "+ray_kwargs.ray_init.include_dashboard=False" \
    "+ray_kwargs.ray_init._temp_dir=/tmp/ray_${SLURM_JOB_ID:-${SLURM_JOBID:-$$}}"
