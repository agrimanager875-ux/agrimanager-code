#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

RUN_NAME="wofost_weather_regime_ood_potato_llm_think_train"
DATASET_CONFIG="$SCRIPT_DIR/config/weather_regime_potato_llm_without_traits_think.yaml"
DATASET_ID="$(basename "$DATASET_CONFIG" .yaml)"
DATASET_DIR="$SCRIPT_DIR/data/${DATASET_ID}"
TRAIN_FILE="$DATASET_DIR/train.parquet"
VAL_ID_FILE="$DATASET_DIR/val_id.parquet"
VAL_DROUGHT_FILE="$DATASET_DIR/val_drought.parquet"
VAL_WET_FILE="$DATASET_DIR/val_wet.parquet"
VAL_HOT_FILE="$DATASET_DIR/val_hot.parquet"
VAL_COLD_FILE="$DATASET_DIR/val_cold.parquet"
VAL_SET_OVERRIDES=(
    "+data.val_sets.id=${VAL_ID_FILE}"
    "+data.val_sets.drought=${VAL_DROUGHT_FILE}"
    "+data.val_sets.wet=${VAL_WET_FILE}"
    "+data.val_sets.hot=${VAL_HOT_FILE}"
    "+data.val_sets.cold=${VAL_COLD_FILE}"
)
DATASET_BUILD_LOG="$SCRIPT_DIR/logs/${RUN_NAME}_build_dataset.log"
LOG_FILE="$SCRIPT_DIR/logs/${RUN_NAME}.log"
MODEL_PATH="Qwen/Qwen3-4B-Instruct-2507"

mkdir -p "$SCRIPT_DIR/logs"

cd "$PROJECT_ROOT"

DATASET_FILES=(
    "$TRAIN_FILE"
    "$VAL_ID_FILE"
    "$VAL_DROUGHT_FILE"
    "$VAL_WET_FILE"
    "$VAL_HOT_FILE"
    "$VAL_COLD_FILE"
)

NEEDS_BUILD=1
for dataset_file in "${DATASET_FILES[@]}"; do
    if [[ 1 -eq 1 || ! -f "$dataset_file" || "$DATASET_CONFIG" -nt "$dataset_file" ]]; then
        NEEDS_BUILD=1
        break
    fi
done

if [[ "$NEEDS_BUILD" -eq 1 ]]; then
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
    "data.validation_axis=weather_regime" \
    "${VAL_SET_OVERRIDES[@]}" \
    "data.gen_batch_size=16" \
    "data.max_response_length=512" \
    "actor_rollout_ref.model.path=${MODEL_PATH}" \
    "actor_rollout_ref.rollout.n=4" \
    "actor_rollout_ref.rollout.gpu_memory_utilization=0.7" \
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32" \
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=64" \
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=64" \
    "trainer.logger=[\"console\", \"wandb\"]" \
    "trainer.project_name=agrimanager_t1_1_weather_regime_shift" \
    "trainer.experiment_name=${RUN_NAME}" \
    "trainer.total_epochs=2" \
    "trainer.n_gpus_per_node=2" \
    "trainer.save_freq=100" \
    "ray_kwargs.ray_init.num_cpus=32" \
    "+ray_kwargs.ray_init.include_dashboard=False" \
    "+ray_kwargs.ray_init._temp_dir=/tmp/ray_${SLURM_JOB_ID:-${SLURM_JOBID:-$$}}"
