#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

RUN_NAME="wofost_weather_regime_ood_chickpea_llm_no_think_train"
DATASET_CONFIG="$SCRIPT_DIR/config/weather_regime_chickpea_llm_without_traits_no_think.yaml"
DATASET_ID="$(basename "$DATASET_CONFIG" .yaml)"
DATASET_DIR="$SCRIPT_DIR/data/${DATASET_ID}"
MANIFEST_FILE="$DATASET_DIR/manifest.json"
TRAIN_FILE="$DATASET_DIR/train.parquet"
VALIDATION_SETS=(id drought wet hot cold)
DATASET_BUILD_LOG="$SCRIPT_DIR/logs/${RUN_NAME}_build_dataset.log"
LOG_FILE="$SCRIPT_DIR/logs/${RUN_NAME}.log"
RESULT_ROOT="${T11_RESULT_ROOT:-$SCRIPT_DIR/results/llm_train}"
RESULT_DIR="${T11_RESULT_DIR:-$RESULT_ROOT/${RUN_NAME}}"
MODEL_PATH="Qwen/Qwen3-4B-Instruct-2507"

mkdir -p "$SCRIPT_DIR/logs" "$RESULT_DIR"

cd "$PROJECT_ROOT"

VAL_SET_OVERRIDES=()
for validation_set in "${VALIDATION_SETS[@]}"; do
    VAL_SET_OVERRIDES+=(
        "+data.val_sets.${validation_set}=${DATASET_DIR}/val_${validation_set}.parquet"
    )
done

NEEDS_BUILD=1
if [[ 1 -eq 1 || ! -f "$MANIFEST_FILE" || ! -f "$TRAIN_FILE" || "$DATASET_CONFIG" -nt "$MANIFEST_FILE" ]]; then
    NEEDS_BUILD=1
fi

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
    "trainer.default_local_dir=${RESULT_DIR}" \
    "trainer.experiment_name=${RUN_NAME}" \
    "trainer.total_epochs=2" \
    "trainer.n_gpus_per_node=2" \
    "trainer.save_freq=100" \
    "ray_kwargs.ray_init.num_cpus=32" \
    "+ray_kwargs.ray_init.include_dashboard=False" \
    "+ray_kwargs.ray_init._temp_dir=/tmp/ray_${SLURM_JOB_ID:-${SLURM_JOBID:-$$}}"
