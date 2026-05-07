#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

RUN_NAME="wofost_observation_schema_shift_maize_llm_think_train"
DATASET_CONFIG="$SCRIPT_DIR/config/t21_observation_schema_shift_llm_think.yaml"
DATASET_ID="$(basename "$DATASET_CONFIG" .yaml)"
DATASET_DIR="$SCRIPT_DIR/data/${DATASET_ID}"
TRAIN_FILE="$DATASET_DIR/train.parquet"
VAL_S1_FILE="$DATASET_DIR/val_s1.parquet"
VAL_S2A_FILE="$DATASET_DIR/val_s2a.parquet"
VAL_S2B_FILE="$DATASET_DIR/val_s2b.parquet"
VAL_S2C_FILE="$DATASET_DIR/val_s2c.parquet"
VAL_S2D_FILE="$DATASET_DIR/val_s2d.parquet"
VAL_S3_FILE="$DATASET_DIR/val_s3.parquet"
VAL_S4_FILE="$DATASET_DIR/val_s4.parquet"
VAL_S5_FILE="$DATASET_DIR/val_s5.parquet"
VAL_SET_OVERRIDES=(
    "+data.val_sets.s1_full_current=${VAL_S1_FILE}"
    "+data.val_sets.s2a_no_stage_time=${VAL_S2A_FILE}"
    "+data.val_sets.s2b_no_resource_state=${VAL_S2B_FILE}"
    "+data.val_sets.s2c_no_management_history=${VAL_S2C_FILE}"
    "+data.val_sets.s2d_no_weather_context=${VAL_S2D_FILE}"
    "+data.val_sets.s3_domain_synonym_rename=${VAL_S3_FILE}"
    "+data.val_sets.s4_compact_growth_superset=${VAL_S4_FILE}"
    "+data.val_sets.s5_anonymous_label_rename=${VAL_S5_FILE}"
)
DATASET_BUILD_LOG="$SCRIPT_DIR/logs/${RUN_NAME}_build_dataset.log"
LOG_FILE="$SCRIPT_DIR/logs/${RUN_NAME}.log"
RESULT_DIR="${T21_RESULT_ROOT:-$SCRIPT_DIR/results/llm_train}/${RUN_NAME}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B-Instruct-2507}"

mkdir -p "$SCRIPT_DIR/logs" "$RESULT_DIR"
cd "$PROJECT_ROOT"

DATASET_FILES=(
    "$TRAIN_FILE"
    "$VAL_S1_FILE"
    "$VAL_S2A_FILE"
    "$VAL_S2B_FILE"
    "$VAL_S2C_FILE"
    "$VAL_S2D_FILE"
    "$VAL_S3_FILE"
    "$VAL_S4_FILE"
    "$VAL_S5_FILE"
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
    "data.validation_axis=observation_schema" \
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
    "trainer.project_name=agrimanager_wofost_observation_schema_shift" \
    "trainer.default_local_dir=${RESULT_DIR}" \
    "trainer.experiment_name=${RUN_NAME}" \
    "trainer.total_epochs=2" \
    "trainer.n_gpus_per_node=2" \
    "trainer.save_freq=100" \
    "ray_kwargs.ray_init.num_cpus=32" \
    "+ray_kwargs.ray_init.include_dashboard=False" \
    "+ray_kwargs.ray_init._temp_dir=/tmp/ray_${SLURM_JOB_ID:-${SLURM_JOBID:-$$}}"
