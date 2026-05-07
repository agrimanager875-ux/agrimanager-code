#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

RUN_NAME="wofost_reward_formulation_shift_maize_llm_yield_only_think_train"
DATASET_CONFIG="$SCRIPT_DIR/config/t23_reward_formulation_shift.yaml"
DATASET_ID="$(basename "$DATASET_CONFIG" .yaml)"
DATASET_DIR="$SCRIPT_DIR/data/${DATASET_ID}"
TRAIN_YIELD_FILE="$DATASET_DIR/train_yield_max.parquet"
VAL_YIELD_FILE="$DATASET_DIR/val_yield_max.parquet"
VAL_PROFIT_FILE="$DATASET_DIR/val_profit_max.parquet"
VAL_WATER_FILE="$DATASET_DIR/val_water_stewardship.parquet"
VAL_NUTRIENT_FILE="$DATASET_DIR/val_nutrient_stewardship.parquet"
VAL_SET_OVERRIDES=(
    "+data.val_sets.yield_max=${VAL_YIELD_FILE}"
    "+data.val_sets.profit_max=${VAL_PROFIT_FILE}"
    "+data.val_sets.water_stewardship=${VAL_WATER_FILE}"
    "+data.val_sets.nutrient_stewardship=${VAL_NUTRIENT_FILE}"
)
DATASET_BUILD_LOG="$SCRIPT_DIR/logs/${RUN_NAME}_build_dataset.log"
LOG_FILE="$SCRIPT_DIR/logs/${RUN_NAME}.log"
RESULT_DIR="$SCRIPT_DIR/results/llm_train/${RUN_NAME}"
MODEL_PATH="Qwen/Qwen3-4B-Instruct-2507"
Y_REF_MAP="$SCRIPT_DIR/analysis/t23_y_ref/calibrated_y_ref.json"
NUM_GPUS="${NUM_GPUS:-2}"

usage() {
    cat <<EOF
Usage: bash $0 [--num-gpus N]

Options:
  --num-gpus N    Set trainer.n_gpus_per_node for this run. Defaults to 2.

Environment overrides:
  NUM_GPUS=N
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --num-gpus=*)
            NUM_GPUS="${1#*=}"
            shift
            ;;
        --num-gpus)
            if [[ $# -lt 2 ]]; then
                echo "Error: --num-gpus requires a value" >&2
                usage >&2
                exit 1
            fi
            NUM_GPUS="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Error: unknown argument '$1'" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ ! "$NUM_GPUS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: GPU count must be a positive integer, got '$NUM_GPUS'" >&2
    exit 1
fi

mkdir -p "$SCRIPT_DIR/logs" "$RESULT_DIR"
cd "$PROJECT_ROOT"

DATASET_FILES=(
    "$TRAIN_YIELD_FILE"
    "$VAL_YIELD_FILE"
    "$VAL_PROFIT_FILE"
    "$VAL_WATER_FILE"
    "$VAL_NUTRIENT_FILE"
)

NEEDS_BUILD=1
for dataset_file in "${DATASET_FILES[@]}"; do
    if [[ 1 -eq 1 || ! -f "$dataset_file" || "$DATASET_CONFIG" -nt "$dataset_file" ]]; then
        NEEDS_BUILD=1
        break
    fi
    if [[ -f "$Y_REF_MAP" && "$Y_REF_MAP" -nt "$dataset_file" ]]; then
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

echo "Using trainer.n_gpus_per_node=${NUM_GPUS}"

exec bash entrypoints/train/train.sh \
    --log-file "$LOG_FILE" \
    --config-name agri_grpo \
    "data.train_files=[${TRAIN_YIELD_FILE},${TRAIN_YIELD_FILE},${TRAIN_YIELD_FILE}]" \
    "data.val_files=null" \
    "data.validation_axis=reward_formulation" \
    "${VAL_SET_OVERRIDES[@]}" \
    "+data.env_config_overrides.require_think=True" \
    "+data.env_config_overrides.thinking_mode=think" \
    "+data.env_config_overrides.think_tag=tool_call" \
    "data.gen_batch_size=16" \
    "data.max_response_length=512" \
    "actor_rollout_ref.model.path=${MODEL_PATH}" \
    "actor_rollout_ref.rollout.n=4" \
    "actor_rollout_ref.rollout.gpu_memory_utilization=0.7" \
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32" \
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=64" \
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=64" \
    "trainer.logger=[\"console\", \"wandb\"]" \
    "trainer.project_name=agrimanager_wofost_reward_formulation_shift" \
    "trainer.default_local_dir=${RESULT_DIR}" \
    "trainer.experiment_name=${RUN_NAME}" \
    "trainer.total_epochs=4" \
    "trainer.n_gpus_per_node=${NUM_GPUS}" \
    "trainer.save_freq=100" \
    "trainer.rollout_filter.enable=False" \
    "ray_kwargs.ray_init.num_cpus=32" \
    "+ray_kwargs.ray_init.include_dashboard=False" \
    "+ray_kwargs.ray_init._temp_dir=/tmp/ray_${SLURM_JOB_ID:-${SLURM_JOBID:-$$}}"
