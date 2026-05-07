#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

RUN_NAME="cross_sim_wofost_llm_think_train"
WOFOST_CONFIG="$SCRIPT_DIR/config/t31_cross_sim_wofost.yaml"
DSSAT_CONFIG="$SCRIPT_DIR/config/t31_cross_sim_gym_dssat.yaml"
CYCLES_CONFIG="$SCRIPT_DIR/config/t31_cross_sim_cycles_gym.yaml"
DATASET_CONFIGS=("$WOFOST_CONFIG" "$DSSAT_CONFIG" "$CYCLES_CONFIG")
WOFOST_DATASET_DIR="$SCRIPT_DIR/data/$(basename "$WOFOST_CONFIG" .yaml)"
DSSAT_DATASET_DIR="$SCRIPT_DIR/data/$(basename "$DSSAT_CONFIG" .yaml)"
CYCLES_DATASET_DIR="$SCRIPT_DIR/data/$(basename "$CYCLES_CONFIG" .yaml)"
WOFOST_TRAIN_FILE="$WOFOST_DATASET_DIR/train.parquet"
WOFOST_VAL_FILE="$WOFOST_DATASET_DIR/val.parquet"
DSSAT_TRAIN_FILE="$DSSAT_DATASET_DIR/train.parquet"
DSSAT_VAL_FILE="$DSSAT_DATASET_DIR/val.parquet"
CYCLES_TRAIN_FILE="$CYCLES_DATASET_DIR/train.parquet"
CYCLES_VAL_FILE="$CYCLES_DATASET_DIR/val.parquet"
TRAIN_FILE="$WOFOST_TRAIN_FILE"
VAL_SET_OVERRIDES=(
    "+data.val_sets.wofost=${WOFOST_VAL_FILE}"
    "+data.val_sets.dssat=${DSSAT_VAL_FILE}"
    "+data.val_sets.cycles=${CYCLES_VAL_FILE}"
)
LOG_DIR="$SCRIPT_DIR/logs"
LOG_FILE="$LOG_DIR/${RUN_NAME}.log"
RESULT_DIR="$SCRIPT_DIR/results/llm_train/${RUN_NAME}"
MODEL_PATH="Qwen/Qwen3-4B-Instruct-2507"
REQUIRE_THINK=True

prepare_dssat_runtime() {
    if [[ -f "$PROJECT_ROOT/smoke_tests/gym_dssat/_activate_spack.sh" ]]; then
        # shellcheck disable=SC1091
        source "$PROJECT_ROOT/smoke_tests/gym_dssat/_activate_spack.sh"
    fi

    local candidate
    for candidate in \
        "$PROJECT_ROOT/AgriManagerExternal/gym_dssat_pdi/gym-dssat-pdi" \
        "$PROJECT_ROOT/../AgriManagerExternal/gym_dssat_pdi/gym-dssat-pdi"
    do
        if [[ -d "$candidate/gym_dssat_pdi" ]]; then
            case ":${PYTHONPATH:-}:" in
                *":$candidate:"*) ;;
                *) export PYTHONPATH="$candidate:${PYTHONPATH:-}" ;;
            esac
            break
        fi
    done
}

dataset_dir_for_config() {
    printf '%s/data/%s\n' "$SCRIPT_DIR" "$(basename "$1" .yaml)"
}

build_dataset_if_needed() {
    local config="$1"
    local dataset_dir
    local dataset_id
    local split_file
    local needs_build=1
    dataset_dir="$(dataset_dir_for_config "$config")"
    dataset_id="$(basename "$config" .yaml)"

    for split_file in "$dataset_dir/train.parquet" "$dataset_dir/val.parquet"; do
        if [[ 1 -eq 1 || ! -f "$split_file" || "$config" -nt "$split_file" ]]; then
            needs_build=1
            break
        fi
    done

    if [[ "$needs_build" -eq 1 ]]; then
        bash entrypoints/dataset/build.sh \
            --config "$config" \
            --num-workers 1 \
            2>&1 | tee "$LOG_DIR/${RUN_NAME}_${dataset_id}_build_dataset.log"
    fi
}

mkdir -p "$LOG_DIR" "$RESULT_DIR"
cd "$PROJECT_ROOT"
prepare_dssat_runtime

for config in "${DATASET_CONFIGS[@]}"; do
    build_dataset_if_needed "$config"
done

exec bash entrypoints/train/train.sh \
    --log-file "$LOG_FILE" \
    --config-name agri_grpo \
    "data.train_files=${TRAIN_FILE}" \
    "data.val_files=null" \
    "data.validation_axis=simulator" \
    "${VAL_SET_OVERRIDES[@]}" \
    "+data.env_config_overrides.require_think=${REQUIRE_THINK}" \
    "+data.env_config_overrides.thinking_mode=think" \
    "+data.env_config_overrides.think_tag=tool_call" \
    "data.max_prompt_length=2048" \
    "data.gen_batch_size=16" \
    "data.max_response_length=512" \
    "actor_rollout_ref.model.path=${MODEL_PATH}" \
    "actor_rollout_ref.rollout.n=4" \
    "actor_rollout_ref.rollout.gpu_memory_utilization=0.7" \
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32" \
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=64" \
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=64" \
    "trainer.logger=[\"console\", \"wandb\"]" \
    "trainer.project_name=agrimanager_cross_simulator_maize_transfer" \
    "trainer.default_local_dir=${RESULT_DIR}" \
    "trainer.experiment_name=${RUN_NAME}" \
    "trainer.total_epochs=2" \
    "trainer.n_gpus_per_node=2" \
    "trainer.save_freq=100" \
    "ray_kwargs.ray_init.num_cpus=32" \
    "+ray_kwargs.ray_init.include_dashboard=False" \
    "+ray_kwargs.ray_init._temp_dir=/tmp/ray_${SLURM_JOB_ID:-${SLURM_JOBID:-$$}}"
