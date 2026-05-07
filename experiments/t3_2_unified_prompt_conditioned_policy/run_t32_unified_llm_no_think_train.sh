#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

RUN_NAME="t32_unified_llm_no_think_train"
WOFOST_CONFIG="$SCRIPT_DIR/config/t32_unified_wofost.yaml"
DSSAT_YIELD_CONFIG="$SCRIPT_DIR/config/t32_unified_gym_dssat_yield.yaml"
DSSAT_PROFIT_CONFIG="$SCRIPT_DIR/config/t32_unified_gym_dssat_profit.yaml"
CYCLES_YIELD_CONFIG="$SCRIPT_DIR/config/t32_unified_cycles_yield.yaml"
CYCLES_PROFIT_CONFIG="$SCRIPT_DIR/config/t32_unified_cycles_profit.yaml"
WOFOST_DATASET_DIR="$SCRIPT_DIR/data/$(basename "$WOFOST_CONFIG" .yaml)"
DSSAT_YIELD_DATASET_DIR="$SCRIPT_DIR/data/$(basename "$DSSAT_YIELD_CONFIG" .yaml)"
DSSAT_PROFIT_DATASET_DIR="$SCRIPT_DIR/data/$(basename "$DSSAT_PROFIT_CONFIG" .yaml)"
CYCLES_YIELD_DATASET_DIR="$SCRIPT_DIR/data/$(basename "$CYCLES_YIELD_CONFIG" .yaml)"
CYCLES_PROFIT_DATASET_DIR="$SCRIPT_DIR/data/$(basename "$CYCLES_PROFIT_CONFIG" .yaml)"

WOFOST_TRAIN_LNPKW_YIELD_FILE="$WOFOST_DATASET_DIR/train_wofost_lnpkw_yield.parquet"
WOFOST_TRAIN_LNPKW_PROFIT_FILE="$WOFOST_DATASET_DIR/train_wofost_lnpkw_profit.parquet"
WOFOST_TRAIN_LNPKW_WATER_FILE="$WOFOST_DATASET_DIR/train_wofost_lnpkw_water.parquet"
WOFOST_TRAIN_LNPK_PROFIT_FILE="$WOFOST_DATASET_DIR/train_wofost_lnpk_profit.parquet"
WOFOST_TRAIN_LNPK_NUTRIENT_FILE="$WOFOST_DATASET_DIR/train_wofost_lnpk_nutrient.parquet"
WOFOST_TRAIN_LNW_YIELD_FILE="$WOFOST_DATASET_DIR/train_wofost_lnw_yield.parquet"
DSSAT_YIELD_TRAIN_FILE="$DSSAT_YIELD_DATASET_DIR/train.parquet"
CYCLES_YIELD_TRAIN_FILE="$CYCLES_YIELD_DATASET_DIR/train.parquet"

WOFOST_VAL_LNPKW_PROFIT_FILE="$WOFOST_DATASET_DIR/val_wofost_lnpkw_profit.parquet"
WOFOST_VAL_LNPK_PROFIT_FILE="$WOFOST_DATASET_DIR/val_wofost_lnpk_profit.parquet"
WOFOST_VAL_LNW_YIELD_FILE="$WOFOST_DATASET_DIR/val_wofost_lnw_yield.parquet"
WOFOST_VAL_LNW_PROFIT_FILE="$WOFOST_DATASET_DIR/val_wofost_lnw_profit.parquet"
WOFOST_VAL_LNPKW_NUTRIENT_FILE="$WOFOST_DATASET_DIR/val_wofost_lnpkw_nutrient.parquet"
DSSAT_YIELD_VAL_FILE="$DSSAT_YIELD_DATASET_DIR/val.parquet"
DSSAT_PROFIT_VAL_FILE="$DSSAT_PROFIT_DATASET_DIR/val.parquet"
CYCLES_YIELD_VAL_FILE="$CYCLES_YIELD_DATASET_DIR/val.parquet"
CYCLES_PROFIT_VAL_FILE="$CYCLES_PROFIT_DATASET_DIR/val.parquet"

LOG_DIR="$SCRIPT_DIR/logs"
LOG_FILE="$LOG_DIR/${RUN_NAME}.log"
RESULT_DIR="$SCRIPT_DIR/results/llm_train/${RUN_NAME}"
MODEL_PATH="Qwen/Qwen3-4B-Instruct-2507"
REQUIRE_THINK=False

prepend_pythonpath() {
    local path="$1"
    if [[ -d "$path" ]]; then
        case ":${PYTHONPATH:-}:" in
            *":$path:"*) ;;
            *) export PYTHONPATH="$path${PYTHONPATH:+:$PYTHONPATH}" ;;
        esac
    fi
}

prepare_pythonpath() {
    prepend_pythonpath "$PROJECT_ROOT"
    prepend_pythonpath "$PROJECT_ROOT/verl"
}

prepare_dssat_runtime() {
    prepare_pythonpath

    if [[ -f "$PROJECT_ROOT/smoke_tests/gym_dssat/_activate_spack.sh" ]]; then
        # shellcheck disable=SC1091
        source "$PROJECT_ROOT/smoke_tests/gym_dssat/_activate_spack.sh"
    fi

    prepare_pythonpath

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

join_hydra_list() {
    local first=1
    printf '['
    for item in "$@"; do
        if [[ "$first" -eq 0 ]]; then
            printf ','
        fi
        printf '%s' "$item"
        first=0
    done
    printf ']'
}

build_dataset_if_needed() {
    local config="$1"
    shift
    local expected_files=("$@")
    local dataset_id
    local expected_file
    local needs_build=1
    dataset_id="$(basename "$config" .yaml)"

    for expected_file in "${expected_files[@]}"; do
        if [[ 1 -eq 1 || ! -f "$expected_file" || "$config" -nt "$expected_file" ]]; then
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

build_dataset_if_needed "$WOFOST_CONFIG" \
    "$WOFOST_TRAIN_LNPKW_YIELD_FILE" \
    "$WOFOST_TRAIN_LNPKW_PROFIT_FILE" \
    "$WOFOST_TRAIN_LNPKW_WATER_FILE" \
    "$WOFOST_TRAIN_LNPK_PROFIT_FILE" \
    "$WOFOST_TRAIN_LNPK_NUTRIENT_FILE" \
    "$WOFOST_TRAIN_LNW_YIELD_FILE" \
    "$WOFOST_VAL_LNPKW_PROFIT_FILE" \
    "$WOFOST_VAL_LNPK_PROFIT_FILE" \
    "$WOFOST_VAL_LNW_YIELD_FILE" \
    "$WOFOST_VAL_LNW_PROFIT_FILE" \
    "$WOFOST_VAL_LNPKW_NUTRIENT_FILE"
build_dataset_if_needed "$DSSAT_YIELD_CONFIG" "$DSSAT_YIELD_TRAIN_FILE" "$DSSAT_YIELD_VAL_FILE"
build_dataset_if_needed "$DSSAT_PROFIT_CONFIG" "$DSSAT_PROFIT_VAL_FILE"
build_dataset_if_needed "$CYCLES_YIELD_CONFIG" "$CYCLES_YIELD_TRAIN_FILE" "$CYCLES_YIELD_VAL_FILE"
build_dataset_if_needed "$CYCLES_PROFIT_CONFIG" "$CYCLES_PROFIT_VAL_FILE"

TRAIN_FILES="$(
    join_hydra_list \
        "$WOFOST_TRAIN_LNPKW_YIELD_FILE" \
        "$WOFOST_TRAIN_LNPKW_PROFIT_FILE" \
        "$WOFOST_TRAIN_LNPKW_WATER_FILE" \
        "$WOFOST_TRAIN_LNPK_PROFIT_FILE" \
        "$WOFOST_TRAIN_LNPK_NUTRIENT_FILE" \
        "$WOFOST_TRAIN_LNW_YIELD_FILE" \
        "$DSSAT_YIELD_TRAIN_FILE" \
        "$CYCLES_YIELD_TRAIN_FILE"
)"
VAL_SET_OVERRIDES=(
    "+data.val_sets.seen_wofost_lnpkw_profit=${WOFOST_VAL_LNPKW_PROFIT_FILE}"
    "+data.val_sets.seen_wofost_lnpk_profit=${WOFOST_VAL_LNPK_PROFIT_FILE}"
    "+data.val_sets.seen_wofost_lnw_yield=${WOFOST_VAL_LNW_YIELD_FILE}"
    "+data.val_sets.seen_dssat_yield=${DSSAT_YIELD_VAL_FILE}"
    "+data.val_sets.seen_cycles_yield=${CYCLES_YIELD_VAL_FILE}"
    "+data.val_sets.heldout_wofost_lnw_profit=${WOFOST_VAL_LNW_PROFIT_FILE}"
    "+data.val_sets.heldout_wofost_lnpkw_nutrient=${WOFOST_VAL_LNPKW_NUTRIENT_FILE}"
    "+data.val_sets.heldout_dssat_profit=${DSSAT_PROFIT_VAL_FILE}"
    "+data.val_sets.heldout_cycles_profit=${CYCLES_PROFIT_VAL_FILE}"
)

exec bash entrypoints/train/train.sh \
    --log-file "$LOG_FILE" \
    --config-name agri_grpo \
    "data.train_files=${TRAIN_FILES}" \
    "data.val_files=null" \
    "data.validation_axis=schema_tuple" \
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
    "trainer.project_name=agrimanager_t32_unified_prompt_conditioned_policy" \
    "trainer.default_local_dir=${RESULT_DIR}" \
    "trainer.experiment_name=${RUN_NAME}" \
    "trainer.total_epochs=4" \
    "trainer.val_before_train=True" \
    "trainer.test_freq=10" \
    "trainer.log_val_generations=4" \
    "trainer.n_gpus_per_node=2" \
    "trainer.save_freq=200" \
    "trainer.max_actor_ckpt_to_keep=4" \
    "trainer.max_critic_ckpt_to_keep=4" \
    "trainer.rollout_filter.enable=False" \
    "ray_kwargs.ray_init.num_cpus=32" \
    "+ray_kwargs.ray_init.include_dashboard=False" \
    "+ray_kwargs.ray_init._temp_dir=/tmp/ray_${SLURM_JOB_ID:-${SLURM_JOBID:-$$}}"
