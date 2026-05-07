#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_cross_sim_train_lib.sh"

MODEL_CONFIG="${CROSS_SIM_EVAL_MODEL_CONFIG:-agrimanager/model_interface/configs/vllm_offline/default.yaml}"
CHECKPOINT_ROOT="${CROSS_SIM_CHECKPOINT_ROOT:-$SCRIPT_DIR/results/llm_train}"

cd "$cross_sim_project_root"

if [[ -f "$cross_sim_project_root/smoke_tests/gym_dssat/_activate_spack.sh" ]]; then
    # shellcheck disable=SC1091
    source "$cross_sim_project_root/smoke_tests/gym_dssat/_activate_spack.sh"
fi

eval_one() {
    local run_name="$1"
    shift
    local configs=("$@")

    for config in "${configs[@]}"; do
        cross_sim_build_dataset "$config"
    done

    local test_files=()
    for config in "${configs[@]}"; do
        test_files+=("$(cross_sim_dataset_dir "$config")/test.parquet")
    done
    local test_files_arg
    test_files_arg="$(cross_sim_join_hydra_list "${test_files[@]}")"

    local latest_file="$CHECKPOINT_ROOT/$run_name/latest_checkpointed_iteration.txt"
    if [[ ! -f "$latest_file" ]]; then
        echo "Missing checkpoint marker: $latest_file" >&2
        exit 1
    fi
    local step
    step="$(tr -d '[:space:]' < "$latest_file")"
    local actor_path="$CHECKPOINT_ROOT/$run_name/global_step_${step}/actor"
    if [[ ! -d "$actor_path" ]]; then
        echo "Missing actor checkpoint: $actor_path" >&2
        exit 1
    fi

    local output_dir="$SCRIPT_DIR/results/llm_eval/${run_name}_latest_eval"
    local log_file="$SCRIPT_DIR/logs/${run_name}_latest_eval.log"
    mkdir -p "$output_dir" "$SCRIPT_DIR/logs"

    bash entrypoints/eval/eval.sh \
        --log-file "$log_file" \
        "data.inference_file=${test_files_arg}" \
        "model.config=${MODEL_CONFIG}" \
        "model.path=${actor_path}" \
        "output.dir=${output_dir}" \
        "runtime.temperature=0" \
        "runtime.max_tokens=512" \
        "runtime.max_retries=1"
}

THINK_TEST_CONFIGS=(
    "$SCRIPT_DIR/config/cross_sim_wofost_llm_without_traits_think.yaml"
    "$SCRIPT_DIR/config/cross_sim_gym_dssat_llm_without_traits_think.yaml"
    "$SCRIPT_DIR/config/cross_sim_cycles_gym_llm_without_traits_think.yaml"
)
NO_THINK_TEST_CONFIGS=(
    "$SCRIPT_DIR/config/cross_sim_wofost_llm_without_traits_no_think.yaml"
    "$SCRIPT_DIR/config/cross_sim_gym_dssat_llm_without_traits_no_think.yaml"
    "$SCRIPT_DIR/config/cross_sim_cycles_gym_llm_without_traits_no_think.yaml"
)

eval_one "cross_sim_wofost_llm_think_train" "${THINK_TEST_CONFIGS[@]}"
eval_one "cross_sim_gym_dssat_llm_think_train" "${THINK_TEST_CONFIGS[@]}"
eval_one "cross_sim_cycles_gym_llm_think_train" "${THINK_TEST_CONFIGS[@]}"
eval_one "cross_sim_joint_llm_think_train" "${THINK_TEST_CONFIGS[@]}"
eval_one "cross_sim_wofost_llm_no_think_train" "${NO_THINK_TEST_CONFIGS[@]}"
eval_one "cross_sim_gym_dssat_llm_no_think_train" "${NO_THINK_TEST_CONFIGS[@]}"
eval_one "cross_sim_cycles_gym_llm_no_think_train" "${NO_THINK_TEST_CONFIGS[@]}"
eval_one "cross_sim_joint_llm_no_think_train" "${NO_THINK_TEST_CONFIGS[@]}"
