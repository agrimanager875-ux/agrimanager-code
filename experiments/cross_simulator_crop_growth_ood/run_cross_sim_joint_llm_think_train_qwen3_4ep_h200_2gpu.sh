#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_cross_sim_train_lib.sh"

export WANDB_MODE="${WANDB_MODE:-online}"
export CROSS_SIM_MODEL_PATH="${CROSS_SIM_MODEL_PATH:-Qwen/Qwen3-4B-Instruct-2507}"
export CROSS_SIM_N_GPUS_PER_NODE="${CROSS_SIM_N_GPUS_PER_NODE:-2}"
export CROSS_SIM_RAY_NUM_CPUS="${CROSS_SIM_RAY_NUM_CPUS:-32}"
export CROSS_SIM_TOTAL_EPOCHS="${CROSS_SIM_TOTAL_EPOCHS:-4}"
export CROSS_SIM_SAVE_FREQ="${CROSS_SIM_SAVE_FREQ:-100}"
export CROSS_SIM_TEST_FREQ="${CROSS_SIM_TEST_FREQ:-10}"
export CROSS_SIM_FORCE_REBUILD_DATASETS="${CROSS_SIM_FORCE_REBUILD_DATASETS:-1}"

TRAIN_CONFIGS=(
    "$SCRIPT_DIR/config/cross_sim_joint_wofost_llm_without_traits_think.yaml"
    "$SCRIPT_DIR/config/cross_sim_joint_gym_dssat_llm_without_traits_think.yaml"
    "$SCRIPT_DIR/config/cross_sim_joint_cycles_gym_llm_without_traits_think.yaml"
)
VAL_CONFIGS=(
    "$SCRIPT_DIR/config/cross_sim_wofost_llm_without_traits_think.yaml"
    "$SCRIPT_DIR/config/cross_sim_gym_dssat_llm_without_traits_think.yaml"
    "$SCRIPT_DIR/config/cross_sim_cycles_gym_llm_without_traits_think.yaml"
)

ray stop --force >/dev/null 2>&1 || true

cross_sim_run_train "cross_sim_joint_llm_think_train_qwen3_4ep_h200_2gpu" TRAIN_CONFIGS VAL_CONFIGS
