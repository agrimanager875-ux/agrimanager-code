#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_cross_sim_train_lib.sh"

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

cross_sim_run_train "cross_sim_joint_llm_think_train" TRAIN_CONFIGS VAL_CONFIGS
