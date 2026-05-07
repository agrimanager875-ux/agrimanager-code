#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

CONFIG_WOFOST="$SCRIPT_DIR/config/t31_cross_sim_wofost.yaml"
CONFIG_DSSAT="$SCRIPT_DIR/config/t31_cross_sim_gym_dssat.yaml"
CONFIG_CYCLES="$SCRIPT_DIR/config/t31_cross_sim_cycles_gym.yaml"

DATA_DIR="${CROSS_SIM_DATA_DIR:-$SCRIPT_DIR/data}"
LOG_DIR="$SCRIPT_DIR/logs"
EVAL_RESULT_DIR="$SCRIPT_DIR/results/llm_eval"
CHECKPOINT_ROOT="${CROSS_SIM_CHECKPOINT_ROOT:-$SCRIPT_DIR/results/llm_train}"
MODEL_CONFIG="${CROSS_SIM_EVAL_MODEL_CONFIG:-agrimanager/model_interface/configs/vllm_offline/default.yaml}"
EXTERNAL_ROOT="${AGRIMANAGER_EXTERNAL_ROOT:-$REPO_ROOT/../AgriManagerExternal}"
DSSAT_RUNTIME_DEFAULT="$REPO_ROOT/spack/gym-dssat-pdi"
if [[ ! -d "$DSSAT_RUNTIME_DEFAULT" && -d "$EXTERNAL_ROOT/gym_dssat_pdi/gym-dssat-pdi" ]]; then
  DSSAT_RUNTIME_DEFAULT="$EXTERNAL_ROOT/gym_dssat_pdi/gym-dssat-pdi"
fi

mkdir -p "$LOG_DIR" "$EVAL_RESULT_DIR"

export DSSAT_HOME="${DSSAT_HOME:-$EXTERNAL_ROOT/dssat-csm-data}"
export DSSAT_BIN="${DSSAT_BIN:-$DSSAT_HOME/bin/dscsm048}"
export DSSAT_MODULE="${DSSAT_MODULE:-gym_dssat_pdi.envs.gym_dssat_env}"
export DSSAT_PDI_PATH="${DSSAT_PDI_PATH:-$DSSAT_RUNTIME_DEFAULT}"
if [[ -d "$DSSAT_PDI_PATH" ]]; then
  export PYTHONPATH="$DSSAT_PDI_PATH:${PYTHONPATH:-}"
fi

export CYCLES_GYM_CYCLESGYM_ROOT="${CYCLES_GYM_CYCLESGYM_ROOT:-$EXTERNAL_ROOT/CyclesGym/cyclesgym}"
export CYCLES_GYM_CYCLES_ROOT="${CYCLES_GYM_CYCLES_ROOT:-$EXTERNAL_ROOT/CyclesGym/cycles}"
export CYCLES_GYM_WORKING_DIR="${CYCLES_GYM_WORKING_DIR:-/tmp/cycles-gym}"

build_dataset_if_needed() {
  local config_path="$1"
  local output_dir="$2"

  echo "[dataset] Building $output_dir from $config_path"
  rm -rf "$output_dir"
  bash entrypoints/dataset/build.sh \
    --config "$config_path" \
    --output-dir "$output_dir" \
    2>&1 | tee "$LOG_DIR/$(basename "$output_dir")_build.log"
}

latest_actor_path() {
  local run_name="$1"
  local run_dir="$CHECKPOINT_ROOT/$run_name"
  local latest_file="$run_dir/latest_checkpointed_iteration.txt"

  if [[ ! -f "$latest_file" ]]; then
    echo "[eval] Missing latest checkpoint marker: $latest_file" >&2
    return 1
  fi

  local step
  step="$(tr -dc '0-9' < "$latest_file")"
  if [[ -z "$step" ]]; then
    echo "[eval] Could not parse checkpoint step from: $latest_file" >&2
    return 1
  fi

  local actor_path="$run_dir/global_step_${step}/actor"
  if [[ ! -d "$actor_path" ]]; then
    echo "[eval] Missing actor checkpoint: $actor_path" >&2
    return 1
  fi

  printf '%s\n' "$actor_path"
}

hydra_list() {
  local sep=""
  local result="["
  local value
  for value in "$@"; do
    result+="${sep}${value}"
    sep=","
  done
  result+="]"
  printf '%s\n' "$result"
}

run_latest_val_eval() {
  local run_name="$1"
  local mode="$2"
  local model_path
  model_path="$(latest_actor_path "$run_name")"

  local output_dir="$EVAL_RESULT_DIR/${run_name}_latest_val_eval"
  local log_file="$LOG_DIR/${run_name}_latest_val_eval.log"
  local val_files
  val_files="$(hydra_list \
    "$DATA_WOFOST_DIR/val.parquet" \
    "$DATA_DSSAT_DIR/val.parquet" \
    "$DATA_CYCLES_DIR/val.parquet")"

  echo "[eval] $run_name"
  echo "[eval] model_path=$model_path"
  bash entrypoints/inference/eval.sh \
    --model-config "$MODEL_CONFIG" \
    --model-path "$model_path" \
    --output-dir "$output_dir" \
    data.inference_file="$val_files" \
    data.use_inference_chat_template=true \
    runtime.temperature=0 \
    runtime.max_tokens=512 \
    runtime.max_retries=1 \
    tracking.project=agrimanager_cross_simulator_crop_growth_ood_eval \
    tracking.run_name="${run_name}_latest_val_eval" \
    "metadata.tags=[T3.1,cross_simulator,maize,latest_checkpoint,val_only,${mode}]" \
    2>&1 | tee "$log_file"
}

DATA_WOFOST_DIR="$DATA_DIR/t31_cross_sim_wofost"
DATA_DSSAT_DIR="$DATA_DIR/t31_cross_sim_gym_dssat"
DATA_CYCLES_DIR="$DATA_DIR/t31_cross_sim_cycles_gym"

build_dataset_if_needed "$CONFIG_WOFOST" "$DATA_WOFOST_DIR"
build_dataset_if_needed "$CONFIG_DSSAT" "$DATA_DSSAT_DIR"
build_dataset_if_needed "$CONFIG_CYCLES" "$DATA_CYCLES_DIR"

run_latest_val_eval "cross_sim_wofost_llm_think_train" "think"
run_latest_val_eval "cross_sim_gym_dssat_llm_think_train" "think"
run_latest_val_eval "cross_sim_cycles_gym_llm_think_train" "think"
run_latest_val_eval "cross_sim_wofost_llm_no_think_train" "no_think"
run_latest_val_eval "cross_sim_gym_dssat_llm_no_think_train" "no_think"
run_latest_val_eval "cross_sim_cycles_gym_llm_no_think_train" "no_think"
