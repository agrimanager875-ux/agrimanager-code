#!/usr/bin/env bash
set -euo pipefail

# Generate datasets for every agent under wheat_agro_daily_wso and drop them
# into a per-model data/ folder next to the agent.pt files.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WOFOST_GYM_PATH="${SCRIPT_DIR}/../../../../../AgriManagerExternal/WOFOSTGym"
BASE_RESULTS_DIR="${SCRIPT_DIR}/../results/wheat_agro_wso_delta/"
DATA_FILE_NAME="data"

if [ ! -d "$WOFOST_GYM_PATH" ]; then
  echo "WOFOST-Gym not found at $WOFOST_GYM_PATH" >&2
  exit 1
fi

if [ ! -d "$BASE_RESULTS_DIR" ]; then
  echo "Results directory not found at $BASE_RESULTS_DIR" >&2
  exit 1
fi

mapfile -t AGENT_FILES < <(find "$BASE_RESULTS_DIR" -type f -name agent.pt | sort)

if [ ${#AGENT_FILES[@]} -eq 0 ]; then
  echo "No agent.pt files found under $BASE_RESULTS_DIR" >&2
  exit 1
fi

for agent_file in "${AGENT_FILES[@]}"; do
  model_dir="$(dirname "$agent_file")"
  algo_dir="$(basename "$(dirname "$model_dir")")"
  agent_type="${algo_dir^^}"
  save_folder="${model_dir}/data/"
  config_path="${model_dir}/config.yaml"

  mkdir -p "$save_folder"

  cmd=(
    python3 -m data_generation.gen_data
    --file-type npz
    --save-folder "$save_folder"
    --data-file "$DATA_FILE_NAME"
    --agent-type "$agent_type"
    --agent-path "$agent_file"
    --no-cuda
  )

  if [ -f "$config_path" ]; then
    config_rel="$(realpath --relative-to="$WOFOST_GYM_PATH" "$config_path")"
    cmd+=(--config-fpath "$config_rel")
  fi

  echo "Generating data for $agent_type model at $model_dir"
  (cd "$WOFOST_GYM_PATH" && "${cmd[@]}")
done

echo "Data generation completed."
