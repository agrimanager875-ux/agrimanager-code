#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

LLM_DATASET_CONFIG="$SCRIPT_DIR/config/crop_planning_llm.yaml"
NN_DATASET_CONFIG="$SCRIPT_DIR/config/crop_planning_nn.yaml"
NUM_WORKERS=1

LLM_LOG_FILE="$SCRIPT_DIR/logs/cycles_gym_crop_planning_smoke_llm_dataset_build.log"
NN_LOG_FILE="$SCRIPT_DIR/logs/cycles_gym_crop_planning_smoke_nn_dataset_build.log"

mkdir -p "$SCRIPT_DIR/logs" "$SCRIPT_DIR/data" "$SCRIPT_DIR/results"

cd "$PROJECT_ROOT"

bash entrypoints/dataset/build.sh \
    --config "$LLM_DATASET_CONFIG" \
    --num-workers "$NUM_WORKERS" \
    2>&1 | tee "$LLM_LOG_FILE"

bash entrypoints/dataset/build.sh \
    --config "$NN_DATASET_CONFIG" \
    --num-workers "$NUM_WORKERS" \
    2>&1 | tee "$NN_LOG_FILE"
