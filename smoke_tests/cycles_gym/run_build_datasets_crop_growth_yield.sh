#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

THINK_DATASET_CONFIG="$SCRIPT_DIR/config/crop_growth_yield_llm_think.yaml"
NO_THINK_DATASET_CONFIG="$SCRIPT_DIR/config/crop_growth_yield_llm_no_think.yaml"
NUM_WORKERS=1

THINK_LOG_FILE="$SCRIPT_DIR/logs/cycles_gym_crop_growth_yield_llm_think_dataset_build.log"
NO_THINK_LOG_FILE="$SCRIPT_DIR/logs/cycles_gym_crop_growth_yield_llm_no_think_dataset_build.log"

mkdir -p "$SCRIPT_DIR/logs" "$SCRIPT_DIR/data" "$SCRIPT_DIR/results"

cd "$PROJECT_ROOT"

bash entrypoints/dataset/build.sh \
    --config "$THINK_DATASET_CONFIG" \
    --num-workers "$NUM_WORKERS" \
    2>&1 | tee "$THINK_LOG_FILE"

bash entrypoints/dataset/build.sh \
    --config "$NO_THINK_DATASET_CONFIG" \
    --num-workers "$NUM_WORKERS" \
    2>&1 | tee "$NO_THINK_LOG_FILE"
