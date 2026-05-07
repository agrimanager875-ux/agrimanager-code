#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
NUM_WORKERS="${NUM_WORKERS:-1}"

mkdir -p "$SCRIPT_DIR/logs" "$SCRIPT_DIR/data" "$SCRIPT_DIR/results"

cd "$PROJECT_ROOT"

for DATASET_CONFIG in \
    "$SCRIPT_DIR/config/crop_growth_yield_llm_think.yaml" \
    "$SCRIPT_DIR/config/crop_growth_yield_llm_no_think.yaml"; do
    DATASET_ID="$(basename "$DATASET_CONFIG" .yaml)"
    LOG_FILE="$SCRIPT_DIR/logs/${DATASET_ID}_build_dataset.log"
    bash entrypoints/dataset/build.sh \
        --config "$DATASET_CONFIG" \
        --num-workers "$NUM_WORKERS" \
        2>&1 | tee "$LOG_FILE"
done
