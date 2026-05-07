#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

RUN_NAME="wofost_variety_diagnostic_potato_v1"
OUTPUT_DIR="$SCRIPT_DIR/results/diagnostics/potato_variant_separability_v1"
LOG_FILE="$SCRIPT_DIR/logs/${RUN_NAME}.log"
NUM_WORKERS="${NUM_WORKERS:-32}"

mkdir -p "$SCRIPT_DIR/logs" "$OUTPUT_DIR"

cd "$PROJECT_ROOT"

python -u experiments/app1_wofost_variety_ood/diagnose_variety_separability.py \
    --crop potato \
    --varieties potato_1 potato_2 potato_3 potato_4 potato_5 potato_6 potato_7 potato_8 potato_9 \
    --pool agrimanager/wofost-weather-pool \
    --pool-split train \
    --num-scenarios 64 \
    --scenario-seed 42 \
    --num-action-seeds 3 \
    --action-seed-base 1729 \
    --num-workers "$NUM_WORKERS" \
    --candidate-pair potato_1 potato_9 \
    --output-dir "$OUTPUT_DIR" \
    2>&1 | tee "$LOG_FILE"
