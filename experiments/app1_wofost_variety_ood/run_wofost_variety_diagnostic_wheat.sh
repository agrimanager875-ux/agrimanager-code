#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

RUN_NAME="wofost_variety_diagnostic_wheat_v1"
OUTPUT_DIR="$SCRIPT_DIR/results/diagnostics/wheat_variant_separability_v1"
LOG_FILE="$SCRIPT_DIR/logs/${RUN_NAME}.log"
NUM_WORKERS="${NUM_WORKERS:-32}"

mkdir -p "$SCRIPT_DIR/logs" "$OUTPUT_DIR"

cd "$PROJECT_ROOT"

python -u experiments/app1_wofost_variety_ood/diagnose_variety_separability.py \
    --crop wheat \
    --varieties wheat_1 wheat_2 wheat_3 wheat_4 wheat_5 wheat_6 wheat_7 wheat_8 \
    --pool agrimanager/wofost-weather-pool \
    --pool-split train \
    --num-scenarios 64 \
    --scenario-seed 42 \
    --num-action-seeds 3 \
    --action-seed-base 1729 \
    --num-workers "$NUM_WORKERS" \
    --candidate-pair wheat_1 wheat_7 \
    --output-dir "$OUTPUT_DIR" \
    2>&1 | tee "$LOG_FILE"
