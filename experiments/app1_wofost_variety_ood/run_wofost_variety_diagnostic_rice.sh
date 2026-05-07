#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

RUN_NAME="wofost_variety_diagnostic_rice_v1"
OUTPUT_DIR="$SCRIPT_DIR/results/diagnostics/rice_variant_separability_v1"
LOG_FILE="$SCRIPT_DIR/logs/${RUN_NAME}.log"
NUM_WORKERS="${NUM_WORKERS:-32}"

mkdir -p "$SCRIPT_DIR/logs" "$OUTPUT_DIR"

cd "$PROJECT_ROOT"

python -u experiments/app1_wofost_variety_ood/diagnose_variety_separability.py \
    --crop rice \
    --varieties rice_1 rice_2 rice_3 rice_4 rice_5 rice_6 rice_7 rice_8 rice_9 \
    --pool agrimanager/wofost-weather-pool \
    --pool-split train \
    --num-scenarios 64 \
    --scenario-seed 42 \
    --num-action-seeds 3 \
    --action-seed-base 1729 \
    --num-workers "$NUM_WORKERS" \
    --candidate-pair rice_1 rice_9 \
    --output-dir "$OUTPUT_DIR" \
    2>&1 | tee "$LOG_FILE"
