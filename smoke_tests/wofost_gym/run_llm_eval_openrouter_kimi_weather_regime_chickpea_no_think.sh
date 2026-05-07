#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

RUN_NAME="wofost_weather_regime_chickpea_llm_no_think_openrouter_kimi_smoke"
DATASET_CONFIG="$SCRIPT_DIR/config/weather_regime_chickpea_llm_without_traits_no_think_kimi_smoke.yaml"
DATASET_ID="weather_regime_chickpea_llm_without_traits_no_think_kimi_smoke"
DATASET_DIR="$SCRIPT_DIR/data/${DATASET_ID}"
VALIDATION_SETS=(id drought wet hot cold)
DATASET_BUILD_LOG="$SCRIPT_DIR/logs/${RUN_NAME}_build_dataset.log"
LOG_FILE="$SCRIPT_DIR/logs/${RUN_NAME}.log"
OUTPUT_DIR="$SCRIPT_DIR/results/llm_eval/${RUN_NAME}"
MODEL_CONFIG="agrimanager/model_interface/configs/openrouter/kimi-k2.6_no_reasoning.yaml"

mkdir -p "$SCRIPT_DIR/logs" "$OUTPUT_DIR"

cd "$PROJECT_ROOT"

DATASET_FILES=()
VAL_FILES=()
for validation_set in "${VALIDATION_SETS[@]}"; do
    dataset_file="$DATASET_DIR/val_${validation_set}.parquet"
    DATASET_FILES+=("$dataset_file")
    VAL_FILES+=("$dataset_file")
done
DATASET_FILES+=("$DATASET_DIR/train.parquet")

NEEDS_BUILD=0
for dataset_file in "${DATASET_FILES[@]}"; do
    if [[ ! -f "$dataset_file" || "$DATASET_CONFIG" -nt "$dataset_file" ]]; then
        NEEDS_BUILD=1
        break
    fi
done

if [[ "$NEEDS_BUILD" -eq 1 ]]; then
    bash entrypoints/dataset/build.sh \
        --config "$DATASET_CONFIG" \
        --num-workers 1 \
        2>&1 | tee "$DATASET_BUILD_LOG"
fi

INFERENCE_FILES="["
for idx in "${!VAL_FILES[@]}"; do
    if [[ "$idx" -gt 0 ]]; then
        INFERENCE_FILES+=","
    fi
    INFERENCE_FILES+="${VAL_FILES[$idx]}"
done
INFERENCE_FILES+="]"

bash entrypoints/eval/eval.sh \
    "data.inference_file=${INFERENCE_FILES}" \
    "data.validation_axis=weather_regime" \
    "model.config=${MODEL_CONFIG}" \
    "output.dir=${OUTPUT_DIR}" \
    "runtime.temperature=0" \
    "runtime.max_tokens=128" \
    "runtime.max_retries=1" \
    2>&1 | tee "$LOG_FILE"
