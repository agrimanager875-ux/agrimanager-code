#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

RUN_NAME="cycles_gym_crop_planning_smoke_llm_eval_qwen3_4b_instruct"
DATASET_CONFIG="$SCRIPT_DIR/config/crop_planning_llm.yaml"
DATASET_DIR="$SCRIPT_DIR/data/cycles_gym_crop_planning_smoke_llm"
TEST_FILE="$DATASET_DIR/test.parquet"
DATASET_BUILD_LOG="$SCRIPT_DIR/logs/cycles_gym_crop_planning_smoke_llm_dataset_build.log"
LOG_FILE="$SCRIPT_DIR/logs/${RUN_NAME}.log"
OUTPUT_DIR="$SCRIPT_DIR/results/llm_eval/${RUN_NAME}"
MODEL_CONFIG="agrimanager/model_interface/configs/vllm_offline/Qwen3-4B-Instruct-2507.yaml"

mkdir -p "$SCRIPT_DIR/logs" "$OUTPUT_DIR"

cd "$PROJECT_ROOT"

if [[ 1 -eq 1 || ! -f "$TEST_FILE" ]]; then
    bash entrypoints/dataset/build.sh \
        --config "$DATASET_CONFIG" \
        --num-workers 1 \
        2>&1 | tee "$DATASET_BUILD_LOG"
fi

bash entrypoints/eval/eval.sh \
    "data.inference_file=${TEST_FILE}" \
    "model.config=${MODEL_CONFIG}" \
    "output.dir=${OUTPUT_DIR}" \
    "runtime.temperature=0" \
    "runtime.max_tokens=256" \
    "runtime.max_retries=1" \
    2>&1 | tee "$LOG_FILE"
