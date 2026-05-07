#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

usage() {
    cat <<'EOF'
Usage: bash run_llm_eval_qwen3_4b_instruct.sh [maize|rice|cotton]
       GYM_DSSAT_CROP=maize bash run_llm_eval_qwen3_4b_instruct.sh
EOF
}

CROP="${GYM_DSSAT_CROP:-maize}"
if [[ $# -gt 0 ]]; then
    case "$1" in
        --crop)
            CROP="${2:?Missing value for --crop}"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            CROP="$1"
            shift
            ;;
    esac
fi
if [[ $# -gt 0 ]]; then
    echo "Unexpected argument: $1" >&2
    usage >&2
    exit 1
fi
CROP="${CROP,,}"
case "$CROP" in
    maize|rice|cotton) ;;
    *)
        echo "Unsupported crop '$CROP'. Choose one of: maize, rice, cotton." >&2
        exit 1
        ;;
esac

DATASET_ID="gym_dssat_smoke_llm_${CROP}"
RUN_NAME="${DATASET_ID}_eval_qwen25_3b_instruct"
DATASET_CONFIG="$SCRIPT_DIR/config/${DATASET_ID}.yaml"
DATASET_DIR="$SCRIPT_DIR/data/${DATASET_ID}"
TEST_FILE="$DATASET_DIR/test.parquet"
DATASET_BUILD_LOG="$SCRIPT_DIR/logs/${DATASET_ID}_dataset_build.log"
LOG_FILE="$SCRIPT_DIR/logs/${RUN_NAME}.log"
OUTPUT_DIR="$SCRIPT_DIR/results/llm_eval/${RUN_NAME}"
MODEL_CONFIG="agrimanager/model_interface/configs/vllm_offline/Qwen2.5-3B-Instruct.yaml"
REPO_DATASET_DIR="$PROJECT_ROOT/data/gym_dssat/${DATASET_ID}"

mkdir -p "$SCRIPT_DIR/logs" "$OUTPUT_DIR"

# shellcheck disable=SC1091
source "$SCRIPT_DIR/_activate_spack.sh"

cd "$PROJECT_ROOT"

if [[ 1 -eq 1 || ! -f "$TEST_FILE" ]]; then
    bash entrypoints/dataset/build.sh \
        --config "$DATASET_CONFIG" \
        --num-workers 1 \
        2>&1 | tee "$DATASET_BUILD_LOG"
fi

mkdir -p "$(dirname "$REPO_DATASET_DIR")"
ln -sfn "$DATASET_DIR" "$REPO_DATASET_DIR"

PYTHON_BIN="${AGRIMANAGER_PYTHON:-${CONDA_PREFIX:+$CONDA_PREFIX/bin/python3}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/verl${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON_BIN" -m agrimanager.rollout.inference.inference_rollout \
    --env-name gym_dssat \
    --dataset-id "$DATASET_ID" \
    --split test \
    --model-config "$MODEL_CONFIG" \
    --output-dir "$OUTPUT_DIR" \
    --temperature 0 \
    --max-tokens 256 \
    --max-retries 1 \
    2>&1 | tee "$LOG_FILE"
