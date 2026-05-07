#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

RUN_NAME="cycles_gym_smoke_nn_eval"
TRAIN_RUN_NAME="cycles_gym_smoke_nn_train"
DATASET_CONFIG="$SCRIPT_DIR/config/nn.yaml"
DATASET_DIR="$SCRIPT_DIR/data/cycles_gym_smoke_nn"
INFERENCE_FILE="$DATASET_DIR/val.parquet"
DATASET_BUILD_LOG="$SCRIPT_DIR/logs/cycles_gym_smoke_nn_dataset_build.log"
LOG_FILE="$SCRIPT_DIR/logs/${RUN_NAME}.log"
AGENT_ROOT="$SCRIPT_DIR/results/nn_train/${TRAIN_RUN_NAME}"
OUTPUT_DIR="$SCRIPT_DIR/results/nn_eval/${RUN_NAME}"

mkdir -p "$SCRIPT_DIR/logs" "$OUTPUT_DIR"

latest_agent() {
    python - "$AGENT_ROOT" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
candidates = list(root.rglob("agent.zip"))
if not candidates:
    raise SystemExit(1)
latest = max(candidates, key=lambda p: p.stat().st_mtime)
print(latest.resolve())
PY
}

cd "$PROJECT_ROOT"

if [[ 1 -eq 1 || ! -f "$INFERENCE_FILE" ]]; then
    bash entrypoints/dataset/build.sh \
        --config "$DATASET_CONFIG" \
        --num-workers 1 \
        2>&1 | tee "$DATASET_BUILD_LOG"
fi

if ! AGENT_PATH="$(latest_agent)"; then
    echo "No agent.zip found under $AGENT_ROOT" >&2
    echo "Run smoke_tests/cycles_gym/run_nn_train.sh first." >&2
    exit 1
fi

bash entrypoints/eval/nn_eval.sh \
    "data.inference_file=${INFERENCE_FILE}" \
    "agent.path=${AGENT_PATH}" \
    "runtime.device=cpu" \
    "runtime.split_by_group=env_id" \
    "output.dir=${OUTPUT_DIR}" \
    2>&1 | tee "$LOG_FILE"
