#!/bin/bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash entrypoints/dataset/build.sh --config <dataset_config.yaml> [--num-workers N] [--output-dir DIR] [--force|--no-force]

Examples:
  bash entrypoints/dataset/build.sh \
    --config entrypoints/dataset/examples/wofost/example_sample.yaml

  bash entrypoints/dataset/build.sh \
    --config experiments/legacy_wofost_weather_generalization/config/weather_wheat_llm_without_traits_think.yaml \
    --num-workers 8

  bash entrypoints/dataset/build.sh \
    --config experiments/.../config.yaml \
    --output-dir experiments/.../my_dataset_output

The config must define `env_name`. The script infers:
  dataset_id = config filename without .yaml
  output_dir = sibling `data/` directory next to the config directory
  if --output-dir is set:
    - use as final dataset directory when basename matches dataset_id
    - otherwise use its parent as generation base and write to <parent>/<dataset_id>
  --force always removes the target dataset directory before generation (default: on)
  --no-force disables cleanup and allows append-like behavior
EOF
}

CONFIG_PATH=""
NUM_WORKERS=""
OUTPUT_DIR=""
FORCE_BUILD=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG_PATH="${2:?Missing value for --config}"
            shift 2
            ;;
        --num-workers)
            NUM_WORKERS="${2:?Missing value for --num-workers}"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="${2:?Missing value for --output-dir}"
            shift 2
            ;;
        --force)
            FORCE_BUILD=1
            shift
            ;;
        --no-force)
            FORCE_BUILD=0
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ -z "$CONFIG_PATH" ]]; then
    echo "--config is required" >&2
    usage >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [[ "$CONFIG_PATH" != /* ]]; then
    CONFIG_PATH="$PROJECT_ROOT/$CONFIG_PATH"
fi
CONFIG_PATH="$(cd "$(dirname "$CONFIG_PATH")" && pwd)/$(basename "$CONFIG_PATH")"

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "Dataset config not found: $CONFIG_PATH" >&2
    exit 1
fi

CONFIG_DIR="$(dirname "$CONFIG_PATH")"
DATASET_ID="$(
    python - "$CONFIG_PATH" <<'PY'
import sys

import yaml

config_path = sys.argv[1]
with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f) or {}

print(config.get("dataset_id") or config_path.rsplit("/", 1)[-1].removesuffix(".yaml"))
PY
)"

DEFAULT_OUTPUT_BASE="$(cd "$CONFIG_DIR/.." && pwd)/data"
if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="$DEFAULT_OUTPUT_BASE"
    OUTPUT_DIR_SPECIFIED=0
else
    OUTPUT_DIR_SPECIFIED=1
fi

if [[ "$OUTPUT_DIR" != /* ]]; then
    OUTPUT_DIR="$PROJECT_ROOT/$OUTPUT_DIR"
fi
OUTPUT_PARENT_DIR="$(dirname "$OUTPUT_DIR")"
mkdir -p "$OUTPUT_PARENT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_PARENT_DIR" && pwd)/$(basename "$OUTPUT_DIR")"

if [[ "$OUTPUT_DIR_SPECIFIED" == "1" ]]; then
    GENERATE_BASE_DIR="$(cd "$(dirname "$OUTPUT_DIR")" && pwd)"
    if [[ "$(basename "$OUTPUT_DIR")" == "$DATASET_ID" ]]; then
        TARGET_DATASET_DIR="$OUTPUT_DIR"
    else
        TARGET_DATASET_DIR="$GENERATE_BASE_DIR/$DATASET_ID"
    fi
else
    GENERATE_BASE_DIR="$OUTPUT_DIR"
    TARGET_DATASET_DIR="$GENERATE_BASE_DIR/$DATASET_ID"
fi

if [[ "$FORCE_BUILD" == "1" ]]; then
    rm -rf "$TARGET_DATASET_DIR"
fi

echo "========================================================================"
echo "Dataset Generation"
echo "========================================================================"
echo "Config file: $CONFIG_PATH"
echo "Dataset id: $DATASET_ID"
echo "Output dir: $TARGET_DATASET_DIR"
echo "Force rebuild: $FORCE_BUILD"
echo "Generate base dir: $GENERATE_BASE_DIR"
if [[ -n "$NUM_WORKERS" ]]; then
    echo "Workers: $NUM_WORKERS"
fi
echo "========================================================================"

PYTHON_CMD=(
    python -c
    "import sys; sys.path.insert(0, '$PROJECT_ROOT'); from agrimanager.env.create_dataset import generate; generate('$CONFIG_PATH', '$GENERATE_BASE_DIR'${NUM_WORKERS:+, num_workers=$NUM_WORKERS})"
)

cd "$PROJECT_ROOT"
"${PYTHON_CMD[@]}"

echo ""
echo "Generated files:"
find "$TARGET_DATASET_DIR" -maxdepth 1 -type f -name "*.parquet" -print | sort | sed 's#^#  - #'
