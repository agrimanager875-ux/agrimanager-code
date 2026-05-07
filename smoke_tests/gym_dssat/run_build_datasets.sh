#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

usage() {
    cat <<'EOF'
Usage: bash run_build_datasets.sh [maize|rice|cotton]
       GYM_DSSAT_CROP=maize bash run_build_datasets.sh
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
DATASET_CONFIG="$SCRIPT_DIR/config/${DATASET_ID}.yaml"
LOG_FILE="$SCRIPT_DIR/logs/${DATASET_ID}_dataset_build.log"

mkdir -p "$SCRIPT_DIR/logs" "$SCRIPT_DIR/data" "$SCRIPT_DIR/results"

# shellcheck disable=SC1091
source "$SCRIPT_DIR/_activate_spack.sh"

cd "$PROJECT_ROOT"

bash entrypoints/dataset/build.sh \
    --config "$DATASET_CONFIG" \
    --num-workers 1 \
    2>&1 | tee "$LOG_FILE"
