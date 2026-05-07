#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEFAULT_CONFIG_DIR="$PROJECT_ROOT/entrypoints/eval/config"
DEFAULT_CONFIG_NAME="nn"
LOG_FILE=""

cd "$PROJECT_ROOT"

CONFIG_PATH_PROVIDED=0
CONFIG_NAME_PROVIDED=0
FORWARD_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config-path|--config-path=*)
            CONFIG_PATH_PROVIDED=1
            FORWARD_ARGS+=("$1")
            if [[ "$1" == "--config-path" ]]; then
                if [[ $# -lt 2 ]]; then
                    echo "Error: --config-path requires a value" >&2
                    exit 1
                fi
                FORWARD_ARGS+=("$2")
                shift
            fi
            shift
            ;;
        --config-name|--config-name=*)
            CONFIG_NAME_PROVIDED=1
            FORWARD_ARGS+=("$1")
            if [[ "$1" == "--config-name" ]]; then
                if [[ $# -lt 2 ]]; then
                    echo "Error: --config-name requires a value" >&2
                    exit 1
                fi
                FORWARD_ARGS+=("$2")
                shift
            fi
            shift
            ;;
        --log-file=*)
            LOG_FILE="${1#*=}"
            shift
            ;;
        --log-file)
            if [[ $# -lt 2 ]]; then
                echo "Error: --log-file requires a value" >&2
                exit 1
            fi
            LOG_FILE="$2"
            shift 2
            ;;
        *)
            FORWARD_ARGS+=("$1")
            shift
            ;;
    esac
done

DEFAULT_ARGS=()
if [ "$CONFIG_PATH_PROVIDED" -eq 0 ]; then
    DEFAULT_ARGS+=(--config-path "$DEFAULT_CONFIG_DIR")
fi
if [ "$CONFIG_NAME_PROVIDED" -eq 0 ]; then
    DEFAULT_ARGS+=(--config-name "$DEFAULT_CONFIG_NAME")
fi

if [[ -n "$LOG_FILE" ]]; then
    if [[ "$LOG_FILE" != /* ]]; then
        LOG_FILE="$PROJECT_ROOT/$LOG_FILE"
    fi
    mkdir -p "$(dirname "$LOG_FILE")"
    python -u -m agrimanager.nn_ppo.eval \
        "${DEFAULT_ARGS[@]}" "${FORWARD_ARGS[@]}" \
        2>&1 | tee "$LOG_FILE"
else
    exec python -u -m agrimanager.nn_ppo.eval \
        "${DEFAULT_ARGS[@]}" "${FORWARD_ARGS[@]}"
fi
