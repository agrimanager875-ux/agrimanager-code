#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEFAULT_CONFIG_DIR="$PROJECT_ROOT/entrypoints/train/config"
DEFAULT_CONFIG_NAME="nn"

cd "$PROJECT_ROOT"

CONFIG_PATH_PROVIDED=0
CONFIG_NAME_PROVIDED=0
for arg in "$@"; do
    case "$arg" in
        --config-path|--config-path=*)
            CONFIG_PATH_PROVIDED=1
            ;;
        --config-name|--config-name=*)
            CONFIG_NAME_PROVIDED=1
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

exec python -u -m agrimanager.nn_ppo.train \
    "${DEFAULT_ARGS[@]}" "$@"
