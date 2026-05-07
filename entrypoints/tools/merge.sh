#!/bin/bash
# Usage: bash entrypoints/tools/merge.sh <local_dir> <target_dir> [hf_model_config_path]
#
# hf_model_config_path: base model path or HF repo id supplying config.json + tokenizer.
# Required when <local_dir>/huggingface/ was not saved by the trainer (e.g. smoke runs).
# Example: bash entrypoints/tools/merge.sh actor/ actor_hf/ Qwen/Qwen2.5-0.5B-Instruct
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="$PROJECT_DIR/verl:${PYTHONPATH:-}"

LOCAL_DIR="${1:?Usage: $0 <local_dir> <target_dir> [hf_model_config_path]}"
TARGET_DIR="${2:?Usage: $0 <local_dir> <target_dir> [hf_model_config_path]}"
HF_CONFIG_PATH="${3:-}"

EXTRA_ARGS=()
if [[ -n "$HF_CONFIG_PATH" ]]; then
    EXTRA_ARGS+=(--hf_model_config_path "$HF_CONFIG_PATH")
fi

python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "$LOCAL_DIR" \
    --target_dir "$TARGET_DIR" \
    "${EXTRA_ARGS[@]}"
