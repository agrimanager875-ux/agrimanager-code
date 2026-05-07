#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

CONFIGS=(
    "$SCRIPT_DIR/config/t21_observation_schema_shift_llm_no_think.yaml"
    "$SCRIPT_DIR/config/t21_observation_schema_shift_llm_think.yaml"
)

cd "$PROJECT_ROOT"

for config in "${CONFIGS[@]}"; do
    dataset_id="$(basename "$config" .yaml)"
    dataset_dir="$SCRIPT_DIR/data/$dataset_id"
    manifest_file="$dataset_dir/manifest.json"
    mkdir -p "$SCRIPT_DIR/logs"
    if [[ 1 -eq 1 || ! -f "$manifest_file" || "$config" -nt "$manifest_file" ]]; then
        bash entrypoints/dataset/build.sh \
            --config "$config" \
            --num-workers 1 \
            2>&1 | tee "$SCRIPT_DIR/logs/${dataset_id}_build_dataset.log"
    fi
done
