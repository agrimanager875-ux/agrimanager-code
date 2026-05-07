#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
EXPERIMENT_DIR="$PROJECT_ROOT/experiments/support_t1_2_cross_crop_trait_selection"

cd "$PROJECT_ROOT"

python "$SCRIPT_DIR/select_traits_from_specialist_transfer.py" \
  --transfer-dir "$EXPERIMENT_DIR/analysis/eight_specialist_transfer" \
  --output-dir "$EXPERIMENT_DIR/analysis/eight_specialist_trait_selection" \
  --trait-schemas traits_v1_23d,traits_v1_6d
