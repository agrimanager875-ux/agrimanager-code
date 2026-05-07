#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
EXPERIMENT_DIR="$PROJECT_ROOT/experiments/support_t1_2_cross_crop_trait_selection"

cd "$PROJECT_ROOT"

python "$SCRIPT_DIR/diagnose_existing_specialist_transfer.py" \
  --policy-set eight \
  --policy-map-file "$SCRIPT_DIR/specialist_transfer_policies_8.json" \
  --device cpu \
  --output-dir "$EXPERIMENT_DIR/analysis/eight_specialist_transfer"
