#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
EXPERIMENT_DIR="$PROJECT_ROOT/experiments/support_t1_2_cross_crop_trait_selection"

cd "$PROJECT_ROOT"

python "$SCRIPT_DIR/discover_strategy_traits.py" \
  --transfer-dir "$EXPERIMENT_DIR/analysis/eight_specialist_transfer" \
  --output-dir "$EXPERIMENT_DIR/analysis/strategy_trait_discovery" \
  --source-trait-schema traits_v1_23d \
  --max-features 8 \
  --candidate-top-k 16 \
  --shuffle-repeats 20 \
  --bootstrap-repeats 50

python "$SCRIPT_DIR/build_selected_trait_schema.py" \
  --discovered-traits "$EXPERIMENT_DIR/analysis/strategy_trait_discovery/discovered_traits.json" \
  --schema-name traits_strategy_selected_v1 \
  --output-schema-dir "$PROJECT_ROOT/agrimanager/env/wofost_gym/crop_traits/traits_strategy_selected_v1" \
  --overwrite
