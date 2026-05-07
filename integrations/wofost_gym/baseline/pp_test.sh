#!/bin/bash
# pp-v0 (Potential Production) baseline test
# All config lives in config/pp_test.yaml
# Override any param via CLI:
#   bash integrations/wofost_gym/baseline/run_pp_test.sh --intvn-interval 7

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

cd "$PROJECT_DIR"

python "$SCRIPT_DIR/test_pp_env.py" \
    --config "$SCRIPT_DIR/config/pp_test.yaml" \
    "$@"
