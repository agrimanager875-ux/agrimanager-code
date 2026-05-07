#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

usage() {
    cat <<'EOF'
Usage: bash smoke_tests/gym_dssat/run_runtime_smoke.sh [--crop maize|rice|cotton] [--timeout seconds]

Checks the live DSSAT runtime in three explicit stages:
  1. import the gym_dssat_pdi bridge,
  2. construct DSSATEnv(config),
  3. run env.reset().

If the repo does not contain a packed DSSAT runtime, set:
  export DSSAT_GYM_PATH=/path/to/spack/gym-dssat-pdi
  export DSSAT_PDI_BRIDGE_DIR=/path/to/gym-dssat-pdi/source
EOF
}

CROP="${GYM_DSSAT_CROP:-maize}"
TIMEOUT_SECONDS="${GYM_DSSAT_RUNTIME_SMOKE_TIMEOUT:-240}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --crop)
            CROP="${2:?Missing value for --crop}"
            shift 2
            ;;
        --timeout)
            TIMEOUT_SECONDS="${2:?Missing value for --timeout}"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        maize|rice|cotton)
            CROP="$1"
            shift
            ;;
        *)
            echo "Unexpected argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

CROP="${CROP,,}"
case "$CROP" in
    maize|rice|cotton) ;;
    *)
        echo "Unsupported crop '$CROP'. Choose one of: maize, rice, cotton." >&2
        exit 1
        ;;
esac

mkdir -p "$SCRIPT_DIR/logs" "$SCRIPT_DIR/results/runtime_smoke"

# shellcheck disable=SC1091
source "$SCRIPT_DIR/_activate_spack.sh"

PYTHON_BIN="${AGRIMANAGER_PYTHON:-python}"
LOG_FILE="$SCRIPT_DIR/logs/gym_dssat_runtime_smoke_${CROP}.log"

cd "$PROJECT_ROOT"

if ! command -v timeout >/dev/null 2>&1; then
    echo "Error: GNU timeout is required for this runtime smoke test." >&2
    exit 2
fi

set +e
timeout "${TIMEOUT_SECONDS}s" "$PYTHON_BIN" - "$CROP" "$SCRIPT_DIR/results/runtime_smoke" <<'PY' 2>&1 | tee "$LOG_FILE"
import importlib
import os
import sys
import time
from pathlib import Path

crop = sys.argv[1]
output_root = Path(sys.argv[2])
output_root.mkdir(parents=True, exist_ok=True)

print(f"DSSAT_RUNTIME_SMOKE_STAGE bridge_import_start crop={crop}", flush=True)
print(f"DSSAT_GYM_PATH={os.environ.get('DSSAT_GYM_PATH')}", flush=True)
print(f"DSSAT_PDI_BRIDGE_DIR={os.environ.get('DSSAT_PDI_BRIDGE_DIR')}", flush=True)
bridge = importlib.import_module("gym_dssat_pdi")
print(f"DSSAT_RUNTIME_SMOKE_STAGE bridge_import_ok file={getattr(bridge, '__file__', None)}", flush=True)

from agrimanager.env.gym_dssat.env import DSSATEnv
from agrimanager.env.gym_dssat.env_config import DSSATEnvConfig

env_id = {
    "maize": "maize-irrigation-v0",
    "rice": "rice-irrigation-v0",
    "cotton": "cotton-irrigation-v0",
}.get(crop, "maize-irrigation-v0")
cfg = DSSATEnvConfig(
    env_id=env_id,
    crop_name=crop,
    llm_mode=True,
    decision_interval=10,
    seed=7300,
    save_folder=str(output_root / crop),
    output_vars=["dap", "grnwt", "nstres", "swfac"],
)

print("DSSAT_RUNTIME_SMOKE_STAGE construct_start", flush=True)
t0 = time.time()
env = DSSATEnv(cfg)
print(f"DSSAT_RUNTIME_SMOKE_STAGE construct_ok seconds={time.time() - t0:.2f}", flush=True)

print("DSSAT_RUNTIME_SMOKE_STAGE reset_start", flush=True)
t0 = time.time()
prompt, info = env.reset()
print(
    "DSSAT_RUNTIME_SMOKE_STAGE reset_ok "
    f"seconds={time.time() - t0:.2f} prompt_type={type(prompt).__name__} "
    f"info_keys={sorted((info or {}).keys())[:8]}",
    flush=True,
)
close = getattr(env, "close", None)
if callable(close):
    close()
print("DSSAT_RUNTIME_SMOKE_OK", flush=True)
PY
status=${PIPESTATUS[0]}
set -e

if [[ "$status" -eq 124 ]]; then
    echo "DSSAT_RUNTIME_SMOKE_TIMEOUT after ${TIMEOUT_SECONDS}s. See $LOG_FILE" >&2
elif [[ "$status" -ne 0 ]]; then
    echo "DSSAT_RUNTIME_SMOKE_FAILED status=$status. See $LOG_FILE" >&2
fi
exit "$status"
