#!/bin/bash
# AgriManager training launcher with VERL
# Default config: entrypoints/train/config/agri_grpo.yaml
# Switch config:
#   bash entrypoints/train/train.sh --config-name agri_ppo
# Custom log file:
#   bash entrypoints/train/train.sh --log-file experiments/my_exp/logs/train.log
# Hydra overrides still work:
#   bash entrypoints/train/train.sh --config-name agri_ppo actor_rollout_ref.model.path=/path/to/model
#   bash entrypoints/train/train.sh trainer.total_epochs=1 trainer.val_before_train=True

set -euo pipefail
set -x

ulimit -n 65535

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
CONFIG_DIR="$PROJECT_DIR/entrypoints/train/config"
DEFAULT_CONFIG_NAME="agri_grpo"
CONFIG_NAME="$DEFAULT_CONFIG_NAME"
LOG_FILE=""
export VERL_CONFIG_PATH="$PROJECT_DIR/verl/verl/trainer/config"

# Parse script-level args and forward the rest to Hydra.
# Supports:
#   --config-name agri_ppo
#   --config-name=agri_ppo
FORWARD_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config-name=*)
            CONFIG_NAME="${1#*=}"
            shift
            ;;
        --config-name)
            if [[ $# -lt 2 ]]; then
                echo "Error: --config-name requires a value" >&2
                exit 1
            fi
            CONFIG_NAME="$2"
            shift 2
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

CONFIG_FILE="$CONFIG_DIR/${CONFIG_NAME}.yaml"
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: config '$CONFIG_NAME' not found at $CONFIG_FILE" >&2
    echo "Available configs:" >&2
    ls "$CONFIG_DIR"/*.yaml | xargs -n1 basename | sed 's/\.yaml$//' >&2
    exit 1
fi

# Extract experiment_name from CLI args, fallback to selected YAML's default.
EXPERIMENT_NAME=""
for arg in "${FORWARD_ARGS[@]}"; do
    if [[ "$arg" == trainer.experiment_name=* ]]; then
        EXPERIMENT_NAME="${arg#*=}"
        break
    fi
done

if [[ -z "$EXPERIMENT_NAME" ]]; then
    EXPERIMENT_NAME="$(
        python3 - "$CONFIG_FILE" "$CONFIG_NAME" <<'PY'
import sys

cfg_path = sys.argv[1]
fallback = sys.argv[2]

try:
    import yaml

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    name = ((cfg.get("trainer") or {}).get("experiment_name") or "").strip()
    print(name if name else fallback)
except Exception:
    print(fallback)
PY
    )"
fi

if [[ -z "$LOG_FILE" ]]; then
    LOG_DIR="$PROJECT_DIR/logs/train"
    mkdir -p "$LOG_DIR"
    LOG_FILE="$LOG_DIR/${EXPERIMENT_NAME}.log"
else
    if [[ "$LOG_FILE" != /* ]]; then
        LOG_FILE="$PROJECT_DIR/$LOG_FILE"
    fi
    mkdir -p "$(dirname "$LOG_FILE")"
fi

# Auto-detect GPU compute capability to speed up CUDA kernel compilation
if python3 -c "import torch" 2>/dev/null; then
    if CUDA_CAP="$(
        python3 -c "import torch; cap = torch.cuda.get_device_capability() if torch.cuda.is_available() else None; print(f'{cap[0]}.{cap[1]}' if cap else '')" 2>/dev/null
    )"; then
        if [ -n "$CUDA_CAP" ]; then
            export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-$CUDA_CAP}"
            echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
        fi
    fi
fi

cd "$PROJECT_DIR"

python3 -m agrimanager.adapter.trainer.main_ppo \
    --config-path="$CONFIG_DIR" \
    --config-name="$CONFIG_NAME" \
    "${FORWARD_ARGS[@]}" \
    2>&1 | tee "$LOG_FILE"
