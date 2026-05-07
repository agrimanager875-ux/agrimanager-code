#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

RUN_NAME="wofost_gym_smoke_nn_train"
DATASET_CONFIG="$SCRIPT_DIR/config/nn.yaml"
DATASET_DIR="$SCRIPT_DIR/data/wofost_gym_smoke_nn"
TRAIN_FILE="$DATASET_DIR/train.parquet"
VAL_FILE="$DATASET_DIR/val.parquet"
DATASET_BUILD_LOG="$SCRIPT_DIR/logs/wofost_gym_smoke_nn_dataset_build.log"
LOG_FILE="$SCRIPT_DIR/logs/${RUN_NAME}.log"
SAVE_ROOT="$SCRIPT_DIR/results/nn_train/${RUN_NAME}"

mkdir -p "$SCRIPT_DIR/logs" "$SAVE_ROOT"

cd "$PROJECT_ROOT"

if [[ 1 -eq 1 || ! -f "$TRAIN_FILE" || ! -f "$VAL_FILE" ]]; then
    bash entrypoints/dataset/build.sh \
        --config "$DATASET_CONFIG" \
        --num-workers 1 \
        2>&1 | tee "$DATASET_BUILD_LOG"
fi

bash entrypoints/train/nn_train.sh \
    "data.train_files=${TRAIN_FILE}" \
    "data.val_files=${VAL_FILE}" \
    "output.save_folder=${SAVE_ROOT}" \
    "agent.exp_name=${RUN_NAME}" \
    "agent.track=false" \
    "agent.total_timesteps=32" \
    "agent.num_envs=1" \
    "agent.num_steps=4" \
    "agent.num_minibatches=1" \
    "agent.checkpoint_mode=timestep" \
    "+agent.checkpoint_frequency=16" \
    "runtime.device=cpu" \
    "runtime.validation.enabled=true" \
    "runtime.validation.val_before_train=true" \
    "runtime.validation.frequency=16" \
    "runtime.validation.log_to_wandb=false" \
    2>&1 | tee "$LOG_FILE"
