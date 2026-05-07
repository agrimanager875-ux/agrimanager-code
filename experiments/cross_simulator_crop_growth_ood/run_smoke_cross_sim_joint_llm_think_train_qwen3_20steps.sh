#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_cross_sim_train_lib.sh"

RUN_NAME="smoke_cross_sim_joint_llm_think_train_qwen3_20steps_joint_format_reward_rerun"
MODEL_PATH="Qwen/Qwen3-4B-Instruct-2507"
RESULT_DIR="$SCRIPT_DIR/results/llm_train/${RUN_NAME}"
LOG_FILE="$SCRIPT_DIR/logs/${RUN_NAME}.log"

CONFIGS=(
    "$SCRIPT_DIR/config/smoke_cross_sim_wofost_llm_think.yaml"
    "$SCRIPT_DIR/config/smoke_cross_sim_gym_dssat_llm_think.yaml"
    "$SCRIPT_DIR/config/smoke_cross_sim_cycles_gym_llm_think.yaml"
)

mkdir -p "$SCRIPT_DIR/logs" "$RESULT_DIR"

cd "$cross_sim_project_root"

if [[ -f "$cross_sim_project_root/smoke_tests/gym_dssat/_activate_spack.sh" ]]; then
    # shellcheck disable=SC1091
    source "$cross_sim_project_root/smoke_tests/gym_dssat/_activate_spack.sh"
fi

for config in "${CONFIGS[@]}"; do
    cross_sim_build_dataset "$config"
done

TRAIN_FILES=()
VAL_FILES=()
for config in "${CONFIGS[@]}"; do
    DATASET_DIR="$(cross_sim_dataset_dir "$config")"
    TRAIN_FILES+=("$DATASET_DIR/train.parquet")
    VAL_FILES+=("$DATASET_DIR/val.parquet")
done

TRAIN_FILES_ARG="$(cross_sim_join_hydra_list "${TRAIN_FILES[@]}")"
VAL_FILES_ARG="$(cross_sim_join_hydra_list "${VAL_FILES[@]}")"

export WANDB_MODE=online
ray stop --force >/dev/null 2>&1 || true

exec bash entrypoints/train/train.sh \
    --log-file "$LOG_FILE" \
    --config-name agri_grpo \
    "data.train_files=${TRAIN_FILES_ARG}" \
    "data.val_files=${VAL_FILES_ARG}" \
    "data.max_prompt_length=2048" \
    "data.gen_batch_size=1" \
    "data.dataloader_num_workers=0" \
    "data.filter_overlong_prompts_workers=1" \
    "actor_rollout_ref.model.path=${MODEL_PATH}" \
    "actor_rollout_ref.rollout.n=1" \
    "actor_rollout_ref.rollout.tensor_model_parallel_size=1" \
    "actor_rollout_ref.rollout.gpu_memory_utilization=0.45" \
    "actor_rollout_ref.rollout.max_num_batched_tokens=4096" \
    "actor_rollout_ref.rollout.max_num_seqs=8" \
    "actor_rollout_ref.rollout.response_length=128" \
    "actor_rollout_ref.actor.ppo_mini_batch_size=1" \
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1" \
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1" \
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1" \
    "trainer.rollout_filter.enable=False" \
    "trainer.total_epochs=4" \
    "trainer.total_training_steps=20" \
    "trainer.val_before_train=True" \
    "trainer.test_freq=10" \
    "trainer.n_gpus_per_node=1" \
    "trainer.save_freq=-1" \
    "trainer.logger=[\"console\", \"wandb\"]" \
    "trainer.project_name=agrimanager_cross_simulator_crop_growth_ood" \
    "trainer.default_local_dir=${RESULT_DIR}" \
    "trainer.experiment_name=${RUN_NAME}"
