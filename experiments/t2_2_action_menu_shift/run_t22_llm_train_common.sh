#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
T22_SMOKE="${T22_SMOKE:-0}"
T22_BASE_TMP="${T22_BASE_TMP:-${TMPDIR:-/tmp}}"
if [[ "$T22_SMOKE" -eq 1 ]]; then
    T22_WORK_DIR="${T22_WORK_DIR:-$T22_BASE_TMP/agrimanager_t22_smoke}"
    DEFAULT_MODEL="Qwen/Qwen2.5-0.5B-Instruct"
else
    T22_WORK_DIR="${T22_WORK_DIR:-$T22_BASE_TMP/agrimanager_t22_run}"
    DEFAULT_MODEL="Qwen/Qwen3-4B-Instruct-2507"
fi

DATASET_CONFIG="${T22_DATASET_CONFIG:?Set T22_DATASET_CONFIG to the dataset YAML path}"
RUN_NAME="${T22_RUN_NAME:?Set T22_RUN_NAME to the experiment name}"
TRAIN_SPLITS_TEXT="${T22_TRAIN_SPLITS:?Set T22_TRAIN_SPLITS, e.g. 'lnpkw lnpk'}"
MODEL_PATH="${MODEL_PATH:-$DEFAULT_MODEL}"
RAY_TMP_DIR="${T22_RAY_TMP_DIR:-/dev/shm/ray_t22_${SLURM_JOB_ID:-${SLURM_JOBID:-$$}}}"

export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$T22_WORK_DIR/hf_datasets_cache}"
export HF_HOME="${HF_HOME:-$T22_WORK_DIR/hf_home}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$T22_WORK_DIR/hf_home/hub}"
export WOFOST_GYM_PATH="${WOFOST_GYM_PATH:-$PROJECT_ROOT/../AgriManagerExternal/WOFOSTGym}"
export PYTHONPATH="$WOFOST_GYM_PATH:$WOFOST_GYM_PATH/pcse_gym:$WOFOST_GYM_PATH/pcse:${PYTHONPATH:-}"

DATASET_ID="$(basename "$DATASET_CONFIG" .yaml)"
DATASET_DIR="$SCRIPT_DIR/data/$DATASET_ID"
DATASET_BUILD_LOG="$T22_WORK_DIR/logs/${RUN_NAME}_build_dataset.log"
LOG_FILE="$T22_WORK_DIR/logs/${RUN_NAME}.log"
RESULT_DIR="$T22_WORK_DIR/results/llm_train/${RUN_NAME}"

join_hydra_list() {
    local first=1
    printf '['
    for item in "$@"; do
        if [[ "$first" -eq 0 ]]; then
            printf ','
        fi
        printf '%s' "$item"
        first=0
    done
    printf ']'
}

mkdir -p "$T22_WORK_DIR/logs" "$RESULT_DIR" "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE" "$RAY_TMP_DIR"
cd "$PROJECT_ROOT"

read -r -a TRAIN_SPLITS <<< "$TRAIN_SPLITS_TEXT"
DATASET_FILES=()
TRAIN_FILES=()
for split in "${TRAIN_SPLITS[@]}"; do
    TRAIN_FILES+=("$DATASET_DIR/train_${split}.parquet")
    DATASET_FILES+=("$DATASET_DIR/train_${split}.parquet")
done
for split in lnpkw lnpk lnw ln lw; do
    DATASET_FILES+=("$DATASET_DIR/val_${split}.parquet")
done

NEEDS_BUILD=1
for dataset_file in "${DATASET_FILES[@]}"; do
    if [[ 1 -eq 1 || ! -f "$dataset_file" || "$DATASET_CONFIG" -nt "$dataset_file" ]]; then
        NEEDS_BUILD=1
        break
    fi
done

if [[ "$NEEDS_BUILD" -eq 1 ]]; then
    bash entrypoints/dataset/build.sh \
        --config "$DATASET_CONFIG" \
        --num-workers 1 \
        2>&1 | tee "$DATASET_BUILD_LOG"
fi

VAL_SET_OVERRIDES=(
    "+data.val_sets.lnpkw=$DATASET_DIR/val_lnpkw.parquet"
    "+data.val_sets.lnpk=$DATASET_DIR/val_lnpk.parquet"
    "+data.val_sets.lnw=$DATASET_DIR/val_lnw.parquet"
    "+data.val_sets.ln=$DATASET_DIR/val_ln.parquet"
    "+data.val_sets.lw=$DATASET_DIR/val_lw.parquet"
)
TRAIN_FILES_ARG="$(join_hydra_list "${TRAIN_FILES[@]}")"

COMMON_ARGS=(
    --log-file "$LOG_FILE"
    --config-name agri_grpo
    "data.train_files=${TRAIN_FILES_ARG}"
    "data.val_files=null"
    "data.validation_axis=action_menu"
    "${VAL_SET_OVERRIDES[@]}"
    "actor_rollout_ref.model.path=${MODEL_PATH}"
    "+actor_rollout_ref.model.override_config.attn_implementation=eager"
    "actor_rollout_ref.model.use_remove_padding=False"
    "critic.model.path=${MODEL_PATH}"
    "+critic.model.override_config.attn_implementation=eager"
    "critic.model.use_remove_padding=False"
    "actor_rollout_ref.rollout.name=vllm"
    "trainer.logger=[\"console\", \"wandb\"]"
    "trainer.project_name=agrimanager_wofost_action_menu_shift"
    "trainer.default_local_dir=${RESULT_DIR}"
    "trainer.experiment_name=${RUN_NAME}"
    "+ray_kwargs.ray_init.include_dashboard=False"
    "+ray_kwargs.ray_init._temp_dir=${RAY_TMP_DIR}"
)

if [[ "$T22_SMOKE" -eq 1 ]]; then
    TRAIN_ARGS=(
        "data.gen_batch_size=1"
        "data.dataloader_num_workers=0"
        "data.filter_overlong_prompts_workers=1"
        "data.max_response_length=128"
        "actor_rollout_ref.rollout.n=1"
        "actor_rollout_ref.rollout.enforce_eager=True"
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1"
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.45"
        "actor_rollout_ref.rollout.max_num_batched_tokens=2048"
        "actor_rollout_ref.rollout.max_num_seqs=4"
        "actor_rollout_ref.rollout.response_length=128"
        "actor_rollout_ref.rollout.agent.num_workers=1"
        "actor_rollout_ref.actor.ppo_mini_batch_size=1"
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1"
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1"
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1"
        "critic.ppo_micro_batch_size_per_gpu=1"
        "trainer.total_epochs=1"
        "trainer.total_training_steps=2"
        "trainer.val_before_train=True"
        "trainer.test_freq=2"
        "trainer.n_gpus_per_node=1"
        "trainer.save_freq=0"
        "ray_kwargs.ray_init.num_cpus=8"
    )
else
    TRAIN_ARGS=(
        "data.gen_batch_size=16"
        "data.max_response_length=512"
        "actor_rollout_ref.rollout.n=4"
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.7"
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32"
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=64"
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=64"
        "trainer.total_epochs=2"
        "trainer.n_gpus_per_node=2"
        "trainer.save_freq=100"
        "ray_kwargs.ray_init.num_cpus=32"
    )
fi

exec bash entrypoints/train/train.sh "${COMMON_ARGS[@]}" "${TRAIN_ARGS[@]}" "$@"
