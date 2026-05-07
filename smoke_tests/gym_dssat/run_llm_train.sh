#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

usage() {
    cat <<'EOF'
Usage: bash run_llm_train.sh [maize|rice|cotton]
       GYM_DSSAT_CROP=maize bash run_llm_train.sh
EOF
}

CROP="${GYM_DSSAT_CROP:-maize}"
if [[ $# -gt 0 ]]; then
    case "$1" in
        --crop)
            CROP="${2:?Missing value for --crop}"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            CROP="$1"
            shift
            ;;
    esac
fi
if [[ $# -gt 0 ]]; then
    echo "Unexpected argument: $1" >&2
    usage >&2
    exit 1
fi
CROP="${CROP,,}"
case "$CROP" in
    maize|rice|cotton) ;;
    *)
        echo "Unsupported crop '$CROP'. Choose one of: maize, rice, cotton." >&2
        exit 1
        ;;
esac

DATASET_ID="gym_dssat_smoke_llm_${CROP}"
RUN_NAME="${DATASET_ID}_train"
DATASET_CONFIG="$SCRIPT_DIR/config/${DATASET_ID}.yaml"
DATASET_DIR="$SCRIPT_DIR/data/${DATASET_ID}"
TRAIN_FILE="$DATASET_DIR/train.parquet"
VAL_FILE="$DATASET_DIR/val.parquet"
DATASET_BUILD_LOG="$SCRIPT_DIR/logs/${DATASET_ID}_dataset_build.log"
LOG_FILE="$SCRIPT_DIR/logs/${RUN_NAME}.log"
# Ray places Unix sockets under RAY_TMPDIR, and AF_UNIX paths must stay under
# 108 bytes. Keep the live Ray session path short, then inspect it via the
# session_latest symlink recorded below.
RAY_LOG_DIR="${RAY_LOG_DIR:-$HOME/ray}"
RAY_LOG_POINTER="$SCRIPT_DIR/logs/${RUN_NAME}_ray_tmpdir.txt"
RESULT_DIR="$SCRIPT_DIR/results/llm_train/${RUN_NAME}"
MODEL_ID="Qwen/Qwen2.5-0.5B-Instruct"
MODEL_PATH="$MODEL_ID"

mkdir -p "$SCRIPT_DIR/logs" "$RAY_LOG_DIR" "$RESULT_DIR"
printf '%s\n' "$RAY_LOG_DIR" > "$RAY_LOG_POINTER"

# shellcheck disable=SC1091
source "$SCRIPT_DIR/_activate_spack.sh"

cd "$PROJECT_ROOT"

if [[ 1 -eq 1 || ! -f "$TRAIN_FILE" || ! -f "$VAL_FILE" ]]; then
    bash entrypoints/dataset/build.sh \
        --config "$DATASET_CONFIG" \
        --num-workers 1 \
        2>&1 | tee "$DATASET_BUILD_LOG"
fi

ray stop --force >/dev/null 2>&1 || true

# Keep the smoke test friendly to login-node style limits.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MALLOC_ARENA_MAX=2
export VLLM_LOGGING_LEVEL=INFO
export RAY_DEDUP_LOGS=0
export RAY_BACKEND_LOG_LEVEL=debug
export RAY_TMPDIR="$RAY_LOG_DIR"
export PYTHONUNBUFFERED=1
export AGRIMANAGER_USE_PLAIN_DATALOADER=1
unset AGRIMANAGER_DISABLE_AGENT_LOOP

RAY_NUM_CPUS=2
if python3 -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)'; then
    RAY_NUM_CPUS=4
fi

LOCAL_MODEL_PATH="$(python3 - <<'PY'
from huggingface_hub import snapshot_download

repo_id = "Qwen/Qwen2.5-0.5B-Instruct"
try:
    print(snapshot_download(repo_id=repo_id, local_files_only=True))
except Exception:
    print("")
PY
)"
if [[ -n "$LOCAL_MODEL_PATH" ]]; then
    MODEL_PATH="$LOCAL_MODEL_PATH"
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
fi

if ! python3 -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)'; then
    echo "gym_dssat llm smoke train requires a CUDA-visible GPU, but torch.cuda.is_available() is false." >&2
    exit 1
fi

exec bash entrypoints/train/train.sh \
    --log-file "$LOG_FILE" \
    --config-name agri_grpo \
    "data.train_files=${TRAIN_FILE}" \
    "data.val_files=${VAL_FILE}" \
    "data.gen_batch_size=1" \
    "data.max_prompt_length=4096" \
    "data.dataloader_num_workers=0" \
    "data.filter_overlong_prompts=False" \
    "data.filter_overlong_prompts_workers=1" \
    "data.shuffle=False" \
    "actor_rollout_ref.model.path=${MODEL_PATH}" \
    "actor_rollout_ref.model.use_remove_padding=False" \
    "+actor_rollout_ref.model.override_config.attn_implementation=sdpa" \
    "critic.enable=False" \
    "actor_rollout_ref.rollout.name=vllm" \
    "actor_rollout_ref.rollout.n=1" \
    "actor_rollout_ref.rollout.prompt_length=4096" \
    "actor_rollout_ref.rollout.tensor_model_parallel_size=1" \
    "actor_rollout_ref.rollout.gpu_memory_utilization=0.45" \
    "actor_rollout_ref.rollout.max_num_batched_tokens=2048" \
    "actor_rollout_ref.rollout.max_num_seqs=8" \
    "actor_rollout_ref.rollout.response_length=128" \
    "actor_rollout_ref.rollout.free_cache_engine=False" \
    "actor_rollout_ref.rollout.agent.num_workers=1" \
    "actor_rollout_ref.actor.ppo_mini_batch_size=1" \
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1" \
    "actor_rollout_ref.actor.use_torch_compile=False" \
    "actor_rollout_ref.actor.fsdp_config.use_torch_compile=False" \
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1" \
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1" \
    "actor_rollout_ref.ref.fsdp_config.use_torch_compile=False" \
    "trainer.n_gpus_per_node=1" \
    "trainer.per_turn_training=False" \
    "ray_kwargs.ray_init.num_cpus=${RAY_NUM_CPUS}" \
    "+ray_kwargs.ray_init.object_store_memory=1073741824" \
    "+ray_kwargs.ray_init.include_dashboard=False" \
    "trainer.total_epochs=1" \
    "trainer.total_training_steps=5" \
    "trainer.val_before_train=False" \
    "trainer.test_freq=10" \
    "trainer.save_freq=-1" \
    'trainer.logger=["console"]' \
    "trainer.default_local_dir=${RESULT_DIR}" \
    "trainer.experiment_name=${RUN_NAME}"
