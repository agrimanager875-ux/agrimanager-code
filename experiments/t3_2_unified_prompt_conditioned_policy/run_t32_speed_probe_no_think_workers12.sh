#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUN_NAME="t32_speed_probe_no_think_workers12"

WOFOST_DATASET_DIR="$SCRIPT_DIR/data/t32_unified_wofost"
DSSAT_YIELD_DATASET_DIR="$SCRIPT_DIR/data/t32_unified_gym_dssat_yield"
CYCLES_YIELD_DATASET_DIR="$SCRIPT_DIR/data/t32_unified_cycles_yield"
TRAIN_FILES="[$WOFOST_DATASET_DIR/train_wofost_lnpkw_yield.parquet,$WOFOST_DATASET_DIR/train_wofost_lnpkw_profit.parquet,$WOFOST_DATASET_DIR/train_wofost_lnpkw_water.parquet,$WOFOST_DATASET_DIR/train_wofost_lnpk_profit.parquet,$WOFOST_DATASET_DIR/train_wofost_lnpk_nutrient.parquet,$WOFOST_DATASET_DIR/train_wofost_lnw_yield.parquet,$DSSAT_YIELD_DATASET_DIR/train.parquet,$CYCLES_YIELD_DATASET_DIR/train.parquet]"

LOG_DIR="$SCRIPT_DIR/logs/speed_probe_${SLURM_JOB_ID:-local}_$(date +%Y%m%d_%H%M%S)"
RESULT_DIR="$SCRIPT_DIR/results/speed_probe/no_think_workers12"
LOG_FILE="$LOG_DIR/optimized_no_think_workers12_batched32768_resp64.log"

prepend_pythonpath() {
    local path="$1"
    if [[ -d "$path" ]]; then
        case ":${PYTHONPATH:-}:" in
            *":$path:"*) ;;
            *) export PYTHONPATH="$path${PYTHONPATH:+:$PYTHONPATH}" ;;
        esac
    fi
}

prepare_pythonpath() {
    prepend_pythonpath "$PROJECT_ROOT"
    prepend_pythonpath "$PROJECT_ROOT/verl"
}

prepare_dssat_runtime() {
    prepare_pythonpath

    if [[ -f "$PROJECT_ROOT/smoke_tests/gym_dssat/_activate_spack.sh" ]]; then
        # shellcheck disable=SC1091
        source "$PROJECT_ROOT/smoke_tests/gym_dssat/_activate_spack.sh"
    fi

    prepare_pythonpath

    local candidate
    for candidate in \
        "$PROJECT_ROOT/AgriManagerExternal/gym_dssat_pdi/gym-dssat-pdi" \
        "$PROJECT_ROOT/../AgriManagerExternal/gym_dssat_pdi/gym-dssat-pdi"
    do
        if [[ -d "$candidate/gym_dssat_pdi" ]]; then
            case ":${PYTHONPATH:-}:" in
                *":$candidate:"*) ;;
                *) export PYTHONPATH="$candidate:${PYTHONPATH:-}" ;;
            esac
            break
        fi
    done
}

mkdir -p "$LOG_DIR" "$RESULT_DIR"
cd "$PROJECT_ROOT"
prepare_dssat_runtime

echo "speed probe node=$(hostname) start=$(date)" | tee "$LOG_DIR/summary.txt"
echo "variant=optimized_no_think_workers12_batched32768_resp64" | tee -a "$LOG_DIR/summary.txt"

bash entrypoints/train/train.sh \
    --log-file "$LOG_FILE" \
    --config-name agri_grpo \
    "data.train_files=${TRAIN_FILES}" \
    "data.val_files=null" \
    "+data.env_config_overrides.require_think=False" \
    "+data.env_config_overrides.thinking_mode=think" \
    "+data.env_config_overrides.think_tag=tool_call" \
    "data.max_prompt_length=2048" \
    "data.gen_batch_size=16" \
    "data.max_response_length=64" \
    "data.dataloader_num_workers=4" \
    "actor_rollout_ref.model.path=Qwen/Qwen3-4B-Instruct-2507" \
    "actor_rollout_ref.rollout.n=4" \
    "actor_rollout_ref.rollout.gpu_memory_utilization=0.75" \
    "actor_rollout_ref.rollout.max_num_batched_tokens=32768" \
    "actor_rollout_ref.rollout.agent.num_workers=12" \
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16" \
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32" \
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32" \
    "trainer.logger=[\"console\"]" \
    "trainer.project_name=agrimanager_t32_speed_probe" \
    "trainer.experiment_name=${RUN_NAME}" \
    "trainer.default_local_dir=${RESULT_DIR}" \
    "trainer.total_epochs=1" \
    "trainer.total_training_steps=1" \
    "trainer.val_before_train=False" \
    "trainer.test_freq=0" \
    "trainer.n_gpus_per_node=1" \
    "trainer.save_freq=-1" \
    "trainer.rollout_filter.enable=False" \
    "ray_kwargs.ray_init.num_cpus=16" \
    "+ray_kwargs.ray_init.include_dashboard=False" \
    "+ray_kwargs.ray_init._temp_dir=/tmp/ray_t32_speed_probe_${SLURM_JOB_ID:-${SLURM_JOBID:-$$}}"

python - "$LOG_FILE" <<'PY' | tee -a "$LOG_DIR/summary.txt"
import datetime
import re
import sys

path = sys.argv[1]
last = None
with open(path, errors="ignore") as handle:
    for line in handle:
        if "step:" in line:
            last = line

print("log_path:", path)
if last is None:
    print("NO_STEP_LINE_FOUND")
else:
    print("step_line_found")
    keys = [
        "timing_s/step",
        "timing_s/gen",
        "timing_s/update_actor",
        "timing_s/old_log_prob",
        "response_length/mean",
        "response_length/max",
        "per_turn/batch_size",
        "perf/throughput",
    ]
    for key in keys:
        match = re.search(re.escape(key) + r":(?:np\.(?:float64|int32)\()?([-+0-9.eE]+)", last)
        if match:
            print(f"{key}={match.group(1)}")
print("end:", datetime.datetime.now())
PY
