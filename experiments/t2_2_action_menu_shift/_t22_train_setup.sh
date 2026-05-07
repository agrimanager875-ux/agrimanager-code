#!/usr/bin/env bash
# Shared setup for T2.2 action-menu-shift train scripts.
# Usage: source this file, then call setup_t22_run family_a|family_b no_think|think full|smoke.

setup_t22_run() {
    local family="${1:?family required}"
    local reasoning="${2:?reasoning required}"
    local profile="${3:?profile required}"

    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[1]}")" && pwd)"
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

    local config_stem train_splits default_work_dir default_model run_prefix
    local default_tmp_root="${T22_BASE_TMP:-${TMPDIR:-/tmp}}"
    case "$family" in
        family_a)
            config_stem="t22_family_a_${reasoning}"
            train_splits=(lnpkw lnpk)
            ;;
        family_b)
            config_stem="t22_family_b_${reasoning}"
            train_splits=(lnpk lw)
            ;;
        *)
            echo "Unknown T2.2 family: $family" >&2
            return 2
            ;;
    esac

    if [[ "$profile" == "smoke" ]]; then
        config_stem="${config_stem}_smoke"
        run_prefix="smoke_"
        default_work_dir="$default_tmp_root/agrimanager_t22_smoke"
        default_model="Qwen/Qwen2.5-0.5B-Instruct"
    else
        run_prefix=""
        default_work_dir="$default_tmp_root/agrimanager_t22_run"
        default_model="Qwen/Qwen3-4B-Instruct-2507"
    fi

    T22_WORK_DIR="${T22_WORK_DIR:-$default_work_dir}"
    DATASET_CONFIG="$SCRIPT_DIR/config/${config_stem}.yaml"
    RUN_NAME="${run_prefix}${family//_/-}_llm_${reasoning}_train"
    RUN_NAME="${RUN_NAME//-/_}"
    MODEL_PATH="${MODEL_PATH:-$default_model}"
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

    mkdir -p "$T22_WORK_DIR/logs" "$RESULT_DIR" "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE" "$RAY_TMP_DIR"
    cd "$PROJECT_ROOT"

    local dataset_files=() train_files=() split dataset_file
    for split in "${train_splits[@]}"; do
        train_files+=("$DATASET_DIR/train_${split}.parquet")
        dataset_files+=("$DATASET_DIR/train_${split}.parquet")
    done
    for split in lnpkw lnpk lnw ln lw; do
        dataset_files+=("$DATASET_DIR/val_${split}.parquet")
    done

    local needs_build=1
    for dataset_file in "${dataset_files[@]}"; do
        if [[ 1 -eq 1 || ! -f "$dataset_file" || "$DATASET_CONFIG" -nt "$dataset_file" ]]; then
            needs_build=1
            break
        fi
    done

    if [[ "$needs_build" -eq 1 ]]; then
        bash entrypoints/dataset/build.sh \
            --config "$DATASET_CONFIG" \
            --num-workers 1 \
            2>&1 | tee "$DATASET_BUILD_LOG"
    fi

    TRAIN_FILE="$(join_hydra_list "${train_files[@]}")"
    VAL_SET_OVERRIDES=(
        "+data.val_sets.lnpkw=$DATASET_DIR/val_lnpkw.parquet"
        "+data.val_sets.lnpk=$DATASET_DIR/val_lnpk.parquet"
        "+data.val_sets.lnw=$DATASET_DIR/val_lnw.parquet"
        "+data.val_sets.ln=$DATASET_DIR/val_ln.parquet"
        "+data.val_sets.lw=$DATASET_DIR/val_lw.parquet"
    )
}

join_hydra_list() {
    local first=1 item
    printf '['
    for item in "$@"; do
        [[ "$first" -eq 0 ]] && printf ','
        printf '%s' "$item"
        first=0
    done
    printf ']'
}
