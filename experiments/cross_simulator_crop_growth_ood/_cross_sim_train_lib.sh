#!/usr/bin/env bash

cross_sim_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cross_sim_project_root="$(cd "$cross_sim_dir/../.." && pwd)"

cross_sim_dataset_id() {
    basename "$1" .yaml
}

cross_sim_dataset_dir() {
    local config="$1"
    local dataset_id
    dataset_id="$(cross_sim_dataset_id "$config")"
    printf '%s/data/%s\n' "$cross_sim_dir" "$dataset_id"
}

cross_sim_build_dataset() {
    local config="$1"
    local dataset_dir
    local dataset_id
    dataset_dir="$(cross_sim_dataset_dir "$config")"
    dataset_id="$(cross_sim_dataset_id "$config")"
    mkdir -p "$cross_sim_dir/logs"
    local needs_build=1
    local split_file
    for split_file in "$dataset_dir/train.parquet" "$dataset_dir/val.parquet" "$dataset_dir/test.parquet"; do
        if [[ 1 -eq 1 || ! -f "$split_file" || "$config" -nt "$split_file" ]]; then
            needs_build=1
            break
        fi
    done
    if [[ "${CROSS_SIM_FORCE_REBUILD_DATASETS:-0}" == "1" ]]; then
        needs_build=1
    fi
    if [[ "$needs_build" -eq 1 ]]; then
        bash entrypoints/dataset/build.sh \
            --config "$config" \
            --num-workers 1 \
            2>&1 | tee "$cross_sim_dir/logs/${dataset_id}_build_dataset.log"
    fi
}

cross_sim_join_hydra_list() {
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

cross_sim_run_train() {
    local run_name="$1"
    local train_configs_name="$2"
    local val_configs_name="$3"
    local -n train_configs_ref="$train_configs_name"
    local -n val_configs_ref="$val_configs_name"

    local run_name_effective="${run_name}${CROSS_SIM_RUN_NAME_SUFFIX:-}"
    local model_path="${CROSS_SIM_MODEL_PATH:-Qwen/Qwen3-4B-Instruct-2507}"
    local n_gpus_per_node="${CROSS_SIM_N_GPUS_PER_NODE:-2}"
    local ray_num_cpus="${CROSS_SIM_RAY_NUM_CPUS:-32}"
    local total_epochs="${CROSS_SIM_TOTAL_EPOCHS:-1}"
    local test_freq="${CROSS_SIM_TEST_FREQ:-10}"
    local save_freq="${CROSS_SIM_SAVE_FREQ:-100}"
    local result_dir="${CROSS_SIM_RESULT_DIR:-$cross_sim_dir/results/llm_train/${run_name_effective}}"
    local log_file="$cross_sim_dir/logs/${run_name_effective}.log"

    mkdir -p "$cross_sim_dir/logs" "$result_dir"

    cd "$cross_sim_project_root"

    if [[ -f "$cross_sim_project_root/smoke_tests/gym_dssat/_activate_spack.sh" ]]; then
        # shellcheck disable=SC1091
        source "$cross_sim_project_root/smoke_tests/gym_dssat/_activate_spack.sh"
    fi

    local config
    for config in "${train_configs_ref[@]}" "${val_configs_ref[@]}"; do
        cross_sim_build_dataset "$config"
    done

    local train_files=()
    for config in "${train_configs_ref[@]}"; do
        train_files+=("$(cross_sim_dataset_dir "$config")/train.parquet")
    done

    local val_files=()
    for config in "${val_configs_ref[@]}"; do
        val_files+=("$(cross_sim_dataset_dir "$config")/val.parquet")
    done

    local train_files_arg
    local val_files_arg
    train_files_arg="$(cross_sim_join_hydra_list "${train_files[@]}")"
    val_files_arg="$(cross_sim_join_hydra_list "${val_files[@]}")"

    exec bash entrypoints/train/train.sh \
        --log-file "$log_file" \
        --config-name agri_grpo \
        "data.train_files=${train_files_arg}" \
        "data.val_files=${val_files_arg}" \
        "data.max_prompt_length=2048" \
        "data.gen_batch_size=16" \
        "data.val_batch_size=48" \
        "data.max_response_length=512" \
        "actor_rollout_ref.model.path=${model_path}" \
        "actor_rollout_ref.rollout.n=4" \
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.7" \
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32" \
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=64" \
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=64" \
        "trainer.logger=[\"console\", \"wandb\"]" \
        "trainer.project_name=agrimanager_cross_simulator_crop_growth_ood" \
        "trainer.default_local_dir=${result_dir}" \
        "trainer.experiment_name=${run_name_effective}" \
        "trainer.total_epochs=${total_epochs}" \
        "trainer.test_freq=${test_freq}" \
        "trainer.n_gpus_per_node=${n_gpus_per_node}" \
        "trainer.save_freq=${save_freq}" \
        "trainer.max_actor_ckpt_to_keep=2" \
        "ray_kwargs.ray_init.num_cpus=${ray_num_cpus}" \
        "+ray_kwargs.ray_init.include_dashboard=False" \
        "+ray_kwargs.ray_init._temp_dir=/tmp/ray_${SLURM_JOB_ID:-${SLURM_JOBID:-$$}}"
}
