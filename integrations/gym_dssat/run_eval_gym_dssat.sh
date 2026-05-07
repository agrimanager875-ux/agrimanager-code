#!/bin/bash
# ==========================================================================
# Master evaluation script for gym_dssat environment
# Runs all LLM models, gpt-oss models, and baselines (random, PPO)
# ==========================================================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
AGRIMANAGER_ROOT="${SCRIPT_DIR}/../.."

# --------------------------------------------------------------------------
# Activate spack environment (required for gym-dssat PDI subprocess)
# --------------------------------------------------------------------------
. "${AGRIMANAGER_ROOT}/spack/share/spack/setup-env.sh"
spack env activate gym-dssat-pdi
echo "Using Python: $(which python3)"

# --------------------------------------------------------------------------
# Common settings
# --------------------------------------------------------------------------
ENV_NAME=gym_dssat
DATASET_ID=maize_phase1
SPLIT=test
TEMPERATURE=0.7
MAX_TOKENS=512
RESULTS_BASE="${SCRIPT_DIR}/results/${ENV_NAME}/${DATASET_ID}/${SPLIT}"

# --------------------------------------------------------------------------
# PPO checkpoint path
# --------------------------------------------------------------------------
PPO_CHECKPOINT="${PPO_CHECKPOINT:-$SCRIPT_DIR/checkpoints/ppo_model.zip}"

# --------------------------------------------------------------------------
# Counters
# --------------------------------------------------------------------------
SUCCESS_COUNT=0
FAIL_COUNT=0
TOTAL_COUNT=0

run_llm_model() {
    local MODEL_PROVIDER=$1
    local MODEL_NAME=$2
    local CONFIG_FILE=$3

    TOTAL_COUNT=$((TOTAL_COUNT + 1))
    local OUTPUT_DIR="${RESULTS_BASE}/${MODEL_NAME}"

    echo ""
    echo "========================================================================"
    echo "[${TOTAL_COUNT}] Running: ${MODEL_NAME} (${MODEL_PROVIDER})"
    echo "  Config: ${CONFIG_FILE}"
    echo "  Output: ${OUTPUT_DIR}"
    echo "========================================================================"

    if [ ! -f "$CONFIG_FILE" ]; then
        echo "ERROR: Config file not found: ${CONFIG_FILE}"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        return 1
    fi

    cd "$AGRIMANAGER_ROOT"
    python3 -m agrimanager.rollout.inference.inference_rollout \
        --env-name "$ENV_NAME" \
        --dataset-id "$DATASET_ID" \
        --split "$SPLIT" \
        --model-config "$CONFIG_FILE" \
        --output-dir "$OUTPUT_DIR" \
        --temperature "$TEMPERATURE" \
        --max-tokens "$MAX_TOKENS"

    if [ $? -eq 0 ]; then
        echo "PASS: ${MODEL_NAME}"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo "FAIL: ${MODEL_NAME}"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
}

# ==========================================================================
# Qwen models (vllm_offline)
# ==========================================================================
CONFIGS_DIR="${AGRIMANAGER_ROOT}/agrimanager/model_interface/configs"

echo ""
echo "##########################################################################"
echo "# Qwen Models (vllm_offline)                                            #"
echo "##########################################################################"

run_llm_model vllm_offline "Qwen2.5-7B-Instruct" \
    "${CONFIGS_DIR}/vllm_offline/Qwen2.5-7B-Instruct.yaml"

run_llm_model vllm_offline "Qwen2.5-32B-Instruct" \
    "${CONFIGS_DIR}/vllm_offline/Qwen2.5-32B-Instruct.yaml"

run_llm_model vllm_offline "Qwen2.5-72B-Instruct" \
    "${CONFIGS_DIR}/vllm_offline/Qwen2.5-72B-Instruct.yaml"

run_llm_model vllm_offline "Qwen3-4B-Instruct-2507" \
    "${CONFIGS_DIR}/vllm_offline/Qwen3-4B-Instruct-2507.yaml"

run_llm_model vllm_offline "Qwen3-30B-A3B-Instruct-2507" \
    "${CONFIGS_DIR}/vllm_offline/Qwen3-30B-A3B-Instruct-2507.yaml"

run_llm_model vllm_offline "Qwen3-235B-A22B-Instruct-2507" \
    "${CONFIGS_DIR}/vllm_offline/Qwen3-235B-A22B-Instruct-2507.yaml"

# ==========================================================================
# gpt-oss models (openai provider with custom base_url)
# ==========================================================================
echo ""
echo "##########################################################################"
echo "# gpt-oss Models (openai)                                               #"
echo "##########################################################################"

run_llm_model openai "gpt-oss-20b_medium" \
    "${CONFIGS_DIR}/openai/gpt-oss-20b_medium.yaml"

run_llm_model openai "gpt-oss-120b_medium" \
    "${CONFIGS_DIR}/openai/gpt-oss-120b_medium.yaml"

# ==========================================================================
# Baselines
# ==========================================================================
echo ""
echo "##########################################################################"
echo "# Baselines                                                             #"
echo "##########################################################################"

# --- Random baseline ---
TOTAL_COUNT=$((TOTAL_COUNT + 1))
echo ""
echo "========================================================================"
echo "[${TOTAL_COUNT}] Running: Random Action Baseline"
echo "========================================================================"

cd "$AGRIMANAGER_ROOT"
python3 integrations/gym_dssat/run_random_inference.py \
    --env-name "$ENV_NAME" \
    --dataset-id "$DATASET_ID" \
    --split "$SPLIT" \
    --output-dir "${RESULTS_BASE}/random" \
    --seed 42

if [ $? -eq 0 ]; then
    echo "PASS: Random Baseline"
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
else
    echo "FAIL: Random Baseline"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

# --- PPO baseline ---
TOTAL_COUNT=$((TOTAL_COUNT + 1))
echo ""
echo "========================================================================"
echo "[${TOTAL_COUNT}] Running: PPO Baseline"
echo "========================================================================"

if [ -f "$PPO_CHECKPOINT" ]; then
    cd "$AGRIMANAGER_ROOT"
    python3 integrations/gym_dssat/run_ppo_inference.py \
        --env-name "$ENV_NAME" \
        --dataset-id "$DATASET_ID" \
        --split "$SPLIT" \
        --ppo-checkpoint "$PPO_CHECKPOINT" \
        --output-dir "${RESULTS_BASE}/ppo"

    if [ $? -eq 0 ]; then
        echo "PASS: PPO Baseline"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo "FAIL: PPO Baseline"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
else
    echo "WARNING: PPO checkpoint not found at ${PPO_CHECKPOINT}, skipping"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

# ==========================================================================
# Summary
# ==========================================================================
echo ""
echo "##########################################################################"
echo "# Evaluation Summary                                                    #"
echo "##########################################################################"
echo "Total runs:  ${TOTAL_COUNT}"
echo "Successful:  ${SUCCESS_COUNT}"
echo "Failed:      ${FAIL_COUNT}"
echo ""
echo "Results directory: ${RESULTS_BASE}/"
echo "##########################################################################"

if [ $FAIL_COUNT -gt 0 ]; then
    exit 1
else
    exit 0
fi
