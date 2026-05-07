#!/bin/bash

# Model name from Hugging Face
# Reasoning LLM
# openai/gpt-oss-120b
# openai/gpt-oss-20b
# Qwen/Qwen3-30B-A3B-Thinking-2507

# LLM
# Qwen/Qwen3-30B-A3B-Instruct-2507

MODEL_NAME="Qwen/Qwen3-30B-A3B-Thinking-2507"

# Specify which GPUs to use (0 means first GPU, use "0,1,2,3" for multiple GPUs)
export CUDA_VISIBLE_DEVICES=0

# Automatically count the number of GPUs based on CUDA_VISIBLE_DEVICES
GPU_NUM=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

# Get the directory where this script is located
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

# Port for the vLLM server to listen on
VLLM_SERVER_PORT=8000

# Directory to store log files
LOG_DIR="$SCRIPT_DIR/logs"

# Create log directory if it doesn't exist
if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR"
fi

# Log file path (replaces / with _ in model name for valid filename)
LOG_PATH="$LOG_DIR/${MODEL_NAME//\//_}.log"
echo "=== Starting vLLM serve on $(date) ===" > $LOG_PATH
echo "Public endpoint: $(curl -s ifconfig.me):$VLLM_SERVER_PORT" >> $LOG_PATH

# Launch vLLM serve.
#   --tensor-parallel-size : number of GPUs used for tensor parallelism
#   --max-num-seqs         : max concurrent sequences
#   --async-scheduling     : enable async scheduling for throughput
#   --port                 : HTTP port for the server
#   --max-model-len        : max context length
#   --gpu-memory-utilization : fraction of GPU memory to allocate

vllm serve $MODEL_NAME \
    --tensor-parallel-size $GPU_NUM \
    --max-num-seqs 100 \
    --port $VLLM_SERVER_PORT \
    --max-model-len 40000 \
    --gpu-memory-utilization 0.95 \
    >> $LOG_PATH 2>&1
