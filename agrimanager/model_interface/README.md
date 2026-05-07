# Unified Model Interface

Unified interface for multiple LLM providers (OpenAI, OpenRouter, Google, vLLM Server, vLLM Offline) with consistent API and automatic batch generation.

## Features

- **Unified API**: Single `generate()` method for all providers
- **Lazy Loading**: Only imports what you use (no vLLM import when using OpenAI)
- **Batch Generation**: Automatic parallel processing with progress bars
- **YAML Configuration**: Simple config files for each model
- **Dynamic Parameters**: Set temperature, top_p, max_tokens at runtime

## Installation

For OpenAI models, set your API key:
```bash
read -rsp "OpenAI API key: " OPENAI_API_KEY
export OPENAI_API_KEY
```

For Google models, set your credentials:
```bash
read -rsp "Gemini API key: " GEMINI_API_KEY
export GEMINI_API_KEY
```

For OpenRouter models, set your API key:
```bash
read -rsp "OpenRouter API key: " OPENROUTER_API_KEY
export OPENROUTER_API_KEY
```

## Quick Start

### 1. Create Config

```yaml
`configs/openai/gpt-4.1-mini.yaml`:
```yaml
model_provider: "openai"
model_name: "gpt-4.1-mini"
num_workers: 64
```

### 2. Use Model

```python
from model_factory import create_model

# Create model from config
model = create_model("configs/openai/gpt-4.1-mini.yaml")

# Prepare messages (OpenAI format)
messages_list = [
    [{"role": "user", "content": "What is the capital of France?"}],
    [{"role": "user", "content": "What is 2+2?"}],
]

# Generate with custom parameters
responses = model.generate(
    messages_list,
    temperature=0.7,
    max_tokens=1024
)
```

## Configuration Examples

### OpenAI
```yaml
model_provider: "openai"
model_name: "gpt-4.1-mini"
num_workers: 64
```

### Google
```yaml
model_provider: "google"
model_name: "gemini-2.0-flash"
num_workers: 64
```

### OpenRouter
```yaml
model_provider: "openrouter"
model_name: "moonshotai/kimi-k2.6"
num_workers: 8
app_title: "AgriManager"
```

### vLLM Server
```yaml
model_provider: "vllm_server"
model_name: "Qwen/Qwen2.5-3B-Instruct"
api_base: "http://localhost:8000/v1"
num_workers: 64
```

Start a local server with a concrete model name:
```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-3B-Instruct \
  --port 8000
```

Alternatively, use the launch script:
```bash
bash server/bash_launch.sh
```

### vLLM Offline
```yaml
model_provider: "vllm_offline"
model_name: "Qwen/Qwen2.5-3B-Instruct"
tensor_parallel_size: 1
gpu_memory_utilization: 0.9
dtype: "auto"
```

## API Reference

```python
model.generate(
    messages_list: List[List[Dict[str, str]]],
    temperature: float = 0.7,
    top_p: float = 1.0,
    max_tokens: int = 2048,
    **kwargs
) -> List[str]
```

Message format (OpenAI standard):
```python
{"role": "system|user|assistant", "content": "..."}
```

## Testing

```bash
python test.py
```

Edit `test.py` to test different providers.


## Next Steps
- Support reasoning models.
- Support Claude models.
