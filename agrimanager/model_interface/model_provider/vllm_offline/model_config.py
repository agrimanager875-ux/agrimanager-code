"""
vLLM Offline provider configuration.
"""
from dataclasses import dataclass
from typing import Optional
from ..base.model_config import ModelConfig


@dataclass
class VLLMOfflineConfig(ModelConfig):
    """Configuration for vLLM offline mode (direct LLM usage)."""

    # Model loading parameters
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    trust_remote_code: bool = False
    dtype: str = "auto"  # "auto", "float16", "bfloat16"

    # Tokenizer settings
    tokenizer_mode: str = "auto"  # "auto", "slow"

    # Performance settings
    max_model_len: int = None  # Maximum sequence length (None for model default)

    # For reasoning models (e.g., o1, QwQ)
    reasoning_effort: Optional[str] = None  # "low", "medium", "high"

    # Enable reasoning extraction in output (splits into content/reasoning).
    # Set to True for models that always think (e.g., Qwen3 Thinking).
    # Auto-enabled when reasoning_effort or reasoning_start_token is set.
    extract_reasoning: bool = False

    # Reasoning extraction tokens (decoded text from chat template).
    # Used to locate the reasoning section in generated text.
    # Examples:
    #   Qwen3-style:  start="<think>",  end="</think>"
    #   gpt-oss-style: start="analysis", end="assistant"
    # When None, falls back to auto-detection of <think>/<analysis> XML tags.
    reasoning_start_token: Optional[str] = None
    reasoning_end_token: Optional[str] = None

    def __post_init__(self):
        """Validate vLLM-specific configuration."""
        if self.dtype not in ["auto", "float16", "bfloat16", "float32"]:
            raise ValueError(
                f"dtype must be one of: auto, float16, bfloat16, float32. Got {self.dtype}"
            )

        if self.gpu_memory_utilization < 0 or self.gpu_memory_utilization > 1:
            raise ValueError(
                f"gpu_memory_utilization must be between 0 and 1, got {self.gpu_memory_utilization}"
            )
