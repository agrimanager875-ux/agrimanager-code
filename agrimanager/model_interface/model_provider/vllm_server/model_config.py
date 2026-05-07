"""
vLLM Server provider configuration.
"""
from dataclasses import dataclass
from typing import Optional
from ..base.model_config import ModelConfig


@dataclass
class VLLMServerConfig(ModelConfig):
    """Configuration for vLLM OpenAI-compatible API server."""

    # vLLM Server-specific parameters
    api_base: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"  # vLLM server default
    timeout: int = 120  # Request timeout in seconds (longer for vLLM)

    # For multiprocessing API calls
    num_workers: int = 4

    # For reasoning models
    reasoning_effort: Optional[str] = None  # "low", "medium", "high"
