"""
OpenAI provider configuration.
"""
from dataclasses import dataclass
from typing import Optional
from ..base.model_config import ModelConfig


@dataclass
class OpenAIConfig(ModelConfig):
    """Configuration for OpenAI API."""

    # For multiprocessing API calls
    num_workers: int = 64

    # Optional custom base URL for OpenAI-compatible endpoints
    base_url: Optional[str] = None

    # Optional API key override (default: uses OPENAI_API_KEY env var)
    api_key: Optional[str] = None