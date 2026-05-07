"""
Base configuration class for all model providers.
"""
from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class ModelConfig:
    """Base configuration for all model providers (model loading/initialization only)."""

    model_provider: str  # e.g., "openai", "openrouter", "vllm_server", "vllm_offline"
    
    # Model identifier (e.g., "gpt-4o", "meta-llama/Llama-2-7b-hf")
    model_name: str

    # Additional provider-specific parameters
    extra_params: Dict[str, Any] = field(default_factory=dict)
