"""
OpenRouter provider configuration.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from ..base.model_config import ModelConfig


@dataclass
class OpenRouterConfig(ModelConfig):
    """Configuration for OpenRouter's OpenAI-compatible chat completions API."""

    # OpenRouter endpoint. The /chat/completions path is added by the OpenAI SDK.
    base_url: str = "https://openrouter.ai/api/v1"

    # Defaults to OPENROUTER_API_KEY when omitted.
    api_key: Optional[str] = None

    # For parallel API calls.
    num_workers: int = 8

    # Request timeout in seconds.
    timeout: int = 120

    # Optional attribution headers recommended by OpenRouter.
    site_url: Optional[str] = None
    app_title: Optional[str] = "AgriManager"
    extra_headers: Dict[str, str] = field(default_factory=dict)

    # OpenRouter request body extensions.
    reasoning: Optional[Dict[str, Any]] = None
    reasoning_effort: Optional[str] = None
    include_reasoning: Optional[bool] = None
    provider: Optional[Dict[str, Any]] = None
    models: Optional[List[str]] = None
    route: Optional[str] = None
