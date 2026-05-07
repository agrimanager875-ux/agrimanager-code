"""
Google Gemini provider configuration.
"""
from dataclasses import dataclass
from typing import Optional
from ..base.model_config import ModelConfig


@dataclass
class GoogleConfig(ModelConfig):
    """Configuration for Google Gemini API."""

    # For multiprocessing API calls
    num_workers: int = 64

    # API key (can also be set via GOOGLE_API_KEY environment variable)
    api_key: Optional[str] = None

    # Request timeout in seconds
    timeout: int = 60
