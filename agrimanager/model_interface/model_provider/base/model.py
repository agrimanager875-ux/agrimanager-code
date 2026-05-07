"""
Base model class for all model providers.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from .model_config import ModelConfig


class BaseModel(ABC):
    """Base class for all model providers."""

    def __init__(self, config: ModelConfig):
        """
        Initialize the model with configuration.

        Args:
            config: Model configuration instance
        """
        self.config = config

    @abstractmethod
    def generate(
        self,
        messages_list: List[List[Dict[str, str]]],
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: int = 2048,
        return_metadata: bool = False,
        **kwargs
    ) -> List[str]:
        """
        Generate responses for a batch of message lists.

        Args:
            messages_list: List of message lists, where each message list is in OpenAI format.
                          Example: [[{"role": "user", "content": "Hello"}], [{"role": "user", "content": "Hi"}]]
            temperature: Temperature parameter (default: 0.7)
            top_p: Top-p sampling parameter (default: 1.0)
            max_tokens: Maximum tokens to generate (default: 2048)
            return_metadata: If True, return dicts with content/reasoning instead of plain strings
            **kwargs: Additional provider-specific parameters

        Returns:
            List of generated response strings or metadata dicts
        """
        pass

    @abstractmethod
    def convert_messages(self, messages: List[Dict[str, str]]) -> Any:
        """
        Convert OpenAI format messages to provider-specific format.

        Args:
            messages: List of messages in OpenAI format
                     Example: [{"role": "user", "content": "Hello"}]

        Returns:
            Provider-specific message format
        """
        pass
