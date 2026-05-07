"""
vLLM Server provider model implementation.
"""
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm import tqdm
from ..base.model import BaseModel
from .model_config import VLLMServerConfig


class VLLMServerModel(BaseModel):
    """vLLM Server model provider implementation."""

    def __init__(self, config: VLLMServerConfig):
        """
        Initialize vLLM Server model.

        Args:
            config: vLLM Server configuration instance
        """
        super().__init__(config)
        self.config: VLLMServerConfig = config

        # Initialize OpenAI client pointing to vLLM server
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.api_base,
            timeout=config.timeout
        )

    def _call_vllm_api(self, args):
        """
        Helper function for multithreading API calls to vLLM server.

        Args:
            args: Tuple of (messages, config_dict)

        Returns:
            Generated response string
        """
        messages, config_dict = args

        # Build API call parameters
        api_params = {
            "model": config_dict["model_name"],
            "messages": messages,
            "temperature": config_dict["temperature"],
            "top_p": config_dict["top_p"],
            "max_tokens": config_dict["max_tokens"],
            **config_dict.get("extra_kwargs", {})
        }

        # Add reasoning_effort for reasoning models
        if config_dict.get("reasoning_effort"):
            api_params["reasoning_effort"] = config_dict["reasoning_effort"]

        # Use shared client (thread-safe)
        response = self.client.chat.completions.create(**api_params)

        message = response.choices[0].message
        content = message.content
        if not config_dict.get("return_metadata"):
            return content

        # First try to get reasoning from message attributes (OpenAI o1-style)
        reasoning = getattr(message, "reasoning", None)
        if reasoning is None:
            reasoning = getattr(message, "reasoning_content", None)

        # If no reasoning attribute, try to extract from content (Qwen3-style <think> tags)
        if reasoning is None and content:
            if "<think>" in content and "</think>" in content:
                start = content.find("<think>") + len("<think>")
                end = content.find("</think>", start)
                if end > start:
                    reasoning = content[start:end].strip()
            elif "</think>" in content:
                # Handle case where content starts with thinking (no opening tag)
                end = content.find("</think>")
                if end > 0:
                    reasoning = content[:end].strip()
            elif "<analysis>" in content and "</analysis>" in content:
                start = content.find("<analysis>") + len("<analysis>")
                end = content.find("</analysis>", start)
                if end > start:
                    reasoning = content[start:end].strip()

        return {"content": content, "reasoning": reasoning}

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
        Generate responses for a batch of message lists using multithreading.

        Args:
            messages_list: List of message lists in OpenAI format
            temperature: Temperature parameter (default: 0.7)
            top_p: Top-p sampling parameter (default: 1.0)
            max_tokens: Maximum tokens to generate (default: 2048)
            **kwargs: Additional vLLM-specific parameters

        Returns:
            List of generated response strings
        """
        if not messages_list:
            return []

        return_metadata = kwargs.pop("return_metadata", return_metadata)
        if not return_metadata and (
            getattr(self.config, "reasoning_effort", None)
            or getattr(self.config, "extract_reasoning", False)
            or getattr(self.config, "reasoning_start_token", None)
        ):
            return_metadata = True

        # Prepare config dict
        config_dict = {
            "model_name": self.config.model_name,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "extra_kwargs": kwargs,
            "reasoning_effort": self.config.reasoning_effort,
            "return_metadata": return_metadata,
        }

        # Prepare arguments for multithreading
        args_list = [(messages, config_dict) for messages in messages_list]

        # Use ThreadPoolExecutor for parallel API calls
        results = [None] * len(messages_list)
        with ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(self._call_vllm_api, args): idx
                for idx, args in enumerate(args_list)
            }

            # Collect results as they complete with progress bar
            for future in tqdm(as_completed(future_to_idx), total=len(messages_list), desc="Generating"):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    print(f"Error processing request {idx}: {str(e)}")
                    results[idx] = f"Error: {str(e)}"

        return results

    def convert_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Convert OpenAI format messages to vLLM format (same as OpenAI).

        Args:
            messages: List of messages in OpenAI format

        Returns:
            Same messages (vLLM uses OpenAI-compatible format)
        """
        return messages
