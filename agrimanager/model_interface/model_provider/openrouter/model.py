"""
OpenRouter provider model implementation.
"""
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from openai import OpenAI
from tqdm import tqdm

from ..base.model import BaseModel
from .model_config import OpenRouterConfig


class OpenRouterModel(BaseModel):
    """OpenRouter model provider implementation."""

    def __init__(self, config: OpenRouterConfig):
        """
        Initialize OpenRouter model.

        Args:
            config: OpenRouter configuration instance
        """
        super().__init__(config)
        self.config: OpenRouterConfig = config

        api_key = config.api_key or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenRouter API key not found. "
                "Set it in config or OPENROUTER_API_KEY environment variable."
            )

        self.default_headers = self._build_default_headers(config)

        client_kwargs = {
            "api_key": api_key,
            "base_url": config.base_url,
            "timeout": config.timeout,
        }
        if self.default_headers:
            client_kwargs["default_headers"] = self.default_headers
        self.client = OpenAI(**client_kwargs)

    @staticmethod
    def _build_default_headers(config: OpenRouterConfig) -> Dict[str, str]:
        headers = {}

        site_url = (
            config.site_url
            or os.getenv("OPENROUTER_HTTP_REFERER")
            or os.getenv("OPENROUTER_SITE_URL")
        )
        if site_url:
            headers["HTTP-Referer"] = site_url

        app_title = config.app_title or os.getenv("OPENROUTER_APP_TITLE")
        if app_title:
            headers["X-OpenRouter-Title"] = app_title

        headers.update(config.extra_headers or {})
        return headers

    @staticmethod
    def _read_extra_field(obj: Any, name: str) -> Any:
        if isinstance(obj, dict):
            return obj.get(name)

        value = getattr(obj, name, None)
        if value is not None:
            return value

        model_extra = getattr(obj, "model_extra", None)
        if isinstance(model_extra, dict):
            return model_extra.get(name)

        return None

    @staticmethod
    def _extract_tagged_reasoning(content: Optional[str]) -> tuple[Optional[str], str]:
        if not content:
            return None, content or ""

        for tag in ("think", "analysis"):
            match = re.search(
                rf"<{tag}>\s*(.*?)\s*</{tag}>",
                content,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if not match:
                continue
            reasoning = match.group(1).strip() or None
            clean_content = re.sub(
                rf"<{tag}>\s*.*?\s*</{tag}>",
                "",
                content,
                count=1,
                flags=re.IGNORECASE | re.DOTALL,
            ).strip()
            return reasoning, clean_content

        return None, content

    @staticmethod
    def _stringify_reasoning_details(reasoning_details: Any) -> Optional[str]:
        if reasoning_details is None:
            return None
        if isinstance(reasoning_details, str):
            return reasoning_details
        if isinstance(reasoning_details, list):
            return "\n".join(str(item) for item in reasoning_details)
        return str(reasoning_details)

    def _build_extra_body(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        extra_body = {}
        configured_params = dict(self.config.extra_params or {})

        if self.config.provider is not None:
            configured_params["provider"] = self.config.provider
        if self.config.models is not None:
            configured_params["models"] = self.config.models
        if self.config.route is not None:
            configured_params["route"] = self.config.route
        if self.config.include_reasoning is not None:
            configured_params["include_reasoning"] = self.config.include_reasoning
        if self.config.reasoning is not None:
            configured_params["reasoning"] = self.config.reasoning
        elif self.config.reasoning_effort:
            configured_params["reasoning"] = {"effort": self.config.reasoning_effort}

        explicit_extra_body = kwargs.pop("extra_body", None)
        if explicit_extra_body:
            extra_body.update(explicit_extra_body)

        extra_body.update(configured_params)
        extra_body.update(kwargs)
        return extra_body

    def _call_openrouter_api(self, args):
        """
        Helper function for multithreading API calls to OpenRouter.

        Args:
            args: Tuple of (messages, config_dict)

        Returns:
            Generated response string or metadata dict.
        """
        messages, config_dict = args
        extra_kwargs = dict(config_dict.get("extra_kwargs", {}))

        api_params = {
            "model": config_dict["model_name"],
            "messages": messages,
            "temperature": config_dict["temperature"],
            "top_p": config_dict["top_p"],
            "max_tokens": config_dict["max_tokens"],
        }

        extra_headers = extra_kwargs.pop("extra_headers", None)
        if extra_headers:
            api_params["extra_headers"] = extra_headers

        extra_query = extra_kwargs.pop("extra_query", None)
        if extra_query:
            api_params["extra_query"] = extra_query

        request_timeout = extra_kwargs.pop("timeout", None)
        if request_timeout is not None:
            api_params["timeout"] = request_timeout

        extra_body = self._build_extra_body(extra_kwargs)
        if extra_body:
            api_params["extra_body"] = extra_body

        response = self.client.chat.completions.create(**api_params)

        message = response.choices[0].message
        content = self._read_extra_field(message, "content") or ""
        if not config_dict.get("return_metadata"):
            return content

        reasoning = self._read_extra_field(message, "reasoning")
        if reasoning is None:
            reasoning = self._read_extra_field(message, "reasoning_content")
        if reasoning is None:
            reasoning = self._stringify_reasoning_details(
                self._read_extra_field(message, "reasoning_details")
            )
        if reasoning is None:
            reasoning, content = self._extract_tagged_reasoning(content)

        return {"content": content, "reasoning": reasoning}

    def generate(
        self,
        messages_list: List[List[Dict[str, str]]],
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: int = 4096,
        return_metadata: bool = False,
        **kwargs
    ) -> List[str]:
        """
        Generate responses for a batch of message lists using multithreading.

        Args:
            messages_list: List of message lists in OpenAI format
            temperature: Temperature parameter (default: 1.0)
            top_p: Top-p sampling parameter (default: 1.0)
            max_tokens: Maximum tokens to generate (default: 4096)
            return_metadata: If True, return dicts with content/reasoning
            **kwargs: Additional OpenRouter request body parameters

        Returns:
            List of generated response strings or metadata dicts
        """
        if not messages_list:
            return []

        return_metadata = kwargs.pop("return_metadata", return_metadata)
        if not return_metadata and (
            self.config.reasoning is not None
            or self.config.reasoning_effort
            or self.config.include_reasoning
        ):
            return_metadata = True

        config_dict = {
            "model_name": self.config.model_name,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "extra_kwargs": kwargs,
            "return_metadata": return_metadata,
        }

        args_list = [(messages, config_dict) for messages in messages_list]

        results = [None] * len(messages_list)
        with ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
            future_to_idx = {
                executor.submit(self._call_openrouter_api, args): idx
                for idx, args in enumerate(args_list)
            }

            for future in tqdm(
                as_completed(future_to_idx),
                total=len(messages_list),
                desc="Generating",
            ):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    print(f"Error processing request {idx}: {str(e)}")
                    results[idx] = f"Error: {str(e)}"

        return results

    def convert_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Convert OpenAI format messages to OpenRouter format.

        OpenRouter chat completions are OpenAI-compatible, so no conversion is
        required for the current AgriManager message format.
        """
        return messages
