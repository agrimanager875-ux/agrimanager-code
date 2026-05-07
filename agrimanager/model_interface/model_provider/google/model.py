"""
Google Gemini provider model implementation.
"""
import os
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
try:
    from google import genai
    from google.genai import types
except ImportError:
    raise ImportError(
        "google-genai is not installed. "
        "Please install it with: pip install google-genai"
    )

from ..base.model import BaseModel
from .model_config import GoogleConfig


class GoogleModel(BaseModel):
    """Google Gemini model provider implementation."""

    def __init__(self, config: GoogleConfig):
        """
        Initialize Google Gemini model.

        Args:
            config: Google configuration instance
        """
        super().__init__(config)
        self.config: GoogleConfig = config

        # Get API key from config or environment
        api_key = config.api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "Google API key not found. "
                "Set it in config or GEMINI_API_KEY environment variable."
            )

        # Initialize the client
        self.client = genai.Client(api_key=api_key)

    def _call_gemini_api(self, args):
        """
        Helper function for multithreading API calls to Gemini.

        Args:
            args: Tuple of (messages, config_dict)

        Returns:
            Generated response string
        """
        messages, config_dict = args

        # Convert messages to Gemini format and extract system instruction
        gemini_contents, system_instruction = self.convert_messages(messages)

        # Build generation config - parameters go directly in GenerateContentConfig
        config_params = {
            "temperature": config_dict["temperature"],
            "top_p": config_dict["top_p"],
            "max_output_tokens": config_dict["max_tokens"],
        }

        # Add system instruction if present
        if system_instruction:
            config_params["system_instruction"] = system_instruction

        gen_config = types.GenerateContentConfig(**config_params)

        # Generate response
        response = self.client.models.generate_content(
            model=self.config.model_name,
            contents=gemini_contents,
            config=gen_config,
        )

        content = response.text
        if not config_dict.get("return_metadata"):
            return content

        reasoning = None
        candidates = getattr(response, "candidates", None)
        if candidates:
            for candidate in candidates:
                content_obj = getattr(candidate, "content", None)
                parts = getattr(content_obj, "parts", None) if content_obj else None
                if not parts:
                    continue
                for part in parts:
                    thought = getattr(part, "thought", None)
                    if thought:
                        reasoning = str(thought)
                        break
                    reason = getattr(part, "reasoning", None)
                    if reason:
                        reasoning = str(reason)
                        break
                if reasoning is not None:
                    break

        return {"content": content, "reasoning": reasoning}

    def generate(
        self,
        messages_list: List[List[Dict[str, str]]],
        temperature: float = 1.0,
        top_p: float = 0.95,
        max_tokens: int = 8192,
        return_metadata: bool = False,
        **kwargs
    ) -> List[str]:
        """
        Generate responses for a batch of message lists using multithreading.

        Args:
            messages_list: List of message lists in OpenAI format
            temperature: Temperature parameter (default: 1.0)
            top_p: Top-p sampling parameter (default: 0.95)
            max_tokens: Maximum tokens to generate (default: 8192)
            **kwargs: Additional Gemini-specific parameters

        Returns:
            List of generated response strings
        """
        if not messages_list:
            return []

        return_metadata = kwargs.pop("return_metadata", return_metadata)
        if not return_metadata and getattr(self.config, "reasoning_effort", None):
            return_metadata = True

        # Prepare config dict
        config_dict = {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "extra_kwargs": kwargs,
            "return_metadata": return_metadata,
        }

        # Prepare arguments for multithreading
        args_list = [(messages, config_dict) for messages in messages_list]

        # Use ThreadPoolExecutor for parallel API calls
        results = [None] * len(messages_list)
        with ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(self._call_gemini_api, args): idx
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

    def convert_messages(self, messages: List[Dict[str, str]]) -> Tuple[any, Optional[str]]:
        """
        Convert OpenAI format messages to Gemini format.

        Gemini uses 'user' and 'model' roles. System messages are extracted
        and returned separately as system_instruction.

        Args:
            messages: List of messages in OpenAI format
                     Example: [{"role": "system", "content": "You are helpful"},
                               {"role": "user", "content": "Hello"}]

        Returns:
            Tuple of (gemini_contents, system_instruction)
            - gemini_contents: String or list of Content objects for Gemini API
            - system_instruction: System instruction string or None
        """
        system_instruction = None
        gemini_contents = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                # Extract system message as system_instruction
                # If multiple system messages, concatenate them
                if system_instruction:
                    system_instruction += "\n\n" + content
                else:
                    system_instruction = content
            elif role == "user":
                # Create Content object with user role
                gemini_contents.append(
                    types.Content(
                        role="user",
                        parts=[types.Part(text=content)]
                    )
                )
            elif role == "assistant":
                # Gemini uses 'model' instead of 'assistant'
                gemini_contents.append(
                    types.Content(
                        role="model",
                        parts=[types.Part(text=content)]
                    )
                )
            else:
                raise ValueError(f"Unknown role: {role}")

        # Gemini requires alternating user/model turns
        # If we have consecutive messages with the same role, merge them
        merged_contents = []
        for content in gemini_contents:
            if merged_contents and merged_contents[-1].role == content.role:
                # Merge parts with previous message
                merged_contents[-1].parts.extend(content.parts)
            else:
                merged_contents.append(content)

        # Gemini API expects simple string for single user message
        if len(merged_contents) == 1 and merged_contents[0].role == "user":
            # Return just the text content as a string
            text_parts = [part.text for part in merged_contents[0].parts if hasattr(part, 'text')]
            content_str = "\n\n".join(text_parts)
            return content_str, system_instruction

        # For multi-turn conversations, return the list of Content objects
        return merged_contents, system_instruction
