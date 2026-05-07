"""
vLLM Offline provider model implementation.
"""
from typing import List, Dict, Any
from vllm import LLM, SamplingParams
from tqdm import tqdm
from ..base.model import BaseModel
from .model_config import VLLMOfflineConfig


class VLLMOfflineModel(BaseModel):
    """vLLM Offline model provider implementation using direct LLM."""

    def __init__(self, config: VLLMOfflineConfig):
        """
        Initialize vLLM Offline model.

        Args:
            config: vLLM Offline configuration instance
        """
        super().__init__(config)
        self.config: VLLMOfflineConfig = config

        # Initialize vLLM LLM
        self.llm = LLM(
            model=config.model_name,
            tensor_parallel_size=config.tensor_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
            trust_remote_code=config.trust_remote_code,
            dtype=config.dtype,
            tokenizer_mode=config.tokenizer_mode,
            max_model_len=config.max_model_len,
        )

        # Get tokenizer for message conversion
        self.tokenizer = self.llm.get_tokenizer()

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
        Generate responses for a batch of message lists using vLLM's batch generation.

        Args:
            messages_list: List of message lists in OpenAI format
            temperature: Temperature parameter (default: 0.7)
            top_p: Top-p sampling parameter (default: 1.0)
            max_tokens: Maximum tokens to generate (default: 2048)
            **kwargs: Additional vLLM sampling parameters

        Returns:
            List of generated response strings
        """
        if not messages_list:
            return []

        # Convert all message lists to prompts
        prompts = []
        for messages in tqdm(messages_list, desc="Converting messages"):
            prompt = self.convert_messages(messages)
            prompts.append(prompt)

        return_metadata = kwargs.pop("return_metadata", return_metadata)
        if not return_metadata and (
            getattr(self.config, "reasoning_effort", None)
            or getattr(self.config, "extract_reasoning", False)
            or getattr(self.config, "reasoning_start_token", None)
        ):
            return_metadata = True

        # Create sampling parameters
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            **kwargs
        )

        # Generate using vLLM's batch generation (vLLM shows its own progress bar)
        print("Generating responses with vLLM...")
        outputs = self.llm.generate(prompts, sampling_params, use_tqdm=True)

        # Extract generated text from outputs
        start_tok = getattr(self.config, "reasoning_start_token", None)
        end_tok = getattr(self.config, "reasoning_end_token", None)

        results = []
        for output in outputs:
            generated_text = output.outputs[0].text
            if not return_metadata:
                results.append(generated_text)
                continue

            reasoning, content = self._split_reasoning(generated_text, start_tok, end_tok)
            results.append({"content": content, "reasoning": reasoning})

        return results

    @staticmethod
    def _split_reasoning(text: str, start_tok: str = None, end_tok: str = None):
        """Split generated text into (reasoning, content).

        Returns the reasoning portion and the clean content with reasoning
        stripped.  When no reasoning is found, returns ``(None, text)``.

        Resolution order:
        1. Configured ``reasoning_start_token`` / ``reasoning_end_token``
        2. ``<think>...</think>`` XML tags (Qwen3, QwQ style)
        3. ``</think>`` only (missing opening tag variant)
        """
        # 1. Config-specified tokens (e.g. gpt-oss: analysis / assistant)
        if start_tok and end_tok:
            si = text.find(start_tok)
            if si != -1:
                reasoning_start = si + len(start_tok)
                ei = text.find(end_tok, reasoning_start)
                if ei > reasoning_start:
                    reasoning = text[reasoning_start:ei].strip() or None
                    content = text[ei + len(end_tok):].strip()
                    return reasoning, content
            return None, text

        # 2. <think>...</think>
        if "<think>" in text and "</think>" in text:
            start = text.find("<think>") + len("<think>")
            end = text.find("</think>", start)
            if end > start:
                reasoning = text[start:end].strip() or None
                content = text[end + len("</think>"):].strip()
                return reasoning, content

        # 3. </think> without opening tag
        if "</think>" in text:
            end = text.find("</think>")
            if end > 0:
                reasoning = text[:end].strip() or None
                content = text[end + len("</think>"):].strip()
                return reasoning, content

        return None, text

    def convert_messages(self, messages: List[Dict[str, str]]) -> str:
        """
        Convert OpenAI format messages to vLLM prompt format using chat template.

        Args:
            messages: List of messages in OpenAI format
                     Example: [{"role": "user", "content": "Hello"}]

        Returns:
            Formatted prompt string
        """
        # Check if tokenizer has apply_chat_template method
        if not hasattr(self.tokenizer, "apply_chat_template"):
            raise NotImplementedError(
                "The tokenizer does not support chat templates. "
                "Please implement a custom conversion method."
            )

        # Build extra kwargs for the chat template (e.g. reasoning_effort)
        template_kwargs = {}
        if getattr(self.config, "reasoning_effort", None):
            template_kwargs["reasoning_effort"] = self.config.reasoning_effort

        # Use the model's chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **template_kwargs
        )

        return prompt
