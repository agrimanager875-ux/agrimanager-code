"""
Factory function to create model instances from YAML configuration files.
"""
import yaml
from pathlib import Path
from typing import Union


def _get_provider_classes(model_provider: str):
    """
    Lazy import provider classes based on model_provider.
    This avoids importing vLLM when only using OpenAI, etc.

    Args:
        model_provider: Provider name ("openai", "openrouter", "vllm_server", "vllm_offline")

    Returns:
        Tuple of (config_class, model_class)

    Raises:
        ValueError: If model_provider is invalid
    """
    if model_provider == "openai":
        from agrimanager.model_interface.model_provider.openai import OpenAIModel, OpenAIConfig
        return OpenAIConfig, OpenAIModel

    elif model_provider == "openrouter":
        from agrimanager.model_interface.model_provider.openrouter import OpenRouterModel, OpenRouterConfig
        return OpenRouterConfig, OpenRouterModel

    elif model_provider == "google":
        from agrimanager.model_interface.model_provider.google import GoogleModel, GoogleConfig
        return GoogleConfig, GoogleModel

    elif model_provider == "vllm_server":
        from agrimanager.model_interface.model_provider.vllm_server import VLLMServerModel, VLLMServerConfig
        return VLLMServerConfig, VLLMServerModel

    elif model_provider == "vllm_offline":
        from agrimanager.model_interface.model_provider.vllm_offline import VLLMOfflineModel, VLLMOfflineConfig
        return VLLMOfflineConfig, VLLMOfflineModel

    else:
        raise ValueError(
            f"Invalid model_provider: {model_provider}. "
            f"Must be one of: openai, openrouter, google, vllm_server, vllm_offline"
        )


def create_model(config_path: Union[str, Path], overrides: dict = None):
    """
    Create a model instance from a YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Model instance for the requested provider

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If model_provider is invalid or missing
        KeyError: If required fields are missing

    Example:
        ```python
        # Create model from config
        model = create_model("configs/openai.yaml")

        # Use the model
        responses = model.generate(
            messages_list,
            temperature=0.7,
            max_tokens=1024
        )
        ```

    Example YAML format:
        ```yaml
        model_provider: "openai"
        model_name: "gpt-4o"
        num_workers: 8
        api_key: "sk-xxx"
        timeout: 60
        ```
    """
    # Convert to Path object
    config_path = Path(config_path)

    # Check if file exists
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load YAML file
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)

    # Validate config_dict
    if not isinstance(config_dict, dict):
        raise ValueError(f"Invalid YAML format: expected dict, got {type(config_dict)}")

    # Apply overrides
    if overrides:
        config_dict.update(overrides)

    # Get model_provider
    model_provider = config_dict.get('model_provider')
    if not model_provider:
        raise ValueError("'model_provider' field is required in config file")

    # Get the appropriate config and model classes (lazy import)
    config_class, model_class = _get_provider_classes(model_provider)

    # Create config instance (keep model_provider in the dict)
    try:
        config = config_class(**config_dict)
    except TypeError as e:
        raise KeyError(f"Error creating {config_class.__name__}: {e}")

    # Create and return model instance
    model = model_class(config)
    return model
