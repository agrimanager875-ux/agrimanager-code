"""Dataset generation router for agrimanager environments.

This module provides the main entry point for generating VERL-compatible
parquet datasets across different AgriManager environments.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Dict

import yaml


def generate(config_path: str, output_dir: str, num_workers: int | None = None):
    """Dispatch dataset generation to the target environment.

    Args:
        config_path: Path to the YAML configuration file.
        output_dir: Directory where dataset files will be saved.
        num_workers: Number of parallel workers. Overrides config value if set.
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config.setdefault("dataset_id", config_file.stem)
    config["_config_path"] = str(config_file.resolve())

    if num_workers is not None:
        config["num_workers"] = num_workers

    env_name = config.get("env_name")
    if not env_name:
        raise ValueError("Config file must specify 'env_name'.")

    module_path = f"agrimanager.env.{env_name}.create_dataset"
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise ImportError(
            f"Could not import create_dataset module for environment '{env_name}'."
        ) from exc

    if not hasattr(module, "generate"):
        raise AttributeError(
            f"Module {module_path} must expose a generate(config, output_dir) function."
        )

    print(f"Generating dataset for environment: {env_name}")
    print(f"Config: {config_path}")
    print(f"Output directory: {output_dir}")
    print("-" * 80)
    module.generate(config, output_dir)
