"""Utility helpers for working with agrimanager environments.

These helpers centralize the common logic needed to discover environment
classes, instantiate them from config dictionaries, and load dataset configs
from the canonical repo-local ``data/`` layout. By keeping this logic here,
adding new environments only requires implementing the BaseEnv interface and
providing parquet configs in the expected location.
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any, Dict, List, Tuple, Type

from .env import BaseEnv
from .env_config import BaseEnvConfig
from .nn_adapter import BaseNNEnvAdapter


class EnvironmentDefinitionError(RuntimeError):
    """Raised when an environment package does not expose the expected classes."""


def _repo_root_from_base() -> Path:
    """Infer the repository root relative to this module."""
    return Path(__file__).resolve().parents[3]


def _resolve_repo_path(path_str: str | Path, repo_root: Path | None = None) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = (repo_root or _repo_root_from_base()) / path
    return path.resolve()


def import_env_module(env_name: str):
    """Import the python module for a given environment name."""
    module_path = f"agrimanager.env.{env_name}"
    try:
        return import_module(module_path)
    except ImportError as exc:  # pragma: no cover - bubble up for clarity
        raise ImportError(
            f"Could not import environment module '{env_name}'."
        ) from exc


def import_nn_adapter_module(env_name: str):
    """Import the NN adapter module for a given environment name."""
    module_path = f"agrimanager.env.{env_name}.nn_adapter"
    try:
        return import_module(module_path)
    except ImportError as exc:  # pragma: no cover - bubble up for clarity
        raise ImportError(
            f"Could not import NN adapter module for environment '{env_name}'."
        ) from exc


def discover_env_classes(module) -> Tuple[Type[BaseEnv], Type[BaseEnvConfig]]:
    """Find the Env and EnvConfig classes exposed by an environment module."""
    env_class = None
    config_class = None

    exports: List[str] = getattr(module, "__all__", []) or []
    for name in exports:
        attr = getattr(module, name, None)
        if isinstance(attr, type):
            if issubclass(attr, BaseEnv) and attr is not BaseEnv:
                env_class = attr
            elif issubclass(attr, BaseEnvConfig) and attr is not BaseEnvConfig:
                config_class = attr

    if env_class and config_class:
        return env_class, config_class

    # Fallback: inspect module attributes if __all__ was not defined correctly
    for attr in module.__dict__.values():
        if isinstance(attr, type):
            if env_class is None and issubclass(attr, BaseEnv) and attr is not BaseEnv:
                env_class = attr
            elif config_class is None and issubclass(attr, BaseEnvConfig) and attr is not BaseEnvConfig:
                config_class = attr

    if not env_class or not config_class:
        raise EnvironmentDefinitionError(
            "Environment module must expose both a BaseEnv subclass and a BaseEnvConfig subclass."
        )

    return env_class, config_class


def discover_nn_adapter_class(module) -> Type[BaseNNEnvAdapter]:
    """Find the NN adapter class exposed by an environment adapter module."""
    adapter_class = None

    exports: List[str] = getattr(module, "__all__", []) or []
    for name in exports:
        attr = getattr(module, name, None)
        if isinstance(attr, type) and issubclass(attr, BaseNNEnvAdapter) and attr is not BaseNNEnvAdapter:
            adapter_class = attr
            break

    if adapter_class is None:
        explicit = getattr(module, "NNEnvAdapter", None)
        if isinstance(explicit, type) and issubclass(explicit, BaseNNEnvAdapter):
            adapter_class = explicit

    if adapter_class is None:
        for attr in module.__dict__.values():
            if isinstance(attr, type) and issubclass(attr, BaseNNEnvAdapter) and attr is not BaseNNEnvAdapter:
                adapter_class = attr
                break

    if not adapter_class:
        raise EnvironmentDefinitionError(
            "Environment NN adapter module must expose a BaseNNEnvAdapter subclass."
        )

    return adapter_class


def create_environment(env_name: str, env_config: Dict[str, Any]) -> Tuple[BaseEnv, BaseEnvConfig]:
    """Instantiate an environment and its config from a config dict."""
    module = import_env_module(env_name)
    EnvClass, ConfigClass = discover_env_classes(module)
    config = ConfigClass(**env_config)
    env = EnvClass(config)
    return env, config


def create_nn_env_adapter(env_name: str) -> BaseNNEnvAdapter:
    """Instantiate the NN adapter for a given environment."""
    module = import_nn_adapter_module(env_name)
    AdapterClass = discover_nn_adapter_class(module)
    return AdapterClass()


def _infer_env_name(rows: List[Dict[str, Any]], dataset_path: Path) -> str:
    if rows:
        data_source = rows[0].get("data_source")
        if isinstance(data_source, str) and "/" in data_source:
            return data_source.split("/", 1)[0]

    parts = dataset_path.parts
    try:
        data_idx = parts.index("data")
        if data_idx >= 2:
            return parts[data_idx - 2]
    except ValueError:
        pass

    raise ValueError(
        f"Unable to infer env_name from dataset rows or path: {dataset_path}"
    )


def _load_env_configs_from_single_parquet(
    dataset_file: str | Path,
    repo_root: Path | None = None,
) -> Tuple[List[Dict[str, Any]], str, Path]:
    """Load env configs from one dataset artifact parquet file."""
    import pandas as pd

    dataset_path = _resolve_repo_path(dataset_file, repo_root=repo_root)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    df = pd.read_parquet(dataset_path)
    rows = df.to_dict(orient="records")
    configs: List[Dict[str, Any]] = []
    for row in rows:
        extra_info = row.get("extra_info") or {}
        interaction_kwargs = extra_info.get("interaction_kwargs") or {}
        env_config = interaction_kwargs.get("env_config")
        if not isinstance(env_config, dict):
            raise ValueError(
                f"Dataset row is missing extra_info.interaction_kwargs.env_config: {dataset_path}"
            )
        configs.append(dict(env_config))

    env_name = _infer_env_name(rows, dataset_path)
    return configs, env_name, dataset_path


def load_env_configs_from_parquet(
    dataset_file: str | Path | Any,
    repo_root: Path | None = None,
) -> Tuple[List[Dict[str, Any]], str, Path]:
    """Load env configs from one or more dataset artifact parquet files."""
    if not isinstance(dataset_file, (str, Path)):
        try:
            dataset_files = list(dataset_file)
        except TypeError:
            dataset_files = None
        if dataset_files is not None:
            if not dataset_files:
                raise ValueError("Dataset file list is empty.")
            all_configs: List[Dict[str, Any]] = []
            env_name: str | None = None
            first_path: Path | None = None
            for path_like in dataset_files:
                configs, path_env_name, dataset_path = _load_env_configs_from_single_parquet(
                    path_like,
                    repo_root=repo_root,
                )
                if env_name is None:
                    env_name = path_env_name
                    first_path = dataset_path
                elif path_env_name != env_name:
                    raise ValueError(
                        f"Dataset env_name mismatch across files: {env_name!r} vs {path_env_name!r}"
                    )
                all_configs.extend(configs)
            assert env_name is not None and first_path is not None
            return all_configs, env_name, first_path

    return _load_env_configs_from_single_parquet(dataset_file, repo_root=repo_root)


def load_dataset_configs(
    env_name: str,
    dataset_id: str,
    split: str,
    repo_root: Path | None = None,
) -> Tuple[List[Dict[str, Any]], Path]:
    """Load dataset configurations from parquet stored under data/<env_name>/.

    The parquet files are generated by ``BaseDatasetGenerator`` and contain
    VERL-compatible records.  The environment configs are extracted from
    ``extra_info.interaction_kwargs.env_config`` in each row.
    """
    base_root = repo_root or _repo_root_from_base()
    dataset_path = base_root / "data" / env_name / dataset_id / f"{split}.parquet"
    configs, _, resolved_path = load_env_configs_from_parquet(dataset_path, repo_root=base_root)
    return configs, resolved_path
