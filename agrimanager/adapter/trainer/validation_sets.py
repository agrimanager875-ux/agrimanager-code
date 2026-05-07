"""Named validation-set helpers for training and evaluation."""

from __future__ import annotations

from bisect import bisect_right
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Iterable

from omegaconf import DictConfig, ListConfig, OmegaConf
from torch.utils.data import Dataset


def _to_plain(value: Any) -> Any:
    if isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(value, resolve=True)
    return value


def normalize_val_sets(raw_val_sets: Any) -> dict[str, list[str]]:
    """Normalize Hydra-style named validation sets to ``name -> files``."""
    raw_val_sets = _to_plain(raw_val_sets)
    if raw_val_sets in (None, "", {}):
        return {}
    if not isinstance(raw_val_sets, dict):
        raise ValueError("data.val_sets must be a mapping from set name to parquet file(s).")

    val_sets: dict[str, list[str]] = {}
    for raw_name, raw_files in raw_val_sets.items():
        name = str(raw_name or "").strip()
        if not name:
            raise ValueError("data.val_sets contains an empty validation-set name.")
        raw_files = _to_plain(raw_files)
        if isinstance(raw_files, (str, Path)):
            files = [str(raw_files)]
        elif isinstance(raw_files, Iterable):
            files = [str(item) for item in raw_files]
        else:
            raise ValueError(f"data.val_sets.{name} must be a path or a list of paths.")
        files = [path for path in files if path]
        if not files:
            raise ValueError(f"data.val_sets.{name} is empty.")
        val_sets[name] = files
    return val_sets


def normalize_env_config_overrides(raw_overrides: Any) -> dict[str, Any]:
    """Normalize optional runtime env_config overrides for prompt-mode runs."""
    raw_overrides = _to_plain(raw_overrides)
    if raw_overrides in (None, "", {}):
        return {}
    if not isinstance(raw_overrides, dict):
        raise ValueError("data.env_config_overrides must be a mapping when set.")
    return dict(raw_overrides)


def flatten_val_set_files(val_sets: dict[str, list[str]]) -> list[str]:
    """Return files in deterministic validation-set order."""
    files: list[str] = []
    for set_files in val_sets.values():
        files.extend(set_files)
    return files


def annotate_env_config_with_validation_set(
    env_config: dict[str, Any],
    validation_set: str,
) -> dict[str, Any]:
    """Return a copy of an env config annotated with a validation-set label."""
    cfg = deepcopy(env_config)
    cfg["dataset_role"] = "validation"
    cfg["validation_set"] = str(validation_set)
    group_labels = dict(cfg.get("trajectory_group_labels") or {})
    group_labels["dataset_role"] = "validation"
    group_labels["validation_set"] = str(validation_set)
    cfg["trajectory_group_labels"] = group_labels
    return cfg


def annotate_sample_with_validation_set(sample: dict[str, Any], validation_set: str) -> dict[str, Any]:
    """Annotate a VERL dataset sample with validation-set metadata."""
    row = dict(sample)
    extra_info = deepcopy(row.get("extra_info") or {})
    interaction_kwargs = deepcopy(extra_info.get("interaction_kwargs") or {})
    env_config = interaction_kwargs.get("env_config")
    if isinstance(env_config, dict):
        interaction_kwargs["env_config"] = annotate_env_config_with_validation_set(
            env_config,
            validation_set,
        )
    extra_info["interaction_kwargs"] = interaction_kwargs
    extra_info["validation_set"] = str(validation_set)
    row["extra_info"] = extra_info
    row["interaction_kwargs"] = interaction_kwargs
    return row


def apply_env_config_overrides_to_sample(
    sample: dict[str, Any],
    env_config_overrides: dict[str, Any],
) -> dict[str, Any]:
    """Return a copy of a dataset row with runtime env_config overrides applied."""
    if not env_config_overrides:
        return sample
    row = dict(sample)
    extra_info = deepcopy(row.get("extra_info") or {})
    interaction_kwargs = deepcopy(extra_info.get("interaction_kwargs") or {})
    env_config = interaction_kwargs.get("env_config")
    if isinstance(env_config, dict):
        cfg = deepcopy(env_config)
        cfg.update(deepcopy(env_config_overrides))
        interaction_kwargs["env_config"] = cfg
    extra_info["interaction_kwargs"] = interaction_kwargs
    row["extra_info"] = extra_info
    row["interaction_kwargs"] = interaction_kwargs
    return row


class EnvConfigOverrideDataset(Dataset):
    """Apply fixed env_config overrides to rows after parquet loading."""

    def __init__(self, dataset: Dataset, env_config_overrides: dict[str, Any]):
        self.dataset = dataset
        self.env_config_overrides = dict(env_config_overrides)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return apply_env_config_overrides_to_sample(
            self.dataset[index],
            self.env_config_overrides,
        )


class NamedValidationDataset(Dataset):
    """Concatenate per-set datasets while annotating each sample in memory."""

    def __init__(self, datasets: dict[str, Dataset]):
        if not datasets:
            raise ValueError("NamedValidationDataset requires at least one dataset.")
        self.datasets = list(datasets.items())
        self.cumulative_lengths: list[int] = []
        total = 0
        for _, dataset in self.datasets:
            total += len(dataset)
            self.cumulative_lengths.append(total)
        if total <= 0:
            raise ValueError("NamedValidationDataset contains no samples.")

    def __len__(self) -> int:
        return self.cumulative_lengths[-1]

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        dataset_idx = bisect_right(self.cumulative_lengths, index)
        previous_end = 0 if dataset_idx == 0 else self.cumulative_lengths[dataset_idx - 1]
        validation_set, dataset = self.datasets[dataset_idx]
        return annotate_sample_with_validation_set(
            dataset[index - previous_end],
            validation_set,
        )


def create_named_validation_dataset(
    raw_val_sets: Any,
    data_config: Any,
    tokenizer: Any,
    processor: Any,
    create_rl_dataset: Callable[..., Dataset],
    *,
    max_samples: int = -1,
) -> NamedValidationDataset | None:
    """Build a concatenated validation dataset from ``data.val_sets``."""
    val_sets = normalize_val_sets(raw_val_sets)
    if not val_sets:
        return None
    datasets = {
        name: create_rl_dataset(
            files,
            data_config,
            tokenizer,
            processor,
            is_train=False,
            max_samples=max_samples,
        )
        for name, files in val_sets.items()
    }
    return NamedValidationDataset(datasets)
