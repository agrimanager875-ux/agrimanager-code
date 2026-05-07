"""Base dataset generation helper for environments.

Environments can subclass :class:`BaseDatasetGenerator` to get standard
validation, directory creation, parquet writing, and summary logging. Only the
logic that produces per-split configuration records needs to be implemented
per environment.

Generated parquet datasets are compatible with VERL's ToolAgentLoop
training framework.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
from abc import ABC, abstractmethod
from functools import partial
from pathlib import Path
from typing import Dict, Any, List, Optional

from agrimanager.env.base.dataset_metadata import (
    normalize_split_metadata,
    write_dataset_manifest,
)


def _worker_convert(args, generator_cls, config, output_dir):
    """Top-level function for multiprocessing (must be picklable)."""
    env_config, idx, split = args
    gen = generator_cls(config, output_dir)
    try:
        return gen._convert_to_verl_format(env_config, idx, split)
    except Exception as e:
        return None


def normalize_prompt_content(content: Any) -> str:
    """Convert prompt content to a stable string representation."""
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if hasattr(content, "tolist"):
        content = content.tolist()
    if isinstance(content, (list, tuple, dict)):
        return json.dumps(content, ensure_ascii=True)
    return str(content)


def build_chat_prompt(system_prompt: Any, observation: Any) -> List[Dict[str, str]]:
    """Build a chat prompt while keeping message content schema-stable."""
    return [
        {"role": "system", "content": normalize_prompt_content(system_prompt)},
        {"role": "user", "content": normalize_prompt_content(observation)},
    ]


class BaseDatasetGenerator(ABC):
    """Template class for dataset generation across environments.

    Attributes:
        config: Configuration dictionary for dataset generation.
        output_dir: Directory where dataset files will be saved.
        dataset_id: Unique identifier for this dataset.
    """

    def __init__(self, config: Dict[str, Any], output_dir: str):
        self.config = config
        self.output_dir = Path(output_dir)
        self.dataset_id = config.get("dataset_id")
        if not self.dataset_id:
            raise ValueError("Config must include 'dataset_id'.")

    def _effective_num_workers(self) -> int:
        """Return the worker count to use for record conversion."""
        return int(self.config.get("num_workers", 64))

    def generate(self) -> None:
        """Generate VERL-compatible parquet dataset."""
        import datasets

        save_dir = self.output_dir / self.dataset_id
        save_dir.mkdir(parents=True, exist_ok=True)

        num_workers = self._effective_num_workers()

        summary = {}
        for split in self.splits():
            from tqdm import tqdm

            target = self.config.get(f"num_{split}_samples")
            records = self.build_split_records(split)
            verl_records = []
            total_failed = 0

            while True:
                work_items = [(r, i, split) for i, r in enumerate(records)]
                desc = f"Generating {split}"
                if num_workers > 1:
                    worker_fn = partial(
                        _worker_convert,
                        generator_cls=type(self),
                        config=self.config,
                        output_dir=str(self.output_dir),
                    )
                    with mp.Pool(num_workers) as pool:
                        results = list(tqdm(
                            pool.imap(worker_fn, work_items),
                            total=len(work_items),
                            desc=desc,
                        ))
                else:
                    results = [
                        self._convert_to_verl_format(r, i, split)
                        for i, r in tqdm(enumerate(records), total=len(records), desc=desc)
                    ]

                failed = sum(1 for r in results if r is None)
                total_failed += failed
                verl_records.extend(r for r in results if r is not None)

                if not target or len(verl_records) >= target:
                    break

                need = target - len(verl_records)
                print(f"  {split}: {failed} failed, need {need} more, retrying...")
                records = self._generate_more_records(split, need)

            if target:
                verl_records = verl_records[:target]
            if total_failed:
                print(f"  {split}: {total_failed} samples failed and replaced")
            summary[split] = len(verl_records)
            ds = datasets.Dataset.from_list(verl_records)
            parquet_path = save_dir / f"{split}.parquet"
            ds.to_parquet(str(parquet_path))

        self._write_manifest(save_dir, summary)
        self._print_summary(save_dir, summary)

    def _convert_to_verl_format(
        self, env_config: Dict[str, Any], idx: int, split: str
    ) -> Dict[str, Any]:
        """Convert environment config to VERL data format.

        Subclasses can override this method to customize the VERL record format.

        Args:
            env_config: Environment configuration dict from build_split_records.
            idx: Index of this record within the split.
            split: Name of the split (e.g., "train", "test").

        Returns:
            Dict with VERL-compatible structure.
        """
        return {
            "data_source": f"{self.config.get('env_name', 'agrimanager')}/{self.dataset_id}",
            "agent_name": "agri_tool_agent",
            "prompt": self._build_initial_prompt(env_config),
            "reward_model": {"style": "rule", "ground_truth": None},
            "extra_info": {
                "split": split,
                "index": idx,
                "interaction_kwargs": {
                    "name": "agri",
                    "env_config": env_config,
                },
            },
        }

    def _placeholder_prompt_system_prompt(self, env_config: Dict[str, Any]) -> str:
        """Return the shared dataset placeholder system message."""
        return "You are an agricultural management expert."

    def _placeholder_prompt_env_label(self, env_config: Dict[str, Any]) -> str:
        """Return the display label used in the shared placeholder prompt."""
        return str(self.config.get("env_name", "AgriManager"))

    def _placeholder_prompt_context(self, env_config: Dict[str, Any]) -> Optional[str]:
        """Return optional environment-specific placeholder context."""
        return None

    def _build_placeholder_prompt(self, env_config: Dict[str, Any]) -> List[Dict[str, str]]:
        """Build the shared dataset-level placeholder prompt.

        Rollout and training derive the live system prompt and current
        observation from the environment at interaction start, so the parquet
        row only needs a schema-stable placeholder.
        """
        env_label = self._placeholder_prompt_env_label(env_config)
        context = self._placeholder_prompt_context(env_config)
        context_suffix = f" ({context})" if context else ""
        return build_chat_prompt(
            self._placeholder_prompt_system_prompt(env_config),
            (
                f"Placeholder dataset prompt for {env_label}{context_suffix}. "
                "Rollout and training derive the live system prompt and current "
                "observation from the environment when interaction starts, so "
                "dataset build avoids calling env.reset()."
            ),
        )

    def _build_initial_prompt(self, env_config: Dict[str, Any]) -> List[Dict[str, str]]:
        """Build the initial conversation prompt stored in the parquet row."""
        return self._build_placeholder_prompt(env_config)

    def splits(self) -> List[str]:
        """Splits to generate. Override for custom split names."""
        return ["train", "test"]

    @abstractmethod
    def build_split_records(self, split: str) -> List[Dict[str, Any]]:
        """Return the list of config dicts for the requested split."""

    def _generate_more_records(self, split: str, count: int) -> List[Dict[str, Any]]:
        """Generate additional records to replace failures. Override in subclass."""
        raise NotImplementedError(
            "Subclass must implement _generate_more_records to support retry on failure."
        )

    def _print_summary(self, save_dir: Path, summary: Dict[str, int]) -> None:
        print(f"\n{'=' * 80}")
        print("Dataset generated successfully!")
        print(f"{'=' * 80}")
        for split, count in summary.items():
            print(f"{split.title()} samples: {count}")
        print(f"\nFiles saved to: {save_dir}")
        for split in summary:
            print(f"  - {split}.parquet")
        print(f"{'=' * 80}\n")

    def _split_config(self, split: str) -> Dict[str, Any]:
        sampling = self.config.get("sampling") or {}
        splits = sampling.get("splits") or {}
        split_cfg = splits.get(split) if isinstance(splits, dict) else None
        return dict(split_cfg) if isinstance(split_cfg, dict) else {}

    def split_metadata(self, split: str) -> Dict[str, Any]:
        return normalize_split_metadata(
            split,
            self._split_config(split),
            dataset_labels=self.config.get("labels"),
        )

    def _write_manifest(self, save_dir: Path, summary: Dict[str, int]) -> None:
        split_metadata = {
            split: self.split_metadata(split)
            for split in summary
        }
        manifest_path = write_dataset_manifest(
            save_dir,
            self.config,
            summary,
            split_metadata=split_metadata,
        )
        print(f"Manifest: {manifest_path}")
