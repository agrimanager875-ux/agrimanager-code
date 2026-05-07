"""Utilities for capturing live prompts from AgriManager environments."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from agrimanager.env.base.utils import create_environment, load_env_configs_from_parquet


DEFAULT_METADATA_KEYS = (
    "env_id",
    "scenario_id",
    "dataset_split",
    "dataset_role",
    "validation_set",
    "simulator",
    "crop",
    "crop_name",
    "weather_regime",
    "crop_regime",
    "action_menu",
    "action_schema",
    "prompt_condition",
    "reward_formulation",
    "train_source",
    "variety",
    "variety_split",
    "objective_id",
    "prompt_objective_id",
    "env_reward",
    "reward_mode",
    "prompt_action_schema_env_id",
    "require_think",
    "thinking_mode",
    "think_tag",
)


def load_live_prompt_env_config_from_parquet(
    dataset_file: str | Path,
    *,
    sample_idx: int = 0,
    repo_root: str | Path | None = None,
) -> tuple[dict[str, Any], str, Path]:
    """Load one environment config from a generated dataset parquet file.

    Generated datasets store the real rollout config under
    ``extra_info.interaction_kwargs.env_config``. The prompt stored in the row is
    usually only a placeholder; live prompt capture must instantiate the
    environment and call ``system_prompt()`` / ``reset()``.
    """

    root = Path(repo_root).resolve() if repo_root is not None else None
    configs, env_name, resolved_path = load_env_configs_from_parquet(
        dataset_file,
        repo_root=root,
    )
    if sample_idx < 0 or sample_idx >= len(configs):
        raise IndexError(
            f"sample_idx={sample_idx} is outside dataset size {len(configs)}: {resolved_path}"
        )
    return dict(configs[sample_idx]), env_name, resolved_path


def capture_live_prompt(
    env_name: str,
    env_config: Mapping[str, Any],
    *,
    label: str | None = None,
    sample_idx: int | None = None,
    steps: int = 0,
    step_action: Any = 0,
    env_config_overrides: Mapping[str, Any] | None = None,
    metadata_keys: Sequence[str] = DEFAULT_METADATA_KEYS,
    ensure_llm_mode: bool = True,
) -> dict[str, Any]:
    """Instantiate an environment and capture its live system/user prompts."""

    config = deepcopy(dict(env_config))
    if env_config_overrides:
        config.update(deepcopy(dict(env_config_overrides)))
    if ensure_llm_mode:
        config["llm_mode"] = True

    env = None
    try:
        env, _ = create_environment(env_name, config)
        system_prompt = env.system_prompt()
        user_prompt, info = env.reset()
        turns = [
            {
                "turn_index": 0,
                "source": "reset",
                "user_prompt": str(user_prompt),
                "info": info,
            }
        ]

        done = False
        for step_idx in range(int(steps)):
            if done:
                break
            next_prompt, reward, done, info = env.step(step_action)
            turns.append(
                {
                    "turn_index": step_idx + 1,
                    "source": f"step(action={step_action!r})",
                    "reward": float(reward),
                    "done": bool(done),
                    "user_prompt": str(next_prompt),
                    "info": info,
                }
            )

        metadata = {
            key: config.get(key)
            for key in metadata_keys
            if config.get(key) is not None
        }
        if label is not None:
            metadata["label"] = label
        if sample_idx is not None:
            metadata["sample_idx"] = sample_idx

        return {
            "label": label or metadata.get("validation_set") or metadata.get("scenario_id") or env_name,
            "env_name": env_name,
            "metadata": metadata,
            "system_prompt": str(system_prompt),
            "turns": turns,
        }
    finally:
        if env is not None:
            env.close()


def capture_live_prompt_from_parquet(
    dataset_file: str | Path,
    *,
    sample_idx: int = 0,
    label: str | None = None,
    steps: int = 0,
    step_action: Any = 0,
    env_name: str | None = None,
    env_config_overrides: Mapping[str, Any] | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Capture one live prompt by loading a scenario from a parquet dataset."""

    env_config, inferred_env_name, resolved_path = load_live_prompt_env_config_from_parquet(
        dataset_file,
        sample_idx=sample_idx,
        repo_root=repo_root,
    )
    capture = capture_live_prompt(
        env_name or inferred_env_name,
        env_config,
        label=label,
        sample_idx=sample_idx,
        steps=steps,
        step_action=step_action,
        env_config_overrides=env_config_overrides,
    )
    capture["dataset_file"] = str(resolved_path)
    return capture


def capture_live_prompts_from_parquet(
    dataset_file: str | Path,
    *,
    sample_indices: Sequence[int] | None = None,
    max_samples: int | None = None,
    label_prefix: str | None = None,
    steps: int = 0,
    step_action: Any = 0,
    env_name: str | None = None,
    env_config_overrides: Mapping[str, Any] | None = None,
    repo_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Capture live prompts for multiple scenarios from one parquet dataset."""

    root = Path(repo_root).resolve() if repo_root is not None else None
    configs, inferred_env_name, resolved_path = load_env_configs_from_parquet(
        dataset_file,
        repo_root=root,
    )
    if sample_indices is None:
        indices = list(range(len(configs)))
    else:
        indices = list(sample_indices)
    if max_samples is not None:
        indices = indices[: int(max_samples)]

    captures = []
    for sample_idx in indices:
        if sample_idx < 0 or sample_idx >= len(configs):
            raise IndexError(
                f"sample_idx={sample_idx} is outside dataset size {len(configs)}: {resolved_path}"
            )
        label = f"{label_prefix}_{sample_idx}" if label_prefix else f"sample_{sample_idx}"
        capture = capture_live_prompt(
            env_name or inferred_env_name,
            configs[sample_idx],
            label=label,
            sample_idx=sample_idx,
            steps=steps,
            step_action=step_action,
            env_config_overrides=env_config_overrides,
        )
        capture["dataset_file"] = str(resolved_path)
        captures.append(capture)
    return captures


def write_live_prompt_markdown(
    path: str | Path,
    captures: Iterable[Mapping[str, Any]],
    *,
    title: str,
    description: str | None = None,
) -> Path:
    """Write captured prompts as a paper-facing markdown appendix file."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [f"# {title}", ""]
    if description:
        lines.extend([description, ""])

    for capture in captures:
        label = str(capture.get("label") or capture.get("env_name") or "scenario")
        metadata = dict(capture.get("metadata") or {})
        dataset_file = capture.get("dataset_file")
        lines.extend([f"## {label}", ""])
        if dataset_file:
            lines.append(f"- dataset_file: `{dataset_file}`")
        if capture.get("env_name"):
            lines.append(f"- env_name: `{capture.get('env_name')}`")
        for key, value in metadata.items():
            lines.append(f"- {key}: `{value}`")
        if lines[-1] != "":
            lines.append("")

        lines.extend(
            [
                "### System Prompt",
                "",
                "```text",
                str(capture.get("system_prompt", "")),
                "```",
                "",
            ]
        )

        for turn in capture.get("turns", []):
            lines.extend(
                [
                    f"### User Prompt: Turn {turn.get('turn_index')} ({turn.get('source')})",
                    "",
                    "```text",
                    str(turn.get("user_prompt", "")),
                    "```",
                    "",
                ]
            )

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def write_live_prompt_artifacts(
    output_dir: str | Path,
    captures: Sequence[Mapping[str, Any]],
    *,
    title: str,
    description: str | None = None,
    json_name: str = "live_prompts.json",
    markdown_name: str = "live_prompts.md",
) -> tuple[Path, Path]:
    """Write both JSON and markdown live-prompt artifacts."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    json_path = output_path / json_name
    md_path = output_path / markdown_name
    json_path.write_text(
        json.dumps(list(captures), indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    write_live_prompt_markdown(md_path, captures, title=title, description=description)
    return json_path, md_path
