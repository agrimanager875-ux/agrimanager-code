"""Shared helpers for framework-native NN PPO training and evaluation."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from agrimanager.adapter.trainer.validation_metrics import (
    add_axis_env_metrics,
    add_env_metrics,
    add_validation_env_metrics,
)
from agrimanager.env.base import BaseNNEnvAdapter


def resolve_repo_path(path_str: str | Path) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    return path.resolve()


def sanitize_filename(name: str) -> str:
    cleaned = [
        ch if ch.isalnum() or ch in ("-", "_", ".") else "_"
        for ch in (name or "unknown").strip().lower()
    ]
    sanitized = "".join(cleaned).strip("._")
    return sanitized or "unknown"


class ScenarioCyclingEnv(gym.Env):
    """Cycle through parquet-defined scenarios while rebuilding the env each episode."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        adapter: BaseNNEnvAdapter,
        env_configs: list[dict[str, Any]],
        *,
        sample_with_replacement: bool = False,
        seed: int = 0,
    ) -> None:
        if not env_configs:
            raise ValueError("env_configs must not be empty")
        self.adapter = adapter
        self.env_configs = [dict(cfg) for cfg in env_configs]
        self.sample_with_replacement = bool(sample_with_replacement)
        self.rng = np.random.default_rng(int(seed))

        prototype_env = self.adapter.make_env(self.env_configs[0])
        try:
            self.observation_space = prototype_env.observation_space
            self.action_space = prototype_env.action_space
        finally:
            prototype_env.close()

        self._current_env: gym.Env | None = None
        self._current_env_config: dict[str, Any] | None = None
        self._order = np.arange(len(self.env_configs), dtype=np.int64)
        self._cursor = 0
        if not self.sample_with_replacement:
            self.rng.shuffle(self._order)

    def _next_index(self) -> int:
        if self.sample_with_replacement:
            return int(self.rng.integers(0, len(self.env_configs)))
        if self._cursor >= len(self._order):
            self.rng.shuffle(self._order)
            self._cursor = 0
        idx = int(self._order[self._cursor])
        self._cursor += 1
        return idx

    def _build_env_for_next_scenario(self) -> gym.Env:
        cfg = dict(self.env_configs[self._next_index()])
        self._current_env_config = cfg
        if self._current_env is not None:
            self._current_env.close()
        self._current_env = self.adapter.make_env(cfg)
        return self._current_env

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        del options
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))
        env = self._build_env_for_next_scenario()
        observation, info = env.reset()
        return observation, dict(info or {})

    def step(self, action):
        if self._current_env is None:
            raise RuntimeError("ScenarioCyclingEnv.step() called before reset().")
        return self._current_env.step(action)

    def close(self):
        if self._current_env is not None:
            self._current_env.close()
            self._current_env = None


def run_episode(
    env: gym.Env,
    model,
    *,
    adapter: BaseNNEnvAdapter | None = None,
    vecnormalize=None,
    deterministic: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    env_config = getattr(env, "env_config", None)
    observation, info = env.reset()
    turns = [
        {
            "turn": 0,
            "action": None,
            "turn_metrics": dict((info or {}).get("turn_metrics") or {}),
        }
    ]

    terminated = False
    truncated = False
    while not (terminated or truncated):
        policy_observation = np.asarray(observation, dtype=np.float32)
        if vecnormalize is not None:
            policy_observation = vecnormalize.normalize_obs(policy_observation[None, ...])[0]
        action, _ = model.predict(policy_observation, deterministic=deterministic)
        serialized_action = (
            adapter.serialize_action(action)
            if adapter is not None
            else _serialize_action_default(action)
        )
        observation, reward, terminated, truncated, info = env.step(action)
        del reward
        turns.append(
            {
                "turn": len(turns),
                "action": serialized_action,
                "turn_metrics": dict((info or {}).get("turn_metrics") or {}),
            }
        )

    if env_config is None:
        env_config = dict((info or {}).get("env_config") or {})
    result = {
        "env_id": 0,
        "env_config": env_config,
        "turns": turns,
    }
    return result, dict(info or {})


def _serialize_action_default(action: Any) -> Any:
    if isinstance(action, dict):
        return {str(key): _serialize_action_default(value) for key, value in action.items()}
    if isinstance(action, (list, tuple)):
        return [_serialize_action_default(value) for value in action]
    if hasattr(action, "tolist"):
        return _serialize_action_default(action.tolist())
    if hasattr(action, "item"):
        return _serialize_action_default(action.item())
    return action


def _json_safe(value: Any) -> Any:
    """Convert NumPy-rich rollout payloads into values accepted by json.dump."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "tolist"):
        try:
            return _json_safe(value.tolist())
        except Exception:
            pass
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:
            pass
    return str(value)


def collect_trajectory_metrics(info: dict[str, Any] | None) -> dict[str, float]:
    """Extract numeric episode-level metrics from a Gymnasium info dict."""
    raw_metrics = dict((info or {}).get("trajectory_metrics") or {})
    metrics: dict[str, float] = {}
    for key, value in raw_metrics.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            metrics[str(key)] = float(value)
    return metrics


def evaluate_model_on_dataset(
    model,
    adapter: BaseNNEnvAdapter,
    env_configs: list[dict[str, Any]],
    *,
    vecnormalize=None,
    deterministic: bool = False,
    seed: int = 0,
    num_repeats: int = 1,
    validation_axis: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, list[Any]], dict[str, float]]:
    results: list[dict[str, Any]] = []
    env_infos: dict[str, list[Any]] = defaultdict(list)

    if vecnormalize is not None:
        prev_training = vecnormalize.training
        prev_norm_reward = vecnormalize.norm_reward
        vecnormalize.training = False
        vecnormalize.norm_reward = False
    else:
        prev_training = None
        prev_norm_reward = None

    try:
        for repeat_idx in range(max(1, int(num_repeats))):
            np.random.seed(int(seed) + repeat_idx)
            for raw_env_config in env_configs:
                env_config = dict(raw_env_config)
                env = adapter.make_env(env_config)
                try:
                    result, final_info = run_episode(
                        env,
                        model,
                        adapter=adapter,
                        vecnormalize=vecnormalize,
                        deterministic=deterministic,
                    )
                    result["group_labels"] = adapter.group_labels(env_config, final_info)
                    result["env_id"] = len(results)
                    results.append(result)

                    for key, value in collect_trajectory_metrics(final_info).items():
                        env_infos[key].append(value)
                    for group_name, group_value in result["group_labels"].items():
                        env_infos[f"group_label/{group_name}"].append(group_value)
                finally:
                    env.close()
    finally:
        if vecnormalize is not None:
            vecnormalize.training = prev_training
            vecnormalize.norm_reward = prev_norm_reward

    metric_dict: dict[str, float] = {}
    axis = str(validation_axis or "").strip()
    if axis:
        add_axis_env_metrics(metric_dict, env_infos, axis=axis, prefix_base="val-env")
    else:
        validation_sets = [
            str(value or "").strip()
            for value in env_infos.get("group_label/validation_set", [])
        ]
        if validation_sets:
            add_env_metrics(metric_dict, env_infos, prefix="val-env/all", include_grouped=False)
            for validation_set in sorted({value for value in validation_sets if value}):
                indices = [idx for idx, value in enumerate(validation_sets) if value == validation_set]
                subset_infos = {
                    key: [values[i] for i in indices]
                    for key, values in env_infos.items()
                    if len(values) > max(indices, default=-1)
                }
                add_env_metrics(
                    metric_dict,
                    subset_infos,
                    prefix=f"val-env/{validation_set}",
                    include_grouped=False,
                )
        add_validation_env_metrics(metric_dict, env_infos)
    return results, env_infos, metric_dict


def save_rollout_results(
    output_dir: str | Path,
    results: list[dict[str, Any]],
    *,
    split_by_group: str | None = None,
) -> Path:
    output_path = resolve_repo_path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results_path = output_path / "results.json"
    safe_results = _json_safe(results)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(safe_results, f, indent=2)

    if not split_by_group:
        return results_path

    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in results:
        group_labels = dict(item.get("group_labels") or {})
        group_value = str(group_labels.get(split_by_group) or "unknown").strip() or "unknown"
        by_group[group_value].append(item)

    group_output_dir = output_path / f"results_by_{sanitize_filename(split_by_group)}"
    group_output_dir.mkdir(parents=True, exist_ok=True)

    group_index: list[dict[str, Any]] = []
    for group_value, group_results in sorted(by_group.items()):
        group_file = group_output_dir / f"{sanitize_filename(group_value)}.json"
        with open(group_file, "w", encoding="utf-8") as f:
            json.dump(_json_safe(group_results), f, indent=2)
        group_index.append(
            {
                "group": split_by_group,
                "value": group_value,
                "num_envs": len(group_results),
                "results_file": str(group_file.relative_to(output_path)),
            }
        )

    with open(group_output_dir / "index.json", "w", encoding="utf-8") as f:
        json.dump(_json_safe(group_index), f, indent=2)

    return results_path
