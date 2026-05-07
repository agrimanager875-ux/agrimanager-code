"""Generic numeric-environment adapter contract for NN trainers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import gymnasium as gym


class BaseNNEnvAdapter(ABC):
    """Contract for framework-native numeric environment adapters.

    Adapters bridge parquet-stored ``env_config`` rows to Gymnasium-compatible
    numeric environments suitable for NN training and evaluation.
    """

    @abstractmethod
    def make_env(self, env_config: dict[str, Any]) -> gym.Env:
        """Build one Gymnasium-compatible numeric environment."""

    def episode_length_hint(self, env_config: dict[str, Any]) -> int | None:
        """Return the expected episode length when it is known.

        The generic trainer uses this only to derive a timestep safety cap when
        training is specified in dataset epochs. Environments with variable
        episode length may return ``None`` and require an explicit timestep cap.
        """
        raw_turn_num = env_config.get("turn_num")
        if raw_turn_num is None:
            return None
        try:
            turn_num = int(raw_turn_num)
        except (TypeError, ValueError):
            return None
        return turn_num if turn_num > 0 else None

    def group_labels(
        self,
        env_config: dict[str, Any],
        info: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        """Return generic group labels used for metric aggregation."""
        labels: dict[str, str] = {}

        raw_config_labels = env_config.get("trajectory_group_labels", {})
        if isinstance(raw_config_labels, dict):
            for key, value in raw_config_labels.items():
                key_str = str(key or "").strip()
                value_str = str(value or "").strip()
                if key_str and value_str:
                    labels[key_str] = value_str

        for key, value in dict(info or {}).items():
            if not isinstance(key, str) or not key.startswith("group_label/"):
                continue
            group_name = key[len("group_label/"):].strip()
            group_value = str(value or "").strip()
            if group_name and group_value:
                labels[group_name] = group_value

        return labels

    def serialize_action(self, action: Any) -> Any:
        """Convert a policy action into JSON-compatible rollout output."""
        return _to_jsonable(action)

    def resolve_spaces(self, env_config: dict[str, Any]) -> tuple[gym.Space, gym.Space]:
        """Infer observation/action spaces by constructing a prototype env."""
        env = self.make_env(env_config)
        try:
            return env.observation_space, env.action_space
        finally:
            env.close()


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return _to_jsonable(value.tolist())
    if hasattr(value, "item"):
        return _to_jsonable(value.item())
    return value
