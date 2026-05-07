"""NN adapter for framework-native CyclesGym numeric training."""

from __future__ import annotations

import copy
import itertools
import os
import tempfile
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from agrimanager.env.base import BaseNNEnvAdapter, create_environment


TMP_ROOT = Path(tempfile.gettempdir()) / "agrimanager" / "cycles_gym"
os.environ.setdefault("MPLCONFIGDIR", str(TMP_ROOT / "mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(TMP_ROOT / "cache"))

_RUNTIME_COUNTER = itertools.count()
_LEGACY_GYM_MODULE: Any | None = None


def _legacy_gym_module() -> Any | None:
    global _LEGACY_GYM_MODULE
    if _LEGACY_GYM_MODULE is not None:
        return _LEGACY_GYM_MODULE
    try:
        import gym as legacy_gym  # type: ignore
    except ModuleNotFoundError:  # pragma: no cover - depends on optional Cycles deps
        return None
    _LEGACY_GYM_MODULE = legacy_gym
    return legacy_gym


def _to_gymnasium_space(space: Any) -> gym.Space:
    """Convert legacy Gym spaces from CyclesGym to Gymnasium spaces."""
    if isinstance(space, gym.spaces.Space):
        return copy.deepcopy(space)

    legacy_gym = _legacy_gym_module()
    if legacy_gym is None:
        raise TypeError(f"Unsupported space type for Gymnasium conversion: {type(space)!r}")

    if isinstance(space, legacy_gym.spaces.Box):
        return gym.spaces.Box(
            low=np.array(space.low, copy=True),
            high=np.array(space.high, copy=True),
            shape=space.shape,
            dtype=space.dtype,
        )
    if isinstance(space, legacy_gym.spaces.Discrete):
        return gym.spaces.Discrete(space.n)
    if isinstance(space, legacy_gym.spaces.MultiDiscrete):
        return gym.spaces.MultiDiscrete(np.array(space.nvec, copy=True))
    if isinstance(space, legacy_gym.spaces.MultiBinary):
        return gym.spaces.MultiBinary(space.n)
    if isinstance(space, legacy_gym.spaces.Dict):
        return gym.spaces.Dict(
            {key: _to_gymnasium_space(value) for key, value in space.spaces.items()}
        )

    raise TypeError(f"Unsupported space type for Gymnasium conversion: {type(space)!r}")


def _get_wrapped_space(env: Any, name: str) -> gym.Space:
    if hasattr(env, name):
        return _to_gymnasium_space(getattr(env, name))
    wrapped = getattr(env, "env", None)
    if wrapped is not None and hasattr(wrapped, name):
        return _to_gymnasium_space(getattr(wrapped, name))
    raise AttributeError(f"CyclesGym environment does not expose {name}.")


def _location_from_env_id(env_id: str) -> str | None:
    for location in ("RockSprings", "NewHolland"):
        if location in env_id:
            return location
    return None


def _task_from_env_id(env_id: str) -> str:
    if env_id.startswith("CropPlanning"):
        return "crop_planning"
    if env_id.startswith("Corn"):
        return "corn"
    return "unknown"


def _compact_env_id_label(env_id: str) -> str:
    parts: list[str] = []
    if env_id.startswith("CropPlanning"):
        parts.append("plan")
    elif env_id.startswith("Corn"):
        parts.append("corn")
        if "Short" in env_id:
            parts.append("short")
    else:
        parts.append("env")

    location = _location_from_env_id(env_id)
    if location == "RockSprings":
        parts.append("rs")
    elif location == "NewHolland":
        parts.append("nh")

    if "RW" in env_id:
        parts.append("rw")
    elif "FW" in env_id:
        parts.append("fw")

    return "_".join(parts)


def _year_window_from_env_config(env_config: dict[str, Any]) -> str | None:
    env_kwargs = env_config.get("env_kwargs") or {}
    start_year = env_kwargs.get("start_year")
    end_year = env_kwargs.get("end_year")
    if start_year is None or end_year is None:
        return None
    return f"{start_year}-{end_year}"


def _default_group_labels(env_config: dict[str, Any]) -> dict[str, str]:
    labels: dict[str, str] = {}

    raw_labels = env_config.get("trajectory_group_labels", {})
    if isinstance(raw_labels, dict):
        for key, value in raw_labels.items():
            key_str = str(key or "").strip()
            value_str = str(value or "").strip()
            if key_str and value_str:
                labels[key_str] = value_str

    env_id = str(env_config.get("env_id") or "").strip()
    if env_id:
        labels.setdefault("env_id", _compact_env_id_label(env_id))
        labels.setdefault("task", _task_from_env_id(env_id))
        location = _location_from_env_id(env_id)
        if location:
            labels.setdefault("location", location)

    dataset_split = str(env_config.get("dataset_split") or "").strip()
    if dataset_split:
        labels.setdefault("dataset_split", dataset_split)

    year_window = _year_window_from_env_config(env_config)
    if year_window:
        labels.setdefault("year_window", year_window)

    return labels


def _runtime_path_for_env(env_config: dict[str, Any]) -> str:
    raw_base = env_config.get("cycles_runtime_path")
    base = Path(str(raw_base)).expanduser() if raw_base else TMP_ROOT / "nn_runtime"
    return str((base / f"pid_{os.getpid()}" / f"env_{next(_RUNTIME_COUNTER)}").resolve())


def _coerce_observation(observation: Any, observation_space: gym.Space) -> Any:
    if isinstance(observation_space, gym.spaces.Box):
        return np.asarray(observation, dtype=observation_space.dtype)
    return observation


def _coerce_action(action: Any, action_space: gym.Space) -> Any:
    if isinstance(action_space, gym.spaces.Discrete):
        return int(np.asarray(action).item())
    if isinstance(action_space, gym.spaces.MultiDiscrete):
        return np.asarray(action, dtype=np.int64)
    if isinstance(action_space, gym.spaces.MultiBinary):
        return np.asarray(action, dtype=np.int8)
    if isinstance(action_space, gym.spaces.Box):
        return np.asarray(action, dtype=action_space.dtype)
    return action


def _as_list(value: Any) -> list[Any]:
    """Convert parquet/Arrow/numpy/list values to a plain Python list."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "tolist"):
        try:
            converted = value.tolist()
            return converted if isinstance(converted, list) else [converted]
        except Exception:
            pass
    try:
        return list(value)
    except TypeError:
        return [value]


def _as_dict(value: Any) -> dict[str, Any]:
    """Convert parquet/Arrow/numpy/dict-like values to a plain Python dict."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "as_py"):
        try:
            converted = value.as_py()
            if isinstance(converted, dict):
                return dict(converted)
        except Exception:
            pass
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if hasattr(value, "tolist") and not isinstance(value, (list, tuple)):
        try:
            value = value.tolist()
        except Exception:
            pass
    try:
        return dict(value)
    except Exception:
        return {}


class CyclesGymSB3Env(gym.Env):
    """Gymnasium-compatible numeric wrapper around AgriManager's CyclesGym env."""

    metadata = {"render_modes": []}

    def __init__(self, env_config: dict[str, Any]) -> None:
        numeric_config = copy.deepcopy(env_config)
        numeric_config["llm_mode"] = False
        numeric_config["cycles_runtime_path"] = _runtime_path_for_env(numeric_config)

        base_env, _ = create_environment("cycles_gym", numeric_config)
        self.base_env = base_env
        self.env_config = numeric_config
        self.group_labels = _default_group_labels(numeric_config)

        self._base_observation_space = _get_wrapped_space(base_env, "observation_space")
        self.action_space = _get_wrapped_space(base_env, "action_space")
        self._augment_crop_planning_observation = str(numeric_config.get("env_id", "")).startswith("CropPlanning")
        env_kwargs = _as_dict(numeric_config.get("env_kwargs"))
        self._rotation_crops = [str(crop) for crop in _as_list(env_kwargs.get("rotation_crops"))]
        self._crop_prices = _as_dict(env_kwargs.get("crop_prices"))
        self._history_records: list[dict[str, float]] = []
        self._year_idx = 0
        self._horizon = int(env_kwargs.get("end_year", 1998)) - int(env_kwargs.get("start_year", 1980)) + 1
        self._horizon = max(1, self._horizon)
        self.observation_space = self._build_observation_space()

    def _build_observation_space(self) -> gym.Space:
        if (
            not self._augment_crop_planning_observation
            or not isinstance(self._base_observation_space, gym.spaces.Box)
        ):
            return self._base_observation_space

        base_dim = int(np.prod(self._base_observation_space.shape))
        n_crops = max(1, len(self._rotation_crops))
        extra_dim = 2 + n_crops + self._horizon * (n_crops + 3)
        return gym.spaces.Box(
            low=np.full((base_dim + extra_dim,), -np.inf, dtype=np.float32),
            high=np.full((base_dim + extra_dim,), np.inf, dtype=np.float32),
            dtype=np.float32,
        )

    def _price_vector(self) -> np.ndarray:
        values = []
        for crop in self._rotation_crops:
            price = self._crop_prices.get(crop, 0.0)
            if isinstance(price, dict):
                price = next(iter(price.values())) if price else 0.0
            try:
                values.append(float(price) / 1000.0)
            except Exception:
                values.append(0.0)
        return np.asarray(values, dtype=np.float32)

    def _augment_observation(self, observation: Any) -> Any:
        observation = _coerce_observation(observation, self._base_observation_space)
        if (
            not self._augment_crop_planning_observation
            or not isinstance(self._base_observation_space, gym.spaces.Box)
        ):
            return observation

        base = np.asarray(observation, dtype=np.float32).reshape(-1)
        year_features = np.asarray(
            [
                self._year_idx / max(1, self._horizon - 1),
                max(0, self._horizon - self._year_idx) / self._horizon,
            ],
            dtype=np.float32,
        )
        history = np.zeros(
            (self._horizon, len(self._rotation_crops) + 3),
            dtype=np.float32,
        )
        for idx, record in enumerate(self._history_records[: self._horizon]):
            crop_idx = int(record.get("crop_idx", -1))
            if 0 <= crop_idx < len(self._rotation_crops):
                history[idx, crop_idx] = 1.0
            offset = len(self._rotation_crops)
            history[idx, offset] = float(record.get("planting_week", 0.0)) / 13.0
            history[idx, offset + 1] = float(record.get("yield_tonnes", 0.0)) / 10.0
            history[idx, offset + 2] = float(record.get("revenue", 0.0)) / 5000.0

        return np.concatenate(
            [base, year_features, self._price_vector(), history.reshape(-1)]
        ).astype(np.float32)

    def _record_transition(self, action: Any, reward: float, info: dict[str, Any]) -> None:
        if not self._augment_crop_planning_observation:
            return

        action_array = np.asarray(action, dtype=np.int64).reshape(-1)
        crop_idx = int(action_array[0]) if action_array.size else -1
        planting_week = int(action_array[1]) if action_array.size > 1 else 0
        metrics = dict((info or {}).get("turn_metrics") or {})
        crop_yield = metrics.get("crop_planning/yield_tonnes", 0.0)
        revenue = metrics.get("crop_planning/revenue", reward)
        self._history_records.append(
            {
                "crop_idx": float(crop_idx),
                "planting_week": float(planting_week),
                "yield_tonnes": float(crop_yield or 0.0),
                "revenue": float(revenue or 0.0),
            }
        )
        self._year_idx = min(self._horizon, self._year_idx + 1)

    def _augment_info(self, info: dict[str, Any] | None) -> dict[str, Any]:
        augmented = dict(info or {})
        augmented.setdefault("env_config", copy.deepcopy(self.env_config))
        for group_name, group_value in self.group_labels.items():
            augmented[f"group_label/{group_name}"] = group_value
        return augmented

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        del seed, options
        self._history_records = []
        self._year_idx = 0
        observation, info = self.base_env.reset()
        return (
            self._augment_observation(observation),
            self._augment_info(info),
        )

    def step(self, action: Any):
        env_action = _coerce_action(action, self.action_space)
        observation, reward, done, info = self.base_env.step(env_action)
        augmented_info = self._augment_info(info)
        self._record_transition(env_action, float(reward), augmented_info)
        return (
            self._augment_observation(observation),
            float(reward),
            bool(done),
            False,
            augmented_info,
        )

    def close(self):
        self.base_env.close()


class CyclesGymNNEnvAdapter(BaseNNEnvAdapter):
    """Build CyclesGym numeric environments for the generic NN PPO path."""

    def make_env(self, env_config: dict[str, Any]) -> gym.Env:
        return CyclesGymSB3Env(env_config=dict(env_config))

    def group_labels(
        self,
        env_config: dict[str, Any],
        info: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        labels = _default_group_labels(env_config)
        for key, value in dict(info or {}).items():
            if not isinstance(key, str) or not key.startswith("group_label/"):
                continue
            group_name = key[len("group_label/"):].strip()
            group_value = str(value or "").strip()
            if group_name and group_value:
                labels[group_name] = group_value
        return labels


NNEnvAdapter = CyclesGymNNEnvAdapter

__all__ = [
    "CyclesGymSB3Env",
    "CyclesGymNNEnvAdapter",
    "NNEnvAdapter",
]
