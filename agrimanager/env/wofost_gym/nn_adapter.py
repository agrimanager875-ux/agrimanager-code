"""NN adapter for framework-native WOFOST numeric training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from agrimanager.env.base import BaseNNEnvAdapter, create_environment

from .crop_traits_observation import (
    CropTraitEncoder,
    append_crop_traits,
    crop_name_from_agro_file,
    crop_variety_from_env_config,
    extend_observation_space,
)
from .crop_trait_schemas import crop_variety_trait_key


def _get_observation_space(env: Any) -> gym.Space:
    if hasattr(env, "env") and hasattr(env.env, "observation_space"):
        return env.env.observation_space
    if hasattr(env, "observation_space"):
        return env.observation_space
    raise AttributeError("Environment does not expose an observation_space.")


def _get_action_space(env: Any) -> gym.Space:
    if hasattr(env, "env") and hasattr(env.env, "action_space"):
        return env.env.action_space
    if hasattr(env, "action_space"):
        return env.action_space
    raise AttributeError("Environment does not expose an action_space.")


def _extract_crop_name(env_config: dict[str, Any]) -> str:
    crop_name = str(env_config.get("crop_name", "") or "").strip()
    if crop_name:
        return crop_name
    return crop_name_from_agro_file(env_config.get("agro_file"))


def _extract_trait_key(
    env_config: dict[str, Any],
    crop_name: str,
    trait_encoder: CropTraitEncoder | None = None,
) -> str:
    crop_variety = crop_variety_from_env_config(env_config)
    variety_trait_key = crop_variety_trait_key(crop_name, crop_variety)
    if bool(env_config.get("include_variety_traits", False)):
        return variety_trait_key

    if trait_encoder is None:
        return crop_name

    available_trait_keys = set(getattr(trait_encoder, "crop_names", ()))
    if crop_name in available_trait_keys:
        return crop_name
    if variety_trait_key in available_trait_keys:
        return variety_trait_key
    return crop_name


def _extract_group_labels(env_config: dict[str, Any]) -> dict[str, str]:
    raw_group_labels = env_config.get("trajectory_group_labels", {})
    if not isinstance(raw_group_labels, dict):
        return {}
    labels: dict[str, str] = {}
    for key, value in raw_group_labels.items():
        key_str = str(key or "").strip()
        value_str = str(value or "").strip()
        if key_str and value_str:
            labels[key_str] = value_str
    return labels


class WOFOSTSB3Env(gym.Env):
    """Gymnasium-compatible numeric wrapper around AgriManager's WOFOST env."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        env_config: dict[str, Any],
        trait_encoder: CropTraitEncoder | None = None,
    ) -> None:
        numeric_config = dict(env_config)
        numeric_config["llm_mode"] = False
        base_env, _ = create_environment("wofost_gym", numeric_config)
        self.base_env = base_env
        self.env_config = numeric_config
        self.crop_name = _extract_crop_name(numeric_config)
        self.crop_variety = crop_variety_from_env_config(numeric_config)
        self.trait_key = _extract_trait_key(numeric_config, self.crop_name, trait_encoder)
        self.group_labels = _extract_group_labels(numeric_config)
        self.trait_encoder = trait_encoder

        raw_observation_space = _get_observation_space(base_env)
        if not isinstance(raw_observation_space, spaces.Box):
            raise TypeError(
                f"WOFOST numeric training expects Box observation space, got {raw_observation_space!r}"
            )
        self._raw_observation_space = raw_observation_space
        self.action_space = _get_action_space(base_env)
        self.observation_space = (
            extend_observation_space(raw_observation_space, trait_encoder)
            if trait_encoder is not None
            else raw_observation_space
        )

    def _transform_observation(self, observation: np.ndarray) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32).reshape(-1)
        if self.trait_encoder is None:
            return obs
        return append_crop_traits(obs, self.trait_key, self.trait_encoder)

    def _augment_info(self, info: dict[str, Any] | None) -> dict[str, Any]:
        augmented = dict(info or {})
        augmented.setdefault("env_config", self.env_config)
        augmented.setdefault("crop_name", self.crop_name)
        if self.crop_variety:
            augmented.setdefault("crop_variety", self.crop_variety)
        for group_name, group_value in self.group_labels.items():
            augmented[f"group_label/{group_name}"] = group_value
        return augmented

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        del seed, options
        observation, info = self.base_env.reset()
        return self._transform_observation(observation), self._augment_info(info)

    def step(self, action: int):
        observation, reward, done, info = self.base_env.step(int(action))
        return self._transform_observation(observation), float(reward), bool(done), False, self._augment_info(info)

    def close(self):
        self.base_env.close()


class WOFOSTNNEnvAdapter(BaseNNEnvAdapter):
    """WOFOST-first adapter for framework-native numeric PPO training."""

    def __init__(self) -> None:
        self._trait_encoder_cache: dict[tuple[str, str], CropTraitEncoder] = {}

    def _trait_encoder_for_config(self, env_config: dict[str, Any]) -> CropTraitEncoder | None:
        if not bool(env_config.get("include_crop_traits", False)):
            return None

        trait_schema = str(env_config.get("trait_schema", "") or "").strip() or "default"
        traits_dir = Path(str(env_config.get("crop_traits_dir") or "")).resolve()
        cache_key = (str(traits_dir), trait_schema)
        encoder = self._trait_encoder_cache.get(cache_key)
        if encoder is None:
            encoder = CropTraitEncoder(traits_dir=traits_dir, trait_schema=trait_schema)
            self._trait_encoder_cache[cache_key] = encoder
        return encoder

    def make_env(self, env_config: dict[str, Any]) -> gym.Env:
        return WOFOSTSB3Env(
            env_config=dict(env_config),
            trait_encoder=self._trait_encoder_for_config(env_config),
        )


NNEnvAdapter = WOFOSTNNEnvAdapter

__all__ = [
    "WOFOSTSB3Env",
    "WOFOSTNNEnvAdapter",
    "NNEnvAdapter",
]
