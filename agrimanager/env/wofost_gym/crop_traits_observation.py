"""Numeric crop-traits observation features for WOFOSTGym RL agents."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import gymnasium as gym
import numpy as np
import yaml
from gymnasium import spaces

from .crop_trait_schemas import (
    DEFAULT_CROP_TRAIT_SCHEMA,
    crop_variety_trait_key,
    normalize_crop_trait_schema,
    resolve_crop_trait_schema_dir,
)
from .env_config import DEFAULT_CROP_TRAITS_DIR, REPO_ROOT


def crop_name_from_agro_file(agro_file: str | None) -> str:
    """Infer crop name from an agromanagement filename."""
    name = Path(str(agro_file or "")).name
    if name.endswith("_agro.yaml"):
        return name[:-10]
    if name.endswith(".yaml"):
        return name[:-5]
    return name


def _candidate_agro_paths(env_config: dict[str, Any]) -> tuple[Path, ...]:
    agro_file = str(env_config.get("agro_file", "") or "").strip()
    if not agro_file:
        return ()

    agro_path = Path(agro_file)
    if agro_path.is_absolute():
        return (agro_path,)

    candidates: list[Path] = []
    wofost_gym_path = str(env_config.get("wofost_gym_path", "") or "").strip()
    if wofost_gym_path:
        wofost_root = Path(wofost_gym_path)
        candidates.append(wofost_root / "env_config" / "agro" / agro_file)
        candidates.append(wofost_root / agro_file)
    candidates.append(REPO_ROOT / agro_file)

    seen: set[str] = set()
    unique_candidates: list[Path] = []
    for candidate in candidates:
        key = str(candidate.expanduser().resolve())
        if key not in seen:
            unique_candidates.append(candidate)
            seen.add(key)
    return tuple(unique_candidates)


@lru_cache(maxsize=128)
def _crop_variety_from_agro_path(agro_path: str) -> str:
    path = Path(agro_path)
    if not path.is_file():
        return ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        return ""
    if not isinstance(loaded, dict):
        return ""
    agro_management = loaded.get("AgroManagement", loaded)
    if not isinstance(agro_management, dict):
        return ""
    crop_calendar = agro_management.get("CropCalendar", {})
    if not isinstance(crop_calendar, dict):
        return ""
    return str(crop_calendar.get("crop_variety", "") or "").strip()


def crop_variety_from_env_config(env_config: dict[str, Any]) -> str:
    """Return crop variety from the canonical dataset env_config shape."""
    crop_variety = str(env_config.get("crop_variety", "") or "").strip()
    if crop_variety:
        return crop_variety
    agro_params = env_config.get("agro_params") or {}
    if isinstance(agro_params, dict):
        crop_variety = str(agro_params.get("crop_variety", "") or "").strip()
        if crop_variety:
            return crop_variety
    for agro_path in _candidate_agro_paths(env_config):
        crop_variety = _crop_variety_from_agro_path(str(agro_path.expanduser().resolve()))
        if crop_variety:
            return crop_variety
    return ""


def _flatten_numeric(prefix: tuple[str, ...], obj: Any) -> dict[str, float]:
    values: dict[str, float] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            values.update(_flatten_numeric((*prefix, str(key)), value))
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        values[".".join(prefix)] = float(obj)
    return values


def _default_traits_dir() -> Path:
    return (REPO_ROOT / DEFAULT_CROP_TRAITS_DIR).resolve()


class CropTraitEncoder:
    """Load structured crop-trait cards and expose fixed numeric vectors."""

    def __init__(
        self,
        traits_dir: str | Path | None = None,
        feature_root: str = "core_facts",
        trait_schema: str = DEFAULT_CROP_TRAIT_SCHEMA,
    ) -> None:
        if traits_dir:
            traits_path = Path(traits_dir)
            if not traits_path.is_absolute():
                traits_path = REPO_ROOT / traits_path
            self._base_traits_dir = traits_path.resolve()
        else:
            self._base_traits_dir = _default_traits_dir()
        self.trait_schema = normalize_crop_trait_schema(trait_schema)
        self.traits_dir = resolve_crop_trait_schema_dir(
            self._base_traits_dir,
            self.trait_schema,
        )
        self.feature_root = feature_root
        self._raw_by_crop = self._load_cards()
        self._trait_key_by_base_crop = self._build_base_crop_index()
        self.feature_names = tuple(sorted(set().union(*self._raw_by_crop.values())))
        if not self.feature_names:
            raise ValueError(f"No numeric crop-trait features found in {self.traits_dir}")

        raw_matrix = np.array(
            [
                [crop_values.get(name, np.nan) for name in self.feature_names]
                for crop_values in self._raw_by_crop.values()
            ],
            dtype=np.float32,
        )
        with np.errstate(all="ignore"):
            mins = np.nanmin(raw_matrix, axis=0)
            maxs = np.nanmax(raw_matrix, axis=0)
        mins = np.nan_to_num(mins, nan=0.0)
        maxs = np.nan_to_num(maxs, nan=0.0)
        scales = maxs - mins
        scales[scales == 0.0] = 1.0

        self.minimums = mins.astype(np.float32)
        self.maximums = maxs.astype(np.float32)
        self.scales = scales.astype(np.float32)
        self.crop_names = tuple(sorted(self._raw_by_crop))

    @property
    def dim(self) -> int:
        return len(self.feature_names)

    def _load_cards(self) -> dict[str, dict[str, float]]:
        if not self.traits_dir.exists():
            raise FileNotFoundError(f"Crop traits directory not found: {self.traits_dir}")

        raw_by_crop: dict[str, dict[str, float]] = {}
        for path in sorted(self.traits_dir.glob("*.json")):
            with open(path, "r", encoding="utf-8") as f:
                card = json.load(f)
            root_obj = card.get(self.feature_root, {}) if self.feature_root else card
            flat = _flatten_numeric((self.feature_root,), root_obj)
            if flat:
                crop_name = str(card.get("crop") or "").strip()
                crop_variety = str(card.get("variety") or "").strip()
                trait_key = str(card.get("trait_key") or "").strip()
                if not trait_key:
                    trait_key = (
                        crop_variety_trait_key(crop_name, crop_variety)
                        if crop_name and crop_variety
                        else (crop_name or path.stem)
                    )
                raw_by_crop[trait_key] = flat

        if not raw_by_crop:
            raise FileNotFoundError(f"No crop trait JSON files found in {self.traits_dir}")
        return raw_by_crop

    def _build_base_crop_index(self) -> dict[str, str]:
        """Map base crop names to their unique crop-variety trait key."""
        keys_by_base_crop: dict[str, list[str]] = {}
        for trait_key in self._raw_by_crop:
            base_crop = trait_key.split("__", 1)[0].strip()
            if not base_crop:
                continue
            keys_by_base_crop.setdefault(base_crop, []).append(trait_key)
        return {
            base_crop: keys[0]
            for base_crop, keys in keys_by_base_crop.items()
            if len(keys) == 1 and keys[0] != base_crop
        }

    def _resolve_trait_key(self, crop_name: str) -> str:
        crop_key = str(crop_name or "").strip()
        if crop_key in self._raw_by_crop:
            return crop_key

        resolved = self._trait_key_by_base_crop.get(crop_key)
        if resolved:
            return resolved

        available = ", ".join(self.crop_names)
        raise KeyError(
            f"Crop traits not found for crop '{crop_name}' in {self.traits_dir}. "
            f"Available crops: {available}"
        )

    def vector_for_crop(self, crop_name: str) -> np.ndarray:
        """Return a min-max normalized trait vector for a crop."""
        values = self._raw_by_crop[self._resolve_trait_key(crop_name)]
        raw = np.array(
            [values.get(name, np.nan) for name in self.feature_names],
            dtype=np.float32,
        )
        raw = np.where(np.isfinite(raw), raw, self.minimums)
        normalized = (raw - self.minimums) / self.scales
        return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def extend_observation_space(
    observation_space: gym.Space,
    encoder: CropTraitEncoder,
) -> spaces.Box:
    """Append normalized trait feature bounds to a 1-D Box observation space."""
    if not isinstance(observation_space, spaces.Box):
        raise TypeError(
            f"Crop trait augmentation expects Box observation space, got {observation_space!r}"
        )
    if len(observation_space.shape) != 1:
        raise ValueError(
            f"Crop trait augmentation expects 1-D observations, got shape={observation_space.shape}"
        )

    low = np.concatenate(
        [
            np.asarray(observation_space.low, dtype=np.float32).reshape(-1),
            np.zeros(encoder.dim, dtype=np.float32),
        ]
    )
    high = np.concatenate(
        [
            np.asarray(observation_space.high, dtype=np.float32).reshape(-1),
            np.ones(encoder.dim, dtype=np.float32),
        ]
    )
    return spaces.Box(low=low, high=high, dtype=np.float32)


def append_crop_traits(
    observation: np.ndarray,
    crop_name: str,
    encoder: CropTraitEncoder,
) -> np.ndarray:
    """Append a crop's normalized trait vector to an observation."""
    obs = np.asarray(observation, dtype=np.float32).reshape(-1)
    traits = encoder.vector_for_crop(crop_name)
    return np.concatenate([obs, traits]).astype(np.float32)


class CropTraitsObservationWrapper(gym.ObservationWrapper):
    """Append normalized structured crop-trait features to each observation."""

    def __init__(
        self,
        env: gym.Env,
        encoder: CropTraitEncoder,
        agro_file: str | None = None,
    ) -> None:
        super().__init__(env)
        self.encoder = encoder
        self._fallback_crop_name = crop_name_from_agro_file(agro_file) if agro_file else None
        self.observation_space = extend_observation_space(env.observation_space, encoder)

        if hasattr(env, "ranges"):
            trait_ranges = np.stack(
                [
                    np.zeros(encoder.dim, dtype=np.float64),
                    np.ones(encoder.dim, dtype=np.float64),
                ],
                axis=1,
            )
            self.ranges = np.vstack([env.ranges, trait_ranges])

        if hasattr(env, "reward_range"):
            self.reward_range = env.reward_range

    def observation(self, observation: np.ndarray) -> np.ndarray:
        crop_name = self._current_crop_name()
        return append_crop_traits(observation, crop_name, self.encoder)

    def _current_crop_name(self) -> str:
        agro_file = self._current_agro_file()
        if agro_file:
            crop_name = crop_name_from_agro_file(agro_file)
            if crop_name:
                return crop_name

        if self._fallback_crop_name:
            return self._fallback_crop_name

        raise RuntimeError("Unable to infer current crop for crop-trait observation features")

    def _current_agro_file(self) -> str | None:
        for wrapped in self._iter_wrappers(self.env):
            for attr in ("_current_agro_file", "agro_file", "agro_fpath"):
                if hasattr(wrapped, attr):
                    value = getattr(wrapped, attr)
                    if value:
                        return str(value)
        return None

    @staticmethod
    def _iter_wrappers(env: gym.Env) -> Iterable[gym.Env]:
        current = env
        seen: set[int] = set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            yield current
            current = getattr(current, "env", None)
