"""Configuration for cycles_gym environments."""

from __future__ import annotations

import os
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

from agrimanager.env.base import BaseEnvConfig

AGRIMANAGER_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = AGRIMANAGER_PACKAGE_ROOT.parent

DEFAULT_CYCLES_GYM_PATH = os.environ.get(
    "CYCLES_GYM_PATH",
    str((REPO_ROOT / ".." / "AgriManagerExternal" / "CyclesGym").resolve()),
)
DEFAULT_CYCLES_RUNTIME_PATH = os.environ.get(
    "CYCLESGYM_RUNTIME_CYCLES_PATH",
    str((Path(tempfile.gettempdir()) / "agrimanager" / "cycles_gym" / "runtime").resolve()),
)


def _ensure_trailing_sep(path: str) -> str:
    if path.endswith(("/", "\\")):
        return path
    return path + os.sep


def _resolve_repo_relative_path(path: str | None) -> str | None:
    if path is None:
        return None
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    return str(resolved.resolve())


def _crop_planning_location_from_env_id(env_id: str) -> Optional[str]:
    for location in ("RockSprings", "NewHolland"):
        if location in env_id:
            return location
    return None


def _normalize_weather_generator_kwargs(weather_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    normalized = deepcopy(weather_kwargs)
    base_weather_file = normalized.get("base_weather_file")
    if isinstance(base_weather_file, str):
        normalized["base_weather_file"] = Path(_resolve_repo_relative_path(base_weather_file))
    return normalized


def _normalize_cycles_env_kwargs(
    env_id: str,
    cycles_gym_path: str,
    env_kwargs: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Normalize env kwargs and auto-fill crop-planning RW weather settings."""
    kwargs = deepcopy(env_kwargs or {})
    weather_kwargs = deepcopy(kwargs.get("weather_generator_kwargs") or {})

    if weather_kwargs:
        kwargs["weather_generator_kwargs"] = _normalize_weather_generator_kwargs(weather_kwargs)

    if not env_id.startswith("CropPlanning") or "RW" not in env_id:
        return kwargs

    start_year = kwargs.get("start_year")
    end_year = kwargs.get("end_year")
    if start_year is None or end_year is None:
        return kwargs

    start_year = int(start_year)
    end_year = int(end_year)
    if end_year < start_year:
        raise ValueError(f"Invalid year window for {env_id}: {start_year}>{end_year}")

    location = _crop_planning_location_from_env_id(env_id)
    if not location:
        return kwargs

    cycles_root = Path(cycles_gym_path).expanduser().resolve()
    weather_kwargs = deepcopy(kwargs.get("weather_generator_kwargs") or {})
    weather_kwargs.setdefault("n_weather_samples", 100)
    weather_kwargs.setdefault("sampling_start_year", start_year)
    weather_kwargs.setdefault("sampling_end_year", end_year)
    weather_kwargs.setdefault(
        "base_weather_file",
        cycles_root / "cycles" / "input" / f"{location}.weather",
    )
    weather_kwargs.setdefault("target_year_range", list(range(start_year, end_year + 1)))
    kwargs["weather_generator_kwargs"] = _normalize_weather_generator_kwargs(weather_kwargs)
    return kwargs


class CyclesEnvConfig(BaseEnvConfig):
    """Configuration wrapper for CyclesGym environments."""

    def __init__(
        self,
        env_id: str = "CornShortRockSpringsFW-v1",
        cycles_gym_path: str = DEFAULT_CYCLES_GYM_PATH,
        cycles_runtime_path: Optional[str] = DEFAULT_CYCLES_RUNTIME_PATH,
        llm_mode: bool = True,
        include_crop_traits: bool = True,
        require_think: bool = False,
        thinking_mode: str = "grounding_decision",
        think_tag: str = "think",
        reward_mode: str = "native",
        valid_action_bonus: float = 0.1,
        invalid_action_penalty: float = 0.0,
        invalid_action_fallback: str = "default",
        reward_scale: float = 1.0,
        env_kwargs: Optional[Dict[str, Any]] = None,
        turn_num: int = 200,
        seed: Optional[int] = None,
        objective_id: str = "profit_max",
        prompt_objective_id: Optional[str] = None,
        prompt_objective_text: Optional[str] = None,
        reward_params: Optional[Dict[str, Any]] = None,
        include_profit_context: bool = False,
        profit_context_params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        super().__init__(seed=seed, llm_mode=llm_mode, turn_num=turn_num, **kwargs)

        self.env_id = env_id
        resolved_cycles_path = _resolve_repo_relative_path(cycles_gym_path)
        self.cycles_gym_path = _ensure_trailing_sep(str(resolved_cycles_path))
        self.cycles_runtime_path = _resolve_repo_relative_path(cycles_runtime_path)
        self.include_crop_traits = include_crop_traits
        self.require_think = require_think
        self.thinking_mode = thinking_mode
        self.think_tag = think_tag
        self.reward_mode = str(reward_mode)
        self.valid_action_bonus = 0.1 if valid_action_bonus is None else float(valid_action_bonus)
        self.invalid_action_penalty = float(invalid_action_penalty)
        self.invalid_action_fallback = str(invalid_action_fallback or "default").strip().lower()
        self.reward_scale = float(reward_scale)
        if self.reward_scale <= 0.0:
            raise ValueError(f"reward_scale must be positive, got {reward_scale}")
        self.objective_id = objective_id
        self.prompt_objective_id = prompt_objective_id
        self.prompt_objective_text = prompt_objective_text
        self.reward_params = deepcopy(reward_params or {})
        self.include_profit_context = bool(include_profit_context)
        self.profit_context_params = deepcopy(profit_context_params or {})

        if env_kwargs is None:
            if env_id.startswith("Corn"):
                env_kwargs = {
                    "delta": 7,
                    "n_actions": 11,
                    "maxN": 150,
                    "start_year": 1980,
                    "end_year": 1980,
                }
            else:
                env_kwargs = {}

        self.env_kwargs = _normalize_cycles_env_kwargs(
            env_id=env_id,
            cycles_gym_path=self.cycles_gym_path,
            env_kwargs=env_kwargs,
        )

    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict.update(
            {
                "env_id": self.env_id,
                "cycles_gym_path": self.cycles_gym_path,
                "cycles_runtime_path": self.cycles_runtime_path,
                "include_crop_traits": self.include_crop_traits,
                "require_think": self.require_think,
                "thinking_mode": self.thinking_mode,
                "think_tag": self.think_tag,
                "reward_mode": self.reward_mode,
                "valid_action_bonus": self.valid_action_bonus,
                "invalid_action_penalty": self.invalid_action_penalty,
                "invalid_action_fallback": self.invalid_action_fallback,
                "reward_scale": self.reward_scale,
                "objective_id": self.objective_id,
                "prompt_objective_id": self.prompt_objective_id,
                "prompt_objective_text": self.prompt_objective_text,
                "reward_params": self.reward_params,
                "include_profit_context": self.include_profit_context,
                "profit_context_params": self.profit_context_params,
                "env_kwargs": self.env_kwargs,
            }
        )
        return base_dict
