"""cycles_gym environment wrapper."""

from __future__ import annotations

import math
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Tuple, Union

from agrimanager.env.base import BaseEnv
from agrimanager.env.base.objective_prompt import profit_cost_params, profit_reward_scale

from .env_config import CyclesEnvConfig
from .prompt import build_prompt_generator


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
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
    if hasattr(value, "tolist"):
        try:
            value = value.tolist()
        except Exception:
            pass
    try:
        return dict(value)
    except Exception:
        return {}


class CyclesEnv(BaseEnv):
    """Adapt CyclesGym environments to the AgriManager BaseEnv contract."""

    @staticmethod
    def _runtime_root_for_process(config: CyclesEnvConfig) -> Path | None:
        if not config.cycles_runtime_path:
            return None
        base_runtime_root = Path(config.cycles_runtime_path).expanduser().resolve()
        return base_runtime_root / f"worker_{os.getpid()}"

    @staticmethod
    def _prepare_runtime_tree(config: CyclesEnvConfig) -> None:
        runtime_root = CyclesEnv._runtime_root_for_process(config)
        if runtime_root is None:
            return

        repo_root = Path(config.cycles_gym_path).expanduser().resolve()
        base_cycles_root = repo_root / "cycles"
        runtime_input = runtime_root / "input"
        runtime_output = runtime_root / "output"
        runtime_binary = runtime_root / "Cycles"
        base_binary = base_cycles_root / "Cycles"
        mpl_config_dir = runtime_root / "mplconfig"

        runtime_root.mkdir(parents=True, exist_ok=True)
        runtime_input.mkdir(parents=True, exist_ok=True)
        runtime_output.mkdir(parents=True, exist_ok=True)
        mpl_config_dir.mkdir(parents=True, exist_ok=True)
        if base_binary.exists():
            try:
                base_binary.chmod(base_binary.stat().st_mode | 0o700)
            except OSError:
                if not os.access(base_binary, os.X_OK):
                    raise
            if runtime_binary.exists() or runtime_binary.is_symlink():
                runtime_binary.chmod(runtime_binary.stat().st_mode | 0o700)
            else:
                tmp_binary = runtime_root / f".Cycles.{os.getpid()}.{id(config)}.tmp"
                shutil.copy2(base_binary, tmp_binary)
                tmp_binary.chmod(tmp_binary.stat().st_mode | 0o700)
                os.replace(tmp_binary, runtime_binary)
            runtime_binary.chmod(runtime_binary.stat().st_mode | 0o700)

        os.environ["CYCLESGYM_BASE_CYCLES_PATH"] = str(base_cycles_root)
        os.environ["CYCLESGYM_RUNTIME_CYCLES_PATH"] = str(runtime_root)
        os.environ["CYCLESGYM_INPUT_PATH"] = str(runtime_input)
        os.environ["CYCLESGYM_OUTPUT_PATH"] = str(runtime_output)
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    @staticmethod
    def _location_from_env_id(env_id: str) -> str:
        if "NewHolland" in env_id:
            return "NewHolland"
        return "RockSprings"

    @classmethod
    def _create_cyclesgym_env(cls, config: CyclesEnvConfig):
        try:
            import gym  # type: ignore
        except ModuleNotFoundError:
            import gymnasium as gym  # type: ignore

            # CyclesGym imports the legacy ``gym`` namespace directly.
            sys.modules.setdefault("gym", gym)

        import cyclesgym
        from cyclesgym.envs import Corn, CropPlanningFixedPlanting

        env_kwargs = dict(config.env_kwargs or {})
        env_id = str(config.env_id)
        location = cls._location_from_env_id(env_id)
        random_weather = "RW" in env_id

        if env_id.startswith("Corn"):
            start_year = int(env_kwargs.pop("start_year", 1980))
            end_year = int(env_kwargs.pop("end_year", start_year))
            weather_generator_class, weather_generator_kwargs = cyclesgym.get_weather(
                start_year,
                end_year,
                random=random_weather,
                location=location,
            )
            explicit_weather_kwargs = env_kwargs.pop("weather_generator_kwargs", None)
            if explicit_weather_kwargs:
                weather_generator_kwargs = explicit_weather_kwargs
            return gym, Corn(
                start_year=start_year,
                end_year=end_year,
                weather_generator_class=weather_generator_class,
                weather_generator_kwargs=weather_generator_kwargs,
                **env_kwargs,
            )

        if env_id.startswith("CropPlanning"):
            start_year = int(env_kwargs.pop("start_year", 1980))
            end_year = int(env_kwargs.pop("end_year", 1998))
            rotation_crops = env_kwargs.pop(
                "rotation_crops",
                ["CornSilageRM.90", "SoybeanMG.3"],
            )
            weather_generator_class, weather_generator_kwargs = cyclesgym.get_weather(
                start_year,
                end_year,
                random=random_weather,
                location=location,
                sampling_start_year=start_year,
                sampling_end_year=end_year,
            )
            explicit_weather_kwargs = env_kwargs.pop("weather_generator_kwargs", None)
            if explicit_weather_kwargs:
                weather_generator_kwargs = explicit_weather_kwargs
            return gym, CropPlanningFixedPlanting(
                start_year=start_year,
                end_year=end_year,
                rotation_crops=rotation_crops,
                weather_generator_class=weather_generator_class,
                weather_generator_kwargs=weather_generator_kwargs,
                **env_kwargs,
            )

        raise ValueError(f"Unsupported CyclesGym env_id: {env_id}")

    def __init__(self, config: CyclesEnvConfig):
        if config.cycles_gym_path not in sys.path:
            sys.path.insert(0, config.cycles_gym_path)
        self._prepare_runtime_tree(config)

        if config.seed is not None:
            try:
                import numpy as np

                np.random.seed(int(config.seed))
            except Exception:
                pass
            random.seed(int(config.seed))

        self._gym, self.env = self._create_cyclesgym_env(config)

        if config.seed is not None:
            try:
                self.env.reset(seed=config.seed)
            except TypeError:
                pass

        self.llm_mode = config.llm_mode
        self.turn_num = config.turn_num
        self.env_id = str(config.env_id)
        self.reward_mode = str(getattr(config, "reward_mode", "native")).strip().lower()
        self.valid_action_bonus = float(getattr(config, "valid_action_bonus", 0.1))
        self.invalid_action_penalty = float(getattr(config, "invalid_action_penalty", 0.0))
        self.invalid_action_fallback = str(
            getattr(config, "invalid_action_fallback", "default") or "default"
        ).strip().lower()
        self.reward_scale = float(getattr(config, "reward_scale", 1.0))
        if self.reward_scale <= 0.0:
            raise ValueError(f"reward_scale must be positive, got {self.reward_scale}")
        self._total_steps = 0
        self._invalid_steps = 0
        self._episode_reward = 0.0
        self._episode_native_reward = 0.0
        self._total_n_applied = 0.0
        self._last_final_yield_tonnes = 0.0
        self._crop_planning_history = []
        self._episode_shaping_reward = 0.0
        self._fallback_steps = 0
        self._last_valid_crop_planning_action: list[int] | None = None
        self._last_target_yield = float("nan")
        self._last_turn_metrics: Dict[str, Any] = {}
        self.prompt_generator = None
        valid_action_bonus = getattr(config, "valid_action_bonus", 0.1)
        self.valid_action_bonus = 0.1 if valid_action_bonus is None else float(valid_action_bonus)
        self.objective_id = str(getattr(config, "objective_id", "profit_max") or "profit_max")
        self.reward_params = dict(getattr(config, "reward_params", {}) or {})
        self._is_corn_cropgrowth = str(config.env_id).startswith("Corn")
        env_reward = str(getattr(config, "env_reward", "") or "").lower()
        self._yield_only_terminal_reward = env_reward in {
            "yield_only",
            "yield_only_terminal",
            "sparse_terminal_yield",
            "rewardfinalyieldwrapper",
        } or self.objective_id == "yield_max" or self.reward_mode in {"final_yield", "yield_only"}
        self._profit_terminal_reward = (
            self._is_corn_cropgrowth
            and self.objective_id == "profit_max"
            and self.reward_mode not in {"final_yield", "yield_only"}
        )

        if self.llm_mode:
            objective_prompt_params = {
                **self.reward_params,
                **dict(getattr(config, "profit_context_params", {}) or {}),
            }
            self.prompt_generator = build_prompt_generator(
                self.env,
                env_id=config.env_id,
                include_crop_traits=getattr(config, "include_crop_traits", True),
                require_think=config.require_think,
                thinking_mode=config.thinking_mode,
                think_tag=config.think_tag,
                reward_mode=self.reward_mode,
                include_profit_context=getattr(config, "include_profit_context", False),
                profit_context_params=objective_prompt_params,
                objective_id=(
                    getattr(config, "prompt_objective_id", None)
                    or ("yield_max" if self.reward_mode in {"final_yield", "yield_only"} else self.objective_id)
                ),
                objective_text=getattr(config, "prompt_objective_text", None),
                reward_params=self.reward_params,
            )

        super().__init__(config)

    @staticmethod
    def _finite_float(value: Any) -> float:
        try:
            value_float = float(value)
        except Exception:
            return float("nan")
        return value_float if math.isfinite(value_float) else float("nan")

    @classmethod
    def _tonnes_per_ha_to_kg_per_ha(cls, value: Any) -> float:
        value_float = cls._finite_float(value)
        if not math.isfinite(value_float):
            return float("nan")
        return value_float * 1000.0

    def _extract_target_yield(self, obs: Any, info: Dict[str, Any] | None = None) -> float:
        """Return CycleGym grain yield as kg/ha.

        CycleGym stores harvest yield in ``season_df["GRAIN YIELD"]`` as
        tonne/ha. AgriManager's cross-simulator canonical metric uses kg/ha, so
        terminal training reward can be computed as ``target_yield / 1000`` and
        stay on WOFOST's normalized reward scale.
        """
        info = info or {}
        for mapping in (info, info.get("trajectory_metrics") or {}, info.get("turn_metrics") or {}):
            if not isinstance(mapping, dict):
                continue
            for key in ("target_yield", "grain_yield_kg_ha", "final_grain_yield_kg_ha"):
                value = self._finite_float(mapping.get(key))
                if math.isfinite(value):
                    return value
            for key in ("grain_yield", "final_grain_yield", "GRAIN YIELD", "yield"):
                value = self._tonnes_per_ha_to_kg_per_ha(mapping.get(key))
                if math.isfinite(value):
                    return value

        obs_names = getattr(getattr(self.env, "observer", None), "obs_names", None)
        if obs_names and hasattr(obs, "__len__"):
            for key in ("GRAIN YIELD", "grain_yield", "yield"):
                if key in obs_names:
                    value = self._tonnes_per_ha_to_kg_per_ha(obs[obs_names.index(key)])
                    if math.isfinite(value):
                        return value

        season_manager = getattr(self.env, "season_manager", None)
        for source in (
            self.env,
            season_manager,
        ):
            if source is None:
                continue
            season_df = getattr(source, "season_df", None)
            if season_df is None:
                continue
            try:
                for key in ("GRAIN YIELD", "grain_yield", "yield"):
                    if key in season_df:
                        values = season_df[key].dropna()
                        if len(values):
                            value = self._tonnes_per_ha_to_kg_per_ha(values.iloc[-1])
                            if math.isfinite(value):
                                return value
            except Exception:
                continue

        return float("nan")

    def _serialize_observation(self, obs: Any) -> Dict[str, Any]:
        obs_names = getattr(getattr(self.env, "observer", None), "obs_names", None) or []
        if not obs_names or not hasattr(obs, "__len__"):
            return {}

        result: Dict[str, Any] = {}
        for name, value in zip(obs_names, obs):
            try:
                result[str(name)] = float(value)
            except Exception:
                result[str(name)] = value
        return result

    def _extract_metrics(self, obs: Any, reward: float, action: Any) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {"reward": float(reward), "action": action}

        obs_names = getattr(getattr(self.env, "observer", None), "obs_names", None)
        if obs_names and hasattr(obs, "__len__"):
            try:
                if "DOY" in obs_names:
                    metrics["doy"] = float(obs[obs_names.index("DOY")])
                if "N TO DATE" in obs_names:
                    metrics["n_to_date"] = float(obs[obs_names.index("N TO DATE")])
            except Exception:
                pass
        return metrics

    def _is_corn_growth(self) -> bool:
        return str(getattr(self, "env_id", "")).startswith("Corn") or bool(
            getattr(self, "_is_corn_cropgrowth", False)
        )

    def _corn_action_to_n(self, action: Any) -> float:
        try:
            action_id = int(action[0] if hasattr(action, "__len__") else action)
        except Exception:
            return 0.0
        n_actions = int(getattr(self.env, "n_actions", 1) or 1)
        max_n = float(getattr(self.env, "maxN", 0.0) or 0.0)
        if n_actions <= 1:
            return 0.0
        action_id = max(0, min(n_actions - 1, action_id))
        return max_n * action_id / (n_actions - 1)

    def _corn_final_yield_tonnes(self) -> float:
        season_manager = getattr(self.env, "season_manager", None)
        season_df = getattr(season_manager, "season_df", None)
        if season_df is None or "GRAIN YIELD" not in getattr(season_df, "columns", []):
            return 0.0
        rows = season_df
        if "CROP" in rows.columns:
            corn_rows = rows.loc[rows["CROP"].isin(["CornRM.90", "CornRM.100"])]
            if not corn_rows.empty:
                rows = corn_rows
        try:
            values = rows["GRAIN YIELD"].astype(float)
        except Exception:
            return 0.0
        if values.empty:
            return 0.0
        return float(values.sum())

    def _format_reward(self, invalid_action: bool) -> float:
        if invalid_action:
            return float(getattr(self, "invalid_action_penalty", 0.0))
        return float(getattr(self, "valid_action_bonus", 0.1))

    def _normalize_reward(self, reward: float) -> float:
        return float(reward) / float(getattr(self, "reward_scale", 1.0))

    def _compute_scalar_reward(
        self,
        *,
        native_reward: float,
        done: bool,
        invalid_action: bool,
    ) -> float:
        if invalid_action:
            return float(getattr(self, "invalid_action_penalty", 0.0))
        format_reward = float(getattr(self, "valid_action_bonus", 0.1)) if self.llm_mode else 0.0
        reward_mode = str(getattr(self, "reward_mode", "native"))
        if not self._is_corn_growth() or reward_mode not in {"final_yield", "yield_only"}:
            return self._normalize_reward(native_reward) + format_reward
        final_yield = self._corn_final_yield_tonnes() if done else 0.0
        self._last_final_yield_tonnes = final_yield
        return self._normalize_reward(final_yield) + format_reward

    def _crop_planning_action_shaping(self, action: Any) -> tuple[float, Dict[str, float]]:
        if not self._is_crop_planning():
            return 0.0, {}

        crop = self._crop_from_action(action)
        if crop is None:
            return 0.0, {}

        history = [row.get("crop") for row in self._crop_planning_history if row.get("crop")]
        if not history:
            return 0.0, {}

        params = self.reward_params
        same_crop_penalty = float(params.get("crop_planning_same_crop_penalty", 0.0) or 0.0)
        same_crop_penalty_cap = float(
            params.get("crop_planning_same_crop_penalty_cap", same_crop_penalty * 8.0) or 0.0
        )
        cereal_run_penalty = float(params.get("crop_planning_cereal_run_penalty", 0.0) or 0.0)
        legume_break_bonus = float(params.get("crop_planning_legume_break_bonus", 0.0) or 0.0)
        cereal_after_legume_bonus = float(
            params.get("crop_planning_cereal_after_legume_bonus", 0.0) or 0.0
        )
        change_crop_bonus = float(params.get("crop_planning_change_crop_bonus", 0.0) or 0.0)

        prev_crop = str(history[-1])
        cereals = {
            str(c)
            for c in _as_list(getattr(self.env, "rotation_crops", None))
            if "soy" not in str(c).lower()
        }
        shaping = 0.0
        diagnostics: Dict[str, float] = {}

        if crop != prev_crop and change_crop_bonus:
            shaping += change_crop_bonus
            diagnostics["crop_planning/change_crop_bonus"] = change_crop_bonus

        if crop == prev_crop and same_crop_penalty:
            run_length = 1
            for prior_crop in reversed(history):
                if str(prior_crop) == crop:
                    run_length += 1
                else:
                    break
            same_penalty = min(same_crop_penalty * max(1, run_length - 1), same_crop_penalty_cap)
            shaping -= same_penalty
            diagnostics["crop_planning/same_crop_penalty"] = -same_penalty
            diagnostics["crop_planning/same_crop_run_length"] = float(run_length)

        if crop in cereals and prev_crop in cereals and cereal_run_penalty:
            shaping -= cereal_run_penalty
            diagnostics["crop_planning/cereal_run_penalty"] = -cereal_run_penalty

        if prev_crop in cereals and "soy" in crop.lower() and legume_break_bonus:
            shaping += legume_break_bonus
            diagnostics["crop_planning/legume_break_bonus"] = legume_break_bonus

        if "soy" in prev_crop.lower() and crop in cereals and cereal_after_legume_bonus:
            shaping += cereal_after_legume_bonus
            diagnostics["crop_planning/cereal_after_legume_bonus"] = cereal_after_legume_bonus

        if shaping:
            diagnostics["crop_planning/shaping_reward"] = shaping
        return shaping, diagnostics

    def _is_crop_planning(self) -> bool:
        return hasattr(self.env, "rotation_crops") and hasattr(self.env, "season_manager")

    def _crop_prices(self) -> Dict[str, float]:
        prices = _as_dict(getattr(self.env, "crop_prices", None))
        result: Dict[str, float] = {}
        for crop in _as_list(getattr(self.env, "rotation_crops", None)):
            price = prices.get(crop)
            if isinstance(price, dict):
                year = getattr(getattr(self.env, "date", None), "year", None)
                if year in price:
                    price = price[year]
                elif price:
                    price = next(iter(price.values()))
            if price is not None:
                try:
                    result[str(crop)] = float(price)
                except Exception:
                    pass
        return result

    def _crop_from_action(self, action: Any) -> str | None:
        crops = [str(crop) for crop in _as_list(getattr(self.env, "rotation_crops", None))]
        if not crops:
            return None
        try:
            crop_idx = int(action[0] if hasattr(action, "__len__") else action)
        except Exception:
            return None
        if crop_idx < 0 or crop_idx >= len(crops):
            return None
        return str(crops[crop_idx])

    def _planting_week_from_action(self, action: Any) -> int | None:
        try:
            if hasattr(action, "__len__") and len(action) >= 2:
                return int(action[1])
        except Exception:
            pass
        return None

    def _fallback_crop_planning_action(self) -> list[int]:
        crops = [str(crop) for crop in _as_list(getattr(self.env, "rotation_crops", None))]
        if not crops:
            return self._default_action()

        prices = self._crop_prices()
        history = [row.get("crop") for row in self._crop_planning_history if row.get("crop")]
        previous_crop = str(history[-1]) if history else None
        soybean_idx = next((idx for idx, crop in enumerate(crops) if "soy" in crop.lower()), None)

        ranked = sorted(
            range(len(crops)),
            key=lambda idx: (float(prices.get(crops[idx], 0.0)), -idx),
            reverse=True,
        )
        chosen_idx = ranked[0]
        if previous_crop is not None and crops[chosen_idx] == previous_crop:
            if soybean_idx is not None and crops[chosen_idx] != crops[soybean_idx]:
                chosen_idx = soybean_idx
            elif len(ranked) > 1:
                chosen_idx = ranked[1]

        default_week = 0
        if self._last_valid_crop_planning_action is not None and len(self._last_valid_crop_planning_action) >= 2:
            try:
                default_week = int(self._last_valid_crop_planning_action[1])
            except Exception:
                default_week = 0
        action_space = getattr(self.env, "action_space", None)
        if action_space is not None and hasattr(action_space, "nvec") and len(action_space.nvec) >= 2:
            max_week = int(action_space.nvec[1]) - 1
            midpoint_week = int(action_space.nvec[1]) // 2
            default_week = max(0, min(max_week, default_week if self._last_valid_crop_planning_action is not None else midpoint_week))
        return [chosen_idx, default_week]

    def _fallback_action(self) -> Any:
        mode = str(getattr(self, "invalid_action_fallback", "default") or "default").strip().lower()
        if not self._is_crop_planning():
            return self._default_action()
        if mode == "repeat_last_valid" and self._last_valid_crop_planning_action is not None:
            return list(self._last_valid_crop_planning_action)
        if mode == "heuristic":
            return self._fallback_crop_planning_action()
        return self._default_action()

    def _latest_crop_yield(
        self,
        crop: str,
        action_year: int | None,
        previous_row_count: int,
    ) -> float | None:
        try:
            from cyclesgym.utils.pricing_utils import crop_type
        except Exception:
            return None

        season_manager = getattr(self.env, "season_manager", None)
        season_df = getattr(season_manager, "season_df", None)
        yield_column = crop_type.get(crop)
        if season_df is None or yield_column is None or yield_column not in season_df.columns:
            return None

        new_rows = season_df.iloc[previous_row_count:].copy()
        if "CROP" in new_rows.columns:
            new_rows = new_rows.loc[new_rows["CROP"] == crop]
        if new_rows.empty:
            candidate_rows = season_df.copy()
            if "CROP" in candidate_rows.columns:
                candidate_rows = candidate_rows.loc[candidate_rows["CROP"] == crop]
            if action_year is not None and "PLANT_YEAR" in candidate_rows.columns:
                candidate_rows = candidate_rows.loc[candidate_rows["PLANT_YEAR"] == action_year]
            elif action_year is not None and "YEAR" in candidate_rows.columns:
                candidate_rows = candidate_rows.loc[candidate_rows["YEAR"] == action_year]
            new_rows = candidate_rows
        if new_rows.empty:
            return None
        try:
            return float(new_rows[yield_column].astype(float).sum())
        except Exception:
            return None

    def _crop_planning_prompt_context(self) -> Dict[str, Any] | None:
        if not self._is_crop_planning():
            return None
        ctrl = getattr(getattr(self.env, "ctrl_base_manager", None), "ctrl_dict", {}) or {}
        current_year = getattr(getattr(self.env, "date", None), "year", None)
        end_year = ctrl.get("SIMULATION_END_YEAR")
        years_remaining = None
        if current_year is not None and end_year is not None:
            try:
                years_remaining = max(0, int(end_year) - int(current_year) + 1)
            except Exception:
                years_remaining = None
        return {
            "current_year": current_year,
            "years_remaining": years_remaining,
            "crop_prices": self._crop_prices(),
            "past_trajectory": list(self._crop_planning_history),
        }

    def _record_crop_planning_transition(
        self,
        *,
        action: Any,
        action_year: int | None,
        previous_row_count: int,
        reward: float,
    ) -> Dict[str, Any]:
        crop = self._crop_from_action(action)
        if crop is None:
            return {}
        crop_yield = self._latest_crop_yield(crop, action_year, previous_row_count)
        prices = self._crop_prices()
        price = prices.get(crop)
        revenue = float(reward)
        if crop_yield is not None and price is not None:
            revenue = crop_yield * price
        row = {
            "year": action_year,
            "crop": crop,
            "planting_week": self._planting_week_from_action(action),
            "yield_tonnes": crop_yield,
            "price_dollars_per_tonne": price,
            "revenue": revenue,
        }
        self._crop_planning_history.append(row)
        return row

    def _rotation_diversity_shannon(self) -> float:
        crops = [row.get("crop") for row in self._crop_planning_history if row.get("crop")]
        total = len(crops)
        if total == 0:
            return 0.0

        counts: Dict[str, int] = {}
        for crop in crops:
            counts[str(crop)] = counts.get(str(crop), 0) + 1

        entropy = 0.0
        for count in counts.values():
            prob = count / total
            entropy -= prob * math.log(prob)
        return float(entropy)
    def _crop_counts(self) -> Dict[str, int]:
        counts = {
            str(crop): 0
            for crop in _as_list(getattr(self.env, "rotation_crops", None))
        }
        for row in self._crop_planning_history:
            crop = row.get("crop")
            if crop is not None:
                counts[str(crop)] = counts.get(str(crop), 0) + 1
        return counts

    @staticmethod
    def _metric_suffix(value: Any) -> str:
        suffix = "".join(
            ch if ch.isalnum() or ch in {".", "_", "-"} else "_"
            for ch in str(value)
        ).strip("._-")
        return suffix or "unknown"

    def _crop_planning_revenue_metrics(self) -> Dict[str, float]:
        crops = [str(crop) for crop in _as_list(getattr(self.env, "rotation_crops", None))]
        totals_by_crop = {crop: 0.0 for crop in crops}
        counts_by_crop = {crop: 0 for crop in crops}
        total_revenue = 0.0
        valid_revenue_count = 0

        for row in self._crop_planning_history:
            crop = row.get("crop")
            crop_key = str(crop) if crop is not None else "unknown"
            if crop_key not in totals_by_crop:
                totals_by_crop[crop_key] = 0.0
                counts_by_crop[crop_key] = 0
            counts_by_crop[crop_key] += 1

            revenue = self._finite_float(row.get("revenue"))
            if not math.isfinite(revenue):
                continue
            totals_by_crop[crop_key] += revenue
            total_revenue += revenue
            valid_revenue_count += 1

        total_years = max(1, len(self._crop_planning_history))
        reward_scale = float(getattr(self, "reward_scale", 1.0))
        metrics: Dict[str, float] = {
            "gross_revenue": float(total_revenue),
            "raw_gross_revenue": float(total_revenue),
            "gross_revenue_pre_normalization": float(total_revenue),
            "cumulative_gross_revenue": float(total_revenue),
            "mean_annual_gross_revenue": float(total_revenue / total_years),
            "normalized_gross_revenue": float(total_revenue / reward_scale),
            "gross_revenue_normalized": float(total_revenue / reward_scale),
            "valid_rev_years": float(valid_revenue_count),
        }

        prices = self._crop_prices()
        for crop in sorted(totals_by_crop):
            suffix = self._metric_suffix(crop)
            crop_revenue = float(totals_by_crop[crop])
            crop_count = int(counts_by_crop.get(crop, 0))
            metrics[f"gross_rev__{suffix}"] = crop_revenue
            metrics[f"raw_gross_rev__{suffix}"] = crop_revenue
            metrics[f"crop_count__{suffix}"] = float(crop_count)
            metrics[f"crop_freq__{suffix}"] = float(crop_count / total_years)
            metrics[f"rev_share__{suffix}"] = (
                float(crop_revenue / total_revenue) if total_revenue > 0.0 else 0.0
            )
            metrics[f"mean_rev__{suffix}"] = (
                float(crop_revenue / crop_count) if crop_count > 0 else 0.0
            )
            price = self._finite_float(prices.get(crop))
            if math.isfinite(price):
                metrics[f"crop_price__{suffix}"] = float(price)
        return metrics

    def _max_consecutive_same_crop(self) -> int:
        best = 0
        current = 0
        previous = None
        for row in self._crop_planning_history:
            crop = row.get("crop")
            if crop is None:
                continue
            if crop == previous:
                current += 1
            else:
                current = 1
                previous = crop
            best = max(best, current)
        return best
    def _max_consecutive_cereals(self) -> int:
        cereals = {
            str(crop)
            for crop in _as_list(getattr(self.env, "rotation_crops", None))
            if "soy" not in str(crop).lower()
        }
        best = 0
        current = 0
        for row in self._crop_planning_history:
            crop = row.get("crop")
            if crop in cereals:
                current += 1
                best = max(best, current)
            else:
                current = 0
        return best

    def _profit_diagnostics(self, total_n_kg_ha: float | None = None) -> Dict[str, float]:
        costs = profit_cost_params(self.reward_params)
        total_n = self._finite_float(total_n_kg_ha)
        if not math.isfinite(total_n):
            total_n = self._finite_float(self._last_turn_metrics.get("n_to_date"))
        if not math.isfinite(total_n):
            total_n = 0.0
        revenue = self._last_target_yield if math.isfinite(self._last_target_yield) else 0.0
        nutrient_cost = costs["cost_n"] * total_n
        profit = revenue - nutrient_cost
        scale = profit_reward_scale(self.reward_params)
        return {
            "objective_reward": float(profit / scale),
            "reward_profit_term": float(profit / scale),
            "revenue_ge_kg_ha": float(revenue),
            "input_cost_ge_kg_ha": float(nutrient_cost),
            "nutrient_cost_ge_kg_ha": float(nutrient_cost),
            "profit_ge_kg_ha": float(profit),
        }

    def _terminal_objective_reward(
        self,
        done: bool,
        native_reward: float,
        total_n_kg_ha: float | None = None,
    ) -> tuple[float, Dict[str, float]]:
        if getattr(self, "_profit_terminal_reward", False):
            if not done:
                return 0.0, {}
            diagnostics = self._profit_diagnostics(total_n_kg_ha)
            return diagnostics["objective_reward"], diagnostics
        if getattr(self, "_yield_only_terminal_reward", False):
            return (
                float(self._last_target_yield / 1000.0 if done and math.isfinite(self._last_target_yield) else 0.0),
                {},
            )
        return float(native_reward), {}

    def reset(self) -> Tuple[Any, Dict[str, Any]]:
        self._total_steps = 0
        self._invalid_steps = 0
        self._episode_reward = 0.0
        self._episode_native_reward = 0.0
        self._total_n_applied = 0.0
        self._last_final_yield_tonnes = 0.0
        self._crop_planning_history = []
        self._episode_shaping_reward = 0.0
        self._fallback_steps = 0
        self._last_valid_crop_planning_action = None
        self._last_target_yield = float("nan")
        self._last_turn_metrics = {}

        reset_out = self.env.reset()
        if isinstance(reset_out, tuple) and len(reset_out) == 2:
            obs, _ = reset_out
        else:
            obs = reset_out

        if self.llm_mode and getattr(self.prompt_generator, "obs_names", None) == []:
            observer = getattr(self.env, "observer", None)
            obs_names = getattr(observer, "obs_names", None)
            if obs_names:
                self.prompt_generator.obs_names = list(obs_names)

        info = {
            "turn_metrics": self._extract_metrics(obs, 0.0, action=None),
            "observation": self._serialize_observation(obs),
        }
        if self._is_corn_growth():
            info["turn_metrics"].update(
                {
                    "reward_mode": str(getattr(self, "reward_mode", "native")),
                    "maize/final_yield_tonnes_ha": 0.0,
                    "maize/n_applied_kg_ha": 0.0,
                    "maize/total_n_kg_ha": 0.0,
                    "maize/total_irrigation": 0.0,
                }
            )
        self._last_turn_metrics = dict(info["turn_metrics"])

        if self.llm_mode:
            return self.prompt_generator.get_turn_prompt(
                obs,
                context=self._crop_planning_prompt_context(),
            ), info
        return obs, info

    def step(self, action: Union[int, float, Any]) -> Tuple[Any, float, bool, Dict[str, Any]]:
        raw_llm_response = None
        invalid_action = False
        valid_action_format = False

        if self.llm_mode and isinstance(action, str):
            raw_llm_response = action
            parsed = self.prompt_generator.parse_action_response(action)
            if parsed is None:
                invalid_action = True
                action = self._fallback_action()
                self._fallback_steps += 1
            else:
                action = parsed
                valid_action_format = True
                if self._is_crop_planning():
                    try:
                        self._last_valid_crop_planning_action = [int(v) for v in action]
                    except Exception:
                        self._last_valid_crop_planning_action = None
        elif self.llm_mode and action is None:
            invalid_action = True
            action = self._fallback_action()
            self._fallback_steps += 1

        self._total_steps += 1
        if invalid_action:
            self._invalid_steps += 1

        action_year = getattr(getattr(self.env, "date", None), "year", None)
        season_manager = getattr(self.env, "season_manager", None)
        season_df = getattr(season_manager, "season_df", None)
        previous_row_count = len(season_df) if season_df is not None else 0

        n_applied = self._corn_action_to_n(action) if self._is_corn_growth() else 0.0
        step_out = self.env.step(action)
        if isinstance(step_out, tuple) and len(step_out) == 5:
            obs, reward, terminated, truncated, info = step_out
            done = bool(terminated or truncated)
        else:
            obs, reward, done, info = step_out

        if self._total_steps >= self.turn_num:
            done = True

        info = info or {}
        native_reward = float(reward)
        target_yield = self._extract_target_yield(obs, info)
        if math.isfinite(target_yield):
            self._last_target_yield = target_yield
        if bool(done) and self._is_corn_growth() and math.isfinite(self._last_target_yield):
            self._last_final_yield_tonnes = self._last_target_yield / 1000.0
        current_metrics = self._extract_metrics(obs, 0.0, action)
        total_n = self._finite_float(current_metrics.get("n_to_date"))
        normalized_env_reward = 0.0 if invalid_action else self._normalize_reward(native_reward)
        format_reward = 0.0 if invalid_action else (self.valid_action_bonus if self.llm_mode else 0.0)
        reward_diagnostics: Dict[str, float] = {}
        shaping_reward = 0.0
        shaping_diagnostics: Dict[str, float] = {}
        if invalid_action:
            scalar_reward = getattr(self, "invalid_action_penalty", 0.0)
        elif getattr(self, "_profit_terminal_reward", False) or getattr(self, "_yield_only_terminal_reward", False):
            objective_reward, reward_diagnostics = self._terminal_objective_reward(
                bool(done),
                native_reward,
                total_n,
            )
            scalar_reward = objective_reward + format_reward
        else:
            scalar_reward = self._compute_scalar_reward(
                native_reward=native_reward,
                done=done,
                invalid_action=invalid_action,
            )
            shaping_reward, shaping_diagnostics = self._crop_planning_action_shaping(action)
            scalar_reward += shaping_reward
        self._episode_native_reward = float(getattr(self, "_episode_native_reward", 0.0)) + native_reward
        self._episode_reward += scalar_reward
        self._episode_shaping_reward += shaping_reward
        if self._is_corn_growth():
            self._total_n_applied = float(getattr(self, "_total_n_applied", 0.0)) + n_applied

        if "turn_metrics" not in info:
            info["turn_metrics"] = self._extract_metrics(obs, scalar_reward, action)
        else:
            info["turn_metrics"] = dict(info["turn_metrics"])
            info["turn_metrics"]["reward"] = float(scalar_reward)
            info["turn_metrics"].setdefault("action", action)
        info["turn_metrics"].update(
            {
                "native_reward": native_reward,
                "normalized_env_reward": normalized_env_reward,
                "format_reward": format_reward,
                "shaping_reward": shaping_reward,
                "reward_scale": float(getattr(self, "reward_scale", 1.0)),
            }
        )
        info["turn_metrics"].update(reward_diagnostics)
        info["turn_metrics"].update(shaping_diagnostics)
        if math.isfinite(self._last_target_yield):
            info["turn_metrics"].setdefault("target_yield", self._last_target_yield)

        if self._is_corn_growth():
            info["turn_metrics"].update(
                {
                    "reward_mode": str(getattr(self, "reward_mode", "native")),
                    "maize/native_reward": native_reward,
                    "maize/final_yield_tonnes_ha": float(getattr(self, "_last_final_yield_tonnes", 0.0)),
                    "maize/n_applied_kg_ha": n_applied,
                    "maize/total_n_kg_ha": float(getattr(self, "_total_n_applied", 0.0)),
                    "maize/total_irrigation": 0.0,
                }
            )

        if self._is_crop_planning():
            transition = self._record_crop_planning_transition(
                action=action,
                action_year=action_year,
                previous_row_count=previous_row_count,
                reward=native_reward,
            )
            info["turn_metrics"].update(
                {
                    f"crop_planning/{key}": value
                    for key, value in transition.items()
                    if value is not None
                }
            )
        self._last_turn_metrics = dict(info["turn_metrics"])

        info["observation"] = self._serialize_observation(obs)
        info["executed_action"] = action
        info["invalid_action"] = invalid_action
        info["fallback_action_used"] = bool(invalid_action)

        if done:
            info["trajectory_metrics"] = self.get_trajectory_metrics()

        if self.llm_mode:
            if raw_llm_response is not None:
                info["raw_llm_response"] = raw_llm_response
            return self.prompt_generator.get_turn_prompt(
                obs,
                context=self._crop_planning_prompt_context(),
            ), float(scalar_reward), bool(done), info

        return obs, float(scalar_reward), bool(done), info

    def _default_action(self) -> Any:
        action_space = getattr(self.env, "action_space", None)
        if action_space is not None and hasattr(action_space, "nvec"):
            return [0 for _ in action_space.nvec]
        return 0

    def system_prompt(self) -> str:
        if not self.llm_mode:
            raise NotImplementedError("CyclesEnv system_prompt is only available in LLM mode.")
        return self.prompt_generator.get_system_prompt()

    def get_trajectory_metrics(self) -> Dict[str, float]:
        total_n = self._finite_float(self._last_turn_metrics.get("n_to_date"))
        if not math.isfinite(total_n):
            total_n = float(getattr(self, "_total_n_applied", 0.0))
        metrics = {
            "invalid_action_rate": (self._invalid_steps / self._total_steps) if self._total_steps else 0.0,
            "invalid_steps": float(self._invalid_steps),
            "total_steps": float(self._total_steps),
            "episode_reward": float(self._episode_reward),
            "native_episode_reward": float(getattr(self, "_episode_native_reward", 0.0)),
            "shaping_episode_reward": float(getattr(self, "_episode_shaping_reward", 0.0)),
            "reward_scale": float(getattr(self, "reward_scale", 1.0)),
            "target_yield": float(self._last_target_yield),
            "grain_yield": float(self._last_target_yield),
            "grain_yield_t_ha": float(self._last_target_yield / 1000.0),
            "total_n_kg_ha": total_n,
            "fallback_steps": float(getattr(self, "_fallback_steps", 0.0)),
        }
        if self._is_crop_planning():
            metrics.update(
                {
                    "crop_planning_history": list(self._crop_planning_history),
                    "crop_counts": self._crop_counts(),
                    "rot_div_shannon": self._rotation_diversity_shannon(),
                    "max_same_crop": self._max_consecutive_same_crop(),
                    "max_cereal_streak": self._max_consecutive_cereals(),
                    "crop_prices": self._crop_prices(),
                }
            )
            metrics.update(self._crop_planning_revenue_metrics())
        if self._is_corn_growth():
            metrics.update(
                {
                    "final_yield_tonnes_ha": float(getattr(self, "_last_final_yield_tonnes", 0.0)),
                    "yield_tonnes_ha": float(getattr(self, "_last_final_yield_tonnes", 0.0)),
                    "total_irrigation": 0.0,
                }
            )
        if getattr(self, "_profit_terminal_reward", False):
            metrics.update(self._profit_diagnostics(total_n))
        return metrics

    def close(self) -> None:
        if hasattr(self, "env") and self.env is not None:
            self.env.close()
