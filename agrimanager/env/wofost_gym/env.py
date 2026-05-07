"""wofost_gym environment wrapper.

This module provides a wrapper around wofost_gym environments to make them
compatible with the BaseEnv interface. It acts as a bridge between WOFOST
simulation and LLM agents by converting observations to natural language
prompts and parsing LLM responses back to actions.
"""

import os
import sys
import math
from pathlib import Path
from typing import Any, Dict, Tuple, Union

from agrimanager.env.base import BaseEnv
from agrimanager.env.base.objective_prompt import profit_reward_scale
from .crop_trait_schemas import crop_variety_trait_key, resolve_crop_trait_artifact_path
from .crop_traits_observation import crop_variety_from_env_config
from .env_config import DEFAULT_CROP_TRAITS_DIR, REPO_ROOT, WOFOSTEnvConfig
from .prompt import WOFOSTPromptGenerator


def _resolve_wofost_gym_path(configured_path: str) -> str:
    """Return a WOFOST-Gym checkout path that contains the legacy utils.py."""
    candidates = [
        configured_path,
        os.environ.get("WOFOST_GYM_PATH"),
        str((REPO_ROOT / ".." / "AgriManagerExternal" / "WOFOSTGym").resolve()),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        candidate_path = Path(candidate).expanduser().resolve()
        if (candidate_path / "utils.py").is_file():
            return str(candidate_path) + os.sep
    return configured_path


def _wofost_python_paths(wofost_gym_path: str) -> list[str]:
    """Return import roots needed by the external WOFOST-Gym checkout."""
    root = Path(wofost_gym_path).expanduser().resolve()
    candidates = [
        root,
        root / "pcse_gym",
        root / "pcse",
        root / "stable-baselines3",
        root / "imitation",
    ]
    return [str(path) for path in candidates if path.exists()]


def _configure_pcse_weather_cache(weather_cache_dir: str | None) -> None:
    """Point PCSE at a bundled meteo cache and disable implicit NASA retrieval."""
    if not weather_cache_dir:
        return

    cache_dir = Path(weather_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["PCSE_METEO_CACHE_DIR"] = str(cache_dir)
    os.environ.setdefault("PCSE_NASAPOWER_NO_RETRIEVE", "1")


def _scale_action_amounts_by_interval(npk_args: Any, intvn_interval: int | float) -> None:
    """Scale per-decision resource amounts by the decision interval."""
    action_amount_scale = float(intvn_interval)
    if action_amount_scale <= 0:
        raise ValueError(f"intvn_interval must be positive, got {intvn_interval!r}")

    npk_args.fert_amount = float(npk_args.fert_amount) * action_amount_scale
    npk_args.irrig_amount = float(npk_args.irrig_amount) * action_amount_scale


def _patch_lw_noop_action_bug(env: Any) -> bool:
    """Patch legacy WOFOST-Gym `lw-v0` action 0, which can reference unset water amount."""
    current = env
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        original_take_action = getattr(current, "_take_action", None)
        if original_take_action is not None:
            def patched_take_action(action: Any, _original_take_action=original_take_action):
                try:
                    action_id = int(action)
                except (TypeError, ValueError):
                    action_id = action
                if action_id == 0:
                    return (0, 0, 0, 0)
                return _original_take_action(action)

            current._take_action = patched_take_action
            return True
        current = getattr(current, "env", None)
    return False


NUTRIENT_STEWARDSHIP_DEFAULTS = {
    "tau_y": 0.80,
    "budget_n_kg_ha": 180.0,
    "budget_p_kg_ha": 180.0,
    "budget_k_kg_ha": 180.0,
    "terminal_nutrient_min": 8.0,
    "terminal_nutrient_max": 45.0,
    "beta_y": 4.0,
    "beta_application": 0.35,
    "beta_low": 0.60,
    "beta_high": 0.50,
}


PROFIT_MAX_DEFAULTS = {
    "cost_n_kg_wso_per_kg": 3.5,
    "cost_p_kg_wso_per_kg": 5.5,
    "cost_k_kg_wso_per_kg": 2.25,
    "cost_irrig_kg_wso_per_mm": 0.05,
}


WATER_STEWARDSHIP_DEFAULTS = {
    "tau_y": 0.80,
    "water_budget_mm": 120.0,
    "lambda_i": 0.10,
    "beta_y": 4.0,
    "beta_budget": 2.0,
}


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _hinge_squared(value: float) -> float:
    return max(0.0, float(value)) ** 2


def _reward_param(params: Dict[str, Any], *names: str, default: float) -> float:
    for name in names:
        if name in params and params[name] is not None:
            return _finite_float(params[name], default)
    return default


def _calibrated_y_ref(
    metrics: Dict[str, float],
    reward_params: Dict[str, Any],
    objective_id: str,
) -> float:
    y_ref = _reward_param(
        reward_params,
        "y_ref",
        "Y_ref",
        default=_finite_float(metrics.get("y_ref"), 0.0),
    )
    if y_ref <= 0.0:
        raise ValueError(f"{objective_id} requires a positive calibrated y_ref.")
    return y_ref


def _yield_ratio_terms(
    metrics: Dict[str, float],
    reward_params: Dict[str, Any],
    objective_id: str,
) -> tuple[float, float, float]:
    y_ref = _calibrated_y_ref(metrics, reward_params, objective_id)
    final_wso = _finite_float(metrics.get("final_wso"), 0.0)
    return final_wso, y_ref, final_wso / y_ref


def compute_yield_max_reward(
    metrics: Dict[str, float],
    reward_params: Dict[str, Any] | None = None,
) -> Dict[str, float]:
    """Compute normalized terminal yield reward for calibrated T2.3 runs."""
    params = dict(reward_params or {})
    final_wso, y_ref, y_ratio = _yield_ratio_terms(metrics, params, "yield_max")
    return {
        "objective_reward": float(y_ratio),
        "final_wso": float(final_wso),
        "y_ref": float(y_ref),
        "y_ratio": float(y_ratio),
        "reward_yield_term": float(y_ratio),
    }


def compute_profit_max_reward(
    metrics: Dict[str, float],
    reward_params: Dict[str, Any] | None = None,
) -> Dict[str, float]:
    """Compute terminal grain-equivalent profit reward."""
    params = dict(reward_params or {})
    final_wso = _finite_float(metrics.get("final_wso"), 0.0)
    y_ref = profit_reward_scale(params)
    y_ratio = final_wso / y_ref

    cost_n = _reward_param(
        params,
        "cost_n_kg_wso_per_kg",
        "cost_n",
        "c_n",
        default=PROFIT_MAX_DEFAULTS["cost_n_kg_wso_per_kg"],
    )
    cost_p = _reward_param(
        params,
        "cost_p_kg_wso_per_kg",
        "cost_p",
        "c_p",
        default=PROFIT_MAX_DEFAULTS["cost_p_kg_wso_per_kg"],
    )
    cost_k = _reward_param(
        params,
        "cost_k_kg_wso_per_kg",
        "cost_k",
        "c_k",
        default=PROFIT_MAX_DEFAULTS["cost_k_kg_wso_per_kg"],
    )
    cost_irrig = _reward_param(
        params,
        "cost_irrig_kg_wso_per_mm",
        "cost_water",
        "c_w",
        default=PROFIT_MAX_DEFAULTS["cost_irrig_kg_wso_per_mm"],
    )

    total_n = _finite_float(metrics.get("total_n_kg_ha"), 0.0)
    total_p = _finite_float(metrics.get("total_p_kg_ha"), 0.0)
    total_k = _finite_float(metrics.get("total_k_kg_ha"), 0.0)
    total_irrig = _finite_float(metrics.get("total_irrig_mm"), 0.0)
    nutrient_cost = cost_n * total_n + cost_p * total_p + cost_k * total_k
    water_cost = cost_irrig * total_irrig
    input_cost = nutrient_cost + water_cost
    profit_ge = final_wso - input_cost
    score = profit_ge / y_ref

    return {
        "objective_reward": float(score),
        "y_ref": float(y_ref),
        "y_ratio": float(y_ratio),
        "reward_yield_term": float(y_ratio),
        "reward_profit_term": float(score),
        "revenue_ge_kg_ha": float(final_wso),
        "input_cost_ge_kg_ha": float(input_cost),
        "nutrient_cost_ge_kg_ha": float(nutrient_cost),
        "irrigation_cost_ge_kg_ha": float(water_cost),
        "profit_ge_kg_ha": float(profit_ge),
    }


def compute_water_stewardship_reward(
    metrics: Dict[str, float],
    reward_params: Dict[str, Any] | None = None,
) -> Dict[str, float]:
    """Compute terminal water-stewardship reward and diagnostics."""
    params = dict(reward_params or {})
    _, y_ref, y_ratio = _yield_ratio_terms(metrics, params, "water_stewardship")
    tau_y = _reward_param(params, "tau_y", "tau_Y", default=WATER_STEWARDSHIP_DEFAULTS["tau_y"])
    water_budget = _reward_param(
        params,
        "water_budget_mm",
        "budget_water_mm",
        "B_W",
        default=WATER_STEWARDSHIP_DEFAULTS["water_budget_mm"],
    )
    lambda_i = _reward_param(
        params,
        "lambda_i",
        "lambda_I",
        default=WATER_STEWARDSHIP_DEFAULTS["lambda_i"],
    )
    beta_y = _reward_param(params, "beta_y", "beta_Y", default=WATER_STEWARDSHIP_DEFAULTS["beta_y"])
    beta_budget = _reward_param(
        params,
        "beta_budget",
        "beta_B",
        default=WATER_STEWARDSHIP_DEFAULTS["beta_budget"],
    )

    total_irrig = _finite_float(metrics.get("total_irrig_mm"), 0.0)
    water_ratio = total_irrig / water_budget if water_budget > 0.0 else 0.0
    yield_floor_penalty = beta_y * _hinge_squared(tau_y - y_ratio)
    water_use_penalty = lambda_i * water_ratio
    water_budget_penalty = beta_budget * _hinge_squared(water_ratio - 1.0)
    score = y_ratio - water_use_penalty - yield_floor_penalty - water_budget_penalty

    return {
        "objective_reward": float(score),
        "y_ref": float(y_ref),
        "y_ratio": float(y_ratio),
        "reward_yield_term": float(y_ratio),
        "reward_yield_floor_penalty": float(yield_floor_penalty),
        "reward_water_use_penalty": float(water_use_penalty),
        "reward_water_budget_penalty": float(water_budget_penalty),
        "water_budget_mm": float(water_budget),
        "water_ratio": float(water_ratio),
    }


def compute_nutrient_stewardship_reward(
    metrics: Dict[str, float],
    reward_params: Dict[str, Any] | None = None,
) -> Dict[str, float]:
    """Compute the v1 terminal nutrient-stewardship reward and diagnostics."""
    params = dict(reward_params or {})
    _, y_ref, y_ratio = _yield_ratio_terms(metrics, params, "nutrient_stewardship")

    tau_y = _reward_param(params, "tau_y", "tau_Y", default=NUTRIENT_STEWARDSHIP_DEFAULTS["tau_y"])
    default_budget = _reward_param(params, "nutrient_budget_kg_ha", "B", default=180.0)
    budget_n = _reward_param(params, "budget_n_kg_ha", "B_N", default=default_budget)
    budget_p = _reward_param(params, "budget_p_kg_ha", "B_P", default=default_budget)
    budget_k = _reward_param(params, "budget_k_kg_ha", "B_K", default=default_budget)
    s_min = _reward_param(params, "terminal_nutrient_min", "S_min", default=8.0)
    s_max = _reward_param(params, "terminal_nutrient_max", "S_max", default=45.0)
    beta_y = _reward_param(params, "beta_y", "beta_Y", default=4.0)
    beta_application = _reward_param(params, "beta_application", "beta_A", default=0.35)
    beta_low = _reward_param(params, "beta_low", default=0.60)
    beta_high = _reward_param(params, "beta_high", default=0.50)

    applications = (
        (_finite_float(metrics.get("total_n_kg_ha"), 0.0), budget_n),
        (_finite_float(metrics.get("total_p_kg_ha"), 0.0), budget_p),
        (_finite_float(metrics.get("total_k_kg_ha"), 0.0), budget_k),
    )
    terminal_states = (
        _finite_float(metrics.get("terminal_navail"), 0.0),
        _finite_float(metrics.get("terminal_pavail"), 0.0),
        _finite_float(metrics.get("terminal_kavail"), 0.0),
    )

    over_application_sum = sum(
        _hinge_squared(application / budget - 1.0)
        for application, budget in applications
        if budget > 0.0
    )
    terminal_low_sum = sum(_hinge_squared(1.0 - state / s_min) for state in terminal_states if s_min > 0.0)
    terminal_high_sum = sum(_hinge_squared(state / s_max - 1.0) for state in terminal_states if s_max > 0.0)

    yield_floor_penalty = beta_y * _hinge_squared(tau_y - y_ratio)
    application_penalty = beta_application * over_application_sum
    terminal_low_penalty = beta_low * terminal_low_sum
    terminal_high_penalty = beta_high * terminal_high_sum
    score = y_ratio - yield_floor_penalty - application_penalty - terminal_low_penalty - terminal_high_penalty

    return {
        "objective_reward": float(score),
        "y_ref": float(y_ref),
        "y_ratio": float(y_ratio),
        "reward_yield_term": float(y_ratio),
        "reward_yield_floor_penalty": float(yield_floor_penalty),
        "reward_application_penalty": float(application_penalty),
        "reward_terminal_low_penalty": float(terminal_low_penalty),
        "reward_terminal_high_penalty": float(terminal_high_penalty),
        "reward_over_application_sum": float(over_application_sum),
        "reward_terminal_low_sum": float(terminal_low_sum),
        "reward_terminal_high_sum": float(terminal_high_sum),
    }


class WOFOSTEnv(BaseEnv):
    """Wrapper for wofost_gym environments with LLM interface.

    This class wraps a wofost_gym environment and provides a natural language
    interface for LLM agents. It automatically converts:
    - Numerical observations → Natural language turn prompts
    - LLM text responses → Numerical action IDs

    Attributes:
        config: WOFOSTEnvConfig object
        env: The underlying wofost_gym environment
        prompt_generator: WOFOSTPromptGenerator for text conversion
        llm_mode: If True, uses natural language interface; if False, uses numerical
    """

    def __init__(self, config: WOFOSTEnvConfig):
        """Initialize the WOFOST environment wrapper.

        Args:
            config: WOFOSTEnvConfig object containing environment configuration
        """
        _configure_pcse_weather_cache(getattr(config, "weather_cache_dir", None))

        config.wofost_gym_path = _resolve_wofost_gym_path(config.wofost_gym_path)

        # Add wofost_gym and its nested vendored packages to Python path.
        for path in reversed(_wofost_python_paths(config.wofost_gym_path)):
            if path not in sys.path:
                sys.path.insert(0, path)

        # Import wofost_gym modules
        import gymnasium as gym
        import utils
        from pcse_gym.args import NPK_Args, WOFOST_Args, Agro_Args

        # Store for later use
        self._gym = gym
        self._utils = utils

        # Create NPK_Args from config
        wofost_args = WOFOST_Args(**config.wofost_params)
        agro_args = Agro_Args(**config.agro_params)
        npk_kwargs = {
            "wf": wofost_args,
            "ag": agro_args,
            "seed": config.seed,
            "intvn_interval": config.intvn_interval,
        }
        if getattr(config, "output_vars", None) is not None:
            npk_kwargs["output_vars"] = list(config.output_vars)
        if getattr(config, "weather_vars", None) is not None:
            npk_kwargs["weather_vars"] = list(config.weather_vars)
        npk_args = NPK_Args(**npk_kwargs)
        if getattr(config, "fert_amount", None) is not None:
            npk_args.fert_amount = float(config.fert_amount)
        if getattr(config, "irrig_amount", None) is not None:
            npk_args.irrig_amount = float(config.irrig_amount)
        if config.scale_action_amounts_by_interval:
            _scale_action_amounts_by_interval(npk_args, config.intvn_interval)

        native_env_reward = getattr(config, "env_reward", None)

        # Create Args object
        args = utils.Args(
            npk=npk_args,
            env_id=config.env_id,
            env_reward=native_env_reward,
            agro_file=config.agro_file,
            base_fpath=config.wofost_gym_path,
            save_folder=config.save_folder,
        )

        # Store args for reference
        self._args = args

        # Create the wofost_gym environment
        self.env = utils.make_gym_env(args)

        # Apply reward wrapper if specified
        if native_env_reward:
            self.env = utils.wrap_env_reward(self.env, args)

        # Normalize reward to [0, 1]
        from pcse_gym.wrappers import NormalizeReward
        self.env = NormalizeReward(self.env)
        if str(config.env_id).strip().lower().removesuffix("-v0") == "lw":
            _patch_lw_noop_action_bug(self.env)

        # Full simulator observation ordering.
        self.output_vars = self.env.unwrapped.get_output_vars()

        prompt_fields = getattr(config, "prompt_observation_fields", None)
        self.prompt_observation_fields = list(prompt_fields) if prompt_fields else list(self.output_vars)
        missing_prompt_fields = [
            field for field in self.prompt_observation_fields if field not in self.output_vars
        ]
        if missing_prompt_fields:
            raise ValueError(
                "prompt_observation_fields contains fields that are not available "
                f"in the simulator observation: {missing_prompt_fields}"
            )

        # Create prompt generator for LLM interface
        self.prompt_generator = WOFOSTPromptGenerator.from_env(
            self.env,
            require_think=getattr(config, "require_think", False),
            thinking_mode=getattr(config, "thinking_mode", "grounding_decision"),
            think_tag=getattr(config, "think_tag", "tool_call"),
            action_schema_env_id=(
                getattr(config, "prompt_action_schema_env_id", None)
                or getattr(config, "env_id", None)
            ),
            objective_id=getattr(config, "prompt_objective_id", None)
            or getattr(config, "objective_id", "profit_max"),
            objective_text=getattr(config, "prompt_objective_text", None),
            include_profit_context=getattr(config, "include_profit_context", False),
            profit_context_params={
                **dict(getattr(config, "reward_params", {}) or {}),
                **dict(getattr(config, "profit_context_params", {}) or {}),
            },
            output_vars=self.prompt_observation_fields,
            field_aliases=getattr(config, "prompt_field_aliases", None),
        )
        self._maybe_load_crop_traits(config)

        # LLM mode: if True, return natural language prompts instead of numpy arrays
        self.llm_mode = getattr(config, 'llm_mode', True)
        self.objective_id = str(getattr(config, "objective_id", "profit_max") or "profit_max")
        self.reward_params = dict(getattr(config, "reward_params", {}) or {})
        y_ref = getattr(config, "y_ref", None)
        if y_ref is not None:
            self.reward_params.setdefault("y_ref", y_ref)

        # Action parsing stats
        self._total_steps = 0
        self._invalid_steps = 0
        self._total_n_kg_ha = 0.0
        self._total_p_kg_ha = 0.0
        self._total_k_kg_ha = 0.0
        self._total_irrig_mm = 0.0

        # Maximum number of turns per episode
        self.turn_num = config.turn_num

        # Bonus reward for valid action format
        self.valid_action_bonus = getattr(config, 'valid_action_bonus', 0.1)

        # Track last valid WSO/DVS for fallback when crop terminates mid-interval
        self._last_valid_wso = 0.0
        self._last_valid_dvs = 0.0
        self._last_observation_values: Dict[str, float] = {}

        # Initialize base class
        super().__init__(config)

    def _uses_terminal_objective_reward(self) -> bool:
        if self.objective_id in {"profit_max", "water_stewardship", "nutrient_stewardship"}:
            return True
        if self.objective_id == "yield_max":
            return _reward_param(self.reward_params, "y_ref", "Y_ref", default=0.0) > 0.0
        return False

    def _compute_objective_diagnostics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        if self.objective_id == "yield_max":
            return compute_yield_max_reward(metrics, self.reward_params)
        if self.objective_id == "profit_max":
            return compute_profit_max_reward(metrics, self.reward_params)
        if self.objective_id == "water_stewardship":
            return compute_water_stewardship_reward(metrics, self.reward_params)
        if self.objective_id == "nutrient_stewardship":
            return compute_nutrient_stewardship_reward(metrics, self.reward_params)
        return {}

    def _action_resource_amounts(self, action_id: int) -> tuple[float, float, float, float]:
        """Return N/P/K kg/ha and irrigation mm implied by an action id."""
        if action_id <= 0:
            return 0.0, 0.0, 0.0, 0.0

        offset = 1
        for component in getattr(self.prompt_generator, "action_components", []):
            if component in {"n", "p", "k"}:
                num_fert = int(getattr(self.prompt_generator, "num_fert", 4))
                if offset <= action_id < offset + num_fert:
                    amount = float(action_id - offset + 1) * float(
                        getattr(self.prompt_generator, "fert_amount", 0.0)
                    )
                    if component == "n":
                        return amount, 0.0, 0.0, 0.0
                    if component == "p":
                        return 0.0, amount, 0.0, 0.0
                    return 0.0, 0.0, amount, 0.0
                offset += num_fert
            elif component == "irrig":
                num_irrig = int(getattr(self.prompt_generator, "num_irrig", 4))
                if offset <= action_id < offset + num_irrig:
                    cm = float(action_id - offset + 1) * float(
                        getattr(self.prompt_generator, "irrig_amount", 0.0)
                    )
                    return 0.0, 0.0, 0.0, cm * 10.0
                offset += num_irrig
        return 0.0, 0.0, 0.0, 0.0

    def _crop_traits_dir(self, config: WOFOSTEnvConfig) -> Path:
        """Return the configured crop traits root directory."""
        raw_dir = getattr(config, "crop_traits_dir", DEFAULT_CROP_TRAITS_DIR)
        traits_dir = Path(str(raw_dir))
        if not traits_dir.is_absolute():
            traits_dir = REPO_ROOT / traits_dir
        return traits_dir.resolve()

    def _infer_crop_name(self, config: WOFOSTEnvConfig) -> str:
        """Infer crop name from environment agromanagement or fallback config."""
        try:
            unwrapped = self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env
            agro = getattr(unwrapped, "agromanagement", {})
            if isinstance(agro, dict):
                crop_name = agro.get("CropCalendar", {}).get("crop_name")
                if crop_name:
                    return str(crop_name)
        except Exception:
            pass

        agro_file = str(getattr(config, "agro_file", "") or "")
        if agro_file.endswith("_agro.yaml"):
            return agro_file[:-10]
        if agro_file.endswith(".yaml"):
            return agro_file[:-5]
        return agro_file or "unknown_crop"

    def _maybe_load_crop_traits(self, config: WOFOSTEnvConfig) -> None:
        """Load crop traits text and inject it into prompt generator when enabled."""
        if not getattr(config, "include_crop_traits", False):
            return

        crop_name = self._infer_crop_name(config)
        trait_key = crop_name
        if getattr(config, "include_variety_traits", False):
            crop_variety = crop_variety_from_env_config(config.to_dict())
            trait_key = crop_variety_trait_key(crop_name, crop_variety)
        traits_path = resolve_crop_trait_artifact_path(
            self._crop_traits_dir(config),
            trait_key,
            ".txt",
            getattr(config, "trait_schema", None),
        )
        if not traits_path.exists():
            raise FileNotFoundError(
                f"Crop traits not found for trait key '{trait_key}': {traits_path}"
            )

        traits_text = traits_path.read_text(encoding="utf-8").strip()
        self.prompt_generator.set_crop_traits(traits_text)

    def _serialize_observation(self, raw_obs: Any) -> Dict[str, float]:
        """Convert an observation array into a JSON-friendly named-value mapping."""
        if hasattr(raw_obs, "tolist"):
            values = raw_obs.tolist()
        else:
            values = list(raw_obs)

        raw_values = [float(value) for value in values]
        return {
            var_name: raw_values[idx]
            for idx, var_name in enumerate(self.output_vars[:len(raw_values)])
        }

    def _project_observation_for_prompt(self, raw_obs: Any) -> Any:
        """Return observation values ordered by prompt_observation_fields."""
        if hasattr(raw_obs, "tolist"):
            values = raw_obs.tolist()
        else:
            values = list(raw_obs)
        value_by_field = {
            var_name: float(values[idx])
            for idx, var_name in enumerate(self.output_vars[:len(values)])
        }
        projected = [value_by_field[field] for field in self.prompt_observation_fields]
        try:
            import numpy as np

            return np.asarray(projected, dtype="float64")
        except Exception:
            return projected

    @staticmethod
    def _obs_value(observation_values: Dict[str, float], key: str, default: float = 0.0) -> float:
        return _finite_float(observation_values.get(key), default)

    def _current_episode_metrics(self) -> Dict[str, float]:
        values = self._last_observation_values
        final_wso = self._last_valid_wso or self._obs_value(values, "WSO", 0.0)
        total_irrig_cm = self._obs_value(values, "TOTIRRIG", self._total_irrig_mm / 10.0)
        metrics = {
            "final_wso": float(final_wso),
            "target_yield": float(final_wso),
            "y_ref": _finite_float(self.reward_params.get("y_ref"), float("nan")),
            "total_n_kg_ha": self._obs_value(values, "TOTN", self._total_n_kg_ha),
            "total_p_kg_ha": self._obs_value(values, "TOTP", self._total_p_kg_ha),
            "total_k_kg_ha": self._obs_value(values, "TOTK", self._total_k_kg_ha),
            "total_irrig_cm": float(total_irrig_cm),
            "total_irrig_mm": float(total_irrig_cm * 10.0),
            "terminal_navail": self._obs_value(values, "NAVAIL", float("nan")),
            "terminal_pavail": self._obs_value(values, "PAVAIL", float("nan")),
            "terminal_kavail": self._obs_value(values, "KAVAIL", float("nan")),
        }
        return metrics

    def _objective_reward_for_step(self, done: bool, native_reward: float) -> tuple[float, Dict[str, float]]:
        if not self._uses_terminal_objective_reward():
            return float(native_reward), {}
        if not done:
            return 0.0, {}

        diagnostics = self._compute_objective_diagnostics(self._current_episode_metrics())
        return diagnostics["objective_reward"], diagnostics

    def reset(self) -> Tuple[Any, Dict[str, Any]]:
        """Reset the environment to initial state.

        Returns:
            obs: Initial observation (natural language prompt if llm_mode=True,
                 numpy array if llm_mode=False)
            info: Dictionary containing turn_metrics plus named observation values
        """
        self._total_steps = 0
        self._invalid_steps = 0
        self._total_n_kg_ha = 0.0
        self._total_p_kg_ha = 0.0
        self._total_k_kg_ha = 0.0
        self._total_irrig_mm = 0.0
        self._last_valid_wso = 0.0
        self._last_valid_dvs = 0.0

        raw_obs, _ = self.env.reset()
        self._last_observation_values = self._serialize_observation(raw_obs)
        prompt_obs = self._project_observation_for_prompt(raw_obs)

        # Extract WSO and DVS values from observation
        wso_idx = self.output_vars.index('WSO') if 'WSO' in self.output_vars else None
        wso_value = float(raw_obs[wso_idx]) if wso_idx is not None else 0.0
        dvs_idx = self.output_vars.index('DVS') if 'DVS' in self.output_vars else None
        dvs_value = float(raw_obs[dvs_idx]) if dvs_idx is not None else 0.0

        # At reset there is no action transition yet, so turn_metrics describes
        # the current initial state rather than a post-step state.
        # Rollout code may choose to persist this separately from per-turn
        # post-step metrics.
        info = {
            'turn_metrics': {
                'wso': wso_value,
                'dvs': dvs_value,
                'reward': 0.0,
                # Living biomass
                'wlv': 0.0,
                'wst': 0.0,
                'wrt': 0.0,
                # Total biomass
                'tagp': 0.0,
                'twlv': 0.0,
                'twst': 0.0,
                'twrt': 0.0,
                'twso': 0.0,
                # Storage organ dynamics
                'dwso': 0.0,
                'grso': 0.0,
                'drso': 0.0,
                'gwso': 0.0,
            },
            'observation': dict(self._last_observation_values),
            'prompt_observation': {
                field: float(value)
                for field, value in zip(self.prompt_observation_fields, prompt_obs)
            },
        }

        if self.llm_mode:
            # Convert observation to natural language turn prompt
            turn_prompt = self.prompt_generator.get_turn_prompt(prompt_obs)
            return turn_prompt, info
        else:
            # Return raw numerical observation
            return raw_obs, info

    def step(self, action: Union[str, int]) -> Tuple[Any, float, bool, Dict[str, Any]]:
        """Execute one step in the environment.

        Args:
            action: Action to execute. Can be:
                    - String (LLM response) if llm_mode=True, will be parsed to action ID
                    - Integer (action ID) if llm_mode=False or for direct control

        Returns:
            obs: Observation after executing the action (natural language prompt if
                 llm_mode=True, numpy array if llm_mode=False)
            reward: Reward obtained from the action
            done: Whether the episode has ended
            info: Dictionary containing:
                  - 'turn_metrics': Dict with per-step metrics (wso, dvs, biomass, etc.)
                  - 'observation': Observation values keyed by output variable name
                  - 'executed_action_id': Actual discrete action applied to the simulator
                  - 'invalid_action': Whether string parsing failed and fell back to action 0
                  - 'trajectory_metrics': Dict with episode-level metrics (only when done=True)
                  - 'raw_llm_response': Original LLM response (if action was a string)
        """
        # Store raw LLM response if action is a string
        raw_llm_response = None
        invalid_action = False
        if isinstance(action, str):
            raw_llm_response = action
            action_id = self.prompt_generator.parse_action_response(action)
            if action_id is None:
                action_id = 0  # fallback to do nothing
                invalid_action = True
        else:
            action_id = int(action)

        self._total_steps += 1
        if invalid_action:
            self._invalid_steps += 1
        n_kg_ha, p_kg_ha, k_kg_ha, irrig_mm = self._action_resource_amounts(action_id)
        self._total_n_kg_ha += n_kg_ha
        self._total_p_kg_ha += p_kg_ha
        self._total_k_kg_ha += k_kg_ha
        self._total_irrig_mm += irrig_mm

        # Execute action in underlying environment
        raw_obs, reward, terminated, truncated, env_info = self.env.step(action_id)
        observation_values = self._serialize_observation(raw_obs)
        prompt_obs = self._project_observation_for_prompt(raw_obs)
        done = terminated or truncated

        # Truncate at turn_num if the env hasn't naturally terminated
        if self._total_steps >= self.turn_num:
            done = True

        # Extract WSO and DVS values from observation
        wso_idx = self.output_vars.index('WSO') if 'WSO' in self.output_vars else None
        wso_value = float(raw_obs[wso_idx]) if wso_idx is not None else 0.0
        dvs_idx = self.output_vars.index('DVS') if 'DVS' in self.output_vars else None
        dvs_value = float(raw_obs[dvs_idx]) if dvs_idx is not None else 0.0

        # Track last valid WSO/DVS. When crop terminates mid-interval
        # (intvn_interval > 1), the observation returns 0 for WSO and DVS.
        # Fall back to the last valid value from a previous step.
        if wso_value > 0:
            self._last_valid_wso = wso_value
        elif (terminated or truncated) and wso_value == 0.0:
            wso_value = self._last_valid_wso
            observation_values["WSO"] = wso_value
            # A native WOFOST reward may have used the terminal zero WSO.
            # Keep the fallback non-negative before AgriManager applies
            # the objective_id reward below.
            reward = max(reward, 0.0)

        if dvs_value > 0:
            self._last_valid_dvs = dvs_value
        elif (terminated or truncated) and dvs_value == 0.0:
            dvs_value = self._last_valid_dvs
            observation_values["DVS"] = dvs_value

        self._last_observation_values = observation_values

        # Helper function to get latest value from date-keyed dict
        def get_latest_value(var_dict):
            if var_dict and len(var_dict) > 0:
                return float(list(var_dict.values())[-1])
            return 0.0

        total_n_kg_ha = self._obs_value(observation_values, "TOTN", self._total_n_kg_ha)
        total_p_kg_ha = self._obs_value(observation_values, "TOTP", self._total_p_kg_ha)
        total_k_kg_ha = self._obs_value(observation_values, "TOTK", self._total_k_kg_ha)
        total_irrig_cm = self._obs_value(observation_values, "TOTIRRIG", self._total_irrig_mm / 10.0)
        objective_reward, reward_diagnostics = self._objective_reward_for_step(done, float(reward))
        reward = objective_reward

        # turn_metrics in step() is explicitly post-step: it describes the
        # simulator state after applying this turn's action.
        turn_metrics = {
            'wso': wso_value,
            'dvs': dvs_value,
            'reward': 0.0,
            # Living biomass
            'wlv': get_latest_value(env_info.get('WLV', {})),
            'wst': get_latest_value(env_info.get('WST', {})),
            'wrt': get_latest_value(env_info.get('WRT', {})),
            # Total biomass
            'tagp': get_latest_value(env_info.get('TAGP', {})),
            'twlv': get_latest_value(env_info.get('TWLV', {})),
            'twst': get_latest_value(env_info.get('TWST', {})),
            'twrt': get_latest_value(env_info.get('TWRT', {})),
            'twso': get_latest_value(env_info.get('TWSO', {})),
            # Storage organ dynamics
            'dwso': get_latest_value(env_info.get('DWSO', {})),
            'grso': get_latest_value(env_info.get('GRSO', {})),
            'drso': get_latest_value(env_info.get('DRSO', {})),
            'gwso': get_latest_value(env_info.get('GWSO', {})),
            'total_n_kg_ha': total_n_kg_ha,
            'total_p_kg_ha': total_p_kg_ha,
            'total_k_kg_ha': total_k_kg_ha,
            'total_irrig_cm': total_irrig_cm,
            'total_irrig_mm': total_irrig_cm * 10.0,
            'terminal_navail': self._obs_value(observation_values, "NAVAIL", float("nan")),
            'terminal_pavail': self._obs_value(observation_values, "PAVAIL", float("nan")),
            'terminal_kavail': self._obs_value(observation_values, "KAVAIL", float("nan")),
            'objective_reward': objective_reward,
        }
        turn_metrics.update(reward_diagnostics)

        # Bonus reward for valid action format; invalid keeps env reward as-is
        if not invalid_action:
            reward += self.valid_action_bonus
        turn_metrics['reward'] = float(reward)

        # Create info with turn_metrics
        info = {
            'turn_metrics': turn_metrics,
            'observation': dict(observation_values),
            'prompt_observation': {
                field: float(value)
                for field, value in zip(self.prompt_observation_fields, prompt_obs)
            },
            'executed_action_id': int(action_id),
            'invalid_action': invalid_action,
        }

        # Add trajectory_metrics only when the episode is done
        if done:
            info['trajectory_metrics'] = self.get_trajectory_metrics()

        # Add raw_llm_response if it exists
        if raw_llm_response is not None:
            info['raw_llm_response'] = raw_llm_response

        if self.llm_mode:
            # Convert observation to natural language turn prompt
            turn_prompt = self.prompt_generator.get_turn_prompt(prompt_obs)
            return turn_prompt, reward, done, info
        else:
            # Return raw numerical observation
            return raw_obs, reward, done, info

    def system_prompt(self) -> str:
        """Get the system prompt for LLM agents.

        Returns:
            System prompt string describing the environment and task
        """
        return self.prompt_generator.get_system_prompt()

    def get_trajectory_metrics(self) -> Dict[str, float]:
        total_steps = max(self._total_steps, 1)
        metrics = self._current_episode_metrics()
        metrics.update({
            'invalid_action_rate': float(self._invalid_steps / total_steps),
            'invalid_steps': float(self._invalid_steps),
            'total_steps': float(self._total_steps),
        })
        if self._uses_terminal_objective_reward():
            metrics.update(self._compute_objective_diagnostics(metrics))
        return metrics

    def close(self) -> None:
        """Close the environment and cleanup resources."""
        if hasattr(self, 'env') and self.env is not None:
            self.env.close()
