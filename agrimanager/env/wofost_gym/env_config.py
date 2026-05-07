"""wofost_gym environment configuration.

This module defines the configuration class for wofost_gym environments.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any, List

from agrimanager.env.base import BaseEnvConfig
from .crop_trait_schemas import DEFAULT_CROP_TRAIT_SCHEMA

AGRIMANAGER_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = AGRIMANAGER_PACKAGE_ROOT.parent
DEFAULT_WOFOST_GYM_PATH = os.environ.get(
    "WOFOST_GYM_PATH",
    str((REPO_ROOT / ".." / "AgriManagerExternal" / "WOFOSTGym").resolve())
)
DEFAULT_CROP_TRAITS_DIR = (Path("agrimanager") / "env" / "wofost_gym" / "crop_traits").as_posix()


def _ensure_trailing_sep(path: str) -> str:
    """Return path with a trailing separator for compatibility with wofost_gym."""
    if path.endswith(('/', '\\')):
        return path
    return path + os.sep


def _resolve_repo_relative_path(path: str) -> str:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    return str(resolved.resolve())


class WOFOSTEnvConfig(BaseEnvConfig):
    """Configuration for wofost_gym environments.

    This class wraps the wofost_gym configuration parameters into a simple
    interface compatible with the BaseEnvConfig.
    """

    def __init__(
        self,
        env_id: str = "lnpkw-v0",
        agro_file: str = "wheat_agro.yaml",
        wofost_gym_path: str = DEFAULT_WOFOST_GYM_PATH,
        save_folder: str = "/tmp/wofost_configs/",
        weather_cache_dir: Optional[str] = None,
        llm_mode: bool = True,
        env_reward: Optional[str] = None,
        require_think: bool = False,
        thinking_mode: str = "grounding_decision",
        think_tag: str = "tool_call",
        output_vars: Optional[List[str]] = None,
        weather_vars: Optional[List[str]] = None,
        prompt_action_schema_env_id: Optional[str] = None,
        objective_id: str = "profit_max",
        prompt_objective_id: Optional[str] = None,
        prompt_objective_text: Optional[str] = None,
        reward_params: Optional[Dict[str, Any]] = None,
        include_profit_context: bool = False,
        profit_context_params: Optional[Dict[str, Any]] = None,
        y_ref: Optional[float] = None,
        wofost_params: Optional[Dict[str, Any]] = None,
        agro_params: Optional[Dict[str, Any]] = None,
        turn_num: int = 241,
        valid_action_bonus: float = 0.1,
        intvn_interval: int = 1,
        scale_action_amounts_by_interval: bool = False,
        include_crop_traits: bool = False,
        include_variety_traits: bool = False,
        crop_traits_dir: str = DEFAULT_CROP_TRAITS_DIR,
        trait_schema: str = DEFAULT_CROP_TRAIT_SCHEMA,
        prompt_observation_fields: Optional[List[str]] = None,
        prompt_field_aliases: Optional[Dict[str, str]] = None,
        observation_schema_name: Optional[str] = None,
        seed: Optional[int] = None,
        **kwargs
    ):
        """Initialize wofost_gym environment configuration.

        Args:
            env_id: Environment template ID
            agro_file: Agromanagement YAML filename
            wofost_gym_path: Path to wofost_gym installation
            save_folder: Folder to save configuration files (must end with '/')
            weather_cache_dir: Optional bundled NASA POWER cache directory
            llm_mode: If True, use natural language interface for LLM agents
            require_think: Whether use thinking mode
            thinking_mode: Thinking prompt variant when require_think=True.
                Supported values: "minimal", "think" (alias of
                "grounding_decision"), "grounding_decision"
            output_vars: Optional WOFOST-Gym simulator output variables.
            weather_vars: Optional WOFOST-Gym weather variables.
            prompt_action_schema_env_id: Optional prompt-only action schema override
                for schema-corruption ablations where the prompt describes a
                different menu than the simulator executes.
            objective_id: True management objective and reward form.
            prompt_objective_id: Optional objective form shown in the prompt.
                Defaults to ``objective_id``; set differently for corruption probes.
            prompt_objective_text: Optional free-form prompt objective override.
            reward_params: Optional objective-specific reward parameters.
            include_profit_context: Backward-compatible flag accepted by older
                configs. New CropGrowth prompts put profit details inside the
                shared management objective block.
            profit_context_params: Backward-compatible profit parameter
                overrides merged into prompt objective parameters.
            y_ref: Optional calibrated yield reference for normalized objectives.
            env_reward: Optional legacy native WOFOST-Gym reward wrapper name.
                New AgriManager experiment configs should use ``objective_id``
                to define the reward objective instead.
            wofost_params: WOFOST model parameters to override
            agro_params: Agromanagement parameters to override
            turn_num: Maximum number of turns per episode
            valid_action_bonus: Bonus reward for correctly formatted actions
            intvn_interval: Decision interval in days (e.g., 7 = one decision per week).
            scale_action_amounts_by_interval: Whether to multiply fertilizer and
                irrigation amounts by ``intvn_interval`` while keeping the number
                of discrete actions unchanged.
            include_crop_traits: Whether to load and inject crop traits into prompts
            include_variety_traits: Whether crop trait lookup should use
                ``(crop_name, crop_variety)`` instead of only ``crop_name``.
            crop_traits_dir: Root directory containing crop-trait cards.
            trait_schema: Schema name used to select crop-trait cards.
            prompt_observation_fields: Optional ordered subset or superset of
                simulator observation fields exposed in the LLM prompt.
            prompt_field_aliases: Optional prompt-only display-name overrides
                for observation fields.
            observation_schema_name: Optional prompt-schema label stored in row
                metadata for grouped validation metrics.
            seed: Random seed
            **kwargs: Additional parameters
        """
        super().__init__(seed=seed, **kwargs)

        self.env_id = env_id
        self.agro_file = agro_file
        # Ensure paths end with '/'
        self.wofost_gym_path = _ensure_trailing_sep(_resolve_repo_relative_path(wofost_gym_path))
        if save_folder is None:
            save_folder = "/tmp/wofost_configs/"
        self.save_folder = _ensure_trailing_sep(_resolve_repo_relative_path(save_folder))
        self.weather_cache_dir = (
            _resolve_repo_relative_path(weather_cache_dir) if weather_cache_dir else None
        )
        self.llm_mode = llm_mode
        self.env_reward = env_reward
        self.require_think = require_think
        self.thinking_mode = thinking_mode
        self.think_tag = think_tag
        self.output_vars = list(output_vars) if output_vars is not None else None
        self.weather_vars = list(weather_vars) if weather_vars is not None else None
        self.prompt_action_schema_env_id = prompt_action_schema_env_id
        self.objective_id = objective_id
        self.prompt_objective_id = prompt_objective_id
        self.prompt_objective_text = prompt_objective_text
        self.reward_params = reward_params or {}
        self.include_profit_context = include_profit_context
        self.profit_context_params = profit_context_params or {}
        self.y_ref = y_ref
        self.wofost_params = wofost_params or {}
        self.agro_params = agro_params or {}
        self.turn_num = turn_num
        self.valid_action_bonus = valid_action_bonus
        self.intvn_interval = intvn_interval
        self.scale_action_amounts_by_interval = scale_action_amounts_by_interval
        self.include_crop_traits = include_crop_traits
        self.include_variety_traits = include_variety_traits
        self.crop_traits_dir = _resolve_repo_relative_path(crop_traits_dir)
        self.trait_schema = trait_schema
        self.prompt_observation_fields = (
            list(prompt_observation_fields) if prompt_observation_fields is not None else None
        )
        self.prompt_field_aliases = (
            {str(key): str(value) for key, value in prompt_field_aliases.items()}
            if prompt_field_aliases is not None
            else None
        )
        self.observation_schema_name = observation_schema_name
