"""
dssat_gym environment configuration.

This module defines the configuration class for gym_dssat environments.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any

from agrimanager.env.base import BaseEnvConfig

# Root resolution relative to AgriManager repo
AGRIMANAGER_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = AGRIMANAGER_PACKAGE_ROOT.parent

DEFAULT_DSSAT_GYM_PATH = os.environ.get(
    "DSSAT_GYM_PATH",
    str(
        (
            (REPO_ROOT / "spack" / "gym-dssat-pdi")
            if (REPO_ROOT / "spack" / "gym-dssat-pdi").exists()
            else (REPO_ROOT / ".." / "AgriManagerExternal" / "DSSATGym")
        ).resolve()
    )
)

# ✅ NEW: persistent DSSAT output directory inside AgriManager
DEFAULT_DSSAT_OUTPUT_PATH = str((REPO_ROOT / "dssat_outputs").resolve())

def _ensure_trailing_sep(path: str) -> str:
    """Return path with trailing slash for compatibility with gym-dssat."""
    if path.endswith(("/", "\\")):
        return path
    return path + os.sep


class DSSATEnvConfig(BaseEnvConfig):
    """Configuration for gym-dssat environments.

    Wraps DSSAT environment parameters into an interface compatible
    with AgriManager's BaseEnvConfig.

    Attributes:
        env_id: Arbitrary environment ID (e.g. "maize-irrigation-v0")
        dssat_gym_path: Path to gym-dssat or DSSAT binary installation
        save_folder: Directory for outputs and logs
        llm_mode: If True, use text interface; if False, numeric
        env_reward: Optional reward wrapper (e.g. "yield_reward")
        dssat_params: DSSAT simulation parameters (fertilizer, irrigation, etc.)
        env_params: General environment parameters (e.g. timestep, mode)
        turn_num: Max steps per episode (days)
        seed: Random seed for reproducibility
        enable_pests: If True, simulate pest pressure and allow pesticide actions
        pest_config: Configuration for pest simulation (pressure_model, damage_model, etc.)
    """

    def __init__(
        self,
        env_id: str = "maize-all-v0",
        dssat_gym_path: str = DEFAULT_DSSAT_GYM_PATH,
        save_folder: str = DEFAULT_DSSAT_OUTPUT_PATH,
        llm_mode: bool = True,
        env_reward: Optional[str] = None,
        dssat_params: Optional[Dict[str, Any]] = None,
        env_params: Optional[Dict[str, Any]] = None,
        turn_num: int = 200,
        seed: Optional[int] = None,
        enable_pests: bool = False,
        pest_config: Optional[Dict[str, Any]] = None,
        require_think: bool = False,
        think_tag: str = "tool_call",
        include_crop_traits: bool = False,
        decision_interval: int = 1,
        num_seasons: int = 1,
        valid_action_bonus: float = 0.1,
        objective_id: str = "profit_max",
        prompt_objective_id: Optional[str] = None,
        prompt_objective_text: Optional[str] = None,
        reward_params: Optional[Dict[str, Any]] = None,
        include_profit_context: bool = False,
        profit_context_params: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Initialize DSSAT environment configuration."""
        super().__init__(seed=seed, **kwargs)

        self.env_id = env_id
        self.dssat_gym_path = _ensure_trailing_sep(dssat_gym_path)
        self.save_folder = _ensure_trailing_sep(save_folder)
        self.llm_mode = llm_mode
        self.env_reward = env_reward
        self.dssat_params = dssat_params or {}
        self.env_params = env_params or {}
        self.turn_num = turn_num
        self.enable_pests = enable_pests
        self.require_think = require_think
        self.think_tag = think_tag
        self.include_crop_traits = bool(include_crop_traits)
        self.decision_interval = decision_interval
        self.num_seasons = max(1, int(num_seasons))
        self.valid_action_bonus = 0.1 if valid_action_bonus is None else float(valid_action_bonus)
        self.objective_id = objective_id
        self.prompt_objective_id = prompt_objective_id
        self.prompt_objective_text = prompt_objective_text
        self.reward_params = reward_params or {}
        self.include_profit_context = include_profit_context
        self.profit_context_params = profit_context_params or {}
        self.pest_config = pest_config or {
            "base_pressure": 0.3,  # Base pest pressure (0-1 scale)
            "weather_sensitivity": 0.5,  # How much weather affects pests
            "damage_rate": 0.02,  # Yield loss per day per pest pressure unit
            "pesticide_efficacy": 0.7,  # How effective pesticides are (0-1)
            "pesticide_cost": 15.0,  # Cost per application ($/ha)
        }

        # ✅ Auto-create persistent save folder
        os.makedirs(self.save_folder, exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        base_dict = super().to_dict()
        base_dict.update({
            "env_id": self.env_id,
            "env_name": "gym_dssat",
            "dssat_gym_path": self.dssat_gym_path,
            "save_folder": self.save_folder,
            "llm_mode": self.llm_mode,
            "dssat_params": self.dssat_params,
            "env_params": self.env_params,
            "turn_num": self.turn_num,
            "require_think": self.require_think,
            "think_tag": self.think_tag,
            "include_crop_traits": self.include_crop_traits,
            "decision_interval": self.decision_interval,
            "num_seasons": self.num_seasons,
            "valid_action_bonus": self.valid_action_bonus,
            "objective_id": self.objective_id,
            "prompt_objective_id": self.prompt_objective_id,
            "prompt_objective_text": self.prompt_objective_text,
            "reward_params": self.reward_params,
            "include_profit_context": self.include_profit_context,
            "profit_context_params": self.profit_context_params,
            "enable_pests": self.enable_pests,
            "pest_config": self.pest_config,
        })
        if self.env_reward:
            base_dict["env_reward"] = self.env_reward
        return base_dict
