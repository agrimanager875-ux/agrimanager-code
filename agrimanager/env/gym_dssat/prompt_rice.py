"""
Prompt generation for gym-dssat Rice (CERES-Rice) environments.

Rice-specific variable set, action bounds, and agronomic hints.
Inherits all structural logic from DSSATPromptGenerator.
"""

from typing import Dict, List, Optional, Tuple
from .prompt import DSSATPromptGenerator, VALID_THINKING_MODES, _normalize_thinking_mode

# Rice observation variables for mode='all'
RICE_OUTPUT_VARS = [
    "dap",
    "istage",
    "grnwt",
    "topwt",
    "lai",
    "tilno",
    "gpp",
    "dyield",
    "chtd",
    "cwad",
    "nstres",
    "swfac",
    "pcnveg",
    "xgnp",
    "xanc",
    "grainn",
    "andem",
    "nuf",
    "unh4",
    "uno3",
    "dtt",
    "cumsumfert",
]

# Rice growth stage labels (istage index → meaning)
RICE_ISTAGE_DESC = {
    0: "pre-planting",
    1: "vegetative (seedling)",
    2: "vegetative (active tillering)",
    3: "reproductive (panicle initiation)",
    4: "reproductive (booting)",
    5: "reproductive (heading)",
    6: "ripening (grain fill)",
    7: "maturity",
    8: "harvest",
}

# Rice-specific description map
RICE_DESC_MAP: Dict[str, Tuple[str, str]] = {
    "dap":      ("Timeline",          "Days after planting"),
    "istage":   ("Crop status",       "Development stage (0=pre-plant … 7=maturity)"),
    "grnwt":    ("Crop status",       "Grain dry weight (kg/ha; harvestable yield biomass)"),
    "topwt":    ("Crop status",       "Total aboveground biomass (kg/ha)"),
    "lai":      ("Crop status",       "Leaf area index (m²_leaf / m²_ground)"),
    "tilno":    ("Crop status",       "Tiller number per plant"),
    "gpp":      ("Crop status",       "Grain number per plant"),
    "dyield":   ("Crop status",       "Dry yield biomass (kg/ha)"),
    "chtd":     ("Crop status",       "Canopy height (m)"),
    "cwad":     ("Crop status",       "Total crop dry weight (kg/ha)"),
    "grainn":   ("Crop status",       "Grain nitrogen content (g_N/plant)"),
    "xgnp":     ("Crop status",       "Nitrogen content of grain (%)"),
    "xanc":     ("Crop status",       "N concentration in aboveground biomass (%)"),
    "pcnveg":   ("Crop status",       "Stem + leaf N concentration (%)"),
    "andem":    ("Crop status",       "Total crop N demand (kg_N/ha)"),
    "nstres":   ("Stress",            "Nitrogen stress factor (0–1, 1 = max stress)"),
    "swfac":    ("Stress",            "Soil water stress factor (0–1, 1 = max stress)"),
    "nuf":      ("N uptake",          "N supply-to-demand ratio (higher = better N availability)"),
    "unh4":     ("N uptake",          "Plant ammonium uptake today (kg_N/ha/day)"),
    "uno3":     ("N uptake",          "Plant nitrate uptake today (kg_N/ha/day)"),
    "dtt":      ("Weather",           "Thermal time today (°C·day)"),
    "cumsumfert": ("Cumulative actions", "Total nitrogen fertilizer applied (kg/ha)"),
}

# Per-step application limits for rice
RICE_FERT_MAX  = 30.0   # kg N/ha per step  (season max ~120–180 kg/ha; no penalty → explore)
RICE_IRRIG_MAX = 80.0   # mm per step        (paddy can absorb large events)


class RicePromptGenerator(DSSATPromptGenerator):
    """
    Prompt generator tuned for DSSAT CERES-Rice.

    Key differences from maize:
    - Uses rice-specific observation variables (tiller number, panicle metrics, grain N)
    - Higher irrigation limit (80 mm vs 40 mm); same fert limit (30 kg/ha)
    - Zero fertilization and irrigation penalties in the reward — model needs to
      learn through exploration, not conservative heuristics
    - System prompt reflects paddy phenology and flood irrigation context
    - istage shown with human-readable label in observation
    """

    def __init__(
        self,
        season_length: int = 150,
        location: str = "the field",
        output_vars: Optional[List[str]] = None,
        desc_map: Optional[Dict[str, Tuple[str, str]]] = None,
        require_think: bool = False,
        thinking_mode: str = "grounding_decision",
        think_tag: str = "tool_call",
        decision_interval: int = 1,
        enable_pests: bool = False,
        fert_max: Optional[float] = None,
        irrig_max: Optional[float] = None,
        include_crop_traits: bool = False,
        include_profit_context: bool = False,
        profit_context_params: Optional[Dict[str, float]] = None,
        objective_id: str = "profit_max",
        objective_text: Optional[str] = None,
        reward_params: Optional[Dict[str, float]] = None,
    ):
        super().__init__(
            crop_name="rice",
            season_length=season_length,
            location=location,
            output_vars=output_vars or RICE_OUTPUT_VARS,
            desc_map=desc_map or RICE_DESC_MAP,
            enable_pests=enable_pests,
            require_think=require_think,
            thinking_mode=thinking_mode,
            think_tag=think_tag,
            decision_interval=decision_interval,
            fert_max=fert_max if fert_max is not None else RICE_FERT_MAX,
            irrig_max=irrig_max if irrig_max is not None else RICE_IRRIG_MAX,
            include_crop_traits=include_crop_traits,
            include_profit_context=include_profit_context,
            profit_context_params=profit_context_params,
            objective_id=objective_id,
            objective_text=objective_text,
            reward_params=reward_params,
        )

    def describe_observation(self, observation, output_vars=None, seed=None) -> str:
        """Override to annotate istage with a human-readable label."""
        text = super().describe_observation(observation, output_vars, seed)
        # Append istage legend as a footnote if istage is in the observation
        vars_ = output_vars or self.output_vars
        obs_dict = observation if isinstance(observation, dict) else dict(zip(vars_, observation))
        if "istage" in obs_dict:
            stage_idx = int(obs_dict["istage"])
            label = RICE_ISTAGE_DESC.get(stage_idx, "unknown")
            text += f"\n[Development stage legend] istage {stage_idx} = {label}"
        return text

    def get_system_prompt(self) -> str:
        return super().get_system_prompt()
