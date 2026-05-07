"""
Prompt generation for gym-dssat Cotton (CROPGRO-Cotton) environments.

Cotton-specific variable set, action bounds, and agronomic hints.
Inherits all structural logic from DSSATPromptGenerator.
"""

from typing import Dict, List, Optional, Tuple
from .prompt import DSSATPromptGenerator, VALID_THINKING_MODES, _normalize_thinking_mode

# Cotton observation variables for mode='all'
COTTON_OUTPUT_VARS = [
    "dap",
    "vstage",
    "grnwt",
    "topwt",
    "xlai",
    "swfac",
    "nstres",
    "turfac",
    "wtnup",
    "pcnl",
    "wtntot",
    "totwt",
    "shelwt",
    "sdwt",
    "podwt",
    "sdlip",
    "pcnsd",
    "nfixn",
    "laimx",
    "cannaa",
    "clw",
    "cumsumfert",
    "totir",
]

# Cotton-specific description map (only the variables cotton actually reports)
COTTON_DESC_MAP: Dict[str, Tuple[str, str]] = {
    "dap":      ("Timeline",          "Days after planting"),
    "vstage":   ("Crop status",       "Vegetative stage (number of nodes on main stem)"),
    "grnwt":    ("Crop status",       "Seed (lint+seed) dry weight (kg/ha; harvestable yield biomass)"),
    "topwt":    ("Crop status",       "Total aboveground biomass (kg/ha)"),
    "xlai":     ("Crop status",       "Leaf area index (m²_leaf / m²_soil)"),
    "totwt":    ("Crop status",       "Total crop weight (g/m²)"),
    "shelwt":   ("Crop status",       "Total shell mass (g/m²)"),
    "sdwt":     ("Crop status",       "Dry seed mass (g/m²)"),
    "podwt":    ("Crop status",       "Dry mass of seeds + shells (g/m²)"),
    "sdlip":    ("Crop status",       "Lipid fraction in seed"),
    "pcnsd":    ("Crop status",       "N percentage in seed tissue (%)"),
    "pcnl":     ("Crop status",       "N percentage in leaf tissue (%)"),
    "laimx":    ("Crop status",       "Maximum LAI reached so far this season"),
    "cannaa":   ("Crop status",       "N weight in total plant at flowering (g_N/m²)"),
    "clw":      ("Crop status",       "Cumulative leaf growth (g_leaf/m²)"),
    "wtntot":   ("Crop status",       "Total plant N content (g_N/m²)"),
    "nfixn":    ("Crop status",       "N fixed today (g_N/m²/day)"),
    "turfac":   ("Stress",            "Water stress for cell expansion (0–1, 1 = max stress)"),
    "swfac":    ("Stress",            "Soil water stress factor (0–1, 1 = max stress)"),
    "nstres":   ("Stress",            "Nitrogen stress factor (0–1, 1 = max stress)"),
    "wtnup":    ("N uptake",          "Cumulative plant N uptake (kg/ha)"),
    "cumsumfert": ("Cumulative actions", "Total nitrogen fertilizer applied (kg/ha)"),
    "totir":    ("Cumulative actions", "Total irrigated water applied (mm)"),
}

# Per-step application limits for cotton
COTTON_FERT_MAX  = 25.0   # kg N/ha per step  (season max ~150 kg/ha)
COTTON_IRRIG_MAX = 80.0   # mm per step        (season max ~100 mm per event)


class CottonPromptGenerator(DSSATPromptGenerator):
    """
    Prompt generator tuned for DSSAT CROPGRO-Cotton.

    Key differences from maize:
    - Uses cotton-specific observation variables (pod/seed/leaf N metrics)
    - Lower per-step fert limit (25 vs 30 kg/ha); higher irrigation limit (80 vs 40 mm)
    - System prompt reflects cotton phenology and boll development
    - Fertilization penalty is higher (0.75) so reward is more sensitive to over-application
    """

    def __init__(
        self,
        season_length: int = 200,
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
            crop_name="cotton",
            season_length=season_length,
            location=location,
            output_vars=output_vars or COTTON_OUTPUT_VARS,
            desc_map=desc_map or COTTON_DESC_MAP,
            enable_pests=enable_pests,
            require_think=require_think,
            thinking_mode=thinking_mode,
            think_tag=think_tag,
            decision_interval=decision_interval,
            fert_max=fert_max if fert_max is not None else COTTON_FERT_MAX,
            irrig_max=irrig_max if irrig_max is not None else COTTON_IRRIG_MAX,
            include_crop_traits=include_crop_traits,
            include_profit_context=include_profit_context,
            profit_context_params=profit_context_params,
            objective_id=objective_id,
            objective_text=objective_text,
            reward_params=reward_params,
        )

    def get_system_prompt(self) -> str:
        return super().get_system_prompt()
