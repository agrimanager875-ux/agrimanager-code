"""
Prompt generation for gym-dssat environments.

This module provides functions to generate system prompts and turn prompts
for LLM-based agents in gym_dssat_pdi environments.
"""

from typing import Dict, List, Tuple, Optional, Union
import numpy as np

from agrimanager.env.base.objective_prompt import (
    build_management_objective_block,
    cropgrowth_system_prompt,
    objective_decision_text,
)

VALID_THINKING_MODES = {"minimal", "grounding_decision"}


def _normalize_thinking_mode(thinking_mode: str) -> str:
    """Normalize and validate the thinking mode name."""
    normalized = str(thinking_mode).strip().lower().replace("-", "_")
    if normalized == "think":
        normalized = "grounding_decision"
    if normalized not in VALID_THINKING_MODES:
        valid = ", ".join(sorted(VALID_THINKING_MODES | {"think"}))
        raise ValueError(f"Invalid thinking_mode '{thinking_mode}'. Expected one of: {valid}")
    return normalized


# Variable descriptions mapping (covers maize, cotton, and rice)
DSSAT_DEFAULT_DESC_MAP = {
    # ── Timeline ──────────────────────────────────────────────────────────────
    "dap":    ("Timeline", "Days after planting"),

    # ── Crop status (shared) ─────────────────────────────────────────────────
    "vstage": ("Crop status", "Vegetative stage index (number of leaves)"),
    "istage": ("Crop status", "Development stage index"),
    "grnwt":  ("Crop status", "Grain dry matter (kg/ha; harvestable yield biomass)"),
    "topwt":  ("Crop status", "Total aboveground biomass (kg/ha)"),
    "xlai":   ("Crop status", "Leaf area index (m²_leaf / m²_soil)"),
    "rtdep":  ("Crop status", "Root depth (cm)"),
    "pltpop": ("Crop status", "Plant population density (plants/m²)"),

    # ── Crop status (cotton-specific) ─────────────────────────────────────────
    "totwt":  ("Crop status", "Total crop weight (g/m²)"),
    "shelwt": ("Crop status", "Total shell mass (g/m²)"),
    "sdwt":   ("Crop status", "Dry seed mass (g/m²)"),
    "podwt":  ("Crop status", "Dry mass of seeds + shells (g/m²)"),
    "sdlip":  ("Crop status", "Lipid composition in seed (fraction)"),
    "clw":    ("Crop status", "Cumulative leaf growth (g_leaf/m²)"),
    "laimx":  ("Crop status", "Maximum leaf area index this season"),
    "cannaa": ("Crop status", "Weight of N in total plant at flowering (g_N/m²)"),
    "wtntot": ("Crop status", "Total plant N content (g_N/m²)"),
    "turfac": ("Crop status", "Water stress factor for cell expansion (0–1)"),
    "pcnsd":  ("Crop status", "Percentage of N in seed tissue (%)"),
    "pcnl":   ("Crop status", "Percentage of N in leaf tissue (%)"),
    "nfixn":  ("Crop status", "N fixed during the day (g_N/m²/d)"),

    # ── Crop status (rice-specific) ───────────────────────────────────────────
    "chtd":   ("Crop status", "Canopy height (m)"),
    "cwad":   ("Crop status", "Total crop dry weight (kg/ha)"),
    "lai":    ("Crop status", "Leaf area index (m²_leaf/m²_ground)"),
    "tilno":  ("Crop status", "Tiller number"),
    "gpp":    ("Crop status", "Grain number per plant"),
    "dyield": ("Crop status", "Dry yield biomass"),
    "grainn": ("Crop status", "Grain nitrogen content (g_N/plant)"),
    "xgnp":   ("Crop status", "Nitrogen content of grain (%)"),
    "xanc":   ("Crop status", "N concentration in aboveground biomass (%)"),
    "pcnveg": ("Crop status", "Stem + leaf N concentration (%)"),
    "andem":  ("Crop status", "Total crop N demand (kg_N/ha)"),

    # ── Stress ────────────────────────────────────────────────────────────────
    "nstres": ("Stress", "Nitrogen stress factor (0–1, 1 = max stress)"),
    "swfac":  ("Stress", "Soil water stress factor (0–1, 1 = max stress)"),

    # ── N uptake ──────────────────────────────────────────────────────────────
    "wtnup":  ("N uptake", "Cumulative plant N uptake (kg/ha)"),
    "trnu":   ("N uptake", "Daily N plant uptake (kg/ha)"),
    "unh4":   ("N uptake", "Plant uptake of ammonium (kg_N/ha/day)"),
    "uno3":   ("N uptake", "Plant uptake of nitrate (kg_N/ha/day)"),
    "nuf":    ("N uptake", "N supply-to-demand ratio used to modify N uptake"),

    # ── Weather ───────────────────────────────────────────────────────────────
    "dtt":    ("Weather", "Thermal time for current day (°C·day)"),
    "tmax":   ("Weather", "Maximum temperature (°C)"),
    "tmin":   ("Weather", "Minimum temperature (°C)"),
    "srad":   ("Weather", "Solar radiation (MJ/m²/day)"),
    "rain":   ("Weather", "Daily rainfall (mm)"),

    # ── Water ─────────────────────────────────────────────────────────────────
    "ep":     ("Water", "Actual plant transpiration (mm/day)"),
    "wtdep":  ("Water", "Depth to water table (cm)"),
    "totir":  ("Cumulative actions", "Total irrigated water (mm)"),

    # ── Cumulative actions ────────────────────────────────────────────────────
    "cumsumfert": ("Cumulative actions", "Total fertilizer applied (kg/ha)"),

    # ── Pest management ───────────────────────────────────────────────────────
    "pest_pressure":      ("Pest Status", "Current pest pressure (0–1, higher = more pests)"),
    "pest_damage":        ("Pest Status", "Cumulative yield loss from pests (kg/ha)"),
    "days_since_pesticide": ("Pest Status", "Days since last pesticide application"),
}

# Crop trait profiles injected into every turn prompt as agronomic prior knowledge.
# Mirrors the <crop traits> block in WOFOST prompts.
CROP_TRAITS = {
    "maize": """\
Crop Name: maize
Profile
  Season type: warm-season (~2400 °C·d accumulated from base 8 °C)
  Temperature adaptation: warm-adapted (optimal growth 25–35 °C; chilling sensitive)
  Development driver: temperature-driven GDD accumulation (no strong photoperiod effect)
  Root/water trait: deep-rooted (120–180 cm), moderate drought tolerance; highly sensitive at silking
  Assimilation trait: C4 photosynthesis — high radiation-use efficiency
Critical growth stages
  V6  (leaf 6, ~25–30 DAP): primary side-dress N window; root system expanding
  VT/R1 (tasseling/silking, ~60–70 DAP): highest water sensitivity — drought here cuts yield most
  R3  (milk, ~80 DAP): active grain fill begins; both N and water still limiting
  R6  (physiological maturity): grain fill complete; no further inputs needed
Typical nutrient requirements (season total)
  Nitrogen: 150–250 kg/ha; split applications reduce leaching loss
  Phosphorus: important at emergence and early vegetative growth
  Potassium: supports stalk strength and water-use efficiency""",

    "cotton": """\
Crop Name: cotton
Profile
  Season type: long-season (180–200 days; indeterminate growth habit)
  Temperature adaptation: warm-adapted (optimal 28–35 °C; frost intolerant; slow below 15 °C)
  Development driver: temperature + water status; boll retention highly sensitive to stress
  Root/water trait: moderately deep-rooted (90–120 cm); drought-tolerant vegetatively but sensitive during boll fill
  Assimilation trait: C3 photosynthesis; canopy closure important for boll-load interception
Critical growth stages
  Squaring (~40–50 DAP): first flower buds; N demand rises sharply
  First flower (~60–70 DAP): peak N uptake window; water stress drops boll retention
  Peak boll set (~80–100 DAP): highest yield sensitivity to both N and water stress
  Boll fill / open (~120–160 DAP): fibre and seed fill; reduce N to avoid rank growth; maintain soil moisture
Typical nutrient requirements (season total)
  Nitrogen: 100–150 kg/ha; excess N causes rank vegetative growth at expense of boll set
  Phosphorus: important at establishment and early square stage
  Potassium: critical for fibre quality and boll fill""",

    "rice": """\
Crop Name: rice
Profile
  Season type: medium-season paddy rice (130–150 days)
  Temperature adaptation: warm-adapted (optimal 25–32 °C; panicle initiation sensitive to cool nights)
  Development driver: temperature + photoperiod (short-day sensitive for tropical varieties)
  Root/water trait: shallow-rooted (30–60 cm); flooded/saturated conditions preferred; aerenchyma for waterlogging
  Assimilation trait: C3 photosynthesis; high tiller number compensates for individual stem productivity
Critical growth stages
  Tillering (~15–40 DAP): tiller production determines potential panicle number; N application here is most efficient
  Panicle initiation / PI (~50–60 DAP): critical N window; stress now reduces spikelet number irreversibly
  Heading/anthesis (~80–90 DAP): highest water sensitivity; cool temperatures reduce fertilisation
  Grain fill (~90–130 DAP): starch accumulation; maintain water and moderate N supply
Typical nutrient requirements (season total)
  Nitrogen: 100–180 kg/ha; split into basal + tillering + PI applications
  Phosphorus: important at transplanting/emergence for root establishment
  Potassium: improves lodging resistance and grain quality""",
}


# Per-crop max per-step application amounts used in prompts
CROP_ACTION_MAX = {
    "maize":  {"fert_max": 30.0, "irrig_max": 40.0},
    "cotton": {"fert_max": 25.0, "irrig_max": 80.0},
    "rice":   {"fert_max": 30.0, "irrig_max": 80.0},
}


class DSSATPromptGenerator:
    """
    Generate prompts for gym-dssat environments.

    Converts a DSSAT observation into natural language context
    + describes available fertilization / irrigation actions.
    Also parses LLM action responses into action indices.
    """

    def __init__(
        self,
        crop_name: str = "the crop",
        season_length: int = 150,
        location: str = "the field",
        num_fert: int = 4,
        num_irrig: int = 4,
        fert_amount: float = 10.0,
        irrig_amount: float = 10.0,
        intervention_interval: int = 7,
        output_vars: Optional[List[str]] = None,
        desc_map: Optional[Dict[str, Tuple[str, str]]] = None,
        enable_pests: bool = False,
        require_think: bool = False,
        thinking_mode: str = "grounding_decision",
        think_tag: str = "tool_call",
        decision_interval: int = 1,
        fert_max: Optional[float] = None,
        irrig_max: Optional[float] = None,
        include_crop_traits: bool = False,
        include_profit_context: bool = False,
        profit_context_params: Optional[Dict[str, float]] = None,
        objective_id: str = "profit_max",
        objective_text: Optional[str] = None,
        reward_params: Optional[Dict[str, float]] = None,
    ):
        self.crop_name = crop_name
        self.season_length = season_length
        self.location = location
        self.num_fert = num_fert
        self.num_irrig = num_irrig
        self.fert_amount = fert_amount
        self.irrig_amount = irrig_amount
        self.intervention_interval = intervention_interval
        self.output_vars = output_vars or []
        self.desc_map = desc_map or DSSAT_DEFAULT_DESC_MAP
        self.enable_pests = enable_pests
        self.require_think = require_think
        self.thinking_mode = _normalize_thinking_mode(thinking_mode)
        self.think_tag = think_tag
        self.decision_interval = decision_interval
        # Per-step action bounds shown in prompts; fall back to crop defaults
        _crop_defaults = CROP_ACTION_MAX.get(crop_name, CROP_ACTION_MAX["maize"])
        self.fert_max = fert_max if fert_max is not None else _crop_defaults["fert_max"]
        self.irrig_max = irrig_max if irrig_max is not None else _crop_defaults["irrig_max"]
        self.include_crop_traits = bool(include_crop_traits)
        self.include_profit_context = bool(include_profit_context)
        self.objective_id = str(objective_id or "profit_max")
        self.objective_text = objective_text
        self.objective_params = {
            **dict(reward_params or {}),
            **dict(profit_context_params or {}),
        }

    def _management_objective_block(self) -> str:
        return build_management_objective_block(
            self.objective_id,
            self.objective_params,
            available_inputs=("n", "irrig"),
            yield_label="final_yield_kg_ha",
            objective_text=self.objective_text,
        )

    # ==========================================================
    # SYSTEM PROMPT
    # ==========================================================
    def get_system_prompt(self) -> str:
        return cropgrowth_system_prompt("DSSAT-Gym")


    # ==========================================================
       # OBSERVATION → NATURAL LANGUAGE
    # ==========================================================
    def describe_observation(
        self,
        observation,  # Remove type hint - can be dict or array
        output_vars: Optional[List[str]] = None,
        seed: Optional[int] = None,
    ) -> str:
        if output_vars is None:
            output_vars = self.output_vars

        if len(output_vars) == 0:
            raise ValueError("output_vars must be provided either in __init__ or as argument")

        # Handle both dict and array observations
        if isinstance(observation, dict):
            obs_dict = observation
        else:
            # Assume it's array-like (numpy array, list, etc.)
            obs_dict = dict(zip(output_vars, observation))
        
        # DAP = key timeline reference
        dap = obs_dict.get("dap", None)

        lines = []
        if dap is not None:
            lines.append(
                f"We are cultivating {self.crop_name}. "
                f"Today is day {int(dap)} after planting out of a planned {self.season_length} days."
            )
        else:
            lines.append(f"We are cultivating {self.crop_name}.")

        lines.append("\nObservation summary:")

        current_section = None
        for key in output_vars:
            if key not in obs_dict:
                continue
                
            val = obs_dict[key]
            section, desc = self.desc_map.get(key, ("Other", key))
            if section != current_section:
                lines.append(f"[{section}]")
                current_section = section
            try:
                val_str = f"{val:.4g}"
            except Exception:
                val_str = str(val)
            lines.append(f"- {desc}: {val_str}")
            if key == 'cumsumfert':
                val_str = f"{val:.1f}"
                # Add warning if over-fertilizing
                if val > 250:
                    val_str += " ⚠️ (VERY HIGH - typical maize: 150-250 kg/ha)"
                elif val > 200:
                    val_str += " (approaching high end)"
                lines.append(f"- {desc}: {val_str}")

        return "\n".join(lines)
    # ==========================================================
    # ACTION DESCRIPTION
    # ==========================================================
    def get_action_options_description(self) -> str:
        max_fert = self.num_fert * self.fert_amount
        max_irrig = self.num_irrig * self.irrig_amount

        lines = [
            "\nAvailable actions (pick exactly one):",
            "- Do nothing.",
            f"- Apply nitrogen fertilizer (specify amount from 0 to {max_fert:.1f} kg/ha).",
            f"  Note: Typical season total is 150-250 kg/ha. Monitor cumulative fertilizer!",
            f"- Irrigate with water (specify amount from 0 to {max_irrig:.1f} mm).",
            "\nRespond using the format <answer>...<answer> "
            "(e.g., <answer>Apply 25.5 kg/ha nitrogen fertilizer.<answer>)"
        ]
        return "\n".join(lines)

    # ==========================================================
    # TURN PROMPT
    # ==========================================================
    def get_turn_prompt(
        self,
        observation: np.ndarray,
        output_vars: Optional[List[str]] = None,
        seed: Optional[int] = None,
        season_num: Optional[int] = None,
        num_seasons: Optional[int] = None,
    ) -> str:
        vars_ = output_vars or self.output_vars
        if not vars_:
            raise ValueError("output_vars must be provided in __init__ or as argument")

        # ── 1. Intro ────────────────────────────────────────────────
        obs_dict = observation if isinstance(observation, dict) else dict(zip(vars_, observation))
        dap = obs_dict.get("dap")
        day_str = f"Today is day {int(dap)} after planting out of a planned {self.season_length} days." if dap is not None else ""

        # Multi-season context
        if season_num is not None and num_seasons is not None and num_seasons > 1:
            season_str = f"Season {season_num} of {num_seasons}. "
        else:
            season_str = ""

        intro = f"{season_str}We are cultivating {self.crop_name}. {day_str}"
        if self.decision_interval > 1:
            intro += f" You make one decision every {self.decision_interval} days."

        # ── 1b. Crop traits block ───────────────────────────────────
        traits_text = CROP_TRAITS.get(self.crop_name) if self.include_crop_traits else None
        traits_block = f"<crop traits>\n{traits_text}\n</crop traits>" if traits_text else ""

        # ── 2. Observation block ────────────────────────────────────
        obs_lines = []
        current_section = None
        for key in vars_:
            if key not in obs_dict:
                continue
            val = obs_dict[key]
            section, desc = self.desc_map.get(key, ("Other", key))
            if section != current_section:
                obs_lines.append(f"[{section}]")
                current_section = section
            try:
                obs_lines.append(f"- {desc}: {val:.4g}")
            except Exception:
                obs_lines.append(f"- {desc}: {val}")
            if key == "cumsumfert":
                if val > 250:
                    obs_lines[-1] += " ⚠️ (VERY HIGH - typical maize: 150-250 kg/ha)"
                elif val > 200:
                    obs_lines[-1] += " (approaching high end)"

        # ── 3. Action options ───────────────────────────────────────
        action_lines = [
            "Available actions (pick exactly one):",
            "- Do nothing.",
            f"- Apply X kg/ha nitrogen fertilizer.  (X must be less than {self.fert_max:.1f})",
            f"- Irrigate with Y mm of water.  (Y must be less than {self.irrig_max:.1f})",
            f"- Irrigate with Y mm of water and apply X kg/ha nitrogen fertilizer.  (Y < {self.irrig_max:.1f}, X < {self.fert_max:.1f})",
        ]
        if self.enable_pests:
            action_lines.append("- Apply pesticide.")

        # ── 4. Decision guidance ────────────────────────────────────
        target_text = objective_decision_text(self.objective_id)
        if not self.require_think:
            action_lines += [
                "",
                "Please consider the following when making a decision:",
                "1. State grounding: describe the current agronomic state and the main limiting factor "
                "based on the current observations",
                f"2. Decision: {target_text} at this step",
            ]
        elif self.thinking_mode == "grounding_decision":
            action_lines += [
                "",
                "Please reason briefly before answering about:",
                "1. State grounding: describe the current agronomic state and the main limiting factor "
                "based on the current observations",
                f"2. Decision: {target_text} at this step",
            ]
        else:  # minimal
            action_lines += [
                "",
                "Please think about your choice before answering.",
            ]

        # ── 5. Response format ──────────────────────────────────────
        t = self.think_tag
        if self.require_think:
            if self.thinking_mode == "grounding_decision":
                action_lines += [
                    "",
                    "Keep the reasoning concise and decision-focused. Do not restate the full input.",
                    "",
                    f"Respond using the exact format: <{t}> ... </{t}> <answer> ... </answer> with no extra text.",
                    "",
                    f"Example: <{t}>[reasoning]</{t}> <answer>Apply {self.fert_max/2:.1f} kg/ha nitrogen fertilizer.</answer>",
                ]
            else:
                action_lines += [
                    "",
                    f"Respond using the exact format: <{t}> ... </{t}> <answer> ... </answer> with no extra text.",
                    "",
                    f"Example: <{t}>[reasoning]</{t}> <answer>Apply {self.fert_max/2:.1f} kg/ha nitrogen fertilizer.</answer>",
                ]
        else:
            action_lines += [
                "",
                "Respond using the exact format: <answer> ... </answer> with no extra text.",
                "",
                f"Example: <answer>Apply {self.fert_max/2:.1f} kg/ha nitrogen fertilizer.</answer>",
            ]

        # ── Assemble ────────────────────────────────────────────────
        obs_block = f"<current observation>\n{chr(10).join(obs_lines)}\n</current observation>"
        parts = [intro, self._management_objective_block()]
        if traits_block:
            parts.append(traits_block)
        parts.append(obs_block)
        parts.append("\n".join(action_lines))
        return "\n\n".join(parts)

    # ==========================================================
    # ACTION → NATURAL LANGUAGE
    # ==========================================================
    def describe_action(self, action_id: int) -> str:
        """
        Convert integer action ID to NL description.
        You must ensure your action_id definition matches your DSSAT wrapper.
        """
        if action_id == 0:
            return "<answer>Take no action.<answer>"

        # fertilizer
        if 1 <= action_id <= self.num_fert:
            amount = (action_id) * self.fert_amount
            return f"<answer>Apply {amount:.1f} kg/ha nitrogen fertilizer.<answer>"

        # irrig
        irr_id = action_id - self.num_fert
        if 1 <= irr_id <= self.num_irrig:
            amt = irr_id * self.irrig_amount
            return f"<answer>Irrigate with {amt:.1f} mm of water.<answer>"

        return f"<answer>Unknown action {action_id}.<answer>"

    # ==========================================================
    # PARSE FROM LLM
    # ==========================================================
    @staticmethod
    def _extract_amount(patterns: List[str], text: str) -> Optional[float]:
        import re

        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
            if match:
                return float(match.group("amount"))
        return None

    @staticmethod
    def _nearest_action_amount(amount: float, step: float, count: int) -> float:
        if amount <= 0.0:
            return 0.0
        level = round(amount / step)
        level = max(1, min(level, count))
        return float(level * step)

    def _extract_nitrogen_amount(self, text: str) -> Optional[float]:
        number = r"(?P<amount>\d+(?:\.\d+)?)"
        kg_ha = r"(?:kg\s*/\s*ha|kg\s*ha\s*-?1|kg/ha|kg)"
        return self._extract_amount(
            [
                rf"{number}\s*{kg_ha}\s*(?:of\s+)?(?:nitrogen|n\b|fertilizer)",
                rf"(?:nitrogen|n\b|fertilizer)[^0-9]{{0,50}}{number}\s*{kg_ha}",
            ],
            text,
        )

    def _extract_irrigation_amount(self, text: str) -> Optional[float]:
        number = r"(?P<amount>\d+(?:\.\d+)?)"
        water_unit = r"(?:mm|millimeter|millimeters|millimetre|millimetres)"
        return self._extract_amount(
            [
                rf"(?:irrigate|irrigation|water)[^0-9]{{0,50}}{number}\s*{water_unit}",
                rf"{number}\s*{water_unit}\s*(?:of\s+)?(?:water|irrigation)",
            ],
            text,
        )

    def parse_action_response(self, response: str) -> Optional[Union[int, Dict[str, float]]]:
        """Extract action ID from the LLM response.

        - require_think=True:  ``<tag>...</tag> <answer>...</answer>``
        - require_think=False: ``<answer>...</answer>`` only
        Uses fullmatch so any extra content outside the tags is invalid.

        Single-input actions return the legacy discrete action id. Combined
        fertilizer + irrigation actions return a clean DSSAT action dict.
        """
        import re

        if self.require_think:
            t = re.escape(self.think_tag)
            m = re.fullmatch(
                rf"\s*<{t}>(.*?)</{t}>\s*<answer>(.*?)</answer>\s*",
                response,
                re.DOTALL,
            )
        else:
            m = re.fullmatch(
                r"\s*<answer>(.*?)</answer>\s*",
                response,
                re.DOTALL,
            )

        if m is None:
            return None
        txt = m.group(m.lastindex).strip().lower()

        if "do nothing" in txt or "take no action" in txt:
            return 0

        if self.enable_pests and "pesticide" in txt:
            return self.num_fert + self.num_irrig + 1

        nitrogen_amount = self._extract_nitrogen_amount(txt)
        irrigation_amount = self._extract_irrigation_amount(txt)

        if nitrogen_amount is not None and irrigation_amount is not None:
            anfer = self._nearest_action_amount(nitrogen_amount, self.fert_amount, self.num_fert)
            amir = self._nearest_action_amount(irrigation_amount, self.irrig_amount, self.num_irrig)
            if anfer <= 0.0 and amir <= 0.0:
                return 0
            if anfer <= 0.0:
                level = round(irrigation_amount / self.irrig_amount)
                return self.num_fert + max(1, min(level, self.num_irrig))
            if amir <= 0.0:
                level = round(nitrogen_amount / self.fert_amount)
                return max(1, min(level, self.num_fert))
            return {"anfer": anfer, "amir": amir}

        if nitrogen_amount is not None:
            if nitrogen_amount == 0:
                return 0
            level = round(nitrogen_amount / self.fert_amount)
            return max(1, min(level, self.num_fert))

        if irrigation_amount is not None:
            if irrigation_amount == 0:
                return 0
            level = round(irrigation_amount / self.irrig_amount)
            return self.num_fert + max(1, min(level, self.num_irrig))

        return None

    # ==========================================================
    # FACTORY: BUILD FROM ENV
    # ==========================================================
    @classmethod
    def from_env(
        cls,
        env,
        require_think: bool = False,
        thinking_mode: str = "grounding_decision",
        think_tag: str = "tool_call",
        include_crop_traits: bool = False,
        include_profit_context: bool = False,
        profit_context_params: Optional[Dict[str, float]] = None,
        objective_id: str = "profit_max",
        objective_text: Optional[str] = None,
        reward_params: Optional[Dict[str, float]] = None,
    ) -> "DSSATPromptGenerator":
        """
        Initialize DSSATPromptGenerator from a gym_dssat_pdi environment.

        Note: gym-dssat doesn't expose metadata as richly as WOFOST,
        so we keep defaults if missing.
        """
        try:
            unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env

            crop_name = getattr(unwrapped, "crop_name", "the crop")
            season_length = getattr(unwrapped, "season_length", 150)
            location = getattr(unwrapped, "location", "the field")

            num_fert = getattr(unwrapped, "num_fert", 4)
            num_irrig = getattr(unwrapped, "num_irrig", 4)
            fert_amount = getattr(unwrapped, "fert_amount", 10.0)
            irrig_amount = getattr(unwrapped, "irrig_amount", 10.0)
            interval = getattr(unwrapped, "intervention_interval", 7)

            # output vars
            output_vars = []
            if hasattr(unwrapped, "observation_variables"):
                output_vars = unwrapped.observation_variables

            return cls(
                crop_name=crop_name,
                season_length=season_length,
                location=location,
                num_fert=num_fert,
                num_irrig=num_irrig,
                fert_amount=fert_amount,
                irrig_amount=irrig_amount,
                intervention_interval=interval,
                output_vars=output_vars,
                require_think=require_think,
                thinking_mode=thinking_mode,
                think_tag=think_tag,
                include_crop_traits=include_crop_traits,
                include_profit_context=include_profit_context,
                profit_context_params=profit_context_params,
                objective_id=objective_id,
                objective_text=objective_text,
                reward_params=reward_params,
            )
        except Exception as e:
            print(f"Warning: Could not extract full config from env: {e}")
            return cls(
                require_think=require_think,
                thinking_mode=thinking_mode,
                think_tag=think_tag,
                include_crop_traits=include_crop_traits,
                include_profit_context=include_profit_context,
                profit_context_params=profit_context_params,
                objective_id=objective_id,
                objective_text=objective_text,
                reward_params=reward_params,
            )
