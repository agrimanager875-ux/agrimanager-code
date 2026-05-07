"""Prompt generation for wofost_gym environments.

This module provides functions to generate system prompts and turn prompts
for LLM-based agents in wofost_gym environments.
"""

from datetime import date, timedelta, datetime
from typing import Dict, List, Optional, Tuple
import numpy as np

from agrimanager.env.base.objective_prompt import (
    build_management_objective_block,
    cropgrowth_system_prompt,
    objective_decision_text,
)


# Variable descriptions mapping
DEFAULT_DESC_MAP = {
    "FIN": ("Crop status", "Finish flag (1 means the season has ended)"),
    "DVS": ("Crop status", "Development stage index"),
    "WSO": (
        "Crop status",
        "Storage organ dry matter (kg/ha; harvestable yield biomass proxy)",
    ),
    "NAVAIL": ("Soil nutrients", "Available soil nitrogen (kg/ha)"),
    "PAVAIL": ("Soil nutrients", "Available soil phosphorus (kg/ha)"),
    "KAVAIL": ("Soil nutrients", "Available soil potassium (kg/ha)"),
    "SM": ("Soil & water", "Root-zone soil moisture (fraction)"),
    "TOTN": ("Cumulative actions", "Cumulative nitrogen applied so far (kg/ha)"),
    "TOTP": ("Cumulative actions", "Cumulative phosphorus applied so far (kg/ha)"),
    "TOTK": ("Cumulative actions", "Cumulative potassium applied so far (kg/ha)"),
    "TOTIRRIG": ("Cumulative actions", "Cumulative irrigation depth applied so far (cm)"),
    "IRRAD": ("Weather", "Daily solar radiation (J/m²/day)"),
    "TEMP": ("Weather", "Mean air temperature (°C)"),
    "RAIN": ("Weather", "Daily rainfall (cm)"),
    "DAYS": ("Timeline", "Days since sowing"),
    "LAI": ("Canopy growth", "Leaf area index"),
    "TAGP": ("Crop biomass", "Total above-ground biomass (kg/ha)"),
    "RD": ("Root growth", "Current rooting depth (cm)"),
    "WLV": ("Crop biomass", "Living leaf dry matter (kg/ha)"),
    "WST": ("Crop biomass", "Living stem dry matter (kg/ha)"),
    "WRT": ("Crop biomass", "Living root dry matter (kg/ha)"),
    "TRA": ("Soil & water", "Actual transpiration rate (cm/day)"),
    "RFTRA": ("Soil & water", "Transpiration reduction factor due to water stress"),
    "IDWST": ("Soil & water", "Cumulative days with water stress"),
    "NUPTAKETOTAL": ("Nutrient uptake", "Cumulative crop nitrogen uptake (kg/ha)"),
    "PUPTAKETOTAL": ("Nutrient uptake", "Cumulative crop phosphorus uptake (kg/ha)"),
    "KUPTAKETOTAL": ("Nutrient uptake", "Cumulative crop potassium uptake (kg/ha)"),
}

VALID_THINKING_MODES = {"minimal", "grounding_decision"}
THINKING_MODE_ALIASES = {
    "think": "grounding_decision",
}
FERTILIZER_COMPONENTS = ("n", "p", "k")
ACTION_COMPONENT_LABELS = {
    "n": "nitrogen",
    "p": "phosphorus",
    "k": "potassium",
}
ACTION_COMPONENT_KEYWORDS = {
    "n": ("nitrogen", " n "),
    "p": ("phosphorus", " p "),
    "k": ("potassium", " k "),
}

ACTION_KIND_LABELS = {
    "n": "nitrogen fertilizer",
    "p": "phosphorus fertilizer",
    "k": "potassium fertilizer",
    "irrig": "irrigation water",
}

ACTION_KIND_ALIASES = {
    "n": ("nitrogen", " n "),
    "p": ("phosphorus", " p "),
    "k": ("potassium", " k "),
    "irrig": ("irrigate", "irrigation", "water"),
}

ACTION_MENUS_BY_ENV_ID = {
    "lnpkw": ("n", "p", "k", "irrig"),
    "lnpk": ("n", "p", "k"),
    "lnw": ("n", "irrig"),
    "ln": ("n",),
    "lw": ("irrig",),
    "pp": (),
}


def action_menu_from_env_id(env_id: Optional[str]) -> Tuple[str, ...]:
    """Return the ordered native action kinds exposed by a WOFOST-Gym env id."""
    if not env_id:
        return ACTION_MENUS_BY_ENV_ID["lnpkw"]
    normalized = str(env_id).strip().lower()
    if normalized.endswith("-v0"):
        normalized = normalized[:-3]
    normalized = normalized.removeprefix("perennial-")
    normalized = normalized[1:] if normalized.startswith("ll") else normalized
    if normalized in ACTION_MENUS_BY_ENV_ID:
        return ACTION_MENUS_BY_ENV_ID[normalized]
    return ACTION_MENUS_BY_ENV_ID["lnpkw"]


def _format_choice_list(values: List[str]) -> str:
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} or {values[1]}"
    return f"{', '.join(values[:-1])}, or {values[-1]}"


def _example_action_text(
    available_action_kinds: Tuple[str, ...],
    *,
    fert_amount: float,
    irrig_amount: float,
) -> str:
    """Return a parser-valid example drawn from the currently available menu."""
    if "n" in available_action_kinds:
        return f"Apply {fert_amount:.1f} kg/ha nitrogen fertilizer."
    if "p" in available_action_kinds:
        return f"Apply {fert_amount:.1f} kg/ha phosphorus fertilizer."
    if "k" in available_action_kinds:
        return f"Apply {fert_amount:.1f} kg/ha potassium fertilizer."
    if "irrig" in available_action_kinds:
        return f"Irrigate with {irrig_amount:.1f} cm of water."
    return "Take no action."


def _normalize_thinking_mode(thinking_mode: str) -> str:
    """Normalize and validate the thinking mode name."""
    normalized = str(thinking_mode).strip().lower().replace("-", "_")
    normalized = THINKING_MODE_ALIASES.get(normalized, normalized)
    if normalized not in VALID_THINKING_MODES:
        valid = ", ".join(sorted(VALID_THINKING_MODES | set(THINKING_MODE_ALIASES)))
        raise ValueError(f"Invalid thinking_mode '{thinking_mode}'. Expected one of: {valid}")
    return normalized


class WOFOSTPromptGenerator:
    """Generate prompts for wofost_gym environments.

    This class handles the creation of system prompts and turn prompts
    for LLM-based agricultural management agents.
    """

    def __init__(
        self,
        crop_name: str = "the crop",
        season_length: int = 241,
        location: str = "the field",
        num_fert: int = 4,
        num_irrig: int = 4,
        fert_amount: float = 2.0,
        irrig_amount: float = 0.5,
        intervention_interval: int = 7,
        output_vars: Optional[List[str]] = None,
        desc_map: Optional[Dict[str, Tuple[str, str]]] = None,
        field_aliases: Optional[Dict[str, str]] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        start_date: Optional[date] = None,
        require_think: bool = False,
        thinking_mode: str = "grounding_decision",
        think_tag: str = "tool_call",
        crop_traits_text: Optional[str] = None,
        available_action_kinds: Optional[Tuple[str, ...]] = None,
        action_components: Optional[List[str]] = None,
        objective_id: str = "profit_max",
        objective_text: Optional[str] = None,
        include_profit_context: bool = False,
        profit_context_params: Optional[Dict[str, float]] = None,
    ):
        """Initialize the prompt generator.

        Args:
            crop_name: Name of the crop being cultivated
            season_length: Total length of growing season in days
            location: Geographic location description
            num_fert: Number of fertilizer levels per nutrient
            num_irrig: Number of irrigation levels
            fert_amount: Base fertilizer amount (kg/ha per level)
            irrig_amount: Base irrigation amount (cm per level)
            intervention_interval: Days between decisions
            output_vars: List of observation variable names
            desc_map: Custom description mapping for variables
            field_aliases: Optional display-name overrides for variables
            latitude: Optional latitude value for the field
            longitude: Optional longitude value for the field
            start_date: Calendar start date of the season (crop_start_date)
            require_think: Whether to require thinking before answering
            thinking_mode: Thinking prompt variant when require_think=True.
                Supported values: "minimal", "think" (alias of
                "grounding_decision"), "grounding_decision"
            think_tag: Tag name for thinking (default: "tool_call")
            crop_traits_text: Optional crop traits text for prompt injection
            available_action_kinds: Ordered WOFOST action kinds exposed by the
                active management menu. Supported values are n, p, k, irrig.
            action_components: Backward-compatible alias for
                ``available_action_kinds``.
            objective_id: Prompt-facing management objective identifier.
            objective_text: Optional free-form objective text override.
            include_profit_context: Backward-compatible flag accepted by older
                configs. Profit details are now rendered inside the management
                objective block.
            profit_context_params: Backward-compatible cost parameters merged
                into the objective block.
        """
        self.crop_name = crop_name
        self.season_length = season_length
        self.location = location
        self.latitude = latitude
        self.longitude = longitude
        self.num_fert = num_fert
        self.num_irrig = num_irrig
        self.fert_amount = fert_amount
        self.irrig_amount = irrig_amount
        self.intervention_interval = intervention_interval
        self.output_vars = output_vars or []
        self.desc_map = desc_map or DEFAULT_DESC_MAP
        self.field_aliases = dict(field_aliases or {})
        self.start_date = start_date
        self.require_think = require_think
        self.thinking_mode = _normalize_thinking_mode(thinking_mode)
        self.think_tag = think_tag
        self.crop_traits_text = crop_traits_text
        self.available_action_kinds = tuple(
            available_action_kinds or action_components or ACTION_MENUS_BY_ENV_ID["lnpkw"]
        )
        self.action_components = list(self.available_action_kinds)
        self.objective_id = str(objective_id or "yield_max")
        self.objective_text = objective_text
        self.include_profit_context = bool(include_profit_context)
        self.profit_context_params = dict(profit_context_params or {})

    def _field_label(self, key: str, desc: str) -> str:
        return str(self.field_aliases.get(key, desc))

    def _uses_observation_glossary(self) -> bool:
        for key in self.output_vars:
            if key.upper().startswith("DAYS"):
                continue
            _, desc = self.desc_map.get(key, ("Other", key))
            if self._field_label(key, desc) != desc:
                return True
        return False

    @classmethod
    def action_menu_from_env_id(cls, env_id: Optional[str]) -> Tuple[str, ...]:
        return action_menu_from_env_id(env_id)

    def _action_id(self, kind: str, level: int) -> Optional[int]:
        if kind not in self.available_action_kinds:
            return None
        if kind == "irrig":
            if not 1 <= level <= self.num_irrig:
                return None
        elif not 1 <= level <= self.num_fert:
            return None
        offset = 1
        for action_kind in self.available_action_kinds:
            if action_kind == kind:
                return offset + level - 1
            offset += self.num_irrig if action_kind == "irrig" else self.num_fert
        return None

    def _action_kind_and_level(self, action_id: int) -> Optional[Tuple[str, int]]:
        if action_id == 0:
            return "none", 0
        offset = 1
        for kind in self.available_action_kinds:
            width = self.num_irrig if kind == "irrig" else self.num_fert
            if offset <= action_id < offset + width:
                return kind, action_id - offset + 1
            offset += width
        return None

    def set_crop_traits(self, text: str) -> None:
        """Set crop traits text used for prompt injection."""
        self.crop_traits_text = text

    def set_objective(self, objective_id: str, objective_text: Optional[str] = None) -> None:
        """Set the prompt-facing management objective."""
        self.objective_id = str(objective_id or "profit_max")
        self.objective_text = objective_text

    def _management_objective_block(self) -> str:
        return build_management_objective_block(
            self.objective_id,
            self.profit_context_params,
            available_inputs=self.action_components,
            yield_label="final_WSO_kg_ha",
            objective_text=self.objective_text,
            irrigation_unit="cm",
        )

    def _objective_decision_text(self) -> str:
        return objective_decision_text(self.objective_id)

    def _season_window_text(self) -> str:
        if str(self.crop_name).strip().lower() == "maize":
            return "around 180 days"
        return f"{self.season_length} days"

    def get_system_prompt(self) -> str:
        """Generate the system prompt for the LLM agent.

        Returns:
            System prompt string
        """
        return cropgrowth_system_prompt("WOFOST-Gym")

    def get_turn_prompt(self, observation: np.ndarray) -> str:
        """Generate the complete per-turn user prompt.

        Supports plain-answer mode plus two thinking guidance variants.
        Structure: intro → objective → crop traits (opt) → observation → actions → guidance → format.
        """
        output_vars = self.output_vars
        if not output_vars:
            raise ValueError("output_vars must be provided in __init__")
        has_traits = bool(self.crop_traits_text and self.crop_traits_text.strip())
        uses_glossary = self._uses_observation_glossary()

        # ── 1. Intro ────────────────────────────────────────────────
        day_val = None
        for i, var in enumerate(output_vars):
            if var.upper() in ("DAYS", "DAYS ELAPSED"):
                day_val = observation[i]
                break
        day_num = day_val if day_val is not None else 0

        calendar_phrase = ""
        if self.start_date is not None and day_val is not None:
            try:
                calendar_date = self.start_date + timedelta(days=max(int(round(float(day_val))) - 1, 0))
                calendar_phrase = f", corresponding to {calendar_date.strftime('%B')} {calendar_date.day}"
            except Exception:
                pass

        intro = (
            f"We are growing {self.crop_name} from sow to maturity. "
            f"The planned growing window spans {self._season_window_text()}, "
            f"and actions are taken every {self.intervention_interval} days."
        )
        if day_val is not None:
            intro += f" Today is day {day_num:.0f} of the season{calendar_phrase}."

        objective_block = self._management_objective_block()

        bridge = (
            "Below are the crop traits and the current observation for this step."
            if has_traits
            else "Below is the current observation for this step."
        )

        # ── 2. Observation block ────────────────────────────────────
        glossary_lines = []
        glossary_section = None
        obs_lines = []
        current_section = None
        for key, value in zip(output_vars, observation):
            if key.upper().startswith("DAYS"):
                continue
            section, desc = self.desc_map.get(key, ("Other", key))
            label = self._field_label(key, desc)
            if uses_glossary:
                if section != glossary_section:
                    glossary_lines.append(f"[{section}]")
                    glossary_section = section
                glossary_lines.append(f"- {label}: {desc}")
            if section != current_section:
                obs_lines.append(f"[{section}]")
                current_section = section
            if key == "DVS":
                if label == desc:
                    obs_lines.append(
                        f"- {label}: {value:.3f} (DVS=1 indicates flowering; DVS=2 indicates maturity)"
                    )
                else:
                    obs_lines.append(
                        f"- {label}: {value:.3f} (value 1 indicates flowering; value 2 indicates maturity)"
                    )
            else:
                obs_lines.append(f"- {label}: {value:.4g}")

        # ── 3. Action options ───────────────────────────────────────
        fert_amounts = [f"{(i + 1) * self.fert_amount:.1f}" for i in range(self.num_fert)]
        irrig_amounts = [f"{(i + 1) * self.irrig_amount:.1f}" for i in range(self.num_irrig)]
        example_action = _example_action_text(
            self.available_action_kinds,
            fert_amount=self.fert_amount,
            irrig_amount=self.irrig_amount,
        )

        action_lines = [
            "Available actions (pick exactly one):",
            "- Do nothing.",
        ]
        for action_kind in self.available_action_kinds:
            if action_kind == "n":
                action_lines.append(
                    f"- Apply nitrogen fertilizer ({_format_choice_list(fert_amounts)} kg/ha)."
                )
            elif action_kind == "p":
                action_lines.append(
                    f"- Apply phosphorus fertilizer ({_format_choice_list(fert_amounts)} kg/ha)."
                )
            elif action_kind == "k":
                action_lines.append(
                    f"- Apply potassium fertilizer ({_format_choice_list(fert_amounts)} kg/ha)."
                )
            elif action_kind == "irrig":
                action_lines.append(
                    f"- Irrigate with {_format_choice_list(irrig_amounts)} cm of water."
                )
        unavailable = [
            label
            for kind, label in ACTION_KIND_LABELS.items()
            if kind not in self.available_action_kinds
        ]
        if unavailable:
            action_lines.append(
                f"Unavailable actions: {_format_choice_list(unavailable)}."
            )

        # ── 4. Decision guidance (varies by require_think × thinking_mode)
        if not self.require_think:
            guidance_intro = "Please consider the following when making a decision:"
            state_verb = "to understand" if has_traits else None
            if has_traits:
                state_grounding = (
                    f"1. State grounding: integrate the crop traits with the current observations "
                    f"{state_verb} the current agronomic state and the main limiting factor"
                )
            else:
                state_grounding = (
                    "1. State grounding: describe the current agronomic state and the main limiting factor "
                    "based on the current observations"
                )

            decision_text = self._objective_decision_text()
            action_lines.extend([
                "",
                guidance_intro,
                state_grounding,
                f"2. Decision: {decision_text} at this step",
            ])
        elif self.thinking_mode == "grounding_decision":
            if has_traits:
                state_grounding = (
                    "1. State grounding: integrate the crop traits with the current observations "
                    "to describe the current agronomic state and the main limiting factor"
                )
            else:
                state_grounding = (
                    "1. State grounding: describe the current agronomic state and the main limiting factor "
                    "based on the current observations"
                )

            decision_text = self._objective_decision_text()
            action_lines.extend([
                "",
                "Please reason briefly before action about:",
                state_grounding,
                f"2. Decision: {decision_text} at this step",
            ])
        else:
            action_lines.extend([
                "",
                "Please think about your choice before answering.",
            ])

        # ── 5. Response format (varies by require_think) ───────────
        if self.require_think:
            t = self.think_tag
            if self.thinking_mode == "grounding_decision":
                action_lines.extend([
                    "",
                    "Keep the reasoning concise and decision-focused. Do not restate the full input.",
                    "",
                    f"Respond using the exact format: <{t}> ... </{t}> <answer> ... </answer> with no extra text.",
                    "",
                    f"Example: <{t}>[reasoning content]</{t}> <answer>{example_action}</answer>",
                ])
            else:
                action_lines.extend([
                    "",
                    f"Respond using the exact format: <{t}> ... </{t}> <answer> ... </answer> with no extra text.",
                    "",
                    f"Example: <{t}>[reasoning content]</{t}> <answer>{example_action}</answer>",
                ])
        else:
            action_lines.extend([
                "",
                "Respond using the exact format: <answer> ... </answer> with no extra text.",
                "",
                f"Example: <answer>{example_action}</answer>",
            ])

        # ── Assemble (sections joined by blank lines) ───────────────
        sections = [intro, objective_block]
        sections.append(bridge)
        if has_traits:
            sections.append(f"<crop traits>\n{self.crop_traits_text.strip()}\n</crop traits>")
        if uses_glossary:
            sections.append(f"<observation glossary>\n{chr(10).join(glossary_lines)}\n</observation glossary>")
        sections.append(f"<current observation>\n{chr(10).join(obs_lines)}\n</current observation>")
        sections.append("\n".join(action_lines))
        return "\n\n".join(sections)

    def describe_action(self, action_id: int) -> str:
        """Convert action ID to natural language description.

        Args:
            action_id: Integer action ID from the environment

        Returns:
            Natural language description in <answer>...</answer> format
        """
        parsed_action = self._action_kind_and_level(action_id)
        if parsed_action is None:
            return f"<answer>Unknown action {action_id}.</answer>"
        kind, level = parsed_action
        if kind == "none":
            return "<answer>Take no action.</answer>"
        if kind == "irrig":
            irrig_depth = level * self.irrig_amount
            return f"<answer>Irrigate with {irrig_depth:.1f} cm of water.</answer>"

        amount = level * self.fert_amount
        if kind == "n":
            return f"<answer>Apply {amount:.1f} kg/ha nitrogen fertilizer.</answer>"
        if kind == "p":
            return f"<answer>Apply {amount:.1f} kg/ha phosphorus fertilizer.</answer>"
        if kind == "k":
            return f"<answer>Apply {amount:.1f} kg/ha potassium fertilizer.</answer>"

        return f"<answer>Unknown action {action_id}.</answer>"

    def parse_action_response(self, response: str) -> Optional[int]:
        """Parse LLM response to extract action ID.

        Both modes use strict fullmatch — any extra content is invalid.
        Model-inherent thinking is extracted by the model interface layer
        before the response reaches this method.

        - ``require_think=True``:  ``<tag>...</tag><answer>...</answer>``
        - ``require_think=False``: ``<answer>...</answer>`` only
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
        action_text = m.group(m.lastindex).strip().lower()

        # Parse action type
        if "take no action" in action_text or "do nothing" in action_text:
            return 0

        # Try to extract amount
        amount_match = re.search(r"(\d+\.?\d*)", action_text)
        if not amount_match:
            return None

        amount = float(amount_match.group(1))

        # Determine action type and level
        matched_kind = None
        padded_action_text = f" {action_text} "
        for kind, aliases in ACTION_KIND_ALIASES.items():
            if any(alias in padded_action_text for alias in aliases):
                matched_kind = kind
                break
        if matched_kind is None:
            return None
        if matched_kind == "irrig":
            level = round(amount / self.irrig_amount)
        else:
            level = round(amount / self.fert_amount)
        return self._action_id(matched_kind, level)

    @staticmethod
    def _infer_action_components(unwrapped) -> List[str]:
        """Infer the ordered non-null action blocks from a WOFOST env."""
        mro_names = {cls.__name__ for cls in type(unwrapped).__mro__}
        if "LNPKW" in mro_names:
            return ["n", "p", "k", "irrig"]
        if "LNPK" in mro_names:
            return ["n", "p", "k"]
        if "LNW" in mro_names:
            return ["n", "irrig"]
        if "LN" in mro_names:
            return ["n"]
        if "LW" in mro_names:
            return ["irrig"]

        num_fert = getattr(unwrapped, "num_fert", 4)
        num_irrig = getattr(unwrapped, "num_irrig", 4)
        action_space = getattr(unwrapped, "action_space", None)
        n_actions = getattr(action_space, "n", None)
        if n_actions is None:
            return ["n", "p", "k", "irrig"]

        if n_actions == 1 + 3 * num_fert + num_irrig:
            return ["n", "p", "k", "irrig"]
        if n_actions == 1 + 3 * num_fert:
            return ["n", "p", "k"]
        if n_actions == 1 + num_fert + num_irrig:
            return ["n", "irrig"]
        if n_actions == 1 + num_fert:
            return ["n"]
        if n_actions == 1 + num_irrig:
            return ["irrig"]
        return ["n", "p", "k", "irrig"]

    @classmethod
    def from_env(
        cls,
        env,
        require_think: bool = False,
        thinking_mode: str = "grounding_decision",
        think_tag: str = "tool_call",
        action_schema_env_id: Optional[str] = None,
        objective_id: str = "profit_max",
        objective_text: Optional[str] = None,
        include_profit_context: bool = False,
        profit_context_params: Optional[Dict[str, float]] = None,
        output_vars: Optional[List[str]] = None,
        field_aliases: Optional[Dict[str, str]] = None,
    ) -> "WOFOSTPromptGenerator":
        """Create prompt generator from a wofost_gym environment.

        Args:
            env: wofost_gym environment instance

        Returns:
            WOFOSTPromptGenerator instance configured for the environment
        """
        # Try to get environment configuration
        try:
            unwrapped = env.unwrapped if hasattr(env, 'unwrapped') else env

            crop_name = "the crop"
            season_length = 241
            location = "the field"
            latitude = None
            longitude = None
            start_date = None

            if hasattr(unwrapped, 'agromanagement'):
                agro = unwrapped.agromanagement
                crop_name = agro.get('CropCalendar', {}).get('crop_name', crop_name)

                # Calculate season length
                try:
                    start = agro.get('CropCalendar', {}).get('crop_start_date')
                    end = agro.get('CropCalendar', {}).get('crop_end_date')
                    if start is not None:
                        try:
                            start_date = start if isinstance(start, date) else datetime.fromisoformat(str(start)).date()
                        except Exception:
                            start_date = None
                    if start and end:
                        # Subtract 1 because end date is exclusive (matches actual growing period)
                        season_length = (end - start).days - 1
                except:
                    pass

                # Get location
                try:
                    site = agro.get('SiteCalendar', {})
                    lat = site.get('latitude')
                    lon = site.get('longitude')
                    if lat is not None and lon is not None:
                        location = f"{lat}°N, {lon}°E"
                        latitude = lat
                        longitude = lon
                except:
                    pass

            # Get action space parameters
            num_fert = getattr(unwrapped, 'num_fert', 4)
            num_irrig = getattr(unwrapped, 'num_irrig', 4)
            fert_amount = getattr(unwrapped, 'fert_amount', 2.0)
            irrig_amount = getattr(unwrapped, 'irrig_amount', 0.5)
            intervention_interval = getattr(unwrapped, 'intervention_interval', 7)
            available_action_kinds = (
                action_menu_from_env_id(action_schema_env_id)
                if action_schema_env_id
                else tuple(cls._infer_action_components(unwrapped))
            )

            # Get output variables
            prompt_output_vars = list(output_vars or [])
            if not prompt_output_vars and hasattr(unwrapped, 'get_output_vars'):
                prompt_output_vars = unwrapped.get_output_vars()

            return cls(
                crop_name=crop_name,
                season_length=season_length,
                location=location,
                num_fert=num_fert,
                num_irrig=num_irrig,
                fert_amount=fert_amount,
                irrig_amount=irrig_amount,
                intervention_interval=intervention_interval,
                output_vars=prompt_output_vars,
                field_aliases=field_aliases,
                latitude=latitude,
                longitude=longitude,
                start_date=start_date,
                require_think=require_think,
                thinking_mode=thinking_mode,
                think_tag=think_tag,
                available_action_kinds=available_action_kinds,
                objective_id=objective_id,
                objective_text=objective_text,
                include_profit_context=include_profit_context,
                profit_context_params=profit_context_params,
            )
        except Exception as e:
            # Return default instance if configuration extraction fails
            print(f"Warning: Could not extract full config from env: {e}")
            return cls(
                require_think=require_think,
                thinking_mode=thinking_mode,
                think_tag=think_tag,
                available_action_kinds=(
                    action_menu_from_env_id(action_schema_env_id)
                    if action_schema_env_id
                    else None
                ),
                output_vars=output_vars,
                field_aliases=field_aliases,
                objective_id=objective_id,
                objective_text=objective_text,
                include_profit_context=include_profit_context,
                profit_context_params=profit_context_params,
            )
