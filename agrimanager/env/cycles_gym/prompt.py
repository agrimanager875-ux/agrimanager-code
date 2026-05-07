"""Prompt generation for cycles_gym environments."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Union

from agrimanager.env.base.objective_prompt import (
    build_management_objective_block,
    cropgrowth_system_prompt,
)

from .crop_traits import build_crop_traits_text


VALID_THINKING_MODES = {"minimal", "grounding_decision"}

_CORN_OBS_DESCRIPTIONS = {
    "CUM. BIOMASS": "Cumulative crop biomass state",
    "AG BIOMASS": "Aboveground biomass state",
    "ROOT BIOMASS": "Root biomass state",
    "GRAIN YIELD": "Grain yield / harvestable yield biomass",
}


def _normalize_thinking_mode(thinking_mode: str) -> str:
    normalized = str(thinking_mode).strip().lower().replace("-", "_")
    if normalized == "think":
        normalized = "grounding_decision"
    if normalized not in VALID_THINKING_MODES:
        valid = ", ".join(sorted(VALID_THINKING_MODES | {"think"}))
        raise ValueError(f"Invalid thinking_mode '{thinking_mode}'. Expected one of: {valid}")
    return normalized


def _normalize_think_tag(think_tag: str) -> str:
    normalized = str(think_tag or "think").strip().lower()
    if normalized == "":
        return "think"
    if not re.fullmatch(r"[a-z][a-z0-9_-]*", normalized):
        return "think"
    return normalized


def _extract_answer_text(response: str) -> str:
    txt = (response or "").strip()
    if not txt:
        return ""

    patterns = [
        r"<answer>\s*(.*?)\s*(?:</answer>|$)",
        r"<answer>\s*(.*?)\s*</answer>",
        r"<answer>\s*(.*?)\s*<answer>",
        r"\(answer\)\s*(.*?)\s*(?:\(/answer\)|\(answer\))",
    ]
    for pattern in patterns:
        match = re.search(pattern, txt, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
    return ""


def _strip_xml_like_tags(text: str) -> str:
    stripped = re.sub(r"</?[^>]+>", " ", text or "", flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", stripped).strip()


def _coerce_sequence(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    try:
        return list(value)
    except TypeError:
        return [value]


class CornPromptGenerator:
    """Prompt generator for CyclesGym corn fertilization environments."""

    def __init__(
        self,
        crop_name: str = "corn",
        n_actions: int = 11,
        max_n: float = 150.0,
        obs_names: Optional[List[str]] = None,
        require_think: bool = False,
        think_tag: str = "tool_call",
        reward_mode: str = "native",
        include_profit_context: bool = False,
        profit_context_params: Optional[dict[str, Any]] = None,
        objective_id: Optional[str] = "profit_max",
        objective_text: Optional[str] = None,
        reward_params: Optional[dict[str, Any]] = None,
    ):
        self.crop_name = crop_name
        self.n_actions = n_actions
        self.max_n = float(max_n)
        self.obs_names = obs_names or []
        self.require_think = require_think
        self.think_tag = _normalize_think_tag(think_tag)
        self.reward_mode = str(reward_mode)
        self.include_profit_context = bool(include_profit_context)
        if objective_id is None:
            objective_id = "yield_max" if self.reward_mode in {"final_yield", "yield_only"} else "profit_max"
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
            available_inputs=("n",),
            yield_label="final_yield_kg_ha",
            objective_text=self.objective_text,
        )

    def get_system_prompt(self) -> str:
        think_tag = self.think_tag
        base_prompt = cropgrowth_system_prompt("CycleGym Corn")
        if self.require_think:
            return (
                f"{base_prompt} Respond with exactly two XML-style blocks in this order: "
                f"<{think_tag}>brief reasoning</{think_tag}> then <answer>final action</answer>. "
                "Never omit the <answer> block."
            )
        return (
            f"{base_prompt} Respond using the exact format <answer>...</answer> and output nothing else."
        )

    def describe_observation(self, obs: Any) -> str:
        if not self.obs_names:
            return "Observation is a numeric array with no variable names available."

        crop_label = self.crop_name
        if self.reward_mode in {"final_yield", "yield_only"} and crop_label.lower() == "corn":
            crop_label = "maize (CycleGym internal crop: corn)"
        lines = [f"Crop: {crop_label}. Observation summary:"]
        for name, value in zip(self.obs_names, obs):
            try:
                val_str = f"{float(value):.4g}"
            except Exception:
                val_str = str(value)
            desc = _CORN_OBS_DESCRIPTIONS.get(str(name).upper())
            label = f"{name} ({desc})" if desc else str(name)
            lines.append(f"- {label}: {val_str}")
        return "\n".join(lines)

    def get_action_options_description(self) -> str:
        step = self.max_n / max(1, self.n_actions - 1)
        max_amt = step * (self.n_actions - 1)
        action_text = (
            "Available actions (pick exactly one):\n"
            "- Do nothing.\n"
            f"- Apply nitrogen fertilizer (amount from 0 to {max_amt:.1f} kg/ha, in steps of {step:.1f}).\n"
            "Use nitrogen when it is likely to improve final maize yield; late or excessive applications may be ineffective."
        )
        if self.require_think:
            think_tag = self.think_tag
            return (
                f"{action_text}\n\n"
                "Keep the reasoning concise and decision-focused. Do not restate the full input.\n\n"
                f"Respond using the exact format: <{think_tag}> ... </{think_tag}> <answer> ... </answer> with no extra text.\n\n"
                f"Example: <{think_tag}>Nitrogen appears limiting.</{think_tag}> <answer>Apply 30 kg/ha nitrogen fertilizer.</answer>"
            )
        return (
            f"{action_text}\n\n"
            "Respond using the exact format: <answer> ... </answer> with no extra text.\n\n"
            "Example: <answer>Apply 30 kg/ha nitrogen fertilizer.</answer>"
        )

    def get_turn_prompt(self, obs: Any, context: Optional[Dict[str, Any]] = None) -> str:
        del context
        sections = [
            f"We are growing {self.crop_name} in a CropGrowth management task.",
            self._management_objective_block(),
            f"<current observation>\n{self.describe_observation(obs)}\n</current observation>",
        ]
        sections.append(self.get_action_options_description())
        return "\n\n".join(sections)

    @staticmethod
    def _extract_answer_text(response: str) -> str:
        return _extract_answer_text(response)

    def parse_action_response(self, response: str) -> Optional[int]:
        if self.require_think:
            t = re.escape(self.think_tag)
            match = re.fullmatch(
                rf"\s*<{t}>(.*?)</{t}>\s*<answer>(.*?)</answer>\s*",
                response or "",
                flags=re.IGNORECASE | re.DOTALL,
            )
            if match is None:
                return None
            txt = match.group(2).strip().lower()
        else:
            txt = self._extract_answer_text(response).lower()
        if not txt:
            return None

        if "do nothing" in txt or "take no action" in txt:
            return 0

        amount_match = re.search(
            r"(-?\d+(?:\.\d+)?)\s*(?:kg\s*/\s*ha|kg\s*ha\s*-?1|kg/ha|kg)",
            txt,
            flags=re.IGNORECASE,
        )
        if not amount_match:
            amount_match = re.search(r"(-?\d+(?:\.\d+)?)", txt)
        if not amount_match:
            return None

        amount = max(0.0, float(amount_match.group(1)))
        step = self.max_n / max(1, self.n_actions - 1)
        action = int(round(amount / step))
        return max(0, min(self.n_actions - 1, action))

    @classmethod
    def from_env(
        cls,
        env: Any,
        require_think: bool = False,
        think_tag: str = "tool_call",
        reward_mode: str = "native",
        include_profit_context: bool = False,
        profit_context_params: Optional[dict[str, Any]] = None,
        objective_id: Optional[str] = None,
        objective_text: Optional[str] = None,
        reward_params: Optional[dict[str, Any]] = None,
    ) -> "CornPromptGenerator":
        n_actions = getattr(env, "n_actions", None)
        if n_actions is None and hasattr(env, "action_space"):
            n_actions = getattr(env.action_space, "n", 11)
        max_n = getattr(env, "maxN", 150.0)

        obs_names = []
        observer = getattr(env, "observer", None)
        if observer is not None:
            obs_names = getattr(observer, "obs_names", []) or []

        return cls(
            crop_name="corn",
            n_actions=int(n_actions) if n_actions is not None else 11,
            max_n=float(max_n),
            obs_names=obs_names,
            require_think=require_think,
            think_tag=think_tag,
            reward_mode=reward_mode,
            include_profit_context=include_profit_context,
            profit_context_params=profit_context_params,
            objective_id=objective_id,
            objective_text=objective_text,
            reward_params=reward_params,
        )


class CropPlanningPromptGenerator:
    """Prompt generator for CyclesGym crop-planning environments."""

    def __init__(
        self,
        rotation_crops: List[str],
        action_nvec: List[int],
        obs_names: Optional[List[str]] = None,
        require_think: bool = False,
        thinking_mode: str = "grounding_decision",
        think_tag: str = "think",
        crop_traits_text: Optional[str] = None,
    ):
        self.rotation_crops = rotation_crops
        self.action_nvec = action_nvec
        self.obs_names = obs_names or []
        self.require_think = require_think
        self.thinking_mode = _normalize_thinking_mode(thinking_mode)
        self.think_tag = _normalize_think_tag(think_tag)
        self.crop_traits_text = crop_traits_text
        self.obs_groups = {
            "ORG SOIL N": "Soil nitrogen pool",
            "PROF SOIL NO3": "Soil nitrogen pool",
            "PROF SOIL NH4": "Soil nitrogen pool",
            "MINERALIZATION": "Nitrogen transformations",
            "IMMOBILIZATION": "Nitrogen transformations",
            "NET MINERALIZ": "Nitrogen transformations",
            "NH4 NITRIFICAT": "Nitrogen transformations",
            "N2O FROM NITRIF": "Nitrogen losses",
            "NH3 VOLATILIZ": "Nitrogen losses",
            "NO3 DENITRIF": "Nitrogen losses",
            "N2O FROM DENIT": "Nitrogen losses",
        }
        self.obs_labels = {
            "ORG SOIL N": "Organic soil nitrogen",
            "PROF SOIL NO3": "Profile soil nitrate",
            "PROF SOIL NH4": "Profile soil ammonium",
            "MINERALIZATION": "Mineralization",
            "IMMOBILIZATION": "Immobilization",
            "NET MINERALIZ": "Net mineralization",
            "NH4 NITRIFICAT": "Ammonium nitrification",
            "N2O FROM NITRIF": "N2O from nitrification",
            "NH3 VOLATILIZ": "Ammonia volatilization",
            "NO3 DENITRIF": "Nitrate denitrification",
            "N2O FROM DENIT": "N2O from denitrification",
        }

    def set_crop_traits(self, text: str) -> None:
        self.crop_traits_text = text

    def get_system_prompt(self) -> str:
        base_prompt = (
            "You are an agricultural planning expert. Your goal is to maximize long-run gross revenue "
            "over a multi-year crop rotation by choosing the crop and planting timing for each year. "
            "Use the current soil nitrogen state to make a decision that balances immediate returns "
            "with future rotation effects. Gross revenue is yield multiplied by the crop price; "
            "there are no production costs in this task. Always emit a valid final action."
        )
        if self.require_think:
            base_prompt += (
                " Keep the reasoning to one short sentence and always close the "
                f"<{self.think_tag}> block before the <answer> block."
            )
        else:
            base_prompt += " Respond only with the final <answer> block."
        if self.crop_traits_text and self.crop_traits_text.strip():
            base_prompt += (
                "\nUse crop traits as prior agronomic knowledge. Combine them with the current soil "
                "nitrogen condition and the long-run rotation objective when deciding which crop to plant next."
            )
        return base_prompt

    def describe_context(self, context: Optional[Dict[str, Any]]) -> str:
        if not context:
            return ""

        lines: List[str] = []
        current_year = context.get("current_year")
        years_remaining = context.get("years_remaining")
        if current_year is not None or years_remaining is not None:
            lines.append("Current rotation context:")
            if current_year is not None:
                lines.append(f"- current_year: {current_year}")
            if years_remaining is not None:
                lines.append(f"- years_remaining: {years_remaining}")

        prices = context.get("crop_prices") or {}
        if prices:
            lines.append("Current crop prices in dollars per tonne:")
            for crop in self.rotation_crops:
                if crop in prices:
                    try:
                        price_text = f"{float(prices[crop]):.2f}"
                    except Exception:
                        price_text = str(prices[crop])
                    lines.append(f"- {crop}: {price_text}")

        history = context.get("past_trajectory") or []
        if history:
            crop_counts = {crop: 0 for crop in self.rotation_crops}
            total_revenue = 0.0
            valid_revenue_count = 0
            for row in history:
                crop = row.get("crop")
                if crop in crop_counts:
                    crop_counts[crop] += 1
                try:
                    total_revenue += float(row.get("revenue"))
                    valid_revenue_count += 1
                except Exception:
                    pass
            count_text = ", ".join(f"{crop}={count}" for crop, count in crop_counts.items())
            if valid_revenue_count:
                lines.append(
                    "Past trajectory summary: "
                    f"{len(history)} years completed; crop counts: {count_text}; "
                    f"cumulative gross revenue: {total_revenue:.2f}."
                )
            else:
                lines.append(
                    "Past trajectory summary: "
                    f"{len(history)} years completed; crop counts: {count_text}."
                )
            recent_history = history[-3:]
            if len(history) > len(recent_history):
                lines.append(f"Recent trajectory (last {len(recent_history)} years):")
            else:
                lines.append("Past trajectory so far:")
            for row in recent_history:
                yield_value = row.get("yield_tonnes")
                revenue = row.get("revenue")
                try:
                    yield_text = f"{float(yield_value):.3f}"
                except Exception:
                    yield_text = str(yield_value)
                try:
                    revenue_text = f"{float(revenue):.2f}"
                except Exception:
                    revenue_text = str(revenue)
                lines.append(
                    "- "
                    f"year: {row.get('year')}, "
                    f"crop: {row.get('crop')}, "
                    f"planting_week: {row.get('planting_week')}, "
                    f"yield_tonnes: {yield_text}, "
                    f"gross_revenue: {revenue_text}"
                )
        else:
            lines.append("Past trajectory so far: none; this is the first year.")

        return "\n".join(lines)

    def describe_observation(self, obs: Any) -> str:
        if not self.obs_names:
            return "Observation is a numeric array with no variable names available."

        lines = ["Current soil-state observation:"]
        current_group = None
        for name, value in zip(self.obs_names, obs):
            group = self.obs_groups.get(name, "Other observations")
            if group != current_group:
                lines.append(f"[{group}]")
                current_group = group
            try:
                val_str = f"{float(value):.4g}"
            except Exception:
                val_str = str(value)
            label = self.obs_labels.get(name, name)
            lines.append(f"- {label}: {val_str}")
        return "\n".join(lines)

    def get_intro(self) -> str:
        if len(self.action_nvec) >= 2:
            planting_choices = self.action_nvec[1]
            planting_window = (
                f"You must also choose a planting-week index from 0 to {planting_choices - 1}, "
                "representing the allowable planting window in spring."
            )
        else:
            planting_window = ""
        return (
            "We are making one crop-planning decision for the current year in a multi-year rotation. "
            "At each decision, choose the crop to plant and its planting timing. "
            "The objective is to maximize cumulative gross revenue across the full rotation horizon, "
            "not just this year. "
            f"{planting_window}"
        ).strip()

    def get_action_options_description(self) -> str:
        crop_list = "\n".join(f"{i}: {crop}" for i, crop in enumerate(self.rotation_crops))
        if len(self.action_nvec) == 2:
            action_text = (
                "Available actions (pick exactly one):\n"
                f"Crop index options:\n{crop_list}\n"
                f"DOY index options: 0 to {self.action_nvec[1] - 1}\n"
            )
            answer_format = "crop_index, doy_index"
            answer_example = "<answer>1, 4</answer>"
        else:
            action_text = (
                "Available actions (pick exactly one):\n"
                f"Crop index options:\n{crop_list}\n"
                f"DOY index options: 0 to {self.action_nvec[1] - 1}\n"
                f"END_DOY index options: 0 to {self.action_nvec[2] - 1}\n"
                f"MAX_SMC index options: 0 to {self.action_nvec[3] - 1}\n"
            )
            answer_format = "crop_index, doy_index, end_doy_index, max_smc_index"
            answer_example = "<answer>1, 4, 2, 0</answer>"

        guidance = (
            "Decision guidance:\n"
            "1. Identify the current crop-price regime and soil nitrogen condition.\n"
            "2. Consider how the crop choice affects both this year's gross revenue and future years in the rotation.\n"
            "3. Avoid blindly choosing the highest-price crop if it creates poor long-run rotation consequences.\n"
            "4. Choose the crop and planting or management settings most likely to maximize cumulative gross revenue.\n"
        )

        if self.require_think:
            think_tag = self.think_tag
            if self.thinking_mode == "grounding_decision":
                think_instructions = (
                    "Reason in one short sentence before answering about:\n"
                    "1. The current price regime, soil nitrogen condition, and main agronomic constraint.\n"
                    "2. The likely short-term gross revenue and future-rotation consequences of candidate crop choices.\n"
                    "3. Why the selected action is the best long-run gross-revenue decision.\n"
                )
            else:
                think_instructions = "Think in one short sentence before answering.\n"
            response_format = (
                "Keep the reasoning concise and decision-focused. Use at most 35 words. "
                "Do not restate the full input. If uncertain, still produce a valid <answer> block.\n\n"
                f"Respond using the exact format: <{think_tag}> ... </{think_tag}> "
                f"<answer>{answer_format}</answer> with no extra text.\n"
                f"Example: <{think_tag}>Soybeans likely improve rotation value under the current nitrogen state.</{think_tag}> "
                f"{answer_example}\n"
                "Output integers only inside the answer tag."
            )
        else:
            think_instructions = ""
            response_format = (
                f"Respond exactly as <answer>{answer_format}</answer>.\n"
                f"Example: {answer_example}\n"
                "Output integers only."
            )

        return f"{action_text}{guidance}{think_instructions}{response_format}"

    def get_turn_prompt(self, obs: Any, context: Optional[Dict[str, Any]] = None) -> str:
        sections = [self.get_intro()]
        context_text = self.describe_context(context)
        if context_text:
            sections.append(context_text)
        if self.crop_traits_text and self.crop_traits_text.strip():
            sections.append(f"<crop traits>\n{self.crop_traits_text.strip()}\n</crop traits>")
        sections.append(f"<current observation>\n{self.describe_observation(obs)}\n</current observation>")
        sections.append(self.get_action_options_description())
        return "\n\n".join(sections)

    @staticmethod
    def _extract_answer_text(response: str) -> str:
        txt = _extract_answer_text(response)
        if txt:
            return txt
        return ""

    def _default_planting_week(self) -> int:
        if len(self.action_nvec) < 2:
            return 0
        return max(0, min(self.action_nvec[1] - 1, self.action_nvec[1] // 2))

    def _extract_action_pair(self, text: str) -> Optional[List[int]]:
        normalized = _strip_xml_like_tags(text or "")
        if not normalized:
            return None

        explicit_patterns = [
            r"crop(?:_index| index)?\s*[:=]?\s*(-?\d+)\D+?(?:doy(?:_index)?|planting(?:[_\s-]*week)?|week)\s*[:=]?\s*(-?\d+)",
            r"(-?\d+)\s*,\s*(-?\d+)",
        ]
        for pattern in explicit_patterns:
            matches = list(re.finditer(pattern, normalized, flags=re.IGNORECASE | re.DOTALL))
            if not matches:
                continue
            last = matches[-1]
            try:
                values = [int(last.group(1)), int(last.group(2))]
            except Exception:
                continue
            if len(self.action_nvec) >= 2 and all(
                0 <= value < size for value, size in zip(values, self.action_nvec[:2])
            ):
                return values

        lowered = normalized.lower()
        crop_matches: List[tuple[int, int]] = []
        for crop_idx, crop_name in enumerate(self.rotation_crops):
            aliases = {
                crop_name.lower(),
                re.sub(r"[^a-z]+", "", crop_name.lower()),
            }
            if "corn" in crop_name.lower():
                aliases.add("corn")
            if "soy" in crop_name.lower():
                aliases.add("soybean")
                aliases.add("soy")
            if "wheat" in crop_name.lower():
                aliases.add("wheat")
                aliases.add("springwheat")
            for alias in aliases:
                if not alias:
                    continue
                pattern = re.escape(alias) if alias == crop_name.lower() else rf"\b{re.escape(alias)}\b"
                for match in re.finditer(pattern, lowered):
                    crop_matches.append((match.start(), crop_idx))
        if not crop_matches:
            return None

        crop_matches.sort()
        _, crop_idx = crop_matches[-1]
        tail = lowered[crop_matches[-1][0] :]
        week_patterns = [
            r"(?:doy(?:_index)?|planting(?:[_\s-]*week)?|week)\s*(?:index\s*)?[:=]?\s*(-?\d+)",
            r"\bat\s*(-?\d+)\b",
        ]
        planting_week: Optional[int] = None
        for pattern in week_patterns:
            matches = list(re.finditer(pattern, tail, flags=re.IGNORECASE))
            if not matches:
                continue
            candidate = int(matches[-1].group(1))
            if len(self.action_nvec) >= 2 and 0 <= candidate < self.action_nvec[1]:
                planting_week = candidate
                break
        if planting_week is None:
            planting_week = self._default_planting_week()

        return [crop_idx, planting_week]

    def parse_action_response(self, response: str) -> Optional[List[int]]:
        needed = len(self.action_nvec)
        response_text = response or ""
        answer_text = ""

        if self.require_think:
            t = re.escape(self.think_tag)
            match = re.fullmatch(
                rf"\s*<{t}>(.*?)</{t}>\s*<answer>(.*?)</answer>\s*",
                response_text,
                flags=re.DOTALL,
            )
            if match is not None:
                answer_text = match.group(2).strip()
        else:
            match = re.fullmatch(
                r"\s*<answer>(.*?)</answer>\s*",
                response_text,
                flags=re.DOTALL,
            )
            if match is not None:
                answer_text = match.group(1).strip()

        if not answer_text:
            answer_text = self._extract_answer_text(response_text)

        if answer_text:
            nums = re.findall(r"-?\d+", answer_text)
            if len(nums) >= needed:
                try:
                    values = [int(x) for x in nums[-needed:]]
                except ValueError:
                    values = []
                if values and all(0 <= value < size for value, size in zip(values, self.action_nvec)):
                    return values

        fallback = self._extract_action_pair(answer_text or response_text)
        if fallback is not None and all(
            0 <= value < size for value, size in zip(fallback, self.action_nvec[: len(fallback)])
        ):
            return fallback
        return None

    @classmethod
    def from_env(
        cls,
        env: Any,
        require_think: bool = False,
        thinking_mode: str = "grounding_decision",
        think_tag: str = "think",
        crop_traits_text: Optional[str] = None,
    ) -> "CropPlanningPromptGenerator":
        rotation_crops = _coerce_sequence(getattr(env, "rotation_crops", None))
        action_nvec = []
        if hasattr(env, "action_space") and hasattr(env.action_space, "nvec"):
            action_nvec = list(env.action_space.nvec)

        obs_names = []
        observer = getattr(env, "observer", None)
        if observer is not None:
            obs_names = getattr(observer, "obs_names", []) or []

        return cls(
            rotation_crops=rotation_crops,
            action_nvec=action_nvec,
            obs_names=obs_names,
            require_think=require_think,
            thinking_mode=thinking_mode,
            think_tag=think_tag,
            crop_traits_text=crop_traits_text,
        )


CyclesPromptGenerator = Union[CornPromptGenerator, CropPlanningPromptGenerator]


def build_prompt_generator(
    env: Any,
    *,
    env_id: str,
    include_crop_traits: bool = True,
    require_think: bool = False,
    thinking_mode: str = "grounding_decision",
    think_tag: str = "think",
    reward_mode: str = "native",
    include_profit_context: bool = False,
    profit_context_params: Optional[dict[str, Any]] = None,
    objective_id: Optional[str] = None,
    objective_text: Optional[str] = None,
    reward_params: Optional[dict[str, Any]] = None,
) -> CyclesPromptGenerator:
    if env_id.startswith("Corn"):
        return CornPromptGenerator.from_env(
            env,
            require_think=require_think,
            think_tag="tool_call",
            reward_mode=reward_mode,
            include_profit_context=include_profit_context,
            profit_context_params=profit_context_params,
            objective_id=objective_id,
            objective_text=objective_text,
            reward_params=reward_params,
        )
    if env_id.startswith("CropPlanning"):
        crop_traits_text = None
        if include_crop_traits:
            rotation_crops = _coerce_sequence(getattr(env, "rotation_crops", None))
            if rotation_crops:
                crop_traits_text = build_crop_traits_text(rotation_crops)
        return CropPlanningPromptGenerator.from_env(
            env,
            require_think=require_think,
            thinking_mode=thinking_mode,
            think_tag=think_tag,
            crop_traits_text=crop_traits_text,
        )
    raise ValueError(f"Unsupported CyclesGym env_id for prompt generation: {env_id}")


__all__ = [
    "CyclesPromptGenerator",
    "CornPromptGenerator",
    "CropPlanningPromptGenerator",
    "build_prompt_generator",
]
