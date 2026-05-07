"""Shared CropGrowth objective prompt helpers."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


DEFAULT_OBJECTIVE_ID = "profit_max"
DEFAULT_YIELD_LABEL = "final_WSO_kg_ha"
DEFAULT_PROFIT_REWARD_SCALE = 1000.0
DEFAULT_YIELD_SCALE_NOTE = (
    "Successful crop seasons usually produce final yield on the order of "
    "thousands of kg/ha."
)

DEFAULT_PROFIT_COSTS = {
    "cost_n": 3.5,
    "cost_p": 5.5,
    "cost_k": 2.25,
    "cost_water": 0.05,
}

INPUT_LABELS = {
    "n": ("Nitrogen", "total_N_kg_ha", "kg/ha N"),
    "p": ("Phosphorus", "total_P_kg_ha", "kg/ha P"),
    "k": ("Potassium", "total_K_kg_ha", "kg/ha K"),
    "irrig": ("Irrigation", "total_irrig_mm", "mm water"),
}

IRRIGATION_DISPLAY_UNITS = {
    "mm": ("total_irrig_mm", "mm water", 1.0),
    "cm": ("total_irrig_cm", "cm water", 10.0),
}

OBJECTIVE_DECISION_TEXT = {
    "yield_max": "choose the action most likely to increase end-of-season yield",
    "profit_max": "choose the action that best balances final-yield gains against input costs",
    "water_stewardship": "choose the action that protects yield while conserving irrigation water",
    "nutrient_stewardship": (
        "choose the action that protects yield while keeping N/P/K inputs and "
        "terminal soil nutrients in agronomic ranges"
    ),
}


def _finite_float(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _param(params: Mapping[str, Any], names: Sequence[str], default: float) -> float:
    for name in names:
        if name in params and params[name] is not None:
            return _finite_float(params[name], default)
    return default


def format_number(value: float) -> str:
    value = float(value)
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def normalize_objective_id(objective_id: str | None = None) -> str:
    normalized = str(objective_id or DEFAULT_OBJECTIVE_ID).strip()
    return normalized or DEFAULT_OBJECTIVE_ID


def profit_cost_params(params: Mapping[str, Any] | None = None) -> dict[str, float]:
    """Normalize paper profit costs from config/reward dictionaries."""

    params = dict(params or {})
    cost_n = _param(
        params,
        ("cost_n_kg_wso_per_kg", "cost_n_kg_grain_per_kg", "cost_n", "c_n"),
        DEFAULT_PROFIT_COSTS["cost_n"],
    )
    cost_p = _param(
        params,
        ("cost_p_kg_wso_per_kg", "cost_p_kg_grain_per_kg", "cost_p", "c_p"),
        DEFAULT_PROFIT_COSTS["cost_p"],
    )
    cost_k = _param(
        params,
        ("cost_k_kg_wso_per_kg", "cost_k_kg_grain_per_kg", "cost_k", "c_k"),
        DEFAULT_PROFIT_COSTS["cost_k"],
    )
    cost_water = _param(
        params,
        (
            "cost_water_kg_wso_per_mm",
            "cost_water_kg_grain_per_mm",
            "cost_irrig_kg_wso_per_mm",
            "cost_irrig",
            "cost_water",
            "c_w",
        ),
        DEFAULT_PROFIT_COSTS["cost_water"],
    )
    return {
        "cost_n": cost_n,
        "cost_p": cost_p,
        "cost_k": cost_k,
        "cost_water": cost_water,
    }


def profit_reward_scale(params: Mapping[str, Any] | None = None) -> float:
    params = dict(params or {})
    scale = _param(
        params,
        ("profit_reward_scale", "reward_scale", "y_ref", "Y_ref"),
        DEFAULT_PROFIT_REWARD_SCALE,
    )
    return scale if scale > 0.0 else DEFAULT_PROFIT_REWARD_SCALE


def cropgrowth_system_prompt(simulator_name: str | None = None) -> str:
    del simulator_name  # Kept for backward-compatible call sites.
    role = "You are an agricultural management expert"
    return (
        f"{role}. Optimize the management objective specified in the user message. "
        "Use only the available actions and respond in the required format."
    )


def objective_decision_text(objective_id: str | None = None) -> str:
    return OBJECTIVE_DECISION_TEXT.get(
        normalize_objective_id(objective_id),
        "choose the action that best advances the stated management objective",
    )


def _ordered_available_inputs(available_inputs: Sequence[str] | None) -> tuple[str, ...]:
    available = set(available_inputs or ())
    ordered = tuple(kind for kind in ("n", "p", "k", "irrig") if kind in available)
    return ordered


def _irrigation_display_terms(irrigation_unit: str | None) -> tuple[str, str, float]:
    normalized = str(irrigation_unit or "mm").strip().lower()
    if normalized not in IRRIGATION_DISPLAY_UNITS:
        normalized = "mm"
    return IRRIGATION_DISPLAY_UNITS[normalized]


def _profit_objective_body(
    params: Mapping[str, Any] | None,
    *,
    available_inputs: Sequence[str] | None,
    yield_label: str,
    irrigation_unit: str = "mm",
) -> str:
    costs = profit_cost_params(params)
    inputs = _ordered_available_inputs(available_inputs)
    irrigation_metric, irrigation_unit_label, irrigation_cost_scale = _irrigation_display_terms(
        irrigation_unit
    )
    formula_lines = ["profit_wso_kg_ha =", f"  {yield_label}"]
    for kind in inputs:
        if kind == "irrig":
            metric_name = irrigation_metric
            display_cost = costs["cost_water"] * irrigation_cost_scale
        else:
            _, metric_name, _ = INPUT_LABELS[kind]
            display_cost = costs[f"cost_{kind}"]
        formula_lines.append(f"  - {format_number(display_cost)} * {metric_name}")

    unit_lines = []
    for kind in inputs:
        if kind == "irrig":
            label = "Irrigation"
            unit = irrigation_unit_label
            display_cost = costs["cost_water"] * irrigation_cost_scale
        else:
            label, _, unit = INPUT_LABELS[kind]
            display_cost = costs[f"cost_{kind}"]
        unit_lines.append(
            f"- {label}: 1 {unit} costs {format_number(display_cost)} kg/ha WSO-equivalent."
        )

    if not unit_lines:
        unit_lines.append("- No priced fertilizer or irrigation input is available in this action space.")

    return "\n".join(
        [
            "Goal: maximize terminal WSO-equivalent profit, not raw yield.",
            "",
            "Score at the end of the season:",
            *formula_lines,
            "",
            "Input unit costs:",
            *unit_lines,
            "",
            DEFAULT_YIELD_SCALE_NOTE,
            "",
            "Decision rule: apply an input only if that input is expected to increase final WSO by more than its WSO-equivalent cost.",
        ]
    )


def _yield_objective_body(yield_label: str) -> str:
    return "\n".join(
        [
            "Goal: maximize terminal crop yield measured by final WSO.",
            "",
            "Score at the end of the season:",
            f"score = {yield_label}",
            "",
            "No fertilizer or irrigation cost is subtracted in this objective.",
            "",
            "Decision rule: choose the action most likely to increase end-of-season WSO.",
        ]
    )


def _water_objective_body(
    params: Mapping[str, Any] | None,
    yield_label: str,
    *,
    irrigation_unit: str = "mm",
) -> str:
    params = dict(params or {})
    budget = _param(params, ("water_budget_mm", "budget_water_mm", "B_W"), 120.0)
    irrigation_metric, irrigation_unit_label, irrigation_cost_scale = _irrigation_display_terms(
        irrigation_unit
    )
    display_budget = budget / irrigation_cost_scale
    return "\n".join(
        [
            "Goal: maintain acceptable terminal yield while conserving irrigation water.",
            "",
            "Score combines terminal yield with irrigation-use and water-budget penalties.",
            f"Yield term uses {yield_label}; irrigation is measured as {irrigation_metric}.",
            f"Reference water budget: {format_number(display_budget)} {irrigation_unit_label.split()[0]}.",
            "",
            "Decision rule: irrigate only when expected yield protection justifies the added water use.",
        ]
    )


def _nutrient_objective_body(params: Mapping[str, Any] | None, yield_label: str) -> str:
    params = dict(params or {})
    budget_n = _param(params, ("budget_n_kg_ha", "B_N", "nutrient_budget_kg_ha", "B"), 180.0)
    budget_p = _param(params, ("budget_p_kg_ha", "B_P", "nutrient_budget_kg_ha", "B"), 180.0)
    budget_k = _param(params, ("budget_k_kg_ha", "B_K", "nutrient_budget_kg_ha", "B"), 180.0)
    return "\n".join(
        [
            "Goal: maintain acceptable terminal yield while avoiding N/P/K over-application and terminal nutrient imbalance.",
            "",
            "Score combines terminal yield with penalties for excessive N/P/K application, terminal nutrient depletion, and terminal nutrient saturation.",
            f"Yield term uses {yield_label}.",
            "Reference nutrient budgets:",
            f"- Nitrogen budget: {format_number(budget_n)} kg/ha.",
            f"- Phosphorus budget: {format_number(budget_p)} kg/ha.",
            f"- Potassium budget: {format_number(budget_k)} kg/ha.",
            "",
            "Decision rule: apply nutrients only when they protect yield while keeping application totals and terminal soil nutrients in agronomic ranges.",
        ]
    )


def build_management_objective_block(
    objective_id: str | None = None,
    params: Mapping[str, Any] | None = None,
    *,
    available_inputs: Sequence[str] | None = None,
    yield_label: str = DEFAULT_YIELD_LABEL,
    objective_text: str | None = None,
    irrigation_unit: str = "mm",
) -> str:
    """Build the shared user-prompt objective block for CropGrowth tasks."""

    objective = normalize_objective_id(objective_id)
    if objective_text and objective_text.strip():
        body = objective_text.strip()
    elif objective == "yield_max":
        body = _yield_objective_body(yield_label)
    elif objective == "profit_max":
        body = _profit_objective_body(
            params,
            available_inputs=available_inputs,
            yield_label=yield_label,
            irrigation_unit=irrigation_unit,
        )
    elif objective == "water_stewardship":
        body = _water_objective_body(params, yield_label, irrigation_unit=irrigation_unit)
    elif objective == "nutrient_stewardship":
        body = _nutrient_objective_body(params, yield_label)
    else:
        body = f"Goal: follow the management objective identified as {objective}."
    return f'<management objective id="{objective}">\n{body}\n</management objective>'
