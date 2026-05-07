"""Generic turn-reward aggregation for AgriManager environments."""

import numbers
import math
from typing import Any, Optional


_GROUP_LABEL_KEYS = (
    "simulator",
    "split",
    "dataset_split",
    "dataset_role",
    "validation_set",
    "crop",
    "weather_regime",
    "crop_regime",
    "price_regime",
    "regime_family",
    "shocked_crop",
    "variety",
    "variety_split",
    "objective_id",
    "observation_schema",
    "observation_schema_family",
    "reward_formulation",
    "action_menu",
    "action_schema",
    "schema_tuple",
    "callback_family",
    "prompt_condition",
    "train_source",
    "env_id",
    "scenario_id",
)


def _extract_env_config(extra_info: dict) -> dict:
    env_config = extra_info.get("interaction_kwargs", {}).get("env_config", {})
    return env_config if isinstance(env_config, dict) else {}


def _extract_turn_num(extra_info: dict, num_steps: int) -> int:
    env_config = _extract_env_config(extra_info)
    turn_num = int(env_config.get("turn_num", 0) or 0)
    return turn_num if turn_num > 0 else max(num_steps, 1)


def _extract_step_idx(extra_info: dict, num_steps: int) -> int:
    if num_steps <= 0:
        return 0
    idx = int(extra_info.get("step_idx", num_steps - 1) or 0)
    return min(max(idx, 0), num_steps - 1)


def _extract_terminal_trajectory_metrics(extra_info: dict) -> dict[str, float]:
    """Extract the terminal env trajectory metrics from interaction metadata.

    In per-turn mode each row carries the full `interaction_metrics` list for the
    trajectory, while validation logging only sees the reward manager's
    `reward_extra_info`. Propagate the terminal env metrics here so trainer-side
    validation can emit them under `val-env/*`.
    """
    interaction_metrics = extra_info.get("interaction_metrics", [])
    if not isinstance(interaction_metrics, (list, tuple)):
        return {}

    for turn_info in reversed(interaction_metrics):
        if not isinstance(turn_info, dict):
            continue
        trajectory_metrics = turn_info.get("trajectory_metrics", {})
        if not isinstance(trajectory_metrics, dict) or not trajectory_metrics:
            continue
        return {
            str(key): float(value)
            for key, value in trajectory_metrics.items()
            if isinstance(value, numbers.Number) and not isinstance(value, bool)
        }
    return {}


def _extract_crop_name(env_config: dict) -> str | None:
    """Extract an optional crop label from explicit environment metadata."""
    crop_name = str(env_config.get("crop_name", "") or "").strip()
    return crop_name or None


def _extract_group_labels(env_config: dict) -> dict[str, str]:
    """Extract optional trajectory group labels from explicit env metadata."""
    raw_group_labels = env_config.get("trajectory_group_labels", {})
    if not isinstance(raw_group_labels, dict):
        return {}

    group_labels: dict[str, str] = {}
    for key, value in raw_group_labels.items():
        group_key = str(key or "").strip()
        group_value = str(value or "").strip()
        if group_key and group_value:
            group_labels[group_key] = group_value
    return group_labels


def _first_numeric(metrics: dict[str, float], keys: tuple[str, ...]) -> float:
    for key in keys:
        value = metrics.get(key)
        if isinstance(value, numbers.Number) and not isinstance(value, bool):
            value = float(value)
            if math.isfinite(value):
                return value
    return float("nan")


def _canonical_trajectory_metrics(env_config: dict, terminal_metrics: dict[str, float]) -> dict[str, float]:
    """Return simulator-portable trajectory metric aliases.

    Missing values are represented as NaN. This keeps the reward-extra schema
    rectangular without requiring changes in VERL, while validation aggregation
    skips unavailable metrics.
    """
    target_yield = _first_numeric(
        terminal_metrics,
        (
            "target_yield",
            "final_wso",
            "yield_kgha",
            "grain_yield",
            "final_grain_yield",
            "GRAIN YIELD",
        ),
    )
    return {
        "target_yield": target_yield,
        "total_n_kg_ha": _first_numeric(
            terminal_metrics,
            (
                "total_n_kg_ha",
                "total_fert",
                "total_n",
                "n_to_date",
                "cumsumfert",
            ),
        ),
        "total_irrig_mm": _first_numeric(
            terminal_metrics,
            (
                "total_irrig_mm",
                "total_irrig",
                "cumsumirrg",
                "totir",
            ),
        ),
        "invalid_action_rate": _first_numeric(terminal_metrics, ("invalid_action_rate",)),
        "total_steps": _first_numeric(terminal_metrics, ("total_steps",)),
    }


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict] = None,
    **kwargs,
) -> float | dict:
    """Aggregate per-turn rewards into trajectory and step rewards.

    The adapter-level contract is intentionally minimal:
    - environments emit a reward on every turn,
    - trajectory-style rewards are represented through that same per-turn stream,
      typically as zeros on earlier turns and a non-zero reward on the final turn,
    - the reward function only aggregates those turn rewards,
    - environment-specific reward semantics stay inside the environment itself.

    Under this contract:
    - ``score`` / ``traj_score`` = ``sum(turn_scores)``
    - ``step_reward`` = reward of the current turn
    """
    del data_source, solution_str, ground_truth, kwargs

    extra_info = extra_info or {}
    env_config = _extract_env_config(extra_info)
    turn_scores = [float(score) for score in extra_info.get("turn_scores", [])]
    num_steps = len(turn_scores)
    turn_num = _extract_turn_num(extra_info, num_steps)
    step_idx = _extract_step_idx(extra_info, num_steps)

    total_reward = sum(turn_scores) if turn_scores else 0.0
    score = total_reward
    step_reward = turn_scores[step_idx] if turn_scores else 0.0

    result = {
        "score": score,
        "traj_score": score,
        "step_reward": step_reward,
        "raw_reward": total_reward,
        "turn_num": turn_num,
        "num_steps": num_steps,
    }
    terminal_metrics = _extract_terminal_trajectory_metrics(extra_info)
    result.update(terminal_metrics)
    result.update(_canonical_trajectory_metrics(env_config, terminal_metrics))
    result["crop_name"] = _extract_crop_name(env_config) or ""
    result["scenario_id"] = str(env_config.get("scenario_id", "") or "").strip()
    group_labels = _extract_group_labels(env_config)
    if result["scenario_id"]:
        group_labels.setdefault("scenario_id", result["scenario_id"])
    env_name = str(env_config.get("env_name", "") or "").strip()
    if env_name:
        group_labels.setdefault("simulator", env_name)
    objective_id = str(env_config.get("objective_id", "") or "").strip()
    if objective_id:
        group_labels.setdefault("objective_id", objective_id)
    for group_name in _GROUP_LABEL_KEYS:
        result[f"group_label/{group_name}"] = group_labels.get(group_name, "")
    return result
