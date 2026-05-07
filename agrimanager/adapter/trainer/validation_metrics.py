"""Helpers for environment metric aggregation."""

from __future__ import annotations

import numbers
from collections import defaultdict
from typing import Any

import numpy as np


_RESERVED_TRAJECTORY_INFO_KEYS = {
    "score",
    "traj_score",
    "step_reward",
    "raw_reward",
    "turn_num",
    "num_steps",
    "step_idx",
    "step_num",
    "is_last_step",
    "crop_name",
    "scenario_id",
    "index",
}
_GROUP_LABEL_PREFIX = "group_label/"


def _to_value_list(values: Any) -> list[Any]:
    """Normalize list-like containers used in env logging into Python lists."""
    if values is None:
        return []
    if isinstance(values, np.ndarray):
        return values.reshape(-1).tolist()
    if isinstance(values, (list, tuple)):
        return list(values)
    return [values]


def _collect_numeric_trajectory_metric_keys(env_infos: dict[str, Any]) -> list[str]:
    """Return env metric keys that look like numeric trajectory-level signals."""
    trajectory_metric_keys: list[str] = []
    for key, raw_values in env_infos.items():
        values = _to_value_list(raw_values)
        if (
            key in _RESERVED_TRAJECTORY_INFO_KEYS
            or key.startswith(_GROUP_LABEL_PREFIX)
            or key.startswith("__")
            or not values
        ):
            continue
        if all(isinstance(value, numbers.Number) and not isinstance(value, bool) for value in values):
            trajectory_metric_keys.append(key)
    return sorted(trajectory_metric_keys)


def _finite_float_array(values: Any) -> np.ndarray:
    arr = np.asarray(_to_value_list(values), dtype=float)
    return arr[np.isfinite(arr)]


def filter_finite_validation_infos(infos_dict: dict[str, Any]) -> dict[str, Any]:
    """Keep only values that VERL validation metric aggregation can average safely.

    Environment metrics with NaN placeholders are still logged through
    ``add_validation_env_metrics``. This filter prevents those placeholders
    from turning duplicate ``val-aux`` aggregates into NaN through VERL's
    generic ``process_validation_metrics`` helper.
    """
    filtered: dict[str, Any] = {}
    for key, raw_values in infos_dict.items():
        values = _to_value_list(raw_values)
        if not values:
            continue
        if all(isinstance(value, str) for value in values):
            filtered[key] = raw_values
            continue
        if all(isinstance(value, numbers.Number) and not isinstance(value, bool) for value in values):
            arr = np.asarray(values, dtype=float)
            if np.all(np.isfinite(arr)):
                filtered[key] = raw_values
    return filtered


def add_env_metrics(
    metric_dict: dict[str, float],
    env_infos: dict[str, Any],
    prefix: str,
    include_grouped: bool = False,
) -> None:
    """Add numeric trajectory-level env metrics with a configurable prefix."""
    trajectory_metric_keys = _collect_numeric_trajectory_metric_keys(env_infos)

    for key in trajectory_metric_keys:
        arr = _finite_float_array(env_infos[key])
        if arr.size == 0:
            continue
        metric_dict[f"{prefix}/{key}/mean"] = float(np.mean(arr))

    if not include_grouped:
        return

    group_label_items = {
        key[len(_GROUP_LABEL_PREFIX):]: _to_value_list(values)
        for key, values in env_infos.items()
        if key.startswith(_GROUP_LABEL_PREFIX) and _to_value_list(values)
    }
    for group_name, group_values in sorted(group_label_items.items()):
        if not group_name:
            continue
        for key in trajectory_metric_keys:
            values = _to_value_list(env_infos.get(key, []))
            if not values or len(values) != len(group_values):
                continue
            grouped_values: dict[str, list[float]] = defaultdict(list)
            for group_value, value in zip(group_values, values):
                group_value_str = str(group_value or "").strip()
                if not group_value_str:
                    continue
                value_float = float(value)
                if np.isfinite(value_float):
                    grouped_values[group_value_str].append(value_float)
            for group_value, group_metric_vals in sorted(grouped_values.items()):
                if group_metric_vals:
                    metric_dict[f"{prefix}-by-{group_name}/{group_value}/{key}/mean"] = float(
                        np.mean(group_metric_vals)
                    )


def add_axis_env_metrics(
    metric_dict: dict[str, float],
    env_infos: dict[str, Any],
    *,
    axis: str,
    prefix_base: str = "val-env",
) -> bool:
    """Add env metrics for one explicit validation axis.

    This avoids logging every available group label as a tracker namespace.  The
    configured axis is the experiment's primary OOD dimension, such as
    ``weather_regime``, ``crop_regime``, ``reward_formulation``, or
    ``simulator``.
    """
    axis_name = str(axis or "").strip()
    if not axis_name:
        return False

    values = _to_value_list(env_infos.get(f"{_GROUP_LABEL_PREFIX}{axis_name}"))
    add_env_metrics(
        metric_dict,
        env_infos,
        prefix=f"{prefix_base}-{axis_name}/all",
        include_grouped=False,
    )
    if not values:
        return False

    trajectory_metric_keys = _collect_numeric_trajectory_metric_keys(env_infos)
    if not trajectory_metric_keys:
        return False

    emitted = False
    for axis_value in sorted({str(value or "").strip() for value in values if str(value or "").strip()}):
        indices = [idx for idx, value in enumerate(values) if str(value or "").strip() == axis_value]
        if not indices:
            continue
        subset_infos = {
            key: [_to_value_list(raw_values)[idx] for idx in indices]
            for key, raw_values in env_infos.items()
            if len(_to_value_list(raw_values)) > max(indices)
        }
        add_env_metrics(
            metric_dict,
            subset_infos,
            prefix=f"{prefix_base}-{axis_name}/{axis_value}",
            include_grouped=False,
        )
        emitted = True
    return emitted


def add_validation_env_metrics(metric_dict: dict[str, float], env_infos: dict[str, list[Any]]) -> None:
    """Add validation-time trajectory-level env metrics and grouped aggregates."""
    add_env_metrics(metric_dict, env_infos, prefix="val-env", include_grouped=True)


def add_training_env_metrics(metric_dict: dict[str, float], env_infos: dict[str, Any]) -> None:
    """Add training-time trajectory-level env metrics and grouped aggregates."""
    add_env_metrics(metric_dict, env_infos, prefix="train-env", include_grouped=True)
