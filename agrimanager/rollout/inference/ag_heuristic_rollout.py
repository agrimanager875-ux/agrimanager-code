"""Source-backed ag-heuristic scripted rollout.

This module is intentionally separate from ``random_rollout.py`` because the
ag-heuristic baseline has a different claim boundary from random,
no-action, and PPV0 references. It evaluates a conservative agronomic reference
policy defined in ``docs/baseline_policy_definitions.md``.
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from tqdm import tqdm

from agrimanager.rollout.inference.inference_rollout import (
    _build_summary,
    _dump_json_atomic,
    _extract_trial_metrics,
    _extract_validation_metric_dict,
    load_dataset,
)
from agrimanager.rollout.inference.random_rollout import (
    _executed_action,
    _scalar_action_id,
    _seed_action_spaces,
    _zero_action,
    create_env,
    get_action_space,
)

VALID_MODES = {"no_action", "ag_heuristic"}
MODE_ALIASES = {
    "no-action": "no_action",
    "noaction": "no_action",
    "zero": "no_action",
    "zero_action": "no_action",
    "ag_heuristic": "ag_heuristic",
    "ag-heuristic": "ag_heuristic",
    "agheuristic": "ag_heuristic",
    "heuristic": "ag_heuristic",
    "expert": "ag_heuristic",
}

def _normalize_mode(mode: str) -> str:
    normalized = str(mode or "ag_heuristic").strip().lower().replace(" ", "_")
    normalized = MODE_ALIASES.get(normalized, normalized)
    if normalized not in VALID_MODES:
        valid = ", ".join(sorted(VALID_MODES | set(MODE_ALIASES)))
        raise ValueError(f"Unsupported ag-heuristic mode {mode!r}. Expected one of: {valid}")
    return normalized


def _numeric_obs_value(payload: Dict[str, Any], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        for candidate in (key, key.upper(), key.lower()):
            if candidate not in payload:
                continue
            try:
                value = float(payload[candidate])
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                return value
    return float(default)


def _config_value(env: Any, key: str, default: Any = None) -> Any:
    for source in (getattr(env, "config", None), getattr(env, "env_config", None)):
        if source is None:
            continue
        if isinstance(source, dict) and key in source:
            return source[key]
        if hasattr(source, key):
            return getattr(source, key)
    return default


def _env_kind(env: Any) -> str:
    env_name = str(_config_value(env, "env_name", "") or "").lower()
    class_name = type(env).__name__.lower()
    if "wofost" in env_name or "wofost" in class_name:
        return "wofost"
    if "dssat" in env_name or "dssat" in class_name:
        return "dssat"
    if "cycles" in env_name or "cycles" in class_name:
        return "cycles"
    return ""


def _objective_id(env: Any) -> str:
    value = _config_value(env, "objective_id")
    if value is None:
        reward_wrapper = str(_config_value(env, "env_reward", "") or "").lower()
        if "wso" in reward_wrapper or "yield" in reward_wrapper:
            value = "yield_max"
    return str(value or "profit_max").strip().lower()


def _select_cycles_action(env: Any, action_space: Any) -> Any:
    heuristic_fn = getattr(env, "_fallback_crop_planning_action", None)
    if callable(heuristic_fn):
        return heuristic_fn()

    inner_env = getattr(env, "env", None)
    heuristic_fn = getattr(inner_env, "_fallback_crop_planning_action", None)
    if callable(heuristic_fn):
        return heuristic_fn()

    return _zero_action(action_space)


def _select_dssat_expert_action(env: Any, observation: Dict[str, Any]) -> Dict[str, float]:
    fert_schedule = {40: 27.0, 45: 35.0, 80: 54.0}
    irrig_schedule = {20: 25.0, 50: 30.0, 80: 25.0}

    dap = int(round(_numeric_obs_value(observation, "dap", "DAP", default=0.0)))
    interval = max(1, int(getattr(env, "decision_interval", 1) or 1))
    applied = getattr(env, "_ag_heuristic_dssat_applied_targets", None)
    if applied is None:
        applied = set()
        setattr(env, "_ag_heuristic_dssat_applied_targets", applied)

    anfer = 0.0
    amir = 0.0
    for target, amount in fert_schedule.items():
        key = ("fert", target)
        if key not in applied and dap <= target < dap + interval:
            anfer = float(amount)
            applied.add(key)
            break

    for target, amount in irrig_schedule.items():
        key = ("irrig", target)
        if key not in applied and dap <= target < dap + interval:
            amir = float(amount)
            applied.add(key)
            break

    return {"anfer": anfer, "amir": amir, "pesticide": 0.0}


def _wofost_action_id_for_level(env: Any, kind: str, preferred_level: int) -> int:
    """Project an ag-heuristic action to the nearest available WOFOST menu level."""
    prompt = getattr(env, "prompt_generator", None)
    action_id_fn = getattr(prompt, "_action_id", None)
    if not callable(action_id_fn):
        return 0

    if kind == "irrig":
        step = float(getattr(prompt, "irrig_amount", 0.5))
        max_level = int(getattr(prompt, "num_irrig", 4))
    else:
        max_level = int(getattr(prompt, "num_fert", 4))
    if max_level <= 0:
        return 0

    level = max(1, min(max_level, int(preferred_level)))
    action_id = action_id_fn(kind, level)
    return int(action_id) if action_id is not None else 0


def _water_policy_says_irrigate(objective: str, rain_cm: float, sm: float) -> bool:
    if rain_cm >= 0.20:
        return False
    if objective == "water_stewardship":
        return sm < 0.25
    return sm < 0.31


def _ag_heuristic_caps(objective: str) -> Dict[str, float]:
    early_n_cap = 50.0
    mid_n_cap = 120.0
    late_n_cap = 160.0
    p_cap = 20.0
    k_cap = 20.0
    if objective == "profit_max":
        late_n_cap = 130.0
    elif objective == "yield_max":
        late_n_cap = 170.0
    elif objective == "nutrient_stewardship":
        early_n_cap = 35.0
        mid_n_cap = 85.0
        late_n_cap = 100.0
        p_cap = 15.0
        k_cap = 15.0
    return {
        "early_n_cap": early_n_cap,
        "mid_n_cap": mid_n_cap,
        "late_n_cap": late_n_cap,
        "p_cap": p_cap,
        "k_cap": k_cap,
    }


def select_ag_heuristic_action(env: Any, action_space: Any, observation: Dict[str, Any]) -> Any:
    env_kind = _env_kind(env)
    if env_kind == "dssat":
        return _select_dssat_expert_action(env, observation or {})
    if env_kind == "cycles":
        return _select_cycles_action(env, action_space)
    if env_kind != "wofost":
        return _zero_action(action_space)

    prompt = getattr(env, "prompt_generator", None)
    available = set(getattr(prompt, "available_action_kinds", ()) or ())
    objective = _objective_id(env)
    caps = _ag_heuristic_caps(objective)

    dvs = _numeric_obs_value(observation, "DVS", default=0.0)
    rain_cm = _numeric_obs_value(observation, "RAIN", default=0.0)
    sm = _numeric_obs_value(observation, "SM", default=1.0)
    total_n = _numeric_obs_value(observation, "TOTN", "total_n_kg_ha", default=0.0)
    total_p = _numeric_obs_value(observation, "TOTP", "total_p_kg_ha", default=0.0)
    total_k = _numeric_obs_value(observation, "TOTK", "total_k_kg_ha", default=0.0)

    if dvs < 0.20:
        return 0

    if 0.20 <= dvs < 0.45:
        if "p" in available and total_p <= 0.0 and caps["p_cap"] > 0.0:
            return _wofost_action_id_for_level(env, "p", 2)
        if "k" in available and total_k <= 0.0 and caps["k_cap"] > 0.0:
            return _wofost_action_id_for_level(env, "k", 2)
        if "n" in available and total_n < caps["early_n_cap"]:
            return _wofost_action_id_for_level(env, "n", 2)
        return 0

    if 0.45 <= dvs < 0.90:
        if "n" in available and total_n < caps["mid_n_cap"]:
            level = 3 if caps["mid_n_cap"] - total_n >= 40.0 else 2
            return _wofost_action_id_for_level(env, "n", level)
        if "irrig" in available and _water_policy_says_irrigate(objective, rain_cm, sm):
            return _wofost_action_id_for_level(env, "irrig", 2)
        return 0

    if 0.90 <= dvs < 1.20:
        if "irrig" in available and _water_policy_says_irrigate(objective, rain_cm, sm):
            return _wofost_action_id_for_level(env, "irrig", 2)
        if objective == "yield_max" and "n" in available and total_n < caps["late_n_cap"]:
            return _wofost_action_id_for_level(env, "n", 1)
        return 0

    return 0


def _select_action(mode: str, env: Any, action_space: Any, observation: Dict[str, Any]) -> Any:
    if mode == "no_action":
        return _zero_action(action_space)
    return select_ag_heuristic_action(env, action_space, observation or {})


def _run_single_trial(
    *,
    envs: List[Any],
    env_configs: List[Dict[str, Any]],
    action_spaces: List[Any],
    turn_num: int,
    mode: str,
) -> List[Dict[str, Any]]:
    results = [{"env_id": i, "env_config": env_configs[i], "turns": []} for i in range(len(envs))]

    observations = []
    done_flags = []
    for env in tqdm(envs, desc="Resetting"):
        _, info = env.reset()
        info = info or {}
        observations.append(info.get("observation") or info.get("turn_metrics") or {})
        done_flags.append(False)

    for turn in tqdm(range(turn_num), desc="Turns", leave=False):
        active_indices = [i for i, done in enumerate(done_flags) if not done]
        if not active_indices:
            break

        for i in active_indices:
            current_observation = observations[i]
            action = _select_action(mode, envs[i], action_spaces[i], current_observation or {})

            try:
                _, reward, done, info = envs[i].step(action)
                info = info or {}
                executed_action = _executed_action(info, action)
                action_id = _scalar_action_id(executed_action)
                turn_record = {
                    "turn": turn,
                    "turn_prompt": None,
                    "observation": current_observation,
                    "raw_llm_response": None,
                    "llm_reasoning": None,
                    "retries": 0,
                    "executed_action": executed_action,
                    "executed_action_id": action_id,
                    "invalid_action": bool(info.get("invalid_action", False)),
                    "reward": float(reward),
                    "done": bool(done),
                    "post_turn_metrics": info.get("turn_metrics", {}),
                }
                if done:
                    turn_record["trajectory_metrics"] = info.get("trajectory_metrics", {})
                results[i]["turns"].append(turn_record)

                if done:
                    done_flags[i] = True
                    observations[i] = None
                else:
                    observations[i] = info.get("observation") or info.get("turn_metrics") or {}
            except Exception as exc:
                results[i]["turns"].append(
                    {
                        "turn": turn,
                        "turn_prompt": None,
                        "observation": current_observation,
                        "raw_llm_response": None,
                        "llm_reasoning": None,
                        "retries": 0,
                        "executed_action": action,
                        "executed_action_id": _scalar_action_id(action),
                        "invalid_action": None,
                        "reward": None,
                        "done": True,
                        "error": str(exc),
                        "post_turn_metrics": {},
                    }
                )
                done_flags[i] = True
                observations[i] = None

        if all(done_flags):
            break

    return results


def run_ag_heuristic_rollout(
    output_dir: str,
    *,
    mode: str = "ag_heuristic",
    seed: int = 42,
    num_trials: int = 1,
    validation_axis: str | None = None,
    test_files: str | list[str] | None = None,
    env_configs: List[Dict[str, Any]] | None = None,
    env_name: str = "wofost_gym",
) -> None:
    mode = _normalize_mode(mode)
    num_trials = max(1, int(num_trials))

    if env_configs is None:
        if not test_files:
            raise ValueError("run_ag_heuristic_rollout requires test_files or env_configs.")
        env_configs, env_name = load_dataset(test_files)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    np.random.seed(seed)

    envs = []
    configs = []
    action_spaces = []
    for env_config in tqdm(env_configs, desc="Creating envs"):
        row_env_name = env_config.get("env_name", env_name)
        env, config = create_env(row_env_name, env_config)
        envs.append(env)
        configs.append(config)
        action_spaces.append(get_action_space(env))

    turn_num = max(int(config.turn_num) for config in configs)
    all_trial_metrics = []
    try:
        for trial in range(num_trials):
            trial_seed = int(seed) + trial
            random.seed(trial_seed)
            np.random.seed(trial_seed)
            _seed_action_spaces(action_spaces, trial_seed)

            results = _run_single_trial(
                envs=envs,
                env_configs=env_configs,
                action_spaces=action_spaces,
                turn_num=turn_num,
                mode=mode,
            )

            trial_dir = output_path if num_trials == 1 else output_path / f"trial_{trial}"
            trial_dir.mkdir(parents=True, exist_ok=True)
            _dump_json_atomic(trial_dir / "results.json", results, "ag heuristic results")

            trial_metrics = _extract_trial_metrics(results)
            trial_metrics["baseline_mode"] = mode
            trial_metrics["seed"] = trial_seed
            all_trial_metrics.append(trial_metrics)
            _dump_json_atomic(trial_dir / "metrics.json", trial_metrics, "ag heuristic metrics")

            validation_metrics = _extract_validation_metric_dict(results, validation_axis=validation_axis)
            _dump_json_atomic(
                trial_dir / "validation_metrics.json",
                validation_metrics,
                "training-style validation metrics",
            )

        if num_trials > 1:
            summary = _build_summary(all_trial_metrics)
            summary["baseline_mode"] = mode
            summary["seed"] = int(seed)
            _dump_json_atomic(output_path / "summary.json", summary, "cross-trial summary")
    finally:
        for env in envs:
            close = getattr(env, "close", None)
            if callable(close):
                close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run no-action or ag-heuristic WOFOST baseline.")
    parser.add_argument("--data", required=True, help="Parquet dataset file.")
    parser.add_argument("--output-dir", required=True, help="Directory for results.")
    parser.add_argument("--mode", default="ag_heuristic", help="no_action or ag_heuristic.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--validation-axis", default=None)
    args = parser.parse_args()

    run_ag_heuristic_rollout(
        output_dir=args.output_dir,
        mode=args.mode,
        seed=args.seed,
        num_trials=args.num_trials,
        validation_axis=args.validation_axis,
        test_files=args.data,
    )


if __name__ == "__main__":
    main()
