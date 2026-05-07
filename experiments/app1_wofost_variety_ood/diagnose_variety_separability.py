"""Diagnose whether WOFOST varieties produce separable outcomes.

This script replays identical action sequences under identical weather
scenarios while changing only ``agro_params.crop_variety``. It is intentionally
kept outside the train/eval entrypoints because it is an experiment gate, not a
model-training operation.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import multiprocessing as mp
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from agrimanager.env.base.utils import create_environment
from agrimanager.env.wofost_gym.create_dataset import (
    env_seed_from_scenario_id,
    scenario_id_for_row,
)
from agrimanager.env.wofost_gym.weather_pool import (
    ensure_pool,
    find_pool_meteo_cache_dir,
    load_pool,
    sample_scenarios,
)


DEFAULT_VARIETIES = [
    "wheat_1",
    "wheat_2",
    "wheat_3",
    "wheat_4",
    "wheat_5",
    "wheat_6",
    "wheat_7",
    "wheat_8",
]
DEFAULT_CANDIDATE_PAIR = ("wheat_1", "wheat_7")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay same weather/action sequences across WOFOST varieties."
    )
    parser.add_argument("--crop", default="wheat")
    parser.add_argument("--varieties", nargs="+", default=DEFAULT_VARIETIES)
    parser.add_argument("--pool", default="agrimanager/wofost-weather-pool")
    parser.add_argument("--pool-revision", default="main")
    parser.add_argument("--pool-split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--num-scenarios", type=int, default=64)
    parser.add_argument("--scenario-seed", type=int, default=42)
    parser.add_argument("--num-action-seeds", type=int, default=3)
    parser.add_argument("--action-seed-base", type=int, default=1729)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--action-space-n", type=int, default=None)
    parser.add_argument("--skip-noop", action="store_true")
    parser.add_argument("--candidate-pair", nargs=2, default=DEFAULT_CANDIDATE_PAIR)
    parser.add_argument(
        "--output-dir",
        default=(
            "experiments/app1_wofost_variety_ood/results/"
            "diagnostics/wheat_variant_separability_v1"
        ),
    )
    parser.add_argument("--wofost-gym-path", default="../AgriManagerExternal/WOFOSTGym")
    parser.add_argument("--env-id", default="lnpkw-v0")
    parser.add_argument("--turn-num", type=int, default=24)
    parser.add_argument("--intvn-interval", type=int, default=10)
    parser.add_argument("--fert-amount", type=float, default=20.0)
    parser.add_argument("--irrig-amount", type=float, default=5.0)
    parser.add_argument("--objective-id", default="profit_max")
    parser.add_argument(
        "--relative-range-pass-threshold",
        type=float,
        default=0.05,
        help="Median across-variety relative range required to pass.",
    )
    parser.add_argument(
        "--relative-range-min-threshold",
        type=float,
        default=0.02,
        help="Below this median relative range, stop the wheat variety experiment.",
    )
    parser.add_argument(
        "--coverage-threshold",
        type=float,
        default=0.75,
        help="Required fraction of random groups with >= min relative range.",
    )
    parser.add_argument(
        "--candidate-pair-threshold",
        type=float,
        default=0.05,
        help="Median paired relative difference required for the candidate pair.",
    )
    return parser.parse_args()


def action_sequence_hash(actions: Iterable[int]) -> str:
    normalized = [int(action) for action in actions]
    payload = json.dumps(normalized, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def build_action_sequences(
    *,
    turn_num: int,
    action_space_n: int,
    num_action_seeds: int,
    action_seed_base: int,
    include_noop: bool,
) -> list[dict[str, Any]]:
    sequences: list[dict[str, Any]] = []
    if include_noop:
        actions = [0] * turn_num
        sequences.append(
            {
                "policy_name": "noop",
                "action_seed": -1,
                "action_sequence_hash": action_sequence_hash(actions),
                "actions": actions,
            }
        )

    for idx in range(num_action_seeds):
        seed = int(action_seed_base + idx)
        rng = np.random.RandomState(seed)
        actions = [int(v) for v in rng.randint(0, action_space_n, size=turn_num)]
        sequences.append(
            {
                "policy_name": "random",
                "action_seed": seed,
                "action_sequence_hash": action_sequence_hash(actions),
                "actions": actions,
            }
        )
    return sequences


def build_env_config(
    *,
    scenario: dict[str, Any],
    variety: str,
    args: argparse.Namespace,
    meteo_cache_dir: Path | None,
) -> dict[str, Any]:
    crop = str(scenario["crop_name"])
    year = int(scenario["year"])
    latitude = round(float(scenario["latitude"]), 2)
    longitude = round(float(scenario["longitude"]), 2)
    scenario_id = scenario_id_for_row(
        str(args.pool_split),
        crop,
        year,
        latitude,
        longitude,
    )

    env_config: dict[str, Any] = {
        "env_id": args.env_id,
        "agro_file": f"{crop}_agro.yaml",
        "wofost_gym_path": args.wofost_gym_path,
        "llm_mode": False,
        "intvn_interval": int(args.intvn_interval),
        "turn_num": int(args.turn_num),
        "scale_action_amounts_by_interval": True,
        "fert_amount": float(args.fert_amount),
        "irrig_amount": float(args.irrig_amount),
        "objective_id": args.objective_id,
        "include_crop_traits": False,
        "crop_name": crop,
        "year": year,
        "seed": env_seed_from_scenario_id(scenario_id),
        "scenario_id": scenario_id,
        "dataset_id": f"variety_diagnostic_{crop}",
        "dataset_split": args.pool_split,
        "trajectory_group_labels": {"crop": crop, "variety": str(variety)},
        "agro_params": {
            "year": year,
            "latitude": latitude,
            "longitude": longitude,
            "crop_variety": str(variety),
        },
    }
    if meteo_cache_dir is not None:
        env_config["weather_cache_dir"] = str(meteo_cache_dir)
    return env_config


def _active_crop_variety(env: Any) -> str | None:
    try:
        unwrapped = env.env.unwrapped
        return str(unwrapped.agromanagement["CropCalendar"]["crop_variety"])
    except Exception:
        return None


def infer_action_space_n(env_config: dict[str, Any]) -> int:
    env, _ = create_environment("wofost_gym", env_config)
    try:
        action_space = getattr(getattr(env, "env", None), "action_space", None)
        if action_space is None or not hasattr(action_space, "n"):
            raise ValueError(
                "Expected a discrete WOFOST action space with attribute 'n'. "
                "Pass --action-space-n explicitly if using a custom env."
            )
        return int(action_space.n)
    finally:
        env.close()


def run_rollout(
    *,
    env_config: dict[str, Any],
    variety: str,
    actions: list[int],
) -> dict[str, Any]:
    env, _ = create_environment("wofost_gym", env_config)
    try:
        active_variety = _active_crop_variety(env)
        if active_variety is not None and active_variety != variety:
            raise RuntimeError(
                f"Expected active crop_variety={variety}, got {active_variety}."
            )

        _, info = env.reset()
        last_info = info
        total_reward = 0.0
        done = False
        steps = 0
        for action in actions:
            _, reward, done, last_info = env.step(int(action))
            total_reward += float(reward)
            steps += 1
            if done:
                break

        turn_metrics = last_info.get("turn_metrics", {})
        trajectory_metrics = last_info.get("trajectory_metrics", {})
        final_wso = float(trajectory_metrics.get("final_wso", turn_metrics.get("wso", 0.0)))
        final_dvs = float(turn_metrics.get("dvs", 0.0))
        invalid_action_rate = float(trajectory_metrics.get("invalid_action_rate", 0.0))

        return {
            "final_wso": final_wso,
            "final_dvs": final_dvs,
            "total_reward": total_reward,
            "terminated": bool(done),
            "steps": int(steps),
            "invalid_action_rate": invalid_action_rate,
            "active_variety": active_variety or variety,
        }
    finally:
        env.close()


def run_rollout_task(task: dict[str, Any]) -> dict[str, Any]:
    rollout = run_rollout(
        env_config=task["env_config"],
        variety=task["variety"],
        actions=task["actions"],
    )
    row = dict(task["row"])
    row.update(rollout)
    return row


def summarize_rollouts(rollouts: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = [
        "policy_name",
        "action_seed",
        "action_sequence_hash",
        "weather_id",
    ]
    for keys, group in rollouts.groupby(group_cols, dropna=False):
        policy_name, action_seed, action_hash, weather_id = keys
        values = group["final_wso"].astype(float)
        mean_wso = float(values.mean())
        max_idx = values.idxmax()
        min_idx = values.idxmin()
        range_wso = float(values.max() - values.min())
        rows.append(
            {
                "policy_name": policy_name,
                "action_seed": int(action_seed),
                "action_sequence_hash": action_hash,
                "weather_id": weather_id,
                "year": int(group.iloc[0]["year"]),
                "latitude": float(group.iloc[0]["latitude"]),
                "longitude": float(group.iloc[0]["longitude"]),
                "num_varieties": int(group["variety"].nunique()),
                "mean_final_wso": mean_wso,
                "min_final_wso": float(values.min()),
                "max_final_wso": float(values.max()),
                "range_final_wso": range_wso,
                "relative_range": range_wso / mean_wso if mean_wso > 0 else math.nan,
                "cv_final_wso": float(values.std(ddof=0) / mean_wso)
                if mean_wso > 0
                else math.nan,
                "max_variety": str(group.loc[max_idx, "variety"]),
                "min_variety": str(group.loc[min_idx, "variety"]),
            }
        )
    return pd.DataFrame(rows)


def build_pair_summary(rollouts: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = [
        "policy_name",
        "action_seed",
        "action_sequence_hash",
        "weather_id",
    ]
    pivot = rollouts.pivot_table(
        index=group_cols,
        columns="variety",
        values="final_wso",
        aggfunc="first",
    )
    varieties = sorted(str(v) for v in rollouts["variety"].unique())
    for policy_name in sorted(str(v) for v in rollouts["policy_name"].unique()):
        policy_pivot = pivot.loc[pivot.index.get_level_values("policy_name") == policy_name]
        for left, right in itertools.combinations(varieties, 2):
            if left not in policy_pivot.columns or right not in policy_pivot.columns:
                continue
            pair_df = policy_pivot[[left, right]].dropna()
            if pair_df.empty:
                continue
            diff = (pair_df[left] - pair_df[right]).abs().astype(float)
            pair_mean = pair_df[[left, right]].mean(axis=1).astype(float)
            rel_diff = diff / pair_mean.replace(0.0, np.nan)
            rows.append(
                {
                    "policy_name": policy_name,
                    "left_variety": left,
                    "right_variety": right,
                    "num_pairs": int(len(pair_df)),
                    "median_abs_diff": float(diff.median()),
                    "mean_abs_diff": float(diff.mean()),
                    "median_relative_diff": float(rel_diff.median()),
                    "mean_relative_diff": float(rel_diff.mean()),
                    "pct_relative_diff_ge_2pct": float((rel_diff >= 0.02).mean()),
                    "pct_relative_diff_ge_5pct": float((rel_diff >= 0.05).mean()),
                }
            )
    return pd.DataFrame(rows)


def evaluate_gate(
    *,
    summary: pd.DataFrame,
    pair_summary: pd.DataFrame,
    candidate_pair: tuple[str, str],
    relative_range_pass_threshold: float,
    relative_range_min_threshold: float,
    coverage_threshold: float,
    candidate_pair_threshold: float,
) -> dict[str, Any]:
    random_summary = summary[summary["policy_name"] == "random"]
    if random_summary.empty:
        raise ValueError("Gate evaluation requires at least one random action sequence.")

    median_relative_range = float(random_summary["relative_range"].median())
    coverage_ge_min = float(
        (random_summary["relative_range"] >= relative_range_min_threshold).mean()
    )
    left, right = candidate_pair
    pair_rows = pair_summary[
        (pair_summary["policy_name"] == "random")
        & (
            (
                (pair_summary["left_variety"] == left)
                & (pair_summary["right_variety"] == right)
            )
            | (
                (pair_summary["left_variety"] == right)
                & (pair_summary["right_variety"] == left)
            )
        )
    ]
    candidate_pair_median_relative_diff = (
        float(pair_rows.iloc[0]["median_relative_diff"]) if not pair_rows.empty else math.nan
    )

    passed = (
        median_relative_range >= relative_range_pass_threshold
        and coverage_ge_min >= coverage_threshold
        and candidate_pair_median_relative_diff >= candidate_pair_threshold
    )
    if median_relative_range < relative_range_min_threshold:
        decision = "stop"
    elif passed:
        decision = "pass"
    else:
        decision = "reconsider"

    return {
        "decision": decision,
        "passed": bool(passed),
        "median_relative_range": median_relative_range,
        "coverage_ge_min_relative_range": coverage_ge_min,
        "candidate_pair": [left, right],
        "candidate_pair_median_relative_diff": candidate_pair_median_relative_diff,
        "thresholds": {
            "relative_range_pass_threshold": relative_range_pass_threshold,
            "relative_range_min_threshold": relative_range_min_threshold,
            "coverage_threshold": coverage_threshold,
            "candidate_pair_threshold": candidate_pair_threshold,
        },
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pool_dir = ensure_pool(args.pool, revision=args.pool_revision)
    split_dir = pool_dir / args.pool_split
    pool = load_pool(split_dir)
    scenarios = sample_scenarios(
        pool=pool,
        crops=[args.crop],
        num_samples=args.num_scenarios,
        seed=args.scenario_seed,
    )
    meteo_cache_dir = find_pool_meteo_cache_dir(pool_dir)

    probe_config = build_env_config(
        scenario=scenarios[0],
        variety=args.varieties[0],
        args=args,
        meteo_cache_dir=meteo_cache_dir,
    )
    action_space_n = args.action_space_n or infer_action_space_n(probe_config)
    action_sequences = build_action_sequences(
        turn_num=args.turn_num,
        action_space_n=action_space_n,
        num_action_seeds=args.num_action_seeds,
        action_seed_base=args.action_seed_base,
        include_noop=not args.skip_noop,
    )

    _write_json(output_dir / "action_sequences.json", action_sequences)
    _write_json(
        output_dir / "run_config.json",
        {
            "crop": args.crop,
            "varieties": list(args.varieties),
            "pool": args.pool,
            "pool_revision": args.pool_revision,
            "pool_split": args.pool_split,
            "num_scenarios": args.num_scenarios,
            "scenario_seed": args.scenario_seed,
            "num_action_seeds": args.num_action_seeds,
            "action_seed_base": args.action_seed_base,
            "action_space_n": action_space_n,
            "include_noop": not args.skip_noop,
            "env_id": args.env_id,
            "turn_num": args.turn_num,
            "intvn_interval": args.intvn_interval,
            "fert_amount": args.fert_amount,
            "irrig_amount": args.irrig_amount,
            "objective_id": args.objective_id,
        },
    )

    tasks: list[dict[str, Any]] = []
    total = len(scenarios) * len(action_sequences) * len(args.varieties)
    for scenario_index, scenario in enumerate(scenarios):
        weather_id = scenario_id_for_row(
            args.pool_split,
            args.crop,
            int(scenario["year"]),
            round(float(scenario["latitude"]), 2),
            round(float(scenario["longitude"]), 2),
        )
        for sequence in action_sequences:
            for variety in args.varieties:
                env_config = build_env_config(
                    scenario=scenario,
                    variety=variety,
                    args=args,
                    meteo_cache_dir=meteo_cache_dir,
                )
                tasks.append(
                    {
                        "env_config": env_config,
                        "variety": str(variety),
                        "actions": list(sequence["actions"]),
                        "row": {
                            "weather_id": weather_id,
                            "scenario_index": int(scenario_index),
                            "crop": args.crop,
                            "year": int(scenario["year"]),
                            "latitude": round(float(scenario["latitude"]), 2),
                            "longitude": round(float(scenario["longitude"]), 2),
                            "policy_name": sequence["policy_name"],
                            "action_seed": int(sequence["action_seed"]),
                            "action_sequence_hash": sequence["action_sequence_hash"],
                            "variety": str(variety),
                        },
                    }
                )

    rows: list[dict[str, Any]] = []
    completed = 0
    num_workers = max(1, int(args.num_workers))
    if num_workers == 1:
        for task in tasks:
            rows.append(run_rollout_task(task))
            completed += 1
            if completed % 25 == 0 or completed == total:
                print(f"Completed {completed}/{total} rollouts", flush=True)
    else:
        print(f"Running rollouts with {num_workers} workers", flush=True)
        with mp.Pool(processes=num_workers) as pool:
            for row in pool.imap_unordered(run_rollout_task, tasks, chunksize=1):
                rows.append(row)
                completed += 1
                if completed % 25 == 0 or completed == total:
                    print(f"Completed {completed}/{total} rollouts", flush=True)

    rollouts = pd.DataFrame(rows)
    summary = summarize_rollouts(rollouts)
    pair_summary = build_pair_summary(rollouts)
    gate = evaluate_gate(
        summary=summary,
        pair_summary=pair_summary,
        candidate_pair=tuple(str(v) for v in args.candidate_pair),
        relative_range_pass_threshold=float(args.relative_range_pass_threshold),
        relative_range_min_threshold=float(args.relative_range_min_threshold),
        coverage_threshold=float(args.coverage_threshold),
        candidate_pair_threshold=float(args.candidate_pair_threshold),
    )

    rollouts.to_csv(output_dir / "rollouts.csv", index=False)
    summary.to_csv(output_dir / "summary_by_weather_action.csv", index=False)
    pair_summary.to_csv(output_dir / "pair_summary.csv", index=False)
    _write_json(output_dir / "gate_decision.json", gate)

    print("\nGate decision:")
    print(json.dumps(gate, indent=2, sort_keys=True))
    print(f"\nWrote diagnostic outputs to: {output_dir}")


if __name__ == "__main__":
    main()
