#!/usr/bin/env python3
"""Paired price-regime heuristic baselines for CycleGym crop planning.

This script implements the non-model part of the CycleGym Price-Regime OOD
experiment: fixed rotation, price-greedy, rotation-aware, and soil-aware
heuristic policies compared on the same exogenous scenarios while only the
price vector changes.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from run_mechanism_sanity_check import (
    CROPS,
    YearWindow,
    configure_cyclesgym,
    default_runtime_dir,
    ensure_runtime_cycles_binary_is_writable,
    import_cyclesgym_modules,
    latest_crop_yield,
    parse_window,
    repo_root,
    soil_obs_to_record,
)


DEFAULT_PLANTING_WEEK_INDEX = 4
CORN = "CornRM.100"
SOYBEAN = "SoybeanMG.3"
CEREALS = {CORN}
ID_CORN_PRICE = 250.0
ID_SOYBEAN_PRICE = 300.0
HIGH_CORN_PRICE = 430.0
HIGH_SOYBEAN_PRICE = 470.0
CORN_DOMINANCE_RATIO = 1.35
SOYBEAN_DOMINANCE_RATIO = 1.45


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    location: str
    window: YearWindow


@dataclass
class PolicyState:
    history: list[str]


def price_regimes(
    *,
    corn_price: float = ID_CORN_PRICE,
    soybean_price: float = ID_SOYBEAN_PRICE,
    high_corn_price: float = HIGH_CORN_PRICE,
    high_soybean_price: float = HIGH_SOYBEAN_PRICE,
) -> dict[str, dict[str, float]]:
    return {
        "id_balanced": {CORN: corn_price, SOYBEAN: soybean_price},
        "high_corn": {CORN: high_corn_price, SOYBEAN: soybean_price},
        "high_soybean": {CORN: corn_price, SOYBEAN: high_soybean_price},
    }


def obs_dict(env: Any, obs: np.ndarray) -> dict[str, float]:
    names = getattr(getattr(env, "observer", None), "obs_names", None) or []
    values: dict[str, float] = {}
    for name, value in zip(names, obs):
        try:
            values[str(name)] = float(value)
        except Exception:
            continue
    return values


def highest_price_crop(prices: dict[str, float], candidates: list[str] | None = None) -> str:
    crop_pool = candidates or list(CROPS)
    return max(crop_pool, key=lambda crop: (prices[crop], -CROPS.index(crop)))


def other_crop(crop: str) -> str:
    return SOYBEAN if crop == CORN else CORN


def price_ratio(prices: dict[str, float], crop: str) -> float:
    other = other_crop(crop)
    denominator = max(float(prices[other]), 1e-9)
    return float(prices[crop]) / denominator


def trailing_count(history: list[str], crop: str) -> int:
    count = 0
    for previous in reversed(history):
        if previous != crop:
            break
        count += 1
    return count


def fixed_rotation_policy(
    *,
    year_idx: int,
    prices: dict[str, float],
    obs_values: dict[str, float],
    state: PolicyState,
) -> str:
    del prices, obs_values, state
    return [CORN, SOYBEAN][year_idx % 2]


def price_greedy_policy(
    *,
    year_idx: int,
    prices: dict[str, float],
    obs_values: dict[str, float],
    state: PolicyState,
) -> str:
    del year_idx, obs_values, state
    return highest_price_crop(prices)


def rotation_aware_policy(
    *,
    year_idx: int,
    prices: dict[str, float],
    obs_values: dict[str, float],
    state: PolicyState,
) -> str:
    del year_idx, obs_values
    top_crop = highest_price_crop(prices)
    top_ratio = price_ratio(prices, top_crop)
    if not state.history:
        return top_crop

    if state.history[-1] != top_crop:
        return top_crop

    if top_crop == CORN and top_ratio >= CORN_DOMINANCE_RATIO:
        return CORN if trailing_count(state.history, CORN) < 2 else SOYBEAN
    if top_crop == SOYBEAN and top_ratio >= SOYBEAN_DOMINANCE_RATIO:
        return SOYBEAN
    return other_crop(top_crop)


def soil_aware_policy(
    *,
    year_idx: int,
    prices: dict[str, float],
    obs_values: dict[str, float],
    state: PolicyState,
) -> str:
    del year_idx
    top_crop = highest_price_crop(prices)
    top_ratio = price_ratio(prices, top_crop)
    profile_no3 = obs_values.get("PROF SOIL NO3", float("nan"))

    if not state.history:
        return top_crop

    if top_crop == CORN:
        if math.isfinite(profile_no3) and profile_no3 < 25.0:
            return SOYBEAN
        if trailing_count(state.history, CORN) >= 2:
            return SOYBEAN
        if state.history[-1] == CORN and top_ratio < CORN_DOMINANCE_RATIO:
            return SOYBEAN
        return CORN

    if state.history[-1] == SOYBEAN and top_ratio < SOYBEAN_DOMINANCE_RATIO:
        return CORN
    return SOYBEAN

POLICIES: dict[str, Callable[..., str]] = {
    "fixed_rotation": fixed_rotation_policy,
    "price_greedy": price_greedy_policy,
    "rotation_aware": rotation_aware_policy,
    "soil_aware": soil_aware_policy,
}


def shannon_diversity(crops: list[str]) -> float:
    if not crops:
        return 0.0
    counts = pd.Series(crops).value_counts()
    probs = counts / counts.sum()
    return float(-(probs * np.log(probs)).sum())


def max_consecutive(crops: list[str], crop_set: set[str] | None = None) -> int:
    best = 0
    current = 0
    previous: str | None = None
    for crop in crops:
        marker = "selected" if crop_set and crop in crop_set else crop
        if crop_set and crop not in crop_set:
            current = 0
            previous = None
            continue
        if marker == previous:
            current += 1
        else:
            current = 1
            previous = marker
        best = max(best, current)
    return best


def run_policy_episode(
    *,
    cyclesgym: Any,
    env_class: Any,
    crop_type: dict[str, str],
    scenario: Scenario,
    regime_name: str,
    prices: dict[str, float],
    policy_name: str,
    policy_fn: Callable[..., str],
    planting_week_index: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    np.random.seed(seed)
    random.seed(seed)
    weather_class, weather_kwargs = cyclesgym.get_weather(
        scenario.window.start,
        scenario.window.end,
        random=False,
        location=scenario.location,
        sampling_start_year=scenario.window.start,
        sampling_end_year=scenario.window.end,
    )
    env = env_class(
        start_year=scenario.window.start,
        end_year=scenario.window.end,
        rotation_crops=list(CROPS),
        crop_prices=prices,
        weather_generator_class=weather_class,
        weather_generator_kwargs=weather_kwargs,
    )
    crop_to_idx = {crop: idx for idx, crop in enumerate(CROPS)}
    obs = env.reset()
    state = PolicyState(history=[])
    yearly_rows: list[dict[str, Any]] = []
    cumulative_revenue = 0.0

    for year_idx, year in enumerate(range(scenario.window.start, scenario.window.end + 1)):
        obs_values = obs_dict(env, obs)
        crop = policy_fn(
            year_idx=year_idx,
            prices=prices,
            obs_values=obs_values,
            state=state,
        )
        previous_row_count = len(env.season_manager.season_df)
        action = [crop_to_idx[crop], planting_week_index]
        obs, reward, done, _info = env.step(action)
        yield_column = crop_type[crop]
        crop_yield = latest_crop_yield(
            env.season_manager.season_df,
            previous_row_count,
            crop,
            year,
            yield_column,
        )
        revenue = crop_yield * prices[crop] if np.isfinite(crop_yield) else float("nan")
        cumulative_revenue += float(reward)
        soil_record = soil_obs_to_record(env, obs)
        yearly_row = {
            "scenario_id": scenario.scenario_id,
            "location": scenario.location,
            "window": scenario.window.label,
            "year": year,
            "year_idx": year_idx,
            "regime": regime_name,
            "policy": policy_name,
            "crop": crop,
            "previous_crop": state.history[-1] if state.history else None,
            "planting_week_index": planting_week_index,
            "yield_tonnes": crop_yield,
            "price_dollars_per_tonne": prices[crop],
            "revenue": revenue,
            "reward": float(reward),
            "done": bool(done),
        }
        yearly_row.update({f"price_{crop}": price for crop, price in prices.items()})
        yearly_row.update(soil_record)
        yearly_rows.append(yearly_row)
        state.history.append(crop)

    crop_counts = pd.Series(state.history).value_counts().to_dict()
    final_soil = {
        key: yearly_rows[-1][key]
        for key in yearly_rows[-1]
        if key.startswith("soil_")
    }
    late_years = yearly_rows[-5:]
    summary = {
        "scenario_id": scenario.scenario_id,
        "location": scenario.location,
        "window": scenario.window.label,
        "regime": regime_name,
        "policy": policy_name,
        "cumulative_gross_revenue": cumulative_revenue,
        "mean_annual_revenue": cumulative_revenue / len(state.history),
        "late_year_mean_revenue": float(np.nanmean([row["revenue"] for row in late_years])),
        "late_year_mean_yield": float(np.nanmean([row["yield_tonnes"] for row in late_years])),
        "rotation_diversity_shannon": shannon_diversity(state.history),
        "unique_crop_count": len(set(state.history)),
        "max_consecutive_same_crop": max_consecutive(state.history),
        "max_consecutive_cereal": max_consecutive(state.history, CEREALS),
        "crop_sequence": " ".join(state.history),
        "corn_count": int(crop_counts.get(CORN, 0)),
        "soybean_count": int(crop_counts.get(SOYBEAN, 0)),
        "corn_frequency": float(crop_counts.get(CORN, 0) / len(state.history)),
        "soybean_frequency": float(crop_counts.get(SOYBEAN, 0) / len(state.history)),
    }
    summary.update({f"price_{crop}": price for crop, price in prices.items()})
    summary.update({f"final_{key}": value for key, value in final_soil.items()})
    return summary, yearly_rows


def build_scenarios(locations: list[str], windows: list[YearWindow]) -> list[Scenario]:
    scenarios: list[Scenario] = []
    for location in locations:
        for window in windows:
            scenarios.append(
                Scenario(
                    scenario_id=f"{location}_{window.label}",
                    location=location,
                    window=window,
                )
            )
    return scenarios


def aggregate_crop_frequency(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (regime, policy), group in summary_df.groupby(["regime", "policy"]):
        for crop, column in [
            (CORN, "corn_frequency"),
            (SOYBEAN, "soybean_frequency"),
        ]:
            rows.append(
                {
                    "regime": regime,
                    "policy": policy,
                    "crop": crop,
                    "mean_frequency": float(group[column].mean()),
                    "std_frequency": float(group[column].std(ddof=0)),
                }
            )
    return pd.DataFrame(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cycles-gym-path",
        type=Path,
        default=repo_root().parent.joinpath("AgriManagerExternal", "CyclesGym"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root().joinpath(
            "experiments", "cycles_gym_price_regime", "results", "price_regime_heuristics"
        ),
    )
    parser.add_argument("--runtime-dir", type=Path, default=default_runtime_dir())
    parser.add_argument("--locations", nargs="+", default=["RockSprings", "NewHolland"])
    parser.add_argument("--windows", nargs="+", type=parse_window, default=[YearWindow(1980, 1998)])
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--corn-price", type=float, default=ID_CORN_PRICE)
    parser.add_argument("--soybean-price", type=float, default=ID_SOYBEAN_PRICE)
    parser.add_argument("--high-corn-price", type=float, default=HIGH_CORN_PRICE)
    parser.add_argument("--high-soybean-price", type=float, default=HIGH_SOYBEAN_PRICE)
    parser.add_argument("--planting-week-index", type=int, default=DEFAULT_PLANTING_WEEK_INDEX)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_cyclesgym(args.cycles_gym_path, args.runtime_dir)
    cyclesgym, env_class, crop_type = import_cyclesgym_modules()
    ensure_runtime_cycles_binary_is_writable()

    scenarios = build_scenarios(args.locations, args.windows)
    regimes = price_regimes(
        corn_price=args.corn_price,
        soybean_price=args.soybean_price,
        high_corn_price=args.high_corn_price,
        high_soybean_price=args.high_soybean_price,
    )
    all_summaries: list[dict[str, Any]] = []
    all_yearly_rows: list[dict[str, Any]] = []

    run_idx = 0
    for scenario in scenarios:
        for regime_name, prices in regimes.items():
            for policy_name, policy_fn in POLICIES.items():
                summary, yearly_rows = run_policy_episode(
                    cyclesgym=cyclesgym,
                    env_class=env_class,
                    crop_type=crop_type,
                    scenario=scenario,
                    regime_name=regime_name,
                    prices=prices,
                    policy_name=policy_name,
                    policy_fn=policy_fn,
                    planting_week_index=args.planting_week_index,
                    seed=args.seed + run_idx,
                )
                all_summaries.append(summary)
                all_yearly_rows.extend(yearly_rows)
                run_idx += 1
                print(
                    f"finished {scenario.scenario_id} {regime_name} {policy_name}: "
                    f"revenue={summary['cumulative_gross_revenue']:.2f}, "
                    f"sequence={summary['crop_sequence']}",
                    flush=True,
                )

    summary_df = pd.DataFrame(all_summaries)
    yearly_df = pd.DataFrame(all_yearly_rows)
    crop_frequency_df = aggregate_crop_frequency(summary_df)

    summary_csv = args.output_dir.joinpath("summary.csv")
    yearly_csv = args.output_dir.joinpath("yearly.csv")
    crop_frequency_csv = args.output_dir.joinpath("crop_frequency.csv")
    metadata_json = args.output_dir.joinpath("metadata.json")

    summary_df.to_csv(summary_csv, index=False)
    yearly_df.to_csv(yearly_csv, index=False)
    crop_frequency_df.to_csv(crop_frequency_csv, index=False)
    metadata_json.write_text(
        json.dumps(
            {
                "crops": list(CROPS),
                "corn_price": args.corn_price,
                "soybean_price": args.soybean_price,
                "high_corn_price": args.high_corn_price,
                "high_soybean_price": args.high_soybean_price,
                "regimes": regimes,
                "policies": list(POLICIES),
                "locations": args.locations,
                "windows": [window.label for window in args.windows],
                "planting_week_index": args.planting_week_index,
                "runtime_dir": str(args.runtime_dir),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {yearly_csv}")
    print(f"Wrote {crop_frequency_csv}")
    print(f"Wrote {metadata_json}")


if __name__ == "__main__":
    main()
