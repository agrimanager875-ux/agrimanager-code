#!/usr/bin/env python3
"""CycleGym crop-rotation mechanism sanity check.

This experiment is a gate before claiming that the CycleGym crop-planning task
tests long-horizon consequence-aware crop selection. It isolates simulator
biology by holding weather, prices, and planting week fixed, then varying only
the crop sequence.

It checks two links:
1. crop sequence -> soil nitrogen state
2. predecessor crop / soil history -> later yield for the same crop
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CROPS = ("CornRM.100", "SoybeanMG.3")
DEFAULT_PRICE_DOLLARS_PER_TONNE = 265.0
RANDOM_SEQUENCE_CODE = {
    "SoybeanMG.3": "0",
    "CornRM.100": "1",
}


@dataclass(frozen=True)
class YearWindow:
    start: int
    end: int

    @property
    def label(self) -> str:
        return f"{self.start}_{self.end}"

    @property
    def horizon(self) -> int:
        return self.end - self.start + 1


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_window(raw: str) -> YearWindow:
    parts = raw.split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Expected START:END, got {raw!r}")
    start, end = int(parts[0]), int(parts[1])
    if end < start:
        raise argparse.ArgumentTypeError(f"Window end must be >= start, got {raw!r}")
    return YearWindow(start=start, end=end)


def slug(value: str) -> str:
    cleaned = re.sub(r"[^0-9a-zA-Z]+", "_", value.strip().lower()).strip("_")
    return cleaned or "value"


def repeat_pattern(pattern: tuple[str, ...], horizon: int) -> list[str]:
    return [pattern[i % len(pattern)] for i in range(horizon)]


def designed_sequences(horizon: int) -> dict[str, list[str]]:
    corn, soybean = CROPS
    return {
        "continuous_corn": repeat_pattern((corn,), horizon),
        "continuous_soybean": repeat_pattern((soybean,), horizon),
        "corn_soy_alternating": repeat_pattern((corn, soybean), horizon),
        "soy_corn_alternating": repeat_pattern((soybean, corn), horizon),
        "corn_heavy_corn_corn_soy": repeat_pattern((corn, corn, soybean), horizon),
        "soy_heavy_soy_soy_corn": repeat_pattern((soybean, soybean, corn), horizon),
    }


def random_sequences(horizon: int, count: int, seed: int) -> dict[str, list[str]]:
    rng = np.random.default_rng(seed)
    sequences: dict[str, list[str]] = {}
    for idx in range(count):
        sequence = list(rng.choice(CROPS, size=horizon, replace=True))
        sequence_name = "".join(RANDOM_SEQUENCE_CODE[crop] for crop in sequence)
        if sequence_name in sequences:
            sequence_name = f"{sequence_name}_dup{idx:03d}"
        sequences[sequence_name] = sequence
    return sequences


def default_runtime_dir() -> Path:
    return Path(os.environ.get("TMPDIR", "/tmp")).joinpath(
        "agrimanager_cycles_gym_mechanism", str(os.getuid())
    )


def configure_cyclesgym(cycles_gym_path: Path, runtime_dir: Path) -> None:
    cycles_gym_path = cycles_gym_path.resolve()
    runtime_dir = runtime_dir.resolve()
    os.environ.setdefault("CYCLESGYM_PROJECT_PATH", str(cycles_gym_path))
    os.environ.setdefault("CYCLESGYM_BASE_CYCLES_PATH", str(cycles_gym_path.joinpath("cycles")))
    os.environ["CYCLESGYM_RUNTIME_CYCLES_PATH"] = str(runtime_dir.joinpath("cycles"))
    os.environ["CYCLESGYM_INPUT_PATH"] = str(runtime_dir.joinpath("cycles", "input"))
    os.environ["CYCLESGYM_OUTPUT_PATH"] = str(runtime_dir.joinpath("cycles", "output"))
    os.environ.setdefault("MPLCONFIGDIR", str(runtime_dir.joinpath("mplconfig")))
    os.environ.setdefault("XDG_CACHE_HOME", str(runtime_dir.joinpath("cache")))

    if str(cycles_gym_path) not in sys.path:
        sys.path.insert(0, str(cycles_gym_path))


def import_cyclesgym_modules() -> tuple[Any, Any, dict[str, str]]:
    import cyclesgym
    from cyclesgym.envs.crop_planning import CropPlanningFixedPlanting
    from cyclesgym.utils.pricing_utils import crop_type

    return cyclesgym, CropPlanningFixedPlanting, crop_type


def ensure_runtime_cycles_binary_is_writable() -> None:
    """Avoid chmod-on-symlink failures when the base Cycles binary is read-only."""
    base_binary = Path(os.environ["CYCLESGYM_BASE_CYCLES_PATH"]).joinpath("Cycles")
    runtime_binary = Path(os.environ["CYCLESGYM_RUNTIME_CYCLES_PATH"]).joinpath("Cycles")
    if runtime_binary.is_symlink():
        runtime_binary.unlink()
    if not runtime_binary.exists():
        shutil.copy2(base_binary, runtime_binary)


def soil_obs_to_record(env: Any, obs: np.ndarray) -> dict[str, float]:
    names = env.observer.obs_names or [f"soil_{idx}" for idx in range(len(obs))]
    record: dict[str, float] = {}
    for idx, (name, value) in enumerate(zip(names, obs)):
        record[f"soil_{idx:02d}_{slug(str(name))}"] = float(value)
    return record


def latest_crop_yield(
    season_df: pd.DataFrame,
    previous_row_count: int,
    crop: str,
    action_year: int,
    yield_column: str,
) -> float:
    if season_df.empty or yield_column not in season_df.columns:
        return float("nan")

    new_rows = season_df.iloc[previous_row_count:].copy()
    if "CROP" in new_rows.columns:
        new_rows = new_rows.loc[new_rows["CROP"] == crop]

    if new_rows.empty:
        candidate_rows = season_df.copy()
        if "CROP" in candidate_rows.columns:
            candidate_rows = candidate_rows.loc[candidate_rows["CROP"] == crop]
        if "PLANT_YEAR" in candidate_rows.columns:
            candidate_rows = candidate_rows.loc[candidate_rows["PLANT_YEAR"] == action_year]
        elif "YEAR" in candidate_rows.columns:
            candidate_rows = candidate_rows.loc[candidate_rows["YEAR"] == action_year]
        new_rows = candidate_rows

    if new_rows.empty:
        return float("nan")
    return float(pd.to_numeric(new_rows[yield_column], errors="coerce").sum())


def run_one_sequence(
    *,
    cyclesgym: Any,
    env_class: Any,
    crop_type: dict[str, str],
    sequence_name: str,
    sequence_type: str,
    sequence: list[str],
    window: YearWindow,
    location: str,
    planting_week_index: int,
    crop_prices: dict[str, float],
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    np.random.seed(seed)
    random.seed(seed)
    weather_class, weather_kwargs = cyclesgym.get_weather(
        window.start,
        window.end,
        random=False,
        location=location,
        sampling_start_year=window.start,
        sampling_end_year=window.end,
    )

    env = env_class(
        start_year=window.start,
        end_year=window.end,
        rotation_crops=list(CROPS),
        crop_prices=crop_prices,
        weather_generator_class=weather_class,
        weather_generator_kwargs=weather_kwargs,
    )

    crop_to_idx = {crop: idx for idx, crop in enumerate(CROPS)}
    obs = env.reset()
    initial_soil = soil_obs_to_record(env, obs)
    yearly_rows: list[dict[str, Any]] = []
    cumulative_reward = 0.0

    for step_idx, crop in enumerate(sequence):
        action_year = window.start + step_idx
        previous_crop = sequence[step_idx - 1] if step_idx > 0 else None
        previous_row_count = len(env.season_manager.season_df)
        action = [crop_to_idx[crop], planting_week_index]
        obs, reward, done, _info = env.step(action)
        cumulative_reward += float(reward)

        yield_column = crop_type[crop]
        crop_yield = latest_crop_yield(
            env.season_manager.season_df,
            previous_row_count,
            crop,
            action_year,
            yield_column,
        )
        soil_record = soil_obs_to_record(env, obs)
        row = {
            "location": location,
            "window": window.label,
            "start_year": window.start,
            "end_year": window.end,
            "sequence_name": sequence_name,
            "sequence_type": sequence_type,
            "step_idx": step_idx,
            "year": action_year,
            "crop": crop,
            "previous_crop": previous_crop,
            "planting_week_index": planting_week_index,
            "yield_column": yield_column,
            "yield_tonnes": crop_yield,
            "price_dollars_per_tonne": crop_prices[crop],
            "revenue": crop_yield * crop_prices[crop] if np.isfinite(crop_yield) else float("nan"),
            "reward": float(reward),
            "done": bool(done),
        }
        row.update(soil_record)
        yearly_rows.append(row)

    final_soil = {key: yearly_rows[-1][key] for key in initial_soil}
    soil_delta = {
        f"delta_{key}": final_soil[key] - initial_soil[key]
        for key in initial_soil
    }
    soil_delta_values = np.array(list(soil_delta.values()), dtype=float)
    summary = {
        "location": location,
        "window": window.label,
        "start_year": window.start,
        "end_year": window.end,
        "horizon_years": window.horizon,
        "sequence_name": sequence_name,
        "sequence_type": sequence_type,
        "sequence": " ".join(sequence),
        "cumulative_reward": cumulative_reward,
        "mean_reward": cumulative_reward / len(sequence),
        "soil_delta_l2": float(np.linalg.norm(soil_delta_values)),
        "initial_soil_sum": float(np.nansum(list(initial_soil.values()))),
        "final_soil_sum": float(np.nansum(list(final_soil.values()))),
        "delta_soil_sum": float(
            np.nansum(list(final_soil.values())) - np.nansum(list(initial_soil.values()))
        ),
    }
    summary.update({f"initial_{key}": value for key, value in initial_soil.items()})
    summary.update({f"final_{key}": value for key, value in final_soil.items()})
    summary.update(soil_delta)
    return summary, yearly_rows


def yield_by_predecessor(yearly_df: pd.DataFrame) -> pd.DataFrame:
    valid = yearly_df.dropna(subset=["previous_crop", "yield_tonnes"]).copy()
    if valid.empty:
        return pd.DataFrame()
    grouped = (
        valid.groupby(["crop", "previous_crop"], dropna=False)
        .agg(
            count=("yield_tonnes", "size"),
            mean_yield_tonnes=("yield_tonnes", "mean"),
            std_yield_tonnes=("yield_tonnes", "std"),
            mean_revenue=("revenue", "mean"),
            mean_reward=("reward", "mean"),
        )
        .reset_index()
        .sort_values(["crop", "previous_crop"])
    )
    return grouped


def soil_spread(summary_df: pd.DataFrame) -> pd.DataFrame:
    final_cols = [col for col in summary_df.columns if col.startswith("final_soil_")]
    rows = []
    for col in final_cols:
        values = pd.to_numeric(summary_df[col], errors="coerce")
        rows.append(
            {
                "soil_field": col.removeprefix("final_"),
                "min": float(values.min()),
                "max": float(values.max()),
                "spread": float(values.max() - values.min()),
                "std": float(values.std(ddof=0)),
            }
        )
    return pd.DataFrame(rows).sort_values("spread", ascending=False)


def pairwise_diagnostics(yield_df: pd.DataFrame) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    if yield_df.empty:
        return diagnostics

    def mean_yield(crop: str, previous_crop: str) -> float | None:
        row = yield_df.loc[
            (yield_df["crop"] == crop) & (yield_df["previous_crop"] == previous_crop)
        ]
        if row.empty:
            return None
        return float(row.iloc[0]["mean_yield_tonnes"])

    corn_after_soy = mean_yield("CornRM.100", "SoybeanMG.3")
    corn_after_corn = mean_yield("CornRM.100", "CornRM.100")
    if corn_after_soy is not None and corn_after_corn is not None:
        lift = corn_after_soy - corn_after_corn
        diagnostics["corn_yield_after_soybean_minus_after_corn"] = lift
        diagnostics["corn_yield_after_soybean_pct_lift_vs_after_corn"] = (
            100.0 * lift / abs(corn_after_corn) if corn_after_corn else float("nan")
        )

    soybean_after_corn = mean_yield("SoybeanMG.3", "CornRM.100")
    soybean_after_soy = mean_yield("SoybeanMG.3", "SoybeanMG.3")
    if soybean_after_corn is not None and soybean_after_soy is not None:
        lift = soybean_after_corn - soybean_after_soy
        diagnostics["soybean_yield_after_corn_minus_after_soybean"] = lift
        diagnostics["soybean_yield_after_corn_pct_lift_vs_after_soybean"] = (
            100.0 * lift / abs(soybean_after_soy) if soybean_after_soy else float("nan")
        )

    return diagnostics


def write_markdown_summary(
    *,
    output_path: Path,
    summary_df: pd.DataFrame,
    spread_df: pd.DataFrame,
    predecessor_df: pd.DataFrame,
    diagnostics: dict[str, Any],
) -> None:
    top_soil = spread_df.head(8).copy()
    lines = [
        "# CycleGym Mechanism Sanity Check",
        "",
        "This run holds weather, prices, and planting week fixed. Only crop sequence changes.",
        "",
        "## Sequence -> Soil",
        "",
        "Largest final-soil spreads across tested sequences:",
        "",
    ]
    if top_soil.empty:
        lines.append("No soil spread rows were produced.")
    else:
        lines.append(top_soil.to_markdown(index=False))

    lines.extend(
        [
            "",
            "Top/bottom sequences by final soil-state aggregate:",
            "",
        ]
    )
    if not summary_df.empty:
        soil_rank = summary_df.sort_values("final_soil_sum")[
            ["sequence_name", "sequence_type", "location", "window", "final_soil_sum", "delta_soil_sum", "cumulative_reward"]
        ]
        lines.append(soil_rank.head(5).to_markdown(index=False))
        lines.append("")
        lines.append(soil_rank.tail(5).to_markdown(index=False))

    lines.extend(
        [
            "",
            "## History -> Yield",
            "",
            "Mean yield grouped by current crop and previous crop:",
            "",
        ]
    )
    if predecessor_df.empty:
        lines.append("No predecessor-yield rows were produced.")
    else:
        lines.append(predecessor_df.to_markdown(index=False))

    lines.extend(["", "## Diagnostics", ""])
    if diagnostics:
        for key, value in diagnostics.items():
            lines.append(f"- `{key}`: {value}")
    else:
        lines.append("No pairwise diagnostics were available.")

    lines.extend(
        [
            "",
            "## Interpretation Rule",
            "",
            "Treat this as a mechanism gate. If designed extreme sequences clearly separate soil states, and the same crop has materially different yield after different predecessors, then CycleGym supports the multi-year rotation-consequence claim. If only revenue changes while soil/yield effects are weak, downgrade the claim to economic-context grounding under price-regime shift.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cycles-gym-path",
        type=Path,
        default=repo_root().parent.joinpath("AgriManagerExternal", "CyclesGym"),
        help="Path to the editable/external CyclesGym checkout.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root().joinpath(
            "experiments", "cycles_gym_price_regime", "results", "mechanism_sanity_check"
        ),
        help="Directory for CSV/JSON/Markdown outputs.",
    )
    parser.add_argument(
        "--runtime-dir",
        type=Path,
        default=default_runtime_dir(),
        help="Scratch directory for CycleGym runtime files and simulator outputs.",
    )
    parser.add_argument(
        "--windows",
        nargs="+",
        type=parse_window,
        default=[YearWindow(1980, 1998)],
        help="One or more fixed-weather year windows, formatted START:END.",
    )
    parser.add_argument(
        "--locations",
        nargs="+",
        default=["RockSprings"],
        help="CycleGym weather locations to test.",
    )
    parser.add_argument(
        "--random-sequences",
        type=int,
        default=16,
        help="Number of random crop sequences to include as a reference distribution.",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--planting-week-index", type=int, default=4)
    parser.add_argument(
        "--price",
        type=float,
        default=DEFAULT_PRICE_DOLLARS_PER_TONNE,
        help="Equalized crop price in dollars per tonne; isolates biology from economics.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_cyclesgym(args.cycles_gym_path, args.runtime_dir)
    cyclesgym, env_class, crop_type = import_cyclesgym_modules()
    ensure_runtime_cycles_binary_is_writable()

    crop_prices = {crop: args.price for crop in CROPS}
    all_summaries: list[dict[str, Any]] = []
    all_yearly_rows: list[dict[str, Any]] = []

    for window in args.windows:
        sequences = designed_sequences(window.horizon)
        sequences.update(random_sequences(window.horizon, args.random_sequences, args.seed))
        for location in args.locations:
            for idx, (sequence_name, sequence) in enumerate(sequences.items()):
                sequence_type = "designed" if sequence_name in designed_sequences(window.horizon) else "random"
                run_seed = args.seed + idx
                summary, yearly_rows = run_one_sequence(
                    cyclesgym=cyclesgym,
                    env_class=env_class,
                    crop_type=crop_type,
                    sequence_name=sequence_name,
                    sequence_type=sequence_type,
                    sequence=sequence,
                    window=window,
                    location=location,
                    planting_week_index=args.planting_week_index,
                    crop_prices=crop_prices,
                    seed=run_seed,
                )
                all_summaries.append(summary)
                all_yearly_rows.extend(yearly_rows)
                print(
                    f"finished {location} {window.label} {sequence_name}: "
                    f"soil_delta_l2={summary['soil_delta_l2']:.3f}, "
                    f"reward={summary['cumulative_reward']:.3f}",
                    flush=True,
                )

    summary_df = pd.DataFrame(all_summaries)
    yearly_df = pd.DataFrame(all_yearly_rows)
    predecessor_df = yield_by_predecessor(yearly_df)
    spread_df = soil_spread(summary_df)
    diagnostics = pairwise_diagnostics(predecessor_df)

    summary_csv = args.output_dir.joinpath("sequence_summary.csv")
    yearly_csv = args.output_dir.joinpath("yearly.csv")
    predecessor_csv = args.output_dir.joinpath("yield_by_predecessor.csv")
    spread_csv = args.output_dir.joinpath("soil_spread.csv")
    diagnostics_json = args.output_dir.joinpath("mechanism_summary.json")
    markdown_path = args.output_dir.joinpath("summary.md")

    summary_df.to_csv(summary_csv, index=False)
    yearly_df.to_csv(yearly_csv, index=False)
    predecessor_df.to_csv(predecessor_csv, index=False)
    spread_df.to_csv(spread_csv, index=False)
    diagnostics_json.write_text(
        json.dumps(
            {
                "crops": list(CROPS),
                "price_dollars_per_tonne": args.price,
                "planting_week_index": args.planting_week_index,
                "locations": args.locations,
                "windows": [window.label for window in args.windows],
                "random_sequences": args.random_sequences,
                "seed": args.seed,
                "runtime_dir": str(args.runtime_dir),
                "diagnostics": diagnostics,
                "largest_soil_spreads": spread_df.head(10).to_dict(orient="records"),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_markdown_summary(
        output_path=markdown_path,
        summary_df=summary_df,
        spread_df=spread_df,
        predecessor_df=predecessor_df,
        diagnostics=diagnostics,
    )

    print("")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {yearly_csv}")
    print(f"Wrote {predecessor_csv}")
    print(f"Wrote {spread_csv}")
    print(f"Wrote {diagnostics_json}")
    print(f"Wrote {markdown_path}")

    if diagnostics:
        print("Key pairwise diagnostics:")
        for key, value in diagnostics.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
