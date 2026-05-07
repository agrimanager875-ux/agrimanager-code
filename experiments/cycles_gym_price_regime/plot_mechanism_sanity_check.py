#!/usr/bin/env python3
"""Plot CycleGym mechanism sanity-check outputs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = str(Path(os.environ.get("TMPDIR", "/tmp")).joinpath("matplotlib"))
if "XDG_CACHE_HOME" not in os.environ:
    os.environ["XDG_CACHE_HOME"] = str(Path(os.environ.get("TMPDIR", "/tmp")).joinpath("fontconfig"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


CROP_COLORS = {
    "CornRM.100": "#c17d11",
    "SoybeanMG.3": "#2f8f4e",
}
DESIGNED_COLOR = "#164e63"
RANDOM_COLOR = "#b7b7b7"
SEQUENCE_KEY_TEXT = "Random sequence key: 0 = SoybeanMG.3, 1 = CornRM.100"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_results_dir() -> Path:
    return repo_root().joinpath(
        "experiments", "cycles_gym_price_regime", "results", "mechanism_sanity_check"
    )


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Wrote {path}")


def add_sequence_key(fig: plt.Figure, *, y: float = 0.01) -> None:
    fig.text(
        0.5,
        y,
        SEQUENCE_KEY_TEXT,
        ha="center",
        va="bottom",
        fontsize=9,
        color="#333333",
    )


def display_name(name: str) -> str:
    return name.replace("_", " ")


def ordered_sequences(summary_df: pd.DataFrame) -> list[str]:
    ordered = summary_df.sort_values("final_soil_sum")["sequence_name"].tolist()
    return ordered


def plot_sequence_heatmap(yearly_df: pd.DataFrame, summary_df: pd.DataFrame, figures_dir: Path) -> None:
    order = ordered_sequences(summary_df)
    crop_to_idx = {crop: idx for idx, crop in enumerate(CROP_COLORS)}
    matrix = []
    for seq in order:
        seq_df = yearly_df.loc[yearly_df["sequence_name"] == seq].sort_values("year")
        matrix.append([crop_to_idx[crop] for crop in seq_df["crop"]])
    values = np.asarray(matrix)
    years = sorted(yearly_df["year"].unique())

    fig_height = max(7, 0.34 * len(order))
    fig, ax = plt.subplots(figsize=(13, fig_height))
    cmap = ListedColormap([CROP_COLORS[crop] for crop in crop_to_idx])
    ax.imshow(
        values,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=-0.5,
        vmax=len(CROP_COLORS) - 0.5,
    )
    ax.set_title("Crop sequences sorted by final soil nitrogen state", fontsize=15, weight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Sequence")
    ax.set_xticks(np.arange(len(years)))
    ax.set_xticklabels(years, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([display_name(seq) for seq in order], fontsize=8)
    ax.set_xticks(np.arange(-0.5, len(years), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(order), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)
    legend_handles = [Patch(facecolor=color, edgecolor="none", label=crop) for crop, color in CROP_COLORS.items()]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=len(CROP_COLORS),
        frameon=False,
    )
    add_sequence_key(fig, y=0.005)
    savefig(figures_dir.joinpath("sequence_crop_heatmap.png"))


def plot_soil_strength(summary_df: pd.DataFrame, figures_dir: Path) -> None:
    df = summary_df.sort_values("delta_soil_sum")
    y = np.arange(len(df))
    colors = [DESIGNED_COLOR if t == "designed" else RANDOM_COLOR for t in df["sequence_type"]]

    fig, axes = plt.subplots(1, 2, figsize=(15, max(7, 0.34 * len(df))), sharey=True)
    axes[0].barh(y, df["delta_soil_sum"], color=colors)
    axes[0].axvline(0, color="#333333", linewidth=0.9)
    axes[0].set_title("Net change in soil state", weight="bold")
    axes[0].set_xlabel("Final soil sum - initial soil sum")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([display_name(seq) for seq in df["sequence_name"]], fontsize=8)

    axes[1].barh(y, df["soil_delta_l2"], color=colors)
    axes[1].set_title("Magnitude of soil movement", weight="bold")
    axes[1].set_xlabel("L2 distance from initial soil vector")
    axes[1].tick_params(axis="y", labelleft=False)

    handles = [
        Patch(facecolor=DESIGNED_COLOR, label="Designed sequence"),
        Patch(facecolor=RANDOM_COLOR, label="Random sequence"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.55, -0.02))
    fig.suptitle("Rotation strength across all tested sequences", fontsize=16, weight="bold", y=0.995)
    add_sequence_key(fig, y=0.035)
    savefig(figures_dir.joinpath("soil_strength_by_sequence.png"))


def plot_soil_trajectories(yearly_df: pd.DataFrame, summary_df: pd.DataFrame, figures_dir: Path) -> None:
    designed = summary_df.loc[summary_df["sequence_type"] == "designed", "sequence_name"].tolist()
    random_sequences = summary_df.loc[summary_df["sequence_type"] == "random", "sequence_name"].tolist()
    fields = [
        ("soil_00_org_soil_n", "Organic soil N"),
        ("soil_01_prof_soil_no3", "Profile soil NO3"),
    ]
    designed_colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(designed))))

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharex=True)
    for ax, (field, label) in zip(axes, fields):
        for seq in random_sequences:
            seq_df = yearly_df.loc[yearly_df["sequence_name"] == seq].sort_values("year")
            ax.plot(seq_df["year"], seq_df[field], color=RANDOM_COLOR, alpha=0.35, linewidth=1)
        for idx, seq in enumerate(designed):
            seq_df = yearly_df.loc[yearly_df["sequence_name"] == seq].sort_values("year")
            ax.plot(
                seq_df["year"],
                seq_df[field],
                color=designed_colors[idx],
                linewidth=2.2,
                label=display_name(seq),
            )
        ax.set_title(label, weight="bold")
        ax.set_xlabel("Year")
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)

    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8)
    fig.suptitle("Soil trajectories under fixed crop sequences", fontsize=16, weight="bold")
    add_sequence_key(fig, y=0.005)
    savefig(figures_dir.joinpath("soil_trajectories.png"))


def plot_predecessor_yield(predecessor_df: pd.DataFrame, figures_dir: Path) -> None:
    crops = list(CROP_COLORS)
    predecessors = list(CROP_COLORS)
    x = np.arange(len(crops))
    width = 0.24

    fig, ax = plt.subplots(figsize=(10.5, 6))
    for idx, predecessor in enumerate(predecessors):
        means = []
        errors = []
        for crop in crops:
            row = predecessor_df.loc[
                (predecessor_df["crop"] == crop)
                & (predecessor_df["previous_crop"] == predecessor)
            ]
            if row.empty:
                means.append(np.nan)
                errors.append(0.0)
            else:
                means.append(float(row.iloc[0]["mean_yield_tonnes"]))
                errors.append(float(row.iloc[0]["std_yield_tonnes"]))
        offset = (idx - 1) * width
        ax.bar(
            x + offset,
            means,
            width,
            yerr=errors,
            capsize=3,
            label=f"Previous: {predecessor}",
            color=CROP_COLORS[predecessor],
            alpha=0.82,
        )

    ax.set_title("Same current crop, different previous crop", fontsize=15, weight="bold")
    ax.set_xlabel("Current crop")
    ax.set_ylabel("Mean yield, tonnes")
    ax.set_xticks(x)
    ax.set_xticklabels(crops)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    add_sequence_key(fig, y=0.005)
    savefig(figures_dir.joinpath("yield_by_previous_crop.png"))


def _with_pre_year_soil(yearly_df: pd.DataFrame, summary_df: pd.DataFrame) -> pd.DataFrame:
    """Attach soil state entering the current crop year.

    ``yearly.csv`` records yield and the soil state after stepping the chosen
    crop. For the soil -> yield mechanism, the cleaner x-axis is the soil state
    available before that crop is grown. We approximate that as the previous
    year's post-step soil state, with the first year filled from
    ``sequence_summary.csv`` initial soil fields.
    """
    soil_fields = [col for col in yearly_df.columns if col.startswith("soil_")]
    key_cols = ["location", "window", "sequence_name"]
    df = yearly_df.sort_values(key_cols + ["step_idx"]).copy()

    for field in soil_fields:
        df[f"pre_{field}"] = df.groupby(key_cols, sort=False)[field].shift(1)

    initial_cols = key_cols + [f"initial_{field}" for field in soil_fields]
    initial_df = summary_df[initial_cols].copy()
    df = df.merge(initial_df, on=key_cols, how="left")

    first_year = df["step_idx"] == 0
    for field in soil_fields:
        pre_col = f"pre_{field}"
        initial_col = f"initial_{field}"
        df.loc[first_year, pre_col] = df.loc[first_year, initial_col]

    df["pre_mineral_n"] = (
        df["pre_soil_01_prof_soil_no3"] + df["pre_soil_02_prof_soil_nh4"]
    )
    return df


def plot_soil_state_to_yield(
    yearly_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    figures_dir: Path,
) -> None:
    df = _with_pre_year_soil(yearly_df, summary_df)
    df = df.loc[df["yield_tonnes"].notna()].copy()

    panels = [
        ("pre_soil_01_prof_soil_no3", "Pre-year profile soil NO3", "Soil nitrate before current crop"),
        ("pre_mineral_n", "Pre-year mineral N: NO3 + NH4", "Mineral nitrogen before current crop"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    for ax, (x_col, x_label, title) in zip(axes, panels):
        for crop, color in CROP_COLORS.items():
            crop_df = df.loc[
                (df["crop"] == crop)
                & df[x_col].notna()
                & df["yield_tonnes"].notna()
            ]
            if crop_df.empty:
                continue
            x = crop_df[x_col].astype(float).to_numpy()
            y = crop_df["yield_tonnes"].astype(float).to_numpy()
            ax.scatter(
                x,
                y,
                s=24,
                color=color,
                alpha=0.42,
                edgecolor="none",
                label=crop,
            )

            if len(crop_df) >= 3 and np.unique(x).size >= 2:
                slope, intercept = np.polyfit(x, y, deg=1)
                x_line = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 80)
                y_line = slope * x_line + intercept
                corr = np.corrcoef(x, y)[0, 1]
                ax.plot(x_line, y_line, color=color, linewidth=2.0)
                ax.annotate(
                    f"{crop}: r={corr:.2f}",
                    xy=(0.02, 0.95 - 0.07 * list(CROP_COLORS).index(crop)),
                    xycoords="axes fraction",
                    color=color,
                    fontsize=8,
                    weight="bold",
                )

        ax.set_title(title, weight="bold")
        ax.set_xlabel(x_label)
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("Yield, tonnes")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.suptitle(
        "Soil state entering the crop year is associated with realized yield",
        fontsize=16,
        weight="bold",
    )
    add_sequence_key(fig, y=0.005)
    savefig(figures_dir.joinpath("soil_state_to_yield.png"))


def plot_soil_reward_scatter(summary_df: pd.DataFrame, figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    for sequence_type, color, alpha in [
        ("random", RANDOM_COLOR, 0.75),
        ("designed", DESIGNED_COLOR, 0.95),
    ]:
        df = summary_df.loc[summary_df["sequence_type"] == sequence_type]
        ax.scatter(
            df["final_soil_sum"],
            df["cumulative_reward"],
            s=70 if sequence_type == "designed" else 45,
            color=color,
            edgecolor="white",
            linewidth=0.7,
            alpha=alpha,
            label=sequence_type,
        )
    for _, row in summary_df.loc[summary_df["sequence_type"] == "designed"].iterrows():
        ax.annotate(
            display_name(row["sequence_name"]),
            (row["final_soil_sum"], row["cumulative_reward"]),
            xytext=(5, 3),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_title("Soil state separation is tied to long-run reward", fontsize=15, weight="bold")
    ax.set_xlabel("Final soil-state aggregate")
    ax.set_ylabel("Cumulative reward")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    add_sequence_key(fig, y=0.005)
    savefig(figures_dir.joinpath("final_soil_vs_reward.png"))


def write_figure_index(figures_dir: Path) -> None:
    lines = [
        "# Mechanism Sanity Check Figures",
        "",
        f"{SEQUENCE_KEY_TEXT}.",
        "",
        "- `sequence_crop_heatmap.png`: all tested sequences, sorted by final soil state.",
        "- `soil_strength_by_sequence.png`: net soil change and soil-vector movement for every sequence.",
        "- `soil_trajectories.png`: yearly organic soil N and profile NO3 trajectories; random sequences are gray.",
        "- `yield_by_previous_crop.png`: yield of the same current crop split by previous crop.",
        "- `soil_state_to_yield.png`: yield of each crop as a function of the soil state entering that crop year.",
        "- `final_soil_vs_reward.png`: final soil state against cumulative reward.",
        "",
    ]
    figures_dir.joinpath("README.md").write_text("\n".join(lines), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=default_results_dir())
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary_df = pd.read_csv(args.results_dir.joinpath("sequence_summary.csv"))
    yearly_df = pd.read_csv(args.results_dir.joinpath("yearly.csv"))
    predecessor_df = pd.read_csv(args.results_dir.joinpath("yield_by_predecessor.csv"))
    figures_dir = args.results_dir.joinpath("figures")

    plot_sequence_heatmap(yearly_df, summary_df, figures_dir)
    plot_soil_strength(summary_df, figures_dir)
    plot_soil_trajectories(yearly_df, summary_df, figures_dir)
    plot_predecessor_yield(predecessor_df, figures_dir)
    plot_soil_state_to_yield(yearly_df, summary_df, figures_dir)
    plot_soil_reward_scatter(summary_df, figures_dir)
    write_figure_index(figures_dir)
    print(f"Wrote {figures_dir.joinpath('README.md')}")


if __name__ == "__main__":
    main()
