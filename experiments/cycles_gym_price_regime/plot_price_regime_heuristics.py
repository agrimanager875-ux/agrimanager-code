#!/usr/bin/env python3
"""Plot paired CycleGym price-regime heuristic baseline results."""

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


CROP_LABELS = {
    "CornRM.100": "Corn",
    "SoybeanMG.3": "Soybean",
}
CROP_COLORS = {
    "CornRM.100": "#c17d11",
    "SoybeanMG.3": "#2f8f4e",
}
POLICY_COLORS = {
    "fixed_rotation": "#64748b",
    "price_greedy": "#dc2626",
    "rotation_aware": "#2563eb",
    "soil_aware": "#16a34a",
}
REGIME_ORDER = ["id_balanced", "high_corn", "high_soybean"]
POLICY_ORDER = ["fixed_rotation", "price_greedy", "rotation_aware", "soil_aware"]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_results_dir() -> Path:
    return repo_root().joinpath(
        "experiments", "cycles_gym_price_regime", "results", "price_regime_heuristics"
    )


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Wrote {path}")


def display_name(value: str) -> str:
    return value.replace("_", " ")


def plot_crop_frequency(crop_frequency_df: pd.DataFrame, figures_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharey=True)
    axes = axes.ravel()
    x = np.arange(len(REGIME_ORDER))
    width = 0.2

    for ax, policy in zip(axes, POLICY_ORDER):
        policy_df = crop_frequency_df.loc[crop_frequency_df["policy"] == policy]
        for idx, crop in enumerate(CROP_LABELS):
            offset = (idx - (len(CROP_LABELS) - 1) / 2) * width
            means = []
            errors = []
            for regime in REGIME_ORDER:
                row = policy_df.loc[
                    (policy_df["regime"] == regime) & (policy_df["crop"] == crop)
                ]
                means.append(float(row.iloc[0]["mean_frequency"]) if not row.empty else np.nan)
                errors.append(float(row.iloc[0]["std_frequency"]) if not row.empty else 0.0)
            ax.bar(
                x + offset,
                means,
                width,
                yerr=errors,
                capsize=3,
                color=CROP_COLORS[crop],
                label=CROP_LABELS[crop],
                alpha=0.9,
            )
        ax.set_title(display_name(policy), weight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([display_name(regime) for regime in REGIME_ORDER], rotation=25, ha="right")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.25)

    axes[0].set_ylabel("Crop frequency")
    axes[2].set_ylabel("Crop frequency")
    axes[0].legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(1.1, 1.45))
    fig.suptitle("Price response under paired price-regime validation", fontsize=16, weight="bold", y=0.99)
    fig.subplots_adjust(top=0.82, hspace=0.55, wspace=0.2)
    savefig(figures_dir.joinpath("price_response_crop_frequency.png"))


def plot_revenue(summary_df: pd.DataFrame, figures_dir: Path) -> None:
    grouped = (
        summary_df.groupby(["regime", "policy"])
        .agg(
            mean_revenue=("cumulative_gross_revenue", "mean"),
            std_revenue=("cumulative_gross_revenue", "std"),
        )
        .reset_index()
    )
    x = np.arange(len(REGIME_ORDER))
    width = 0.18
    fig, ax = plt.subplots(figsize=(12, 6.5))
    for idx, policy in enumerate(POLICY_ORDER):
        values = []
        errors = []
        for regime in REGIME_ORDER:
            row = grouped.loc[(grouped["regime"] == regime) & (grouped["policy"] == policy)]
            values.append(float(row.iloc[0]["mean_revenue"]) if not row.empty else np.nan)
            errors.append(float(row.iloc[0]["std_revenue"]) if not row.empty else 0.0)
        ax.bar(
            x + (idx - 1.5) * width,
            values,
            width,
            yerr=errors,
            capsize=3,
            color=POLICY_COLORS[policy],
            label=display_name(policy),
            alpha=0.9,
        )
    ax.set_title("Long-run cumulative gross revenue", fontsize=15, weight="bold")
    ax.set_ylabel("19-year cumulative gross revenue")
    ax.set_xticks(x)
    ax.set_xticklabels([display_name(regime) for regime in REGIME_ORDER])
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    savefig(figures_dir.joinpath("long_run_revenue_by_regime.png"))


def plot_rotation_quality(summary_df: pd.DataFrame, figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6.5))
    for policy in POLICY_ORDER:
        df = summary_df.loc[summary_df["policy"] == policy]
        ax.scatter(
            df["rotation_diversity_shannon"],
            df["cumulative_gross_revenue"],
            s=75,
            color=POLICY_COLORS[policy],
            label=display_name(policy),
            alpha=0.85,
            edgecolor="white",
            linewidth=0.6,
        )
    ax.set_title("Revenue vs. rotation diversity", fontsize=15, weight="bold")
    ax.set_xlabel("Rotation diversity, Shannon index")
    ax.set_ylabel("19-year cumulative gross revenue")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    savefig(figures_dir.joinpath("revenue_vs_rotation_diversity.png"))


def plot_final_soil(summary_df: pd.DataFrame, figures_dir: Path) -> None:
    soil_col = "final_soil_00_org_soil_n"
    if soil_col not in summary_df.columns:
        return
    grouped = (
        summary_df.groupby(["regime", "policy"])
        .agg(mean_final_soil=(soil_col, "mean"), std_final_soil=(soil_col, "std"))
        .reset_index()
    )
    x = np.arange(len(REGIME_ORDER))
    width = 0.18
    fig, ax = plt.subplots(figsize=(12, 6.5))
    for idx, policy in enumerate(POLICY_ORDER):
        values = []
        errors = []
        for regime in REGIME_ORDER:
            row = grouped.loc[(grouped["regime"] == regime) & (grouped["policy"] == policy)]
            values.append(float(row.iloc[0]["mean_final_soil"]) if not row.empty else np.nan)
            errors.append(float(row.iloc[0]["std_final_soil"]) if not row.empty else 0.0)
        ax.bar(
            x + (idx - 1.5) * width,
            values,
            width,
            yerr=errors,
            capsize=3,
            color=POLICY_COLORS[policy],
            label=display_name(policy),
            alpha=0.9,
        )
    ax.set_title("Final organic soil N by regime", fontsize=15, weight="bold")
    ax.set_ylabel("Final organic soil N")
    ax.set_xticks(x)
    ax.set_xticklabels([display_name(regime) for regime in REGIME_ORDER])
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    savefig(figures_dir.joinpath("final_soil_n_by_regime.png"))


def write_summary(summary_df: pd.DataFrame, figures_dir: Path) -> None:
    grouped = (
        summary_df.groupby(["regime", "policy"])
        .agg(
            mean_revenue=("cumulative_gross_revenue", "mean"),
            mean_diversity=("rotation_diversity_shannon", "mean"),
            mean_final_soil=("final_soil_00_org_soil_n", "mean"),
            mean_max_cereal_run=("max_consecutive_cereal", "mean"),
        )
        .reset_index()
        .sort_values(["regime", "mean_revenue"], ascending=[True, False])
    )
    lines = [
        "# Price-Regime Heuristic Figures",
        "",
        "These figures use paired scenarios: for a given location/year window, only the crop-price vector changes.",
        "",
        "## Output Files",
        "",
        "- `price_response_crop_frequency.png`: crop frequencies by regime and heuristic policy.",
        "- `long_run_revenue_by_regime.png`: cumulative gross revenue by regime and heuristic policy.",
        "- `revenue_vs_rotation_diversity.png`: long-run revenue against sequence diversity.",
        "- `final_soil_n_by_regime.png`: final organic soil N by regime and heuristic policy.",
        "",
        "## Aggregated Results",
        "",
        grouped.to_markdown(index=False),
        "",
    ]
    figures_dir.joinpath("README.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {figures_dir.joinpath('README.md')}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=default_results_dir())
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary_df = pd.read_csv(args.results_dir.joinpath("summary.csv"))
    crop_frequency_df = pd.read_csv(args.results_dir.joinpath("crop_frequency.csv"))
    figures_dir = args.results_dir.joinpath("figures")
    plot_crop_frequency(crop_frequency_df, figures_dir)
    plot_revenue(summary_df, figures_dir)
    plot_rotation_quality(summary_df, figures_dir)
    plot_final_soil(summary_df, figures_dir)
    write_summary(summary_df, figures_dir)


if __name__ == "__main__":
    main()
