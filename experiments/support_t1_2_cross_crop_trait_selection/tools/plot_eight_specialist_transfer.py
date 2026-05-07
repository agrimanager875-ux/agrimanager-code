#!/usr/bin/env python
"""Plot the eight-specialist cross-crop transfer diagnostic."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ANALYSIS_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "support_t1_2_cross_crop_trait_selection"
    / "analysis"
    / "eight_specialist_transfer"
)
FIGURE_DIR = ANALYSIS_DIR / "figures"

POLICY_ORDER = [
    "maize_weather_specialist",
    "wheat_weather_specialist",
    "cotton_specialist",
    "rice_specialist",
    "potato_specialist",
    "sugarbeet_specialist",
    "barley_specialist",
    "seed_onion_specialist",
]

POLICY_LABELS = {
    "maize_weather_specialist": "Maize",
    "wheat_weather_specialist": "Wheat",
    "cotton_specialist": "Cotton",
    "rice_specialist": "Rice",
    "potato_specialist": "Potato",
    "sugarbeet_specialist": "Sugarbeet",
    "barley_specialist": "Barley",
    "seed_onion_specialist": "Seed onion",
}

POLICY_COLORS = {
    "maize_weather_specialist": "#E69F00",
    "wheat_weather_specialist": "#56B4E9",
    "cotton_specialist": "#009E73",
    "rice_specialist": "#F0E442",
    "potato_specialist": "#0072B2",
    "sugarbeet_specialist": "#D55E00",
    "barley_specialist": "#CC79A7",
    "seed_onion_specialist": "#999999",
}


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 160,
            "axes.grid": False,
        }
    )


def despine(ax: plt.Axes, *, left: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if left:
        ax.spines["left"].set_visible(False)


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(ANALYSIS_DIR / "policy_transfer_matrix.csv")
    normalized_long = pd.read_csv(ANALYSIS_DIR / "policy_transfer_matrix_normalized.csv")
    ranking = pd.read_csv(ANALYSIS_DIR / "policy_ranking_by_crop.csv")
    return raw, normalized_long, ranking


def best_policy_table(ranking: pd.DataFrame) -> pd.DataFrame:
    specialist_ranking = ranking[ranking["policy"].isin(POLICY_ORDER)].copy()
    rows = []
    for crop, group in specialist_ranking.groupby("crop", sort=False):
        ordered = group.sort_values("final_wso_mean", ascending=False).reset_index(drop=True)
        best = ordered.iloc[0]
        second = ordered.iloc[1]
        rows.append(
            {
                "crop": crop,
                "best_policy": best["policy"],
                "best_wso": float(best["final_wso_mean"]),
                "second_policy": second["policy"],
                "second_wso": float(second["final_wso_mean"]),
                "margin_wso": float(best["final_wso_mean"] - second["final_wso_mean"]),
                "margin_pct_second": float(
                    (best["final_wso_mean"] - second["final_wso_mean"])
                    / max(abs(second["final_wso_mean"]), 1.0)
                    * 100.0
                ),
            }
        )
    return pd.DataFrame(rows)


def crop_order(best: pd.DataFrame) -> list[str]:
    policy_counts = best["best_policy"].value_counts()
    policy_rank = {
        policy: rank
        for rank, policy in enumerate(
            sorted(POLICY_ORDER, key=lambda item: (-policy_counts.get(item, 0), POLICY_LABELS[item]))
        )
    }
    ordered = best.sort_values(
        by=["best_policy", "margin_pct_second", "crop"],
        key=lambda column: column.map(policy_rank) if column.name == "best_policy" else column,
        ascending=[True, False, True],
    )
    return ordered["crop"].tolist()


def normalized_matrix(normalized_long: pd.DataFrame, crops: list[str]) -> pd.DataFrame:
    matrix = normalized_long.pivot(
        index="crop",
        columns="policy",
        values="crop_normalized_score",
    )
    matrix = matrix.loc[crops, POLICY_ORDER]
    matrix = matrix.rename(columns=POLICY_LABELS)
    return matrix


def draw_best_cell_boxes(ax: plt.Axes, matrix: pd.DataFrame, best: pd.DataFrame) -> None:
    crop_to_row = {crop: idx for idx, crop in enumerate(matrix.index)}
    policy_to_col = {POLICY_LABELS[policy]: idx for idx, policy in enumerate(POLICY_ORDER)}
    for _, row in best.iterrows():
        crop = row["crop"]
        label = POLICY_LABELS[row["best_policy"]]
        if crop not in crop_to_row or label not in policy_to_col:
            continue
        y = crop_to_row[crop]
        x = policy_to_col[label]
        ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, fill=False, edgecolor="black", linewidth=1.2))
        ax.text(x, y, "*", ha="center", va="center", color="black", fontsize=10)


def plot_heatmap(matrix: pd.DataFrame, best: pd.DataFrame, output_prefix: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 7.8), constrained_layout=True)
    norm = TwoSlopeNorm(vmin=-0.15, vcenter=0.0, vmax=1.0)
    im = ax.imshow(matrix.to_numpy(dtype=float), cmap="RdBu_r", norm=norm, aspect="auto")
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_xticklabels(matrix.columns)
    ax.set_yticklabels(matrix.index)
    ax.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Crop-normalized WSO (noop=0, best=1)")
    draw_best_cell_boxes(ax, matrix, best)
    ax.set_title("Eight-specialist transfer profile")
    ax.set_xlabel("Source specialist policy")
    ax.set_ylabel("Target crop")
    ax.tick_params(axis="x", rotation=35)
    ax.tick_params(axis="y", rotation=0)
    fig.savefig(output_prefix.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_best_counts(best: pd.DataFrame, output_prefix: Path) -> None:
    counts = best["best_policy"].value_counts().reindex(POLICY_ORDER, fill_value=0)
    counts = counts[counts > 0].sort_values(ascending=True)
    labels = [POLICY_LABELS[item] for item in counts.index]
    colors = [POLICY_COLORS[item] for item in counts.index]

    fig, ax = plt.subplots(figsize=(5.0, 3.1), constrained_layout=True)
    bars = ax.barh(labels, counts.to_numpy(), color=colors, edgecolor="black", linewidth=0.4)
    ax.set_xlabel("Number of target crops where policy is best")
    ax.set_title("Best specialist distribution")
    ax.set_xlim(0, max(counts.max() + 0.8, 5.8))
    for bar, value in zip(bars, counts.to_numpy(), strict=True):
        ax.text(value + 0.12, bar.get_y() + bar.get_height() / 2, str(int(value)), va="center")
    despine(ax, left=True)
    fig.savefig(output_prefix.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_margins(best: pd.DataFrame, output_prefix: Path) -> None:
    ordered = best.sort_values("margin_pct_second", ascending=True)
    labels = ordered["crop"].tolist()
    colors = [POLICY_COLORS[policy] for policy in ordered["best_policy"]]

    fig, ax = plt.subplots(figsize=(7.0, 6.2), constrained_layout=True)
    bars = ax.barh(labels, ordered["margin_pct_second"], color=colors, edgecolor="black", linewidth=0.3)
    ax.set_xlabel("Best minus second-best WSO (% of second-best)")
    ax.set_title("Transfer margin by target crop")
    ax.axvline(10, color="#333333", linewidth=0.8, linestyle="--", alpha=0.75)
    ax.text(10.5, -0.7, "10% margin", fontsize=8, color="#333333")
    for bar, value in zip(bars, ordered["margin_pct_second"], strict=True):
        if value >= 12:
            x = value - 1.0
            ha = "right"
            color = "white"
        else:
            x = value + 0.8
            ha = "left"
            color = "black"
        ax.text(x, bar.get_y() + bar.get_height() / 2, f"{value:.1f}", va="center", ha=ha, color=color, fontsize=7)

    handles = [
        Rectangle((0, 0), 1, 1, facecolor=POLICY_COLORS[policy], edgecolor="black", linewidth=0.3)
        for policy in POLICY_ORDER
        if policy in set(ordered["best_policy"])
    ]
    legend_labels = [
        POLICY_LABELS[policy]
        for policy in POLICY_ORDER
        if policy in set(ordered["best_policy"])
    ]
    ax.legend(handles, legend_labels, title="Best specialist", loc="lower right", frameon=True)
    despine(ax, left=True)
    fig.savefig(output_prefix.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_summary(matrix: pd.DataFrame, best: pd.DataFrame, output_prefix: Path) -> None:
    fig = plt.figure(figsize=(11.5, 10.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[3.2, 1.35], height_ratios=[2.1, 1.55])
    ax_heat = fig.add_subplot(gs[0, 0])
    ax_counts = fig.add_subplot(gs[0, 1])
    ax_margin = fig.add_subplot(gs[1, :])

    norm = TwoSlopeNorm(vmin=-0.15, vcenter=0.0, vmax=1.0)
    im = ax_heat.imshow(matrix.to_numpy(dtype=float), cmap="RdBu_r", norm=norm, aspect="auto")
    ax_heat.set_xticks(np.arange(matrix.shape[1]))
    ax_heat.set_yticks(np.arange(matrix.shape[0]))
    ax_heat.set_xticklabels(matrix.columns)
    ax_heat.set_yticklabels(matrix.index)
    ax_heat.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
    ax_heat.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
    ax_heat.grid(which="minor", color="white", linestyle="-", linewidth=0.6)
    ax_heat.tick_params(which="minor", bottom=False, left=False)
    cbar = fig.colorbar(im, ax=ax_heat, shrink=0.72)
    cbar.set_label("Normalized WSO")
    draw_best_cell_boxes(ax_heat, matrix, best)
    ax_heat.set_title("A. Specialist transfer heatmap")
    ax_heat.set_xlabel("")
    ax_heat.set_ylabel("Target crop")
    ax_heat.tick_params(axis="x", rotation=35)
    ax_heat.tick_params(axis="y", rotation=0)

    counts = best["best_policy"].value_counts().reindex(POLICY_ORDER, fill_value=0)
    counts = counts[counts > 0].sort_values(ascending=True)
    count_labels = [POLICY_LABELS[item] for item in counts.index]
    count_colors = [POLICY_COLORS[item] for item in counts.index]
    bars = ax_counts.barh(count_labels, counts.to_numpy(), color=count_colors, edgecolor="black", linewidth=0.4)
    ax_counts.set_title("B. Best-policy counts")
    ax_counts.set_xlabel("Crops")
    ax_counts.set_xlim(0, max(counts.max() + 1.0, 6.0))
    for bar, value in zip(bars, counts.to_numpy(), strict=True):
        ax_counts.text(value + 0.12, bar.get_y() + bar.get_height() / 2, str(int(value)), va="center")
    despine(ax_counts, left=True)

    ordered = best.sort_values("margin_pct_second", ascending=True)
    colors = [POLICY_COLORS[policy] for policy in ordered["best_policy"]]
    ax_margin.barh(ordered["crop"], ordered["margin_pct_second"], color=colors, edgecolor="black", linewidth=0.3)
    ax_margin.axvline(10, color="#333333", linewidth=0.8, linestyle="--", alpha=0.75)
    ax_margin.set_title("C. Best-vs-second specialist margin")
    ax_margin.set_xlabel("Best minus second-best WSO (% of second-best)")
    ax_margin.set_ylabel("Target crop")
    despine(ax_margin, left=True)

    fig.suptitle("Eight specialist policies reveal crop-dependent management strategies", fontsize=13, fontweight="bold")
    fig.savefig(output_prefix.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    set_style()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    _, normalized_long, ranking = load_tables()
    best = best_policy_table(ranking)
    crops = crop_order(best)
    matrix = normalized_matrix(normalized_long, crops)

    plot_heatmap(matrix, best, FIGURE_DIR / "eight_specialist_transfer_heatmap")
    plot_best_counts(best, FIGURE_DIR / "eight_specialist_best_policy_counts")
    plot_margins(best, FIGURE_DIR / "eight_specialist_transfer_margins")
    plot_summary(matrix, best, FIGURE_DIR / "eight_specialist_transfer_summary")

    print(f"wrote figures to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
