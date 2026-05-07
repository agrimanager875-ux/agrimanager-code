#!/usr/bin/env python
"""Select crop-trait schemas from a specialist-transfer matrix."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from agrimanager.env.wofost_gym.crop_traits_observation import CropTraitEncoder

from diagnose_existing_specialist_transfer import (
    ANALYSIS_DIR,
    DEFAULT_TRAITS_DIR,
    PROJECT_ROOT,
    resolve_path,
    spearman_corr,
    trait_key_for_crop,
)


DEFAULT_TRANSFER_DIR = ANALYSIS_DIR / "eight_specialist_transfer"
DEFAULT_OUTPUT_DIR = ANALYSIS_DIR / "eight_specialist_trait_selection"
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run nested leave-one-crop-out trait-schema selection from a "
            "specialist-transfer matrix."
        )
    )
    parser.add_argument("--transfer-dir", type=Path, default=DEFAULT_TRANSFER_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--traits-dir", type=Path, default=DEFAULT_TRAITS_DIR)
    parser.add_argument("--trait-schemas", default="traits_v1_23d,traits_v1_6d")
    parser.add_argument("--exclude-policies", default="noop_template")
    parser.add_argument("--shuffle-repeats", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_transfer_matrix(
    transfer_dir: Path,
    *,
    exclude_policies: set[str],
) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    raw_path = transfer_dir / "policy_transfer_matrix.csv"
    normalized_path = transfer_dir / "policy_transfer_matrix_normalized.csv"
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Missing raw transfer matrix: {raw_path}. Run the 8x20 transfer diagnostic first."
        )
    if not normalized_path.exists():
        raise FileNotFoundError(
            f"Missing normalized transfer matrix: {normalized_path}. Run the 8x20 transfer diagnostic first."
        )

    raw = pd.read_csv(raw_path)
    normalized_long = pd.read_csv(normalized_path)
    normalized = normalized_long.pivot(
        index="crop",
        columns="policy",
        values="crop_normalized_score",
    ).reset_index()

    crops = raw["crop"].tolist()
    normalized = normalized.set_index("crop").loc[crops].reset_index()
    raw = raw.set_index("crop").loc[crops].reset_index()

    policies = [
        column
        for column in raw.columns
        if column != "crop"
        and column in normalized.columns
        and column not in exclude_policies
        and raw[column].notna().any()
    ]
    if len(policies) < 2:
        raise ValueError(f"Need at least two specialist policies after exclusions, got {policies}")

    y_norm = normalized[policies].to_numpy(dtype=float)
    y_raw = raw[policies].to_numpy(dtype=float)
    return crops, policies, y_norm, y_raw


def encode_traits(crops: list[str], *, traits_dir: Path, trait_schema: str) -> tuple[np.ndarray, list[str]]:
    encoder = CropTraitEncoder(traits_dir=traits_dir, trait_schema=trait_schema)
    trait_keys = [trait_key_for_crop(encoder, crop) for crop in crops]
    X = np.vstack([encoder.vector_for_crop(key) for key in trait_keys]).astype(float)
    return X, list(encoder.feature_names)


def fit_ridge_predict(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0.0] = 1.0
    X_train_std = (X_train - mean) / std
    X_test_std = (X_test - mean) / std
    design = np.column_stack([np.ones(len(X_train_std)), X_train_std])
    reg = np.eye(design.shape[1], dtype=float) * float(alpha)
    reg[0, 0] = 0.0
    coef = np.linalg.pinv(design.T @ design + reg) @ design.T @ Y_train
    return np.column_stack([np.ones(len(X_test_std)), X_test_std]) @ coef


def choose_alpha_inner_cv(X: np.ndarray, Y: np.ndarray, alphas: tuple[float, ...]) -> float:
    if len(X) < 4:
        return 1.0

    scores = []
    for alpha in alphas:
        fold_errors = []
        for heldout in range(len(X)):
            mask = np.ones(len(X), dtype=bool)
            mask[heldout] = False
            pred = fit_ridge_predict(
                X[mask],
                Y[mask],
                X[heldout : heldout + 1],
                alpha=alpha,
            )
            fold_errors.append(float(np.mean((pred[0] - Y[heldout]) ** 2)))
        scores.append((float(np.mean(fold_errors)), alpha))
    scores.sort(key=lambda item: (item[0], item[1]))
    return float(scores[0][1])


def nested_loocv_predict(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, list[float]]:
    preds = np.zeros_like(Y, dtype=float)
    selected_alphas: list[float] = []
    for heldout in range(len(X)):
        train_mask = np.ones(len(X), dtype=bool)
        train_mask[heldout] = False
        alpha = choose_alpha_inner_cv(X[train_mask], Y[train_mask], ALPHAS)
        selected_alphas.append(alpha)
        preds[heldout] = fit_ridge_predict(
            X[train_mask],
            Y[train_mask],
            X[heldout : heldout + 1],
            alpha=alpha,
        )[0]
    return preds, selected_alphas


def mean_profile_predict(Y: np.ndarray) -> np.ndarray:
    preds = np.zeros_like(Y, dtype=float)
    for heldout in range(len(Y)):
        mask = np.ones(len(Y), dtype=bool)
        mask[heldout] = False
        preds[heldout] = Y[mask].mean(axis=0)
    return preds


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.std(a) == 0.0 or np.std(b) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_raw: np.ndarray,
) -> dict[str, float]:
    true_best = np.argmax(y_true, axis=1)
    pred_best = np.argmax(y_pred, axis=1)
    top1 = float(np.mean(pred_best == true_best))
    top3 = float(
        np.mean(
            [
                true_idx in np.argsort(pred)[-min(3, len(pred)) :]
                for true_idx, pred in zip(true_best, y_pred, strict=True)
            ]
        )
    )
    regret_norm = y_true[np.arange(len(y_true)), true_best] - y_true[np.arange(len(y_true)), pred_best]
    regret_raw = y_raw[np.arange(len(y_raw)), true_best] - y_raw[np.arange(len(y_raw)), pred_best]

    spearman_values = [spearman_corr(pred, true) for pred, true in zip(y_pred, y_true, strict=True)]
    pearson_values = [pearson_corr(pred, true) for pred, true in zip(y_pred, y_true, strict=True)]
    finite_spearman = [value for value in spearman_values if math.isfinite(value)]
    finite_pearson = [value for value in pearson_values if math.isfinite(value)]

    return {
        "top1_accuracy": top1,
        "top3_accuracy": top3,
        "mean_regret_norm": float(np.mean(regret_norm)),
        "median_regret_norm": float(np.median(regret_norm)),
        "mean_regret_raw_wso": float(np.mean(regret_raw)),
        "median_regret_raw_wso": float(np.median(regret_raw)),
        "mean_profile_spearman": float(np.mean(finite_spearman)) if finite_spearman else float("nan"),
        "mean_profile_pearson": float(np.mean(finite_pearson)) if finite_pearson else float("nan"),
    }


def detail_rows(
    crops: list[str],
    policies: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    trait_schema: str,
) -> list[dict[str, Any]]:
    rows = []
    for idx, crop in enumerate(crops):
        true_best_idx = int(np.argmax(y_true[idx]))
        pred_best_idx = int(np.argmax(y_pred[idx]))
        rows.append(
            {
                "trait_schema": trait_schema,
                "crop": crop,
                "true_best_policy": policies[true_best_idx],
                "predicted_best_policy": policies[pred_best_idx],
                "top1_correct": true_best_idx == pred_best_idx,
                "profile_spearman": spearman_corr(y_pred[idx], y_true[idx]),
                "profile_pearson": pearson_corr(y_pred[idx], y_true[idx]),
                "predicted_profile_json": json.dumps(
                    {policy: float(value) for policy, value in zip(policies, y_pred[idx], strict=True)},
                    sort_keys=True,
                ),
                "true_profile_json": json.dumps(
                    {policy: float(value) for policy, value in zip(policies, y_true[idx], strict=True)},
                    sort_keys=True,
                ),
            }
        )
    return rows


def run_schema(
    trait_schema: str,
    *,
    crops: list[str],
    policies: list[str],
    y_norm: np.ndarray,
    y_raw: np.ndarray,
    traits_dir: Path,
    rng: np.random.Generator,
    shuffle_repeats: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    X, feature_names = encode_traits(crops, traits_dir=traits_dir, trait_schema=trait_schema)
    pred, alphas = nested_loocv_predict(X, y_norm)
    metrics = evaluate_predictions(y_norm, pred, y_raw)

    mean_pred = mean_profile_predict(y_norm)
    mean_metrics = evaluate_predictions(y_norm, mean_pred, y_raw)

    shuffled_top1 = []
    shuffled_regret = []
    shuffled_spearman = []
    for _ in range(max(1, shuffle_repeats)):
        shuffled_X = X[rng.permutation(len(X))]
        shuffled_pred, _ = nested_loocv_predict(shuffled_X, y_norm)
        shuffled_metrics = evaluate_predictions(y_norm, shuffled_pred, y_raw)
        shuffled_top1.append(shuffled_metrics["top1_accuracy"])
        shuffled_regret.append(shuffled_metrics["mean_regret_norm"])
        shuffled_spearman.append(shuffled_metrics["mean_profile_spearman"])

    summary = {
        "trait_schema": trait_schema,
        "num_crops": len(crops),
        "num_policies": len(policies),
        "trait_dim": len(feature_names),
        "selected_alpha_mode": max(set(alphas), key=alphas.count),
        **metrics,
        "mean_profile_top1_accuracy": mean_metrics["top1_accuracy"],
        "mean_profile_regret_norm": mean_metrics["mean_regret_norm"],
        "shuffled_top1_accuracy_mean": float(np.mean(shuffled_top1)),
        "shuffled_top1_accuracy_std": float(np.std(shuffled_top1)),
        "shuffled_regret_norm_mean": float(np.mean(shuffled_regret)),
        "shuffled_regret_norm_std": float(np.std(shuffled_regret)),
        "shuffled_profile_spearman_mean": float(np.mean(shuffled_spearman)),
        "top1_lift_vs_shuffled": metrics["top1_accuracy"] - float(np.mean(shuffled_top1)),
        "regret_reduction_vs_shuffled": float(np.mean(shuffled_regret)) - metrics["mean_regret_norm"],
    }
    return summary, detail_rows(crops, policies, y_norm, pred, trait_schema=trait_schema)


def write_report(summary: pd.DataFrame, output_dir: Path, transfer_dir: Path) -> None:
    ranked = summary.sort_values(
        ["top1_accuracy", "mean_profile_spearman", "mean_regret_norm"],
        ascending=[False, False, True],
    )
    best = ranked.iloc[0].to_dict()
    lines = [
        "# Specialist Transfer Trait Selection",
        "",
        f"- Transfer directory: `{transfer_dir}`",
        f"- Selected schema by this diagnostic: `{best['trait_schema']}`",
        f"- Top-1 specialist accuracy: `{best['top1_accuracy']:.3f}`",
        f"- Lift vs shuffled traits: `{best['top1_lift_vs_shuffled']:.3f}`",
        f"- Mean normalized regret: `{best['mean_regret_norm']:.3f}`",
        f"- Mean profile Spearman: `{best['mean_profile_spearman']:.3f}`",
        "",
        "## All Candidate Schemas",
        "",
        summary.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "This is a diagnostic for whether a trait schema predicts specialist-transfer",
        "profiles under crop-held-out evaluation. It should be used before training",
        "new trait-conditioned policies; final policy results should not be used to",
        "choose the trait schema.",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    transfer_dir = resolve_path(args.transfer_dir)
    output_dir = resolve_path(args.output_dir)
    traits_dir = resolve_path(args.traits_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    schemas = [schema.strip() for schema in args.trait_schemas.split(",") if schema.strip()]
    exclude_policies = {
        policy.strip()
        for policy in args.exclude_policies.split(",")
        if policy.strip()
    }
    crops, policies, y_norm, y_raw = load_transfer_matrix(
        transfer_dir,
        exclude_policies=exclude_policies,
    )

    rng = np.random.default_rng(args.seed)
    summaries = []
    details = []
    for schema in schemas:
        summary, schema_details = run_schema(
            schema,
            crops=crops,
            policies=policies,
            y_norm=y_norm,
            y_raw=y_raw,
            traits_dir=traits_dir,
            rng=rng,
            shuffle_repeats=args.shuffle_repeats,
        )
        summaries.append(summary)
        details.extend(schema_details)

    summary_df = pd.DataFrame(summaries).sort_values(
        ["top1_accuracy", "mean_profile_spearman", "mean_regret_norm"],
        ascending=[False, False, True],
    )
    details_df = pd.DataFrame(details)
    summary_df.to_csv(output_dir / "trait_schema_nested_loocv_summary.csv", index=False)
    details_df.to_csv(output_dir / "trait_schema_nested_loocv_by_crop.csv", index=False)
    write_report(summary_df, output_dir, transfer_dir)

    print(f"wrote {output_dir / 'trait_schema_nested_loocv_summary.csv'}")
    print(f"wrote {output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
