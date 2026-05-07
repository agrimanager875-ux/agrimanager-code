#!/usr/bin/env python
"""Discover strategy-predictive crop traits from specialist transfer."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agrimanager.env.wofost_gym.crop_trait_schemas import DEFAULT_CROP_TRAIT_SCHEMA
from agrimanager.env.wofost_gym.crop_traits_observation import CropTraitEncoder

from diagnose_existing_specialist_transfer import (
    ANALYSIS_DIR,
    DEFAULT_TRAITS_DIR,
    resolve_path,
    spearman_corr,
    trait_key_for_crop,
)
from select_traits_from_specialist_transfer import (
    ALPHAS,
    DEFAULT_TRANSFER_DIR,
    evaluate_predictions,
    fit_ridge_predict,
    load_transfer_matrix,
    pearson_corr,
)


DEFAULT_OUTPUT_DIR = ANALYSIS_DIR / "strategy_trait_discovery"
EPS = 1e-8


@dataclass(frozen=True)
class CandidatePool:
    crops: list[str]
    trait_keys: list[str]
    feature_names: list[str]
    X: np.ndarray
    metadata: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class SubsetEvaluation:
    selected_indices: tuple[int, ...]
    alpha: float
    objective_mean: float
    objective_se: float
    top1_accuracy: float
    mean_regret_norm: float
    mean_profile_spearman: float


@dataclass(frozen=True)
class SelectionResult:
    selected_indices: tuple[int, ...]
    alpha: float
    trace: list[dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Discover a compact crop-trait set that predicts specialist-transfer "
            "strategy profiles."
        )
    )
    parser.add_argument("--transfer-dir", type=Path, default=DEFAULT_TRANSFER_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--traits-dir", type=Path, default=DEFAULT_TRAITS_DIR)
    parser.add_argument("--source-trait-schema", default=DEFAULT_CROP_TRAIT_SCHEMA)
    parser.add_argument("--exclude-policies", default="noop_template")
    parser.add_argument("--max-features", type=int, default=8)
    parser.add_argument("--min-features", type=int, default=1)
    parser.add_argument(
        "--candidate-top-k",
        type=int,
        default=30,
        help="Fast univariate prefilter before greedy selection; use 0 to keep all candidates.",
    )
    parser.add_argument("--shuffle-repeats", type=int, default=10)
    parser.add_argument("--bootstrap-repeats", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--regret-weight", type=float, default=1.0)
    parser.add_argument("--top1-weight", type=float, default=0.5)
    parser.add_argument("--spearman-weight", type=float, default=1.0)
    parser.add_argument("--feature-penalty", type=float, default=0.01)
    parser.add_argument("--no-derived", action="store_true")
    parser.add_argument("--disable-one-se-rule", action="store_true")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a small CPU smoke-test search budget.",
    )
    return parser.parse_args()


def _safe_div(numerator: float, denominator: float) -> float:
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        return float("nan")
    if abs(denominator) < EPS:
        return float("nan")
    return float(numerator / denominator)


def derived_feature_values(raw: dict[str, float]) -> dict[str, float]:
    """Return derived crop-management candidate features from raw trait facts."""

    def value(name: str) -> float:
        return float(raw.get(name, float("nan")))

    tsumem = value("core_facts.phenology.TSUMEM_Cd")
    tsum1 = value("core_facts.phenology.TSUM1_Cd")
    tsum2 = value("core_facts.phenology.TSUM2_Cd")
    tsum3 = value("core_facts.phenology.TSUM3_Cd")
    total = value("core_facts.phenology.TSUM_total_Cd")
    storage_onset = value("core_facts.assimilation_and_partition.FOTB_storage_onset_DVS")
    rdmcr = value("core_facts.root_and_water.RDMCR_cm")
    rri = value("core_facts.root_and_water.RRI_cm_per_day")
    depnr = value("core_facts.root_and_water.DEPNR")
    cfet = value("core_facts.root_and_water.CFET")
    amax = value("core_facts.assimilation_and_partition.AMAX_peak")

    n0 = value("core_facts.nutrient_capacity_leaf.NMAXLV_DVS0")
    n1 = value("core_facts.nutrient_capacity_leaf.NMAXLV_DVS1")
    p0 = value("core_facts.nutrient_capacity_leaf.PMAXLV_DVS0")
    p1 = value("core_facts.nutrient_capacity_leaf.PMAXLV_DVS1")
    k0 = value("core_facts.nutrient_capacity_leaf.KMAXLV_DVS0")
    k1 = value("core_facts.nutrient_capacity_leaf.KMAXLV_DVS1")

    early_total = n0 + p0 + k0
    late_total = n1 + p1 + k1

    return {
        "derived.phenology.emergence_fraction_of_total": _safe_div(tsumem, total),
        "derived.phenology.vegetative_fraction_of_total": _safe_div(tsum1, total),
        "derived.phenology.reproductive_fraction_of_total": _safe_div(tsum2, total),
        "derived.phenology.late_fraction_of_total": _safe_div(tsum3, total),
        "derived.phenology.post_anthesis_fraction_of_total": _safe_div(tsum2 + tsum3, total),
        "derived.phenology.storage_onset_thermal_proxy": storage_onset * total,
        "derived.root_water.root_depth_per_thermal_time": _safe_div(rdmcr, total),
        "derived.root_water.root_growth_per_thermal_time": _safe_div(rri, total),
        "derived.root_water.water_buffer_proxy": rdmcr * depnr,
        "derived.root_water.evap_demand_per_root_depth": _safe_div(cfet, rdmcr),
        "derived.assimilation.amax_per_thermal_time": _safe_div(amax, total),
        "derived.nutrient.early_total_capacity": early_total,
        "derived.nutrient.late_total_capacity": late_total,
        "derived.nutrient.total_capacity_decline": early_total - late_total,
        "derived.nutrient.N_decline": n0 - n1,
        "derived.nutrient.P_decline": p0 - p1,
        "derived.nutrient.K_decline": k0 - k1,
        "derived.nutrient.N_late_to_early_ratio": _safe_div(n1, n0),
        "derived.nutrient.P_late_to_early_ratio": _safe_div(p1, p0),
        "derived.nutrient.K_late_to_early_ratio": _safe_div(k1, k0),
    }


def _minmax_normalize(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    with np.errstate(all="ignore"):
        mins = np.nanmin(arr, axis=0)
        maxs = np.nanmax(arr, axis=0)
    mins = np.nan_to_num(mins, nan=0.0)
    maxs = np.nan_to_num(maxs, nan=0.0)
    arr = np.where(np.isfinite(arr), arr, mins)
    scales = maxs - mins
    scales[scales == 0.0] = 1.0
    return np.clip((arr - mins) / scales, 0.0, 1.0)


def _filter_candidate_columns(
    X: np.ndarray,
    feature_names: list[str],
    metadata: dict[str, dict[str, Any]],
) -> tuple[np.ndarray, list[str], dict[str, dict[str, Any]]]:
    keep_indices: list[int] = []
    seen_columns: set[tuple[float, ...]] = set()
    for idx, name in enumerate(feature_names):
        column = np.asarray(X[:, idx], dtype=float)
        if np.nanstd(column) <= 1e-12:
            continue
        key = tuple(np.round(column, 12).tolist())
        if key in seen_columns:
            continue
        seen_columns.add(key)
        keep_indices.append(idx)

    kept_names = [feature_names[idx] for idx in keep_indices]
    kept_metadata = {name: metadata[name] for name in kept_names}
    return X[:, keep_indices], kept_names, kept_metadata


def load_candidate_pool(
    crops: list[str],
    *,
    traits_dir: Path,
    source_trait_schema: str,
    include_derived: bool,
) -> CandidatePool:
    encoder = CropTraitEncoder(traits_dir=traits_dir, trait_schema=source_trait_schema)
    trait_keys = [trait_key_for_crop(encoder, crop) for crop in crops]
    raw_feature_names = list(encoder.feature_names)
    raw_X = np.vstack([encoder.vector_for_crop(key) for key in trait_keys]).astype(float)

    feature_names = list(raw_feature_names)
    X_parts = [raw_X]
    metadata = {
        name: {
            "kind": "raw",
            "source_trait_schema": source_trait_schema,
            "source_feature": name,
        }
        for name in raw_feature_names
    }

    if include_derived:
        derived_rows = [
            derived_feature_values(encoder._raw_by_crop[key])  # noqa: SLF001 - diagnostic script
            for key in trait_keys
        ]
        derived_names = sorted(set().union(*(row.keys() for row in derived_rows)))
        derived_raw = np.array(
            [[row.get(name, float("nan")) for name in derived_names] for row in derived_rows],
            dtype=float,
        )
        if derived_names:
            X_parts.append(_minmax_normalize(derived_raw))
            feature_names.extend(derived_names)
            metadata.update(
                {
                    name: {
                        "kind": "derived",
                        "source_trait_schema": source_trait_schema,
                        "source_feature": name,
                    }
                    for name in derived_names
                }
            )

    X = np.column_stack(X_parts)
    X, feature_names, metadata = _filter_candidate_columns(X, feature_names, metadata)
    if X.shape[1] == 0:
        raise ValueError("No non-constant candidate trait features are available.")
    return CandidatePool(crops=crops, trait_keys=trait_keys, feature_names=feature_names, X=X, metadata=metadata)


def loocv_predict_fixed_alpha(X: np.ndarray, Y: np.ndarray, *, alpha: float) -> np.ndarray:
    preds = np.zeros_like(Y, dtype=float)
    for heldout in range(len(X)):
        mask = np.ones(len(X), dtype=bool)
        mask[heldout] = False
        preds[heldout] = fit_ridge_predict(X[mask], Y[mask], X[heldout : heldout + 1], alpha=alpha)[0]
    return preds


def objective_values(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    regret_weight: float,
    top1_weight: float,
    spearman_weight: float,
    feature_penalty: float,
    num_features: int,
) -> np.ndarray:
    true_best = np.argmax(y_true, axis=1)
    pred_best = np.argmax(y_pred, axis=1)
    regrets = y_true[np.arange(len(y_true)), true_best] - y_true[np.arange(len(y_true)), pred_best]

    values: list[float] = []
    for idx, (pred, true) in enumerate(zip(y_pred, y_true, strict=True)):
        corr = spearman_corr(pred, true)
        if not math.isfinite(corr):
            corr = 0.0
        correct = 1.0 if pred_best[idx] == true_best[idx] else 0.0
        values.append(
            spearman_weight * float(corr)
            + top1_weight * correct
            - regret_weight * float(regrets[idx])
            - feature_penalty * float(num_features)
        )
    return np.asarray(values, dtype=float)


def evaluate_subset_cv(
    X: np.ndarray,
    Y: np.ndarray,
    selected_indices: tuple[int, ...],
    *,
    regret_weight: float,
    top1_weight: float,
    spearman_weight: float,
    feature_penalty: float,
) -> SubsetEvaluation:
    if not selected_indices:
        raise ValueError("selected_indices must not be empty.")
    X_subset = X[:, selected_indices]

    alpha_candidates: list[tuple[float, float, np.ndarray]] = []
    for alpha in ALPHAS:
        pred = loocv_predict_fixed_alpha(X_subset, Y, alpha=float(alpha))
        mse = float(np.mean((pred - Y) ** 2))
        alpha_candidates.append((mse, float(alpha), pred))
    alpha_candidates.sort(key=lambda item: (item[0], item[1]))
    _, alpha, pred = alpha_candidates[0]

    metrics = evaluate_predictions(Y, pred, Y)
    per_crop_objective = objective_values(
        Y,
        pred,
        regret_weight=regret_weight,
        top1_weight=top1_weight,
        spearman_weight=spearman_weight,
        feature_penalty=feature_penalty,
        num_features=len(selected_indices),
    )
    se = float(np.std(per_crop_objective, ddof=1) / math.sqrt(len(per_crop_objective))) if len(per_crop_objective) > 1 else 0.0
    return SubsetEvaluation(
        selected_indices=selected_indices,
        alpha=alpha,
        objective_mean=float(np.mean(per_crop_objective)),
        objective_se=se,
        top1_accuracy=float(metrics["top1_accuracy"]),
        mean_regret_norm=float(metrics["mean_regret_norm"]),
        mean_profile_spearman=float(metrics["mean_profile_spearman"]),
    )


def _corr_abs_score(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) <= 1e-12:
        return 0.0
    scores = []
    for column in range(y.shape[1]):
        target = y[:, column]
        if np.std(target) <= 1e-12:
            continue
        corr = float(np.corrcoef(x, target)[0, 1])
        if math.isfinite(corr):
            scores.append(abs(corr))
    return float(np.mean(scores)) if scores else 0.0


def prefilter_feature_indices(
    X: np.ndarray,
    Y: np.ndarray,
    feature_names: list[str],
    *,
    candidate_top_k: int,
) -> list[int]:
    if candidate_top_k <= 0 or candidate_top_k >= X.shape[1]:
        return list(range(X.shape[1]))
    scored = [
        (_corr_abs_score(X[:, idx], Y), feature_names[idx], idx)
        for idx in range(X.shape[1])
    ]
    scored.sort(key=lambda item: (-item[0], item[1]))
    return sorted(idx for _, _, idx in scored[: max(1, candidate_top_k)])


def select_feature_subset(
    X: np.ndarray,
    Y: np.ndarray,
    feature_names: list[str],
    *,
    max_features: int,
    min_features: int,
    candidate_top_k: int,
    regret_weight: float,
    top1_weight: float,
    spearman_weight: float,
    feature_penalty: float,
    one_se_rule: bool,
) -> SelectionResult:
    allowed_indices = prefilter_feature_indices(
        X,
        Y,
        feature_names,
        candidate_top_k=int(candidate_top_k),
    )
    max_features = max(1, min(int(max_features), len(allowed_indices)))
    min_features = max(1, min(int(min_features), max_features))

    selected: list[int] = []
    remaining = set(allowed_indices)
    trace: list[dict[str, Any]] = []
    step_evaluations: list[SubsetEvaluation] = []

    for step in range(1, max_features + 1):
        candidate_evaluations: list[SubsetEvaluation] = []
        for candidate in sorted(remaining, key=lambda idx: feature_names[idx]):
            subset = tuple(selected + [candidate])
            candidate_evaluations.append(
                evaluate_subset_cv(
                    X,
                    Y,
                    subset,
                    regret_weight=regret_weight,
                    top1_weight=top1_weight,
                    spearman_weight=spearman_weight,
                    feature_penalty=feature_penalty,
                )
            )

        candidate_evaluations.sort(
            key=lambda item: (
                -item.objective_mean,
                item.mean_regret_norm,
                -item.top1_accuracy,
                feature_names[item.selected_indices[-1]],
            )
        )
        best = candidate_evaluations[0]
        chosen_idx = best.selected_indices[-1]
        selected.append(chosen_idx)
        remaining.remove(chosen_idx)
        step_evaluations.append(best)
        trace.append(
            {
                "num_features": step,
                "added_feature": feature_names[chosen_idx],
                "selected_features_json": json.dumps([feature_names[idx] for idx in best.selected_indices]),
                "selected_indices_json": json.dumps(list(best.selected_indices)),
                "alpha": best.alpha,
                "objective_mean": best.objective_mean,
                "objective_se": best.objective_se,
                "top1_accuracy": best.top1_accuracy,
                "mean_regret_norm": best.mean_regret_norm,
                "mean_profile_spearman": best.mean_profile_spearman,
            }
        )

    eligible = [item for item in step_evaluations if len(item.selected_indices) >= min_features]
    best_eval = max(
        eligible,
        key=lambda item: (item.objective_mean, -len(item.selected_indices), -item.mean_regret_norm),
    )
    if one_se_rule:
        threshold = best_eval.objective_mean - best_eval.objective_se
        one_se_eligible = [
            item for item in eligible if item.objective_mean >= threshold
        ]
        chosen = min(
            one_se_eligible,
            key=lambda item: (len(item.selected_indices), -item.objective_mean, item.mean_regret_norm),
        )
    else:
        chosen = best_eval

    return SelectionResult(
        selected_indices=chosen.selected_indices,
        alpha=chosen.alpha,
        trace=trace,
    )


def nested_discovery_predict(
    X: np.ndarray,
    Y_norm: np.ndarray,
    Y_raw: np.ndarray,
    crops: list[str],
    policies: list[str],
    feature_names: list[str],
    *,
    max_features: int,
    min_features: int,
    candidate_top_k: int,
    regret_weight: float,
    top1_weight: float,
    spearman_weight: float,
    feature_penalty: float,
    one_se_rule: bool,
    keep_trace: bool,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    preds = np.zeros_like(Y_norm, dtype=float)
    fold_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []

    for heldout, crop in enumerate(crops):
        train_mask = np.ones(len(crops), dtype=bool)
        train_mask[heldout] = False
        selection = select_feature_subset(
            X[train_mask],
            Y_norm[train_mask],
            feature_names,
            max_features=max_features,
            min_features=min_features,
            candidate_top_k=candidate_top_k,
            regret_weight=regret_weight,
            top1_weight=top1_weight,
            spearman_weight=spearman_weight,
            feature_penalty=feature_penalty,
            one_se_rule=one_se_rule,
        )
        preds[heldout] = fit_ridge_predict(
            X[train_mask][:, selection.selected_indices],
            Y_norm[train_mask],
            X[heldout : heldout + 1][:, selection.selected_indices],
            alpha=selection.alpha,
        )[0]

        true_best_idx = int(np.argmax(Y_norm[heldout]))
        pred_best_idx = int(np.argmax(preds[heldout]))
        selected_features = [feature_names[idx] for idx in selection.selected_indices]
        fold_rows.append(
            {
                "crop": crop,
                "true_best_policy": policies[true_best_idx],
                "predicted_best_policy": policies[pred_best_idx],
                "top1_correct": true_best_idx == pred_best_idx,
                "regret_norm": float(Y_norm[heldout, true_best_idx] - Y_norm[heldout, pred_best_idx]),
                "regret_raw_wso": float(Y_raw[heldout, true_best_idx] - Y_raw[heldout, pred_best_idx]),
                "profile_spearman": spearman_corr(preds[heldout], Y_norm[heldout]),
                "profile_pearson": pearson_corr(preds[heldout], Y_norm[heldout]),
                "num_selected_features": len(selected_features),
                "selected_alpha": selection.alpha,
                "selected_features_json": json.dumps(selected_features),
                "predicted_profile_json": json.dumps(
                    {policy: float(value) for policy, value in zip(policies, preds[heldout], strict=True)},
                    sort_keys=True,
                ),
                "true_profile_json": json.dumps(
                    {policy: float(value) for policy, value in zip(policies, Y_norm[heldout], strict=True)},
                    sort_keys=True,
                ),
            }
        )
        if keep_trace:
            for row in selection.trace:
                trace_row = dict(row)
                trace_row["outer_heldout_crop"] = crop
                trace_rows.append(trace_row)

    return preds, pd.DataFrame(fold_rows), pd.DataFrame(trace_rows)


def selection_frequency(
    feature_names: list[str],
    selected_json_values: pd.Series,
) -> dict[str, float]:
    counts = {name: 0 for name in feature_names}
    total = 0
    for raw in selected_json_values:
        selected = json.loads(raw)
        total += 1
        for name in selected:
            counts[name] = counts.get(name, 0) + 1
    if total == 0:
        return {name: 0.0 for name in feature_names}
    return {name: count / total for name, count in counts.items()}


def bootstrap_feature_frequency(
    X: np.ndarray,
    Y_norm: np.ndarray,
    feature_names: list[str],
    *,
    repeats: int,
    rng: np.random.Generator,
    max_features: int,
    min_features: int,
    candidate_top_k: int,
    regret_weight: float,
    top1_weight: float,
    spearman_weight: float,
    feature_penalty: float,
    one_se_rule: bool,
) -> dict[str, float]:
    if repeats <= 0:
        return {name: float("nan") for name in feature_names}

    counts = {name: 0 for name in feature_names}
    for _ in range(repeats):
        indices = rng.integers(0, len(X), size=len(X))
        selection = select_feature_subset(
            X[indices],
            Y_norm[indices],
            feature_names,
            max_features=max_features,
            min_features=min_features,
            candidate_top_k=candidate_top_k,
            regret_weight=regret_weight,
            top1_weight=top1_weight,
            spearman_weight=spearman_weight,
            feature_penalty=feature_penalty,
            one_se_rule=one_se_rule,
        )
        for idx in selection.selected_indices:
            counts[feature_names[idx]] += 1
    return {name: counts[name] / repeats for name in feature_names}


def run_shuffled_discovery(
    X: np.ndarray,
    Y_norm: np.ndarray,
    Y_raw: np.ndarray,
    crops: list[str],
    policies: list[str],
    feature_names: list[str],
    *,
    repeats: int,
    rng: np.random.Generator,
    max_features: int,
    min_features: int,
    candidate_top_k: int,
    regret_weight: float,
    top1_weight: float,
    spearman_weight: float,
    feature_penalty: float,
    one_se_rule: bool,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for repeat in range(max(0, repeats)):
        shuffled_X = X[rng.permutation(len(X))]
        preds, _, _ = nested_discovery_predict(
            shuffled_X,
            Y_norm,
            Y_raw,
            crops,
            policies,
            feature_names,
            max_features=max_features,
            min_features=min_features,
            candidate_top_k=candidate_top_k,
            regret_weight=regret_weight,
            top1_weight=top1_weight,
            spearman_weight=spearman_weight,
            feature_penalty=feature_penalty,
            one_se_rule=one_se_rule,
            keep_trace=False,
        )
        metrics = evaluate_predictions(Y_norm, preds, Y_raw)
        metrics["repeat"] = repeat
        rows.append(metrics)
    return pd.DataFrame(rows)


def write_report(
    *,
    output_dir: Path,
    transfer_dir: Path,
    source_trait_schema: str,
    policies: list[str],
    metrics: dict[str, float],
    shuffled_summary: dict[str, float],
    final_features: pd.DataFrame,
    global_selection: SelectionResult,
) -> None:
    lines = [
        "# Strategy-Supervised Trait Discovery",
        "",
        f"- Transfer directory: `{transfer_dir}`",
        f"- Source trait schema: `{source_trait_schema}`",
        f"- Specialist policies: `{', '.join(policies)}`",
        f"- Selected feature count: `{len(global_selection.selected_indices)}`",
        f"- Nested LOOCV top-1 specialist accuracy: `{metrics['top1_accuracy']:.3f}`",
        f"- Nested LOOCV mean normalized regret: `{metrics['mean_regret_norm']:.3f}`",
        f"- Nested LOOCV mean profile Spearman: `{metrics['mean_profile_spearman']:.3f}`",
    ]
    if shuffled_summary:
        lines.extend(
            [
                f"- Shuffled-traits top-1 accuracy mean: `{shuffled_summary['top1_accuracy_mean']:.3f}`",
                f"- Top-1 lift vs shuffled: `{metrics['top1_accuracy'] - shuffled_summary['top1_accuracy_mean']:.3f}`",
                f"- Regret reduction vs shuffled: `{shuffled_summary['mean_regret_norm_mean'] - metrics['mean_regret_norm']:.3f}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Final Selected Candidate Traits",
            "",
            final_features.head(30).to_markdown(index=False),
            "",
            "## Interpretation",
            "",
            "This script discovers traits using specialist-transfer profiles as the",
            "supervision target. It does not use final LLM/NN OOD policy performance",
            "to choose traits. The selected set should be treated as a candidate",
            "schema for the next policy-training stage, and should still be compared",
            "against shuffled traits and existing schemas.",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    transfer_dir = resolve_path(args.transfer_dir)
    output_dir = resolve_path(args.output_dir)
    traits_dir = resolve_path(args.traits_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exclude_policies = {
        policy.strip()
        for policy in str(args.exclude_policies).split(",")
        if policy.strip()
    }
    crops, policies, y_norm, y_raw = load_transfer_matrix(
        transfer_dir,
        exclude_policies=exclude_policies,
    )
    pool = load_candidate_pool(
        crops,
        traits_dir=traits_dir,
        source_trait_schema=str(args.source_trait_schema),
        include_derived=not bool(args.no_derived),
    )
    one_se_rule = not bool(args.disable_one_se_rule)
    max_features = 3 if bool(args.quick) else int(args.max_features)
    min_features = int(args.min_features)
    candidate_top_k = 12 if bool(args.quick) else int(args.candidate_top_k)
    shuffle_repeats = 1 if bool(args.quick) else int(args.shuffle_repeats)
    bootstrap_repeats = 0 if bool(args.quick) else int(args.bootstrap_repeats)

    rng = np.random.default_rng(int(args.seed))
    preds, folds, trace = nested_discovery_predict(
        pool.X,
        y_norm,
        y_raw,
        crops,
        policies,
        pool.feature_names,
        max_features=max_features,
        min_features=min_features,
        candidate_top_k=candidate_top_k,
        regret_weight=float(args.regret_weight),
        top1_weight=float(args.top1_weight),
        spearman_weight=float(args.spearman_weight),
        feature_penalty=float(args.feature_penalty),
        one_se_rule=one_se_rule,
        keep_trace=True,
    )
    metrics = evaluate_predictions(y_norm, preds, y_raw)

    shuffled = run_shuffled_discovery(
        pool.X,
        y_norm,
        y_raw,
        crops,
        policies,
        pool.feature_names,
        repeats=shuffle_repeats,
        rng=rng,
        max_features=max_features,
        min_features=min_features,
        candidate_top_k=candidate_top_k,
        regret_weight=float(args.regret_weight),
        top1_weight=float(args.top1_weight),
        spearman_weight=float(args.spearman_weight),
        feature_penalty=float(args.feature_penalty),
        one_se_rule=one_se_rule,
    )
    shuffled_summary = {}
    if not shuffled.empty:
        shuffled_summary = {
            f"{column}_mean": float(shuffled[column].mean())
            for column in [
                "top1_accuracy",
                "mean_regret_norm",
                "mean_profile_spearman",
            ]
        }
        shuffled_summary.update(
            {
                f"{column}_std": float(shuffled[column].std(ddof=0))
                for column in [
                    "top1_accuracy",
                    "mean_regret_norm",
                    "mean_profile_spearman",
                ]
            }
        )

    global_selection = select_feature_subset(
        pool.X,
        y_norm,
        pool.feature_names,
        max_features=max_features,
        min_features=min_features,
        candidate_top_k=candidate_top_k,
        regret_weight=float(args.regret_weight),
        top1_weight=float(args.top1_weight),
        spearman_weight=float(args.spearman_weight),
        feature_penalty=float(args.feature_penalty),
        one_se_rule=one_se_rule,
    )

    outer_freq = selection_frequency(pool.feature_names, folds["selected_features_json"])
    bootstrap_freq = bootstrap_feature_frequency(
        pool.X,
        y_norm,
        pool.feature_names,
        repeats=bootstrap_repeats,
        rng=rng,
        max_features=max_features,
        min_features=min_features,
        candidate_top_k=candidate_top_k,
        regret_weight=float(args.regret_weight),
        top1_weight=float(args.top1_weight),
        spearman_weight=float(args.spearman_weight),
        feature_penalty=float(args.feature_penalty),
        one_se_rule=one_se_rule,
    )

    final_selected = [pool.feature_names[idx] for idx in global_selection.selected_indices]
    final_rows = []
    for rank, name in enumerate(final_selected, start=1):
        row = {
            "rank": rank,
            "feature_name": name,
            "outer_fold_selection_frequency": outer_freq.get(name, 0.0),
            "bootstrap_selection_frequency": bootstrap_freq.get(name, float("nan")),
        }
        row.update(pool.metadata.get(name, {}))
        final_rows.append(row)
    final_features = pd.DataFrame(final_rows)

    all_feature_rows = []
    final_set = set(final_selected)
    for name in pool.feature_names:
        row = {
            "feature_name": name,
            "selected_in_final_schema": name in final_set,
            "outer_fold_selection_frequency": outer_freq.get(name, 0.0),
            "bootstrap_selection_frequency": bootstrap_freq.get(name, float("nan")),
        }
        row.update(pool.metadata.get(name, {}))
        all_feature_rows.append(row)
    feature_stability = pd.DataFrame(all_feature_rows).sort_values(
        ["selected_in_final_schema", "outer_fold_selection_frequency", "bootstrap_selection_frequency", "feature_name"],
        ascending=[False, False, False, True],
    )

    summary_row = {
        "source_trait_schema": str(args.source_trait_schema),
        "num_crops": len(crops),
        "num_policies": len(policies),
        "num_candidate_features": len(pool.feature_names),
        "max_features": max_features,
        "min_features": min_features,
        "candidate_top_k": candidate_top_k,
        "one_se_rule": one_se_rule,
        **metrics,
    }
    summary_row.update({f"shuffled_{key}": value for key, value in shuffled_summary.items()})
    pd.DataFrame([summary_row]).to_csv(output_dir / "discovery_summary.csv", index=False)
    folds.to_csv(output_dir / "nested_loocv_predictions_by_crop.csv", index=False)
    trace.to_csv(output_dir / "selection_trace_by_outer_fold.csv", index=False)
    final_features.to_csv(output_dir / "final_selected_features.csv", index=False)
    feature_stability.to_csv(output_dir / "feature_stability.csv", index=False)
    if not shuffled.empty:
        shuffled.to_csv(output_dir / "shuffled_trait_discovery_summary.csv", index=False)

    artifact = {
        "transfer_dir": str(transfer_dir),
        "source_trait_schema": str(args.source_trait_schema),
        "selected_features": final_selected,
        "selected_feature_metadata": {
            name: pool.metadata.get(name, {}) for name in final_selected
        },
        "policies": policies,
        "crops": crops,
        "trait_keys": pool.trait_keys,
        "nested_loocv_metrics": metrics,
        "shuffled_summary": shuffled_summary,
        "selection_config": {
            "max_features": int(args.max_features),
            "effective_max_features": max_features,
            "min_features": min_features,
            "candidate_top_k": candidate_top_k,
            "quick": bool(args.quick),
            "regret_weight": float(args.regret_weight),
            "top1_weight": float(args.top1_weight),
            "spearman_weight": float(args.spearman_weight),
            "feature_penalty": float(args.feature_penalty),
            "one_se_rule": one_se_rule,
            "include_derived": not bool(args.no_derived),
        },
    }
    (output_dir / "discovered_traits.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    write_report(
        output_dir=output_dir,
        transfer_dir=transfer_dir,
        source_trait_schema=str(args.source_trait_schema),
        policies=policies,
        metrics=metrics,
        shuffled_summary=shuffled_summary,
        final_features=final_features,
        global_selection=global_selection,
    )

    print(f"wrote {output_dir / 'discovery_summary.csv'}")
    print(f"wrote {output_dir / 'discovered_traits.json'}")
    print(f"wrote {output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
