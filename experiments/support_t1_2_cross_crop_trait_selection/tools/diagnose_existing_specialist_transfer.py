#!/usr/bin/env python
"""Diagnose cross-crop transfer from existing maize/wheat specialist NN policies."""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from agrimanager.env.base import create_nn_env_adapter, load_env_configs_from_parquet
from agrimanager.env.wofost_gym.crop_trait_schemas import DEFAULT_CROP_TRAIT_SCHEMA
from agrimanager.env.wofost_gym.crop_traits_observation import CropTraitEncoder
from agrimanager.nn_ppo.common import run_episode


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / "support_t1_2_cross_crop_trait_selection"
ANALYSIS_DIR = EXPERIMENT_DIR / "analysis"
DEFAULT_OUTPUT_DIR = ANALYSIS_DIR / "existing_specialist_transfer"
DEFAULT_TEST_FILE = (
    PROJECT_ROOT
    / "experiments"
    / "t1_2_cross_crop_trait_shift"
    / "data"
    / "cross_crop_16id_nn_without_traits"
    / "test.parquet"
)
DEFAULT_TRAITS_DIR = PROJECT_ROOT / "agrimanager" / "env" / "wofost_gym" / "crop_traits"

SPECIALIST_POLICIES = {
    "maize_weather_specialist": PROJECT_ROOT
    / "experiments"
    / "legacy_wofost_weather_generalization"
    / "results"
    / "nn_train"
    / "wofost_generalization_weather_maize_nn_16ep_matched_train",
    "wheat_weather_specialist": PROJECT_ROOT
    / "experiments"
    / "legacy_wofost_weather_generalization"
    / "results"
    / "nn_train"
    / "wofost_generalization_weather_wheat_nn_16ep_matched_train",
    "cotton_specialist": PROJECT_ROOT
    / "experiments"
    / "support_t1_2_cross_crop_trait_selection"
    / "results"
    / "nn_train"
    / "wofost_specialist_transfer_cotton_nn_train_16ep_n1",
    "rice_specialist": PROJECT_ROOT
    / "experiments"
    / "support_t1_2_cross_crop_trait_selection"
    / "results"
    / "nn_train"
    / "wofost_specialist_transfer_rice_nn_train_16ep_n1",
    "potato_specialist": PROJECT_ROOT
    / "experiments"
    / "support_t1_2_cross_crop_trait_selection"
    / "results"
    / "nn_train"
    / "wofost_specialist_transfer_potato_nn_train_16ep_n1",
    "sugarbeet_specialist": PROJECT_ROOT
    / "experiments"
    / "support_t1_2_cross_crop_trait_selection"
    / "results"
    / "nn_train"
    / "wofost_specialist_transfer_sugarbeet_nn_train_16ep_n1",
    "barley_specialist": PROJECT_ROOT
    / "experiments"
    / "support_t1_2_cross_crop_trait_selection"
    / "results"
    / "nn_train"
    / "wofost_specialist_transfer_barley_nn_train_16ep_n1",
    "seed_onion_specialist": PROJECT_ROOT
    / "experiments"
    / "support_t1_2_cross_crop_trait_selection"
    / "results"
    / "nn_train"
    / "wofost_specialist_transfer_seed_onion_nn_train_16ep_n1",
}

POLICY_ORDER = ["noop_template", "maize_weather_specialist", "wheat_weather_specialist"]
EIGHT_SPECIALIST_POLICY_ORDER = [
    "noop_template",
    "maize_weather_specialist",
    "wheat_weather_specialist",
    "cotton_specialist",
    "rice_specialist",
    "potato_specialist",
    "sugarbeet_specialist",
    "barley_specialist",
    "seed_onion_specialist",
]
RESOURCE_ORDER = ["N", "P", "K", "IRR"]
DVS_BINS = ["early", "vegetative", "reproductive", "late"]


@dataclass
class LoadedPolicy:
    name: str
    model: Any
    vecnormalize: VecNormalize | None = None


class NoopPolicy:
    """Stable-Baselines-like policy that always applies action 0."""

    def predict(self, observation, deterministic: bool = True):
        del observation, deterministic
        return np.asarray(0, dtype=np.int64), None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate existing maize/wheat weather-specialist NN policies on "
            "the cross-crop no-traits test set."
        )
    )
    parser.add_argument("--test-file", type=Path, default=DEFAULT_TEST_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--traits-dir", type=Path, default=DEFAULT_TRAITS_DIR)
    parser.add_argument("--trait-schema", default=DEFAULT_CROP_TRAIT_SCHEMA)
    parser.add_argument(
        "--policy-set",
        choices=("two", "eight"),
        default="two",
        help="Named policy set to evaluate when --policies is omitted.",
    )
    parser.add_argument(
        "--policy-map-file",
        type=Path,
        default=None,
        help="Optional JSON mapping policy names to result directories.",
    )
    parser.add_argument(
        "--policies",
        default=None,
        help="Comma-separated policy names to evaluate. Overrides --policy-set.",
    )
    parser.add_argument(
        "--max-scenarios-per-crop",
        type=int,
        default=None,
        help="Optional cap for fast smoke tests. Omit for full 128/crop evaluation.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shuffle-repeats", type=int, default=200)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def display_path(path: Path) -> str:
    resolved = resolve_path(path)
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def load_policy_map(policy_map_file: Path | None) -> dict[str, Path]:
    policies = dict(SPECIALIST_POLICIES)
    if policy_map_file is None:
        return policies

    resolved = resolve_path(policy_map_file)
    with resolved.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise TypeError(f"Policy map must be a JSON object: {resolved}")

    for name, value in raw.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Invalid policy name in {resolved}: {name!r}")
        if name == "noop_template":
            continue
        if not isinstance(value, str):
            raise TypeError(f"Policy path for {name!r} must be a string")
        policies[name] = resolve_path(Path(value))
    return policies


def policy_names_from_args(args: argparse.Namespace) -> list[str]:
    if args.policies:
        return [item.strip() for item in args.policies.split(",") if item.strip()]
    if args.policy_set == "eight":
        return list(EIGHT_SPECIALIST_POLICY_ORDER)
    return list(POLICY_ORDER)


def crop_from_config(env_config: dict[str, Any]) -> str:
    crop_name = str(env_config.get("crop_name", "") or "").strip()
    if crop_name:
        return crop_name
    agro_file = Path(str(env_config.get("agro_file", "") or "")).name
    if agro_file.endswith("_agro.yaml"):
        return agro_file[:-10]
    if agro_file.endswith(".yaml"):
        return agro_file[:-5]
    return agro_file or "unknown"


def scenario_from_config(env_config: dict[str, Any]) -> str:
    return str(env_config.get("scenario_id") or env_config.get("seed") or "")


def force_no_traits(env_config: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(env_config)
    cfg["include_crop_traits"] = False
    cfg["include_variety_traits"] = False
    cfg["llm_mode"] = False
    return cfg


def subset_env_configs(
    env_configs: list[dict[str, Any]],
    max_scenarios_per_crop: int | None,
) -> list[dict[str, Any]]:
    if max_scenarios_per_crop is None:
        return [force_no_traits(cfg) for cfg in env_configs]

    selected: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for raw_cfg in env_configs:
        cfg = force_no_traits(raw_cfg)
        crop = crop_from_config(cfg)
        current_count = counts.get(crop, 0)
        if current_count >= max_scenarios_per_crop:
            continue
        selected.append(cfg)
        counts[crop] = current_count + 1
    return selected


def load_vecnormalize(adapter, env_configs: list[dict[str, Any]], stats_path: Path) -> VecNormalize:
    first_env_config = dict(env_configs[0])
    dummy_vec_env = DummyVecEnv([lambda: adapter.make_env(first_env_config)])
    vecnormalize = VecNormalize.load(str(stats_path), dummy_vec_env)
    vecnormalize.training = False
    vecnormalize.norm_reward = False
    return vecnormalize


def load_policy(
    name: str,
    policy_map: dict[str, Path],
    adapter,
    env_configs: list[dict[str, Any]],
    *,
    device: str,
) -> LoadedPolicy:
    if name == "noop_template":
        return LoadedPolicy(name=name, model=NoopPolicy(), vecnormalize=None)
    if name not in policy_map:
        raise ValueError(f"Unknown policy {name!r}. Available: {sorted(policy_map)}")

    policy_dir = policy_map[name]
    agent_path = policy_dir / "agent.zip"
    vecnormalize_path = policy_dir / "vecnormalize.pkl"
    if not agent_path.exists():
        raise FileNotFoundError(agent_path)
    if not vecnormalize_path.exists():
        raise FileNotFoundError(vecnormalize_path)

    model = PPO.load(str(agent_path), device=device)
    vecnormalize = load_vecnormalize(adapter, env_configs, vecnormalize_path)
    return LoadedPolicy(name=name, model=model, vecnormalize=vecnormalize)


def action_to_int(action: Any) -> int:
    if action is None:
        return 0
    if isinstance(action, (int, np.integer)):
        return int(action)
    if isinstance(action, float) and float(action).is_integer():
        return int(action)
    if isinstance(action, (list, tuple)) and action:
        return action_to_int(action[0])
    if hasattr(action, "item"):
        return int(action.item())
    if hasattr(action, "tolist"):
        return action_to_int(action.tolist())
    return int(action)


def action_amount_scale(env_config: dict[str, Any]) -> float:
    if bool(env_config.get("scale_action_amounts_by_interval", False)):
        return float(env_config.get("intvn_interval", 1.0) or 1.0)
    return 1.0


def decode_action(action_id: int, env_config: dict[str, Any]) -> dict[str, float | str]:
    if action_id <= 0:
        return {"resource": "NONE", "amount": 0.0}

    num_fert = 4
    scale = action_amount_scale(env_config)
    fert_amount = float(env_config.get("fert_amount", 20.0) or 20.0) * scale
    irrig_amount = float(env_config.get("irrig_amount", 5.0) or 5.0) * scale

    if action_id <= num_fert:
        return {"resource": "N", "amount": float(action_id * fert_amount)}
    if action_id <= 2 * num_fert:
        slot = action_id - num_fert
        return {"resource": "P", "amount": float(slot * fert_amount)}
    if action_id <= 3 * num_fert:
        slot = action_id - 2 * num_fert
        return {"resource": "K", "amount": float(slot * fert_amount)}

    slot = action_id - 3 * num_fert
    return {"resource": "IRR", "amount": float(slot * irrig_amount)}


def dvs_bin(dvs: float | None) -> str:
    if dvs is None or not math.isfinite(float(dvs)):
        return "unknown"
    value = float(dvs)
    if value < 0.5:
        return "early"
    if value < 1.0:
        return "vegetative"
    if value < 1.5:
        return "reproductive"
    return "late"


def summarize_episode(
    *,
    policy_name: str,
    result: dict[str, Any],
    final_info: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    env_config = dict(result.get("env_config") or {})
    crop = crop_from_config(env_config)
    scenario_id = scenario_from_config(env_config)
    turns = list(result.get("turns") or [])
    final_wso = (
        dict(final_info.get("trajectory_metrics") or {}).get("final_wso")
        if final_info
        else None
    )
    if final_wso is None and turns:
        final_wso = dict(turns[-1].get("turn_metrics") or {}).get("wso")
    final_wso = float(final_wso or 0.0)

    totals = {f"total_{resource.lower()}": 0.0 for resource in RESOURCE_ORDER}
    counts = {f"count_{resource.lower()}": 0 for resource in RESOURCE_ORDER}
    turn_rows: list[dict[str, Any]] = []
    for turn in turns:
        if turn.get("action") is None:
            continue
        action_id = action_to_int(turn.get("action"))
        decoded = decode_action(action_id, env_config)
        resource = str(decoded["resource"])
        amount = float(decoded["amount"])
        metrics = dict(turn.get("turn_metrics") or {})
        current_dvs = float(metrics.get("dvs", float("nan")))
        bin_name = dvs_bin(current_dvs)
        if resource in RESOURCE_ORDER:
            totals[f"total_{resource.lower()}"] += amount
            counts[f"count_{resource.lower()}"] += 1
        turn_rows.append(
            {
                "policy": policy_name,
                "crop": crop,
                "scenario_id": scenario_id,
                "turn": int(turn.get("turn", len(turn_rows) + 1)),
                "action_id": action_id,
                "resource": resource,
                "amount": amount,
                "dvs": current_dvs,
                "dvs_bin": bin_name,
                "wso": float(metrics.get("wso", float("nan"))),
            }
        )

    episode_row: dict[str, Any] = {
        "policy": policy_name,
        "crop": crop,
        "scenario_id": scenario_id,
        "final_wso": final_wso,
        "num_turns": max(0, len(turns) - 1),
    }
    episode_row.update(totals)
    episode_row.update(counts)
    return episode_row, turn_rows


def evaluate_policies(
    policies: list[LoadedPolicy],
    adapter,
    env_configs: list[dict[str, Any]],
    *,
    deterministic: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    episode_rows: list[dict[str, Any]] = []
    turn_rows: list[dict[str, Any]] = []

    for policy in policies:
        print(f"[diagnostic] evaluating policy={policy.name} on {len(env_configs)} scenarios")
        for idx, env_config in enumerate(env_configs, start=1):
            if idx % 100 == 0:
                print(f"  {policy.name}: {idx}/{len(env_configs)}")
            env = adapter.make_env(dict(env_config))
            try:
                result, final_info = run_episode(
                    env,
                    policy.model,
                    adapter=adapter,
                    vecnormalize=policy.vecnormalize,
                    deterministic=deterministic,
                )
                result["env_config"] = dict(env_config)
                episode_row, episode_turn_rows = summarize_episode(
                    policy_name=policy.name,
                    result=result,
                    final_info=final_info,
                )
                episode_rows.append(episode_row)
                turn_rows.extend(episode_turn_rows)
            finally:
                env.close()

    return pd.DataFrame(episode_rows), pd.DataFrame(turn_rows)


def make_summary_tables(
    episodes: pd.DataFrame,
    turns: pd.DataFrame,
    output_dir: Path,
    policy_order: list[str],
) -> dict[str, pd.DataFrame]:
    policy_summary = (
        episodes.groupby(["crop", "policy"], as_index=False)
        .agg(
            num_scenarios=("final_wso", "count"),
            final_wso_mean=("final_wso", "mean"),
            final_wso_std=("final_wso", "std"),
            total_n_mean=("total_n", "mean"),
            total_p_mean=("total_p", "mean"),
            total_k_mean=("total_k", "mean"),
            total_irr_mean=("total_irr", "mean"),
            count_n_mean=("count_n", "mean"),
            count_p_mean=("count_p", "mean"),
            count_k_mean=("count_k", "mean"),
            count_irr_mean=("count_irr", "mean"),
        )
        .sort_values(["crop", "policy"])
    )
    policy_summary["policy_rank_in_crop"] = (
        policy_summary.groupby("crop")["final_wso_mean"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    transfer = policy_summary.pivot(
        index="crop",
        columns="policy",
        values="final_wso_mean",
    ).reset_index()
    transfer = transfer[["crop", *[p for p in policy_order if p in transfer.columns]]]

    policy_ranking = policy_summary[
        ["crop", "policy", "policy_rank_in_crop", "final_wso_mean", "num_scenarios"]
    ].sort_values(["crop", "policy_rank_in_crop", "policy"])
    best_wso_by_crop = policy_ranking.groupby("crop")["final_wso_mean"].transform("max")
    policy_ranking["gap_to_crop_best_wso"] = policy_ranking["final_wso_mean"] - best_wso_by_crop
    policy_ranking["relative_gap_to_crop_best_pct"] = (
        policy_ranking["gap_to_crop_best_wso"] / best_wso_by_crop.clip(lower=1.0) * 100.0
    )

    normalized_rows = []
    for _, row in transfer.iterrows():
        crop = row["crop"]
        values = {
            policy: float(row[policy])
            for policy in transfer.columns
            if policy != "crop" and pd.notna(row[policy])
        }
        base = values.get("noop_template", min(values.values()))
        best = max(values.values())
        denom = best - base
        for policy, value in values.items():
            normalized = 0.0 if abs(denom) < 1e-8 else (value - base) / denom
            normalized_rows.append(
                {
                    "crop": crop,
                    "policy": policy,
                    "final_wso_mean": value,
                    "crop_normalized_score": normalized,
                }
            )
    normalized_transfer = pd.DataFrame(normalized_rows)

    gap_rows = []
    for _, row in transfer.iterrows():
        if "maize_weather_specialist" not in row or "wheat_weather_specialist" not in row:
            continue
        maize = float(row["maize_weather_specialist"])
        wheat = float(row["wheat_weather_specialist"])
        gap = maize - wheat
        gap_pct = gap / max(abs(wheat), 1.0) * 100.0
        gap_rows.append(
            {
                "crop": row["crop"],
                "maize_wso": maize,
                "wheat_wso": wheat,
                "maize_minus_wheat": gap,
                "maize_minus_wheat_pct_of_wheat": gap_pct,
                "best_specialist": (
                    "maize_weather_specialist"
                    if gap >= 0
                    else "wheat_weather_specialist"
                ),
                "abs_gap_gt_10pct": abs(gap_pct) > 10.0,
            }
        )
    if gap_rows:
        policy_gap = pd.DataFrame(gap_rows).sort_values("maize_minus_wheat_pct_of_wheat")
    else:
        policy_gap = pd.DataFrame(
            columns=[
                "crop",
                "maize_wso",
                "wheat_wso",
                "maize_minus_wheat",
                "maize_minus_wheat_pct_of_wheat",
                "best_specialist",
                "abs_gap_gt_10pct",
            ]
        )

    if not turns.empty:
        sequence_rows = []
        for keys, group in turns.sort_values("turn").groupby(
            ["policy", "crop", "scenario_id"],
            dropna=False,
        ):
            policy, crop, scenario_id = keys
            sequence_rows.append(
                {
                    "policy": policy,
                    "crop": crop,
                    "scenario_id": scenario_id,
                    "action_ids": " ".join(str(int(value)) for value in group["action_id"]),
                    "resources": " ".join(str(value) for value in group["resource"]),
                    "amounts": " ".join(f"{float(value):.2f}" for value in group["amount"]),
                }
            )
        action_sequences = pd.DataFrame(sequence_rows).sort_values(["crop", "policy"])
        action_by_bin = (
            turns[turns["resource"].isin(RESOURCE_ORDER)]
            .groupby(["policy", "crop", "resource", "dvs_bin"], as_index=False)
            .agg(action_count=("action_id", "count"), amount_sum=("amount", "sum"))
            .sort_values(["policy", "crop", "resource", "dvs_bin"])
        )
    else:
        action_sequences = pd.DataFrame(
            columns=["policy", "crop", "scenario_id", "action_ids", "resources", "amounts"]
        )
        action_by_bin = pd.DataFrame(
            columns=["policy", "crop", "resource", "dvs_bin", "action_count", "amount_sum"]
        )

    tables = {
        "episode_rollouts": episodes,
        "turn_actions": turns,
        "episode_action_sequences": action_sequences,
        "per_crop_policy_summary": policy_summary,
        "policy_ranking_by_crop": policy_ranking,
        "policy_transfer_matrix": transfer,
        "policy_transfer_matrix_normalized": normalized_transfer,
        "maize_vs_wheat_gap_by_crop": policy_gap,
        "action_by_dvs_bin": action_by_bin,
    }
    for name, frame in tables.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)
    return tables


def rankdata(values: np.ndarray) -> np.ndarray:
    series = pd.Series(values)
    return series.rank(method="average").to_numpy(dtype=float)


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def ridge_loocv(X: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    preds = np.zeros_like(y, dtype=float)
    for heldout in range(len(y)):
        train_mask = np.ones(len(y), dtype=bool)
        train_mask[heldout] = False
        X_train = X[train_mask]
        y_train = y[train_mask]
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        std[std == 0.0] = 1.0
        X_train_std = (X_train - mean) / std
        X_test_std = (X[heldout : heldout + 1] - mean) / std
        design = np.column_stack([np.ones(len(X_train_std)), X_train_std])
        reg = np.eye(design.shape[1], dtype=float) * float(alpha)
        reg[0, 0] = 0.0
        coef = np.linalg.pinv(design.T @ design + reg) @ design.T @ y_train
        preds[heldout] = float((np.column_stack([np.ones(1), X_test_std]) @ coef).item())
    return preds


def majority_loocv_accuracy(y: np.ndarray) -> float:
    labels = y >= 0.0
    preds = np.zeros_like(labels, dtype=bool)
    for heldout in range(len(labels)):
        train = np.delete(labels, heldout)
        preds[heldout] = np.mean(train) >= 0.5
    return float(np.mean(preds == labels))


def trait_key_for_crop(encoder: CropTraitEncoder, crop: str) -> str:
    """Map dataset crop names to crop-variety trait keys."""
    if crop in encoder.crop_names:
        return crop

    prefix = f"{crop}__"
    matches = [name for name in encoder.crop_names if name.startswith(prefix)]
    if len(matches) == 1:
        return matches[0]

    available = ", ".join(encoder.crop_names)
    if not matches:
        raise KeyError(f"No trait card found for crop '{crop}'. Available: {available}")
    raise KeyError(f"Ambiguous trait cards for crop '{crop}': {', '.join(matches)}")


def analyze_traits(
    policy_gap: pd.DataFrame,
    *,
    traits_dir: Path,
    trait_schema: str,
    output_dir: Path,
    seed: int,
    shuffle_repeats: int,
) -> dict[str, Any]:
    if policy_gap.empty:
        return {"available": False, "reason": "maize/wheat policy gap table is empty"}

    encoder = CropTraitEncoder(traits_dir=traits_dir, trait_schema=trait_schema)
    crops = policy_gap["crop"].tolist()
    trait_keys = [trait_key_for_crop(encoder, crop) for crop in crops]
    X = np.vstack([encoder.vector_for_crop(trait_key) for trait_key in trait_keys]).astype(float)
    y = policy_gap["maize_minus_wheat_pct_of_wheat"].to_numpy(dtype=float)
    labels = y >= 0.0

    trait_rows = []
    for crop, trait_key, vector in zip(crops, trait_keys, X, strict=True):
        row = {"crop": crop, "trait_key": trait_key}
        row.update({name: float(value) for name, value in zip(encoder.feature_names, vector)})
        trait_rows.append(row)
    pd.DataFrame(trait_rows).to_csv(output_dir / "trait_features.csv", index=False)

    maize_vec = encoder.vector_for_crop(trait_key_for_crop(encoder, "maize")).astype(float)
    wheat_vec = encoder.vector_for_crop(trait_key_for_crop(encoder, "wheat")).astype(float)
    dist_to_maize = np.linalg.norm(X - maize_vec, axis=1)
    dist_to_wheat = np.linalg.norm(X - wheat_vec, axis=1)
    anchor_score = dist_to_wheat - dist_to_maize
    anchor_pred = anchor_score >= 0.0
    anchor_accuracy = float(np.mean(anchor_pred == labels))
    anchor_spearman = spearman_corr(anchor_score, y)

    rng = np.random.default_rng(seed)
    null_corrs = []
    for _ in range(max(1, shuffle_repeats)):
        null_corrs.append(spearman_corr(anchor_score, rng.permutation(y)))
    null_corrs_arr = np.asarray(null_corrs, dtype=float)
    corr_p = float(np.mean(np.abs(null_corrs_arr) >= abs(anchor_spearman)))

    ridge_pred = ridge_loocv(X, y, alpha=1.0)
    ridge_accuracy = float(np.mean((ridge_pred >= 0.0) == labels))
    ridge_mae = float(np.mean(np.abs(ridge_pred - y)))
    majority_accuracy = majority_loocv_accuracy(y)

    shuffled_accuracies = []
    shuffled_maes = []
    for _ in range(max(1, shuffle_repeats)):
        shuffled_X = X[rng.permutation(len(X))]
        shuffled_pred = ridge_loocv(shuffled_X, y, alpha=1.0)
        shuffled_accuracies.append(float(np.mean((shuffled_pred >= 0.0) == labels)))
        shuffled_maes.append(float(np.mean(np.abs(shuffled_pred - y))))

    details = policy_gap[["crop", "maize_minus_wheat_pct_of_wheat"]].copy()
    details["trait_key"] = trait_keys
    details["dist_to_maize"] = dist_to_maize
    details["dist_to_wheat"] = dist_to_wheat
    details["trait_anchor_score"] = anchor_score
    details["trait_anchor_prediction"] = np.where(anchor_pred, "maize", "wheat")
    details["actual_better_specialist"] = np.where(labels, "maize", "wheat")
    details["ridge_predicted_gap_pct"] = ridge_pred
    details["ridge_prediction"] = np.where(ridge_pred >= 0.0, "maize", "wheat")
    details.to_csv(output_dir / "trait_predictability_by_crop.csv", index=False)

    summary = {
        "available": True,
        "num_crops": int(len(crops)),
        "trait_schema": trait_schema,
        "trait_dim": int(encoder.dim),
        "anchor_accuracy": anchor_accuracy,
        "anchor_spearman": anchor_spearman,
        "anchor_spearman_permutation_p": corr_p,
        "ridge_loocv_accuracy": ridge_accuracy,
        "ridge_loocv_mae": ridge_mae,
        "majority_loocv_accuracy": majority_accuracy,
        "shuffled_ridge_accuracy_mean": float(np.mean(shuffled_accuracies)),
        "shuffled_ridge_accuracy_std": float(np.std(shuffled_accuracies)),
        "shuffled_ridge_mae_mean": float(np.mean(shuffled_maes)),
        "ridge_accuracy_lift_vs_shuffled": ridge_accuracy - float(np.mean(shuffled_accuracies)),
    }
    pd.DataFrame([summary]).to_csv(output_dir / "trait_predictability_summary.csv", index=False)
    return summary


def plot_heatmap(transfer: pd.DataFrame, figure_dir: Path, policy_order: list[str]) -> None:
    if transfer.empty:
        return
    frame = transfer.set_index("crop")
    policies = [policy for policy in policy_order if policy in frame.columns]
    values = frame[policies].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(max(7.8, 0.82 * len(policies)), max(5.0, 0.34 * len(frame))))
    im = ax.imshow(values, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(policies)))
    ax.set_xticklabels([p.replace("_", "\n") for p in policies], fontsize=8)
    ax.set_yticks(np.arange(len(frame.index)))
    ax.set_yticklabels(frame.index)
    ax.set_title("Cross-Crop Final WSO by Existing Specialist Policy")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Final WSO mean")
    fig.tight_layout()
    fig.savefig(figure_dir / "crop_policy_wso_heatmap.png", dpi=220)
    plt.close(fig)


def plot_policy_gap(policy_gap: pd.DataFrame, figure_dir: Path) -> None:
    if policy_gap.empty:
        return
    frame = policy_gap.sort_values("maize_minus_wheat_pct_of_wheat")
    colors = [
        "#2C6E49" if value >= 0.0 else "#C04B32"
        for value in frame["maize_minus_wheat_pct_of_wheat"]
    ]
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.bar(frame["crop"], frame["maize_minus_wheat_pct_of_wheat"], color=colors)
    ax.axhline(0, color="#222222", linewidth=0.8)
    ax.axhline(10, color="#888888", linewidth=0.8, linestyle="--")
    ax.axhline(-10, color="#888888", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Maize specialist gap vs wheat (%)")
    ax.set_title("Specialist Transfer Gap by Crop")
    ax.tick_params(axis="x", labelrotation=45)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "maize_minus_wheat_gap_by_crop.png", dpi=220)
    plt.close(fig)


def plot_resource_totals(summary: pd.DataFrame, figure_dir: Path, policy_order: list[str]) -> None:
    if summary.empty:
        return
    policies = [p for p in policy_order if p in set(summary["policy"])]
    crops = sorted(summary["crop"].unique())
    fig, axes = plt.subplots(len(RESOURCE_ORDER), 1, figsize=(11.5, 9.0), sharex=True)
    colors = {
        "noop_template": "#6B7280",
        "maize_weather_specialist": "#2C6E49",
        "wheat_weather_specialist": "#C04B32",
        "cotton_specialist": "#2563EB",
        "rice_specialist": "#7C3AED",
        "potato_specialist": "#D97706",
        "sugarbeet_specialist": "#DB2777",
        "barley_specialist": "#059669",
        "seed_onion_specialist": "#0F766E",
    }
    x = np.arange(len(crops), dtype=float)
    width = 0.8 / max(1, len(policies))
    for ax, resource in zip(axes, RESOURCE_ORDER, strict=True):
        col = f"total_{resource.lower()}_mean"
        for offset_idx, policy in enumerate(policies):
            values = []
            for crop in crops:
                match = summary[(summary["crop"] == crop) & (summary["policy"] == policy)]
                values.append(float(match[col].iloc[0]) if not match.empty else 0.0)
            offsets = x - 0.4 + width / 2 + offset_idx * width
            ax.bar(
                offsets,
                values,
                width=width,
                label=policy.replace("_", " "),
                color=colors.get(policy, "#4B5563"),
            )
        ax.set_ylabel(resource)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, ncol=min(4, max(1, len(policies))), fontsize=8)
    axes[0].set_title("Mean Seasonal Resource Input by Crop and Policy")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(crops, rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(figure_dir / "resource_totals_by_crop_policy.png", dpi=220)
    plt.close(fig)


def plot_trait_scatter(trait_details_path: Path, figure_dir: Path) -> None:
    if not trait_details_path.exists():
        return
    details = pd.read_csv(trait_details_path)
    if details.empty:
        return
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    colors = [
        "#2C6E49" if value >= 0.0 else "#C04B32"
        for value in details["maize_minus_wheat_pct_of_wheat"]
    ]
    ax.scatter(
        details["trait_anchor_score"],
        details["maize_minus_wheat_pct_of_wheat"],
        color=colors,
        s=42,
    )
    for _, row in details.iterrows():
        ax.annotate(
            str(row["crop"]),
            (row["trait_anchor_score"], row["maize_minus_wheat_pct_of_wheat"]),
            fontsize=7,
            xytext=(3, 3),
            textcoords="offset points",
        )
    ax.axhline(0, color="#222222", linewidth=0.8)
    ax.axvline(0, color="#222222", linewidth=0.8)
    ax.set_xlabel("Trait anchor score (positive = closer to maize than wheat)")
    ax.set_ylabel("Maize specialist gap vs wheat (%)")
    ax.set_title("Trait Distance Anchor vs Specialist Transfer Gap")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "trait_anchor_vs_policy_gap.png", dpi=220)
    plt.close(fig)


def write_report(
    *,
    output_dir: Path,
    args: argparse.Namespace,
    policy_order: list[str],
    tables: dict[str, pd.DataFrame],
    trait_summary: dict[str, Any],
) -> None:
    policy_gap = tables["maize_vs_wheat_gap_by_crop"]
    summary = tables["per_crop_policy_summary"]
    num_crops = int(policy_gap["crop"].nunique()) if not policy_gap.empty else 0
    gap_gt_10 = int(policy_gap["abs_gap_gt_10pct"].sum()) if not policy_gap.empty else 0
    maize_better = int((policy_gap["maize_minus_wheat"] >= 0).sum()) if not policy_gap.empty else 0
    wheat_better = int((policy_gap["maize_minus_wheat"] < 0).sum()) if not policy_gap.empty else 0
    claim1_pass = gap_gt_10 >= 6 and maize_better > 0 and wheat_better > 0
    ranking = tables["policy_ranking_by_crop"]
    best_counts = {}
    if not ranking.empty:
        best_counts = (
            ranking[ranking["policy_rank_in_crop"] == 1]["policy"]
            .value_counts()
            .sort_index()
            .to_dict()
        )

    resource_cols = [f"total_{resource.lower()}_mean" for resource in RESOURCE_ORDER]
    resource_gap_lines = []
    for resource, col in zip(RESOURCE_ORDER, resource_cols, strict=True):
        maize = summary[summary["policy"] == "maize_weather_specialist"][col].mean()
        wheat = summary[summary["policy"] == "wheat_weather_specialist"][col].mean()
        resource_gap_lines.append(f"- `{resource}` mean seasonal input: maize={maize:.2f}, wheat={wheat:.2f}")

    claim2_pass = False
    if trait_summary.get("available"):
        claim2_pass = (
            float(trait_summary["ridge_accuracy_lift_vs_shuffled"]) >= 0.15
            and math.isfinite(float(trait_summary["anchor_spearman"]))
        )

    lines = [
        "# Existing Specialist Transfer Diagnostic",
        "",
        "This is a quick gate, not final paper-level evidence. It reuses two existing no-traits",
        "single-crop NN policies and evaluates them on the same cross-crop no-traits test set.",
        "",
        "## Run Configuration",
        "",
        f"- Test file: `{display_path(args.test_file)}`",
        f"- Max scenarios per crop: `{args.max_scenarios_per_crop if args.max_scenarios_per_crop is not None else 'all'}`",
        f"- Deterministic policy evaluation: `{args.deterministic}`",
        f"- Policy set: `{args.policy_set}`",
        f"- Policy map file: `{display_path(args.policy_map_file) if args.policy_map_file else 'built-in'}`",
        f"- Policies: `{','.join(policy_order)}`",
        "",
        "## Strategy Difference Check",
        "",
        f"- Crops evaluated: `{num_crops}`",
        f"- Crops with |maize-vs-wheat gap| > 10%: `{gap_gt_10}`",
        f"- Maize specialist better on: `{maize_better}` crops",
        f"- Wheat specialist better on: `{wheat_better}` crops",
        f"- Claim 1 quick-gate status: `{'PASS' if claim1_pass else 'WEAK/FAIL'}`",
        "",
        *resource_gap_lines,
        "",
        "## Specialist Transfer Matrix",
        "",
        "- Best-policy crop counts:",
        *[
            f"  - `{policy}`: `{count}`"
            for policy, count in best_counts.items()
        ],
        "",
        "This section is descriptive for policy sets larger than the two maize/wheat",
        "anchors. Full trait-schema selection still requires the nested crop-held-out",
        "procedure associated with `research/paper/experiment_cards/T1_2_cross_crop_trait_shift.md`.",
        "",
        "Figures:",
        "",
        "- ![Crop policy WSO heatmap](./figures/crop_policy_wso_heatmap.png)",
        "- ![Maize minus wheat gap by crop](./figures/maize_minus_wheat_gap_by_crop.png)",
        "- ![Resource totals by crop and policy](./figures/resource_totals_by_crop_policy.png)",
        "",
        "## Trait Predictability Check",
        "",
    ]
    if trait_summary.get("available"):
        lines.extend(
            [
                f"- Trait schema: `{trait_summary['trait_schema']}`",
                f"- Trait dimension: `{trait_summary['trait_dim']}`",
                f"- Ridge LOOCV accuracy: `{trait_summary['ridge_loocv_accuracy']:.3f}`",
                f"- Shuffled-trait ridge accuracy mean: `{trait_summary['shuffled_ridge_accuracy_mean']:.3f}`",
                f"- Accuracy lift vs shuffled: `{trait_summary['ridge_accuracy_lift_vs_shuffled']:.3f}`",
                f"- Trait anchor accuracy: `{trait_summary['anchor_accuracy']:.3f}`",
                f"- Trait anchor Spearman: `{trait_summary['anchor_spearman']:.3f}`",
                f"- Permutation p-value: `{trait_summary['anchor_spearman_permutation_p']:.3f}`",
                f"- Claim 2 quick-gate status: `{'PASS' if claim2_pass else 'WEAK/FAIL'}`",
                "",
                "Figure:",
                "",
                "- ![Trait anchor vs policy gap](./figures/trait_anchor_vs_policy_gap.png)",
            ]
        )
    else:
        lines.append(f"- Trait analysis unavailable: `{trait_summary.get('reason', 'unknown')}`")

    lines.extend(
        [
            "",
            "## Output Tables",
            "",
            "- `episode_rollouts.csv`",
            "- `turn_actions.csv`",
            "- `episode_action_sequences.csv`",
            "- `per_crop_policy_summary.csv`",
            "- `policy_ranking_by_crop.csv`",
            "- `policy_transfer_matrix.csv`",
            "- `policy_transfer_matrix_normalized.csv`",
            "- `maize_vs_wheat_gap_by_crop.csv`",
            "- `action_by_dvs_bin.csv`",
            "- `trait_predictability_summary.csv`",
            "- `trait_predictability_by_crop.csv`",
            "",
            "## Limit Check",
            "",
            "Only two specialist policies are tested here. A positive result is useful as a quick",
            "sanity check, but it does not prove that `traits_v1_23d` can infer arbitrary crop",
            "strategies. If this gate is positive, the next step should train 4-8 crop specialists",
            "and build a full specialist-transfer matrix.",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    status = {
        "claim1_strategy_difference_pass": claim1_pass,
        "claim2_trait_predictability_pass": claim2_pass,
        "num_crops": num_crops,
        "gap_gt_10pct_crops": gap_gt_10,
        "maize_better_crops": maize_better,
        "wheat_better_crops": wheat_better,
        "trait_summary": trait_summary,
    }
    with (output_dir / "quick_gate_status.json").open("w", encoding="utf-8") as handle:
        json.dump(status, handle, indent=2, sort_keys=True)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    test_file = resolve_path(args.test_file)
    output_dir = resolve_path(args.output_dir)
    figure_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    env_configs, env_name, dataset_path = load_env_configs_from_parquet(test_file)
    env_configs = subset_env_configs(env_configs, args.max_scenarios_per_crop)
    adapter = create_nn_env_adapter(env_name)

    policy_map = load_policy_map(args.policy_map_file)
    selected_policy_names = policy_names_from_args(args)
    policies = [
        load_policy(name, policy_map, adapter, env_configs, device=args.device)
        for name in selected_policy_names
    ]

    print("=" * 80)
    print("Existing Specialist Transfer Diagnostic")
    print("=" * 80)
    print(f"Dataset: {dataset_path}")
    print(f"Env: {env_name}")
    print(f"Scenarios: {len(env_configs)}")
    print(f"Output: {output_dir}")
    print(f"Policies: {selected_policy_names}")
    print("=" * 80)

    episodes, turns = evaluate_policies(
        policies,
        adapter,
        env_configs,
        deterministic=bool(args.deterministic),
    )
    tables = make_summary_tables(episodes, turns, output_dir, selected_policy_names)
    trait_summary = analyze_traits(
        tables["maize_vs_wheat_gap_by_crop"],
        traits_dir=resolve_path(args.traits_dir),
        trait_schema=str(args.trait_schema),
        output_dir=output_dir,
        seed=int(args.seed),
        shuffle_repeats=int(args.shuffle_repeats),
    )

    plot_heatmap(tables["policy_transfer_matrix"], figure_dir, selected_policy_names)
    plot_policy_gap(tables["maize_vs_wheat_gap_by_crop"], figure_dir)
    plot_resource_totals(tables["per_crop_policy_summary"], figure_dir, selected_policy_names)
    plot_trait_scatter(output_dir / "trait_predictability_by_crop.csv", figure_dir)
    write_report(
        output_dir=output_dir,
        args=args,
        policy_order=selected_policy_names,
        tables=tables,
        trait_summary=trait_summary,
    )

    for policy in policies:
        if policy.vecnormalize is not None:
            policy.vecnormalize.venv.close()

    print(f"wrote {output_dir / 'report.md'}")
    print(f"wrote {output_dir / 'quick_gate_status.json'}")


if __name__ == "__main__":
    main()
