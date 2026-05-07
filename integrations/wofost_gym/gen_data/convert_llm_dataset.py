#!/usr/bin/env python3
"""
Convert WOFOST-Gym rollout data (.npz) into LLM SFT train/test JSON using the
prompt helper from agrimanager.env.wofost_gym.prompt and dataset split config.

Example:
    python convert_llm_dataset.py \\
        --data-path results/wheat_agro_daily_wso/DQN/DQN_wheat_agro_daily_wso_bs128/data/test_data.npz \\
        --env-config results/wheat_agro_daily_wso/DQN/DQN_wheat_agro_daily_wso_bs128/config.yaml \\
        --dataset-config /path/to/legacy_year_split.yaml \\
        --output-dir llm_data
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, date
from pathlib import Path
from typing import Iterable, List, Tuple, Optional

import numpy as np
from omegaconf import OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from agrimanager.env.wofost_gym.prompt import WOFOSTPromptGenerator
except Exception as exc:  # pragma: no cover - import guard
    raise SystemExit(f"Failed to import prompt generator: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert WOFOST-Gym npz to LLM SFT JSON.")
    parser.add_argument("--data-path", type=Path, required=True, help="Path to data npz file.")
    parser.add_argument(
        "--env-config",
        type=Path,
        help="Path to environment/config.yaml (defaults to sibling of data/ directory).",
    )
    parser.add_argument(
        "--dataset-config",
        type=Path,
        required=True,
        help="Legacy dataset split config with train_years/test_years.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "integrations/wofost_gym/llm_data",
        help="Directory to write train/test/meta JSON files.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="wofost_wheat_sft",
        help="Prefix for output files (e.g., <prefix>_train.json).",
    )
    return parser.parse_args()


def compute_season_length(ag_cfg: dict) -> int:
    """Return season length from agro config."""
    start_date = ag_cfg.get("crop_start_date")
    end_date = ag_cfg.get("crop_end_date")
    if start_date and end_date:
        try:
            start = datetime.fromisoformat(str(start_date))
            end = datetime.fromisoformat(str(end_date))
            days = (end - start).days - 1  # end is exclusive in WOFOST
            if days > 0:
                return days
        except Exception:
            pass
    max_duration = ag_cfg.get("max_duration")
    return int(max_duration) if max_duration else 241


def parse_date(val) -> Optional[date]:
    """Parse a value to date if possible."""
    if val is None:
        return None
    if isinstance(val, date):
        return val
    try:
        return datetime.fromisoformat(str(val)).date()
    except Exception:
        return None


def extract_start_date_from_infos(infos: np.ndarray) -> Optional[date]:
    """Extract earliest calendar date from infos dicts (if present)."""
    earliest = None
    for info in infos:
        if not isinstance(info, dict):
            continue
        for val in info.values():
            if isinstance(val, dict):
                for k, v in val.items():
                    for candidate in (k, v):
                        if isinstance(candidate, date):
                            if earliest is None or candidate < earliest:
                                earliest = candidate
    return earliest


def build_prompt_generator(
    env_cfg: dict,
    dataset_cfg: dict,
    output_vars: List[str],
    start_date_override: Optional[date] = None,
) -> WOFOSTPromptGenerator:
    """Instantiate prompt generator using env config values."""
    npk_cfg = env_cfg.get("npk", {})
    ag_cfg = npk_cfg.get("ag", {}) or {}
    fert_amount = npk_cfg.get("fert_amount", 2.0)
    irrig_amount = npk_cfg.get("irrig_amount", 0.5)
    num_fert = npk_cfg.get("num_fert", 4)
    num_irrig = npk_cfg.get("num_irrig", 4)
    intervention_interval = npk_cfg.get("intvn_interval", 1)

    # Crop and location info
    crop_name = ag_cfg.get("crop_name") or dataset_cfg.get("dataset_id") or "the crop"
    season_length = compute_season_length(ag_cfg)
    lat = ag_cfg.get("latitude") or dataset_cfg.get("agro_params", {}).get("latitude")
    lon = ag_cfg.get("longitude") or dataset_cfg.get("agro_params", {}).get("longitude")
    location = "the field"
    if lat is not None and lon is not None:
        location = f"{lat}°N, {lon}°E"
    start_date = start_date_override or parse_date(ag_cfg.get("crop_start_date"))

    return WOFOSTPromptGenerator(
        crop_name=crop_name,
        season_length=season_length,
        location=location,
        num_fert=num_fert,
        num_irrig=num_irrig,
        fert_amount=fert_amount,
        irrig_amount=irrig_amount,
        intervention_interval=intervention_interval,
        output_vars=output_vars,
        latitude=lat,
        longitude=lon,
        start_date=start_date,
    )


def split_episodes(obs: np.ndarray, actions: np.ndarray, dones: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Split trajectories by terminal flags."""
    split_points = [s for s in (np.where(dones)[0] + 1) if 0 < s < len(obs)]
    obs_eps = np.split(obs, split_points)
    actions_eps = np.split(actions, split_points)
    return obs_eps, actions_eps


def load_year_splits(dataset_cfg: dict, num_episodes: int) -> Tuple[List[int], set[int], set[int]]:
    """Validate and return year ordering plus train/test sets."""
    train_years = list(dataset_cfg.get("train_years", []))
    test_years = list(dataset_cfg.get("test_years", []))
    all_years = sorted(train_years + test_years)

    if num_episodes != len(all_years):
        raise ValueError(
            f"Episode count ({num_episodes}) does not match total years in split config ({len(all_years)})."
        )
    return all_years, set(train_years), set(test_years)


def compute_episode_stats(ep_obs: np.ndarray, wso_index: int) -> dict:
    peak = float(np.max(ep_obs[:, wso_index])) if len(ep_obs) > 0 else float("nan")
    peak_idx = int(np.argmax(ep_obs[:, wso_index])) if len(ep_obs) > 0 else -1
    cumulative = float(np.sum(ep_obs[:, wso_index])) if len(ep_obs) > 0 else float("nan")
    return {"peak_wso": peak, "peak_day_index": peak_idx, "cumulative_wso": cumulative}


def build_records(
    prompt_gen: WOFOSTPromptGenerator,
    output_vars: List[str],
    obs_eps: Iterable[np.ndarray],
    actions_eps: Iterable[np.ndarray],
    years: Iterable[int],
) -> Tuple[List[dict], List[dict]]:
    """Return (records, metadata) for provided episodes."""
    records: List[dict] = []
    meta: List[dict] = []
    wso_idx = output_vars.index("WSO") if "WSO" in output_vars else -1

    for ep_obs, ep_actions, year in zip(obs_eps, actions_eps, years):
        ep_meta = compute_episode_stats(ep_obs, wso_idx) if wso_idx >= 0 else {}
        ep_meta.update({"year": int(year)})
        meta.append(ep_meta)

        system_msg = {"role": "system", "content": prompt_gen.get_system_prompt()}
        for row, act in zip(ep_obs, ep_actions):
            user_content = prompt_gen.get_turn_prompt(row, output_vars=output_vars, year=year)
            assistant_content = prompt_gen.describe_action(int(act))
            records.append(
                {
                    "messages": [
                        system_msg,
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": assistant_content},
                    ]
                }
            )
    return records, meta


def main() -> None:
    args = parse_args()
    data_path = args.data_path.resolve()
    if not data_path.is_file():
        raise SystemExit(f"Data file not found: {data_path}")

    env_config_path = args.env_config or data_path.parent.parent / "config.yaml"
    if not env_config_path.is_file():
        raise SystemExit(f"Environment config not found: {env_config_path}")

    dataset_config_path = args.dataset_config.resolve()
    if not dataset_config_path.is_file():
        raise SystemExit(f"Dataset config not found: {dataset_config_path}")

    npz = np.load(data_path, allow_pickle=True)
    obs = npz["obs"]
    actions = npz["actions"]
    dones = npz["dones"]
    output_vars = [str(v) for v in npz["output_vars"]]

    env_cfg = OmegaConf.to_container(OmegaConf.load(env_config_path), resolve=True)
    dataset_cfg = OmegaConf.to_container(OmegaConf.load(dataset_config_path), resolve=True)

    # Prefer actual calendar start date from data if available
    start_date_override = extract_start_date_from_infos(npz["infos"]) if "infos" in npz else None

    prompt_gen = build_prompt_generator(env_cfg, dataset_cfg, output_vars, start_date_override=start_date_override)
    obs_eps, actions_eps = split_episodes(obs, actions, dones)
    years, train_set, test_set = load_year_splits(dataset_cfg, len(obs_eps))

    # Map episodes to train/test using year membership
    train_indices = [i for i, y in enumerate(years) if y in train_set]
    test_indices = [i for i, y in enumerate(years) if y in test_set]

    def select(items: List, indices: List[int]) -> List:
        return [items[i] for i in indices]

    train_records, train_meta = build_records(
        prompt_gen, output_vars, select(obs_eps, train_indices), select(actions_eps, train_indices), select(years, train_indices)
    )
    test_records, test_meta = build_records(
        prompt_gen, output_vars, select(obs_eps, test_indices), select(actions_eps, test_indices), select(years, test_indices)
    )

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix

    (out_dir / f"{prefix}_train.json").write_text(json.dumps(train_records, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / f"{prefix}_test.json").write_text(json.dumps(test_records, ensure_ascii=False, indent=2), encoding="utf-8")

    meta = {
        "source_data": str(data_path),
        "env_config": str(env_config_path),
        "dataset_config": str(dataset_config_path),
        "train_meta": train_meta,
        "test_meta": test_meta,
    }
    (out_dir / f"{prefix}_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"✓ Wrote {len(train_records)} train records and {len(test_records)} test records "
        f"to {out_dir} (prefix: {prefix})"
    )


if __name__ == "__main__":
    main()
