from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest


MODULE_PATH = Path(
    "experiments/app1_wofost_variety_ood/diagnose_variety_separability.py"
).resolve()
SPEC = importlib.util.spec_from_file_location("wofost_variety_diagnostic", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load module from {MODULE_PATH}")
diagnostic = importlib.util.module_from_spec(SPEC)
sys.modules["wofost_variety_diagnostic"] = diagnostic
SPEC.loader.exec_module(diagnostic)


def _args() -> argparse.Namespace:
    return argparse.Namespace(
        pool_split="train",
        env_id="lnpkw-v0",
        wofost_gym_path="../AgriManagerExternal/WOFOSTGym",
        turn_num=24,
        intvn_interval=10,
        fert_amount=20.0,
        irrig_amount=5.0,
        objective_id="profit_max",
    )


def _rollout_rows() -> pd.DataFrame:
    rows = []
    for weather_id, base in [("weather_a", 100.0), ("weather_b", 200.0)]:
        for variety, offset in [("wheat_1", 0.0), ("wheat_2", 10.0), ("wheat_7", 20.0)]:
            rows.append(
                {
                    "weather_id": weather_id,
                    "year": 2001,
                    "latitude": 51.1,
                    "longitude": 5.1,
                    "policy_name": "random",
                    "action_seed": 1729,
                    "action_sequence_hash": "abc",
                    "variety": variety,
                    "final_wso": base + offset * (base / 100.0),
                }
            )
    return pd.DataFrame(rows)


def test_action_sequence_hash_and_sequence_metadata_are_stable() -> None:
    first = diagnostic.action_sequence_hash([1, 2, 3])
    second = diagnostic.action_sequence_hash([1, 2, 3])

    assert first == second
    assert first != diagnostic.action_sequence_hash([1, 2, 4])

    sequences = diagnostic.build_action_sequences(
        turn_num=4,
        action_space_n=17,
        num_action_seeds=2,
        action_seed_base=100,
        include_noop=True,
    )

    assert sequences[0]["policy_name"] == "noop"
    assert sequences[0]["actions"] == [0, 0, 0, 0]
    assert [s["action_seed"] for s in sequences[1:]] == [100, 101]
    assert all(len(s["actions"]) == 4 for s in sequences)


def test_env_config_sets_variety_without_changing_weather_identity() -> None:
    scenario = {
        "crop_name": "wheat",
        "year": 2001,
        "latitude": 51.1,
        "longitude": 5.1,
    }
    args = _args()

    wheat_1 = diagnostic.build_env_config(
        scenario=scenario,
        variety="wheat_1",
        args=args,
        meteo_cache_dir=Path("/tmp/meteo_cache"),
    )
    wheat_7 = diagnostic.build_env_config(
        scenario=scenario,
        variety="wheat_7",
        args=args,
        meteo_cache_dir=Path("/tmp/meteo_cache"),
    )

    assert wheat_7["agro_params"]["crop_variety"] == "wheat_7"
    assert wheat_7["trajectory_group_labels"] == {"crop": "wheat", "variety": "wheat_7"}
    assert wheat_1["scenario_id"] == wheat_7["scenario_id"]
    assert wheat_1["seed"] == wheat_7["seed"]
    assert wheat_1["agro_params"]["year"] == wheat_7["agro_params"]["year"]
    assert wheat_1["agro_params"]["latitude"] == wheat_7["agro_params"]["latitude"]
    assert wheat_1["agro_params"]["longitude"] == wheat_7["agro_params"]["longitude"]


def test_summary_pair_summary_and_gate_pass_for_separable_varieties() -> None:
    rollouts = _rollout_rows()
    summary = diagnostic.summarize_rollouts(rollouts)
    pair_summary = diagnostic.build_pair_summary(rollouts)
    gate = diagnostic.evaluate_gate(
        summary=summary,
        pair_summary=pair_summary,
        candidate_pair=("wheat_1", "wheat_7"),
        relative_range_pass_threshold=0.05,
        relative_range_min_threshold=0.02,
        coverage_threshold=0.75,
        candidate_pair_threshold=0.05,
    )

    assert len(summary) == 2
    assert summary["relative_range"].median() == pytest.approx(20.0 / 110.0)
    assert gate["decision"] == "pass"
    assert gate["passed"] is True
    assert gate["candidate_pair_median_relative_diff"] == pytest.approx(20.0 / 110.0)


def test_gate_stops_when_varieties_are_not_separable() -> None:
    rollouts = _rollout_rows()
    rollouts["final_wso"] = 100.0
    rollouts.loc[rollouts["variety"] == "wheat_7", "final_wso"] = 101.0
    summary = diagnostic.summarize_rollouts(rollouts)
    pair_summary = diagnostic.build_pair_summary(rollouts)
    gate = diagnostic.evaluate_gate(
        summary=summary,
        pair_summary=pair_summary,
        candidate_pair=("wheat_1", "wheat_7"),
        relative_range_pass_threshold=0.05,
        relative_range_min_threshold=0.02,
        coverage_threshold=0.75,
        candidate_pair_threshold=0.05,
    )

    assert gate["decision"] == "stop"
    assert gate["passed"] is False
