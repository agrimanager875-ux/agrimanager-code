import math

import pytest

from agrimanager.adapter.reward.agri_reward import compute_score
from agrimanager.adapter.trainer.validation_metrics import (
    add_env_metrics,
    add_validation_env_metrics,
    filter_finite_validation_infos,
)


def test_compute_score_aggregates_turn_rewards():
    result = compute_score(
        data_source="agrimanager",
        solution_str="",
        ground_truth=None,
        extra_info={
            "turn_scores": [0.0, 0.0, 0.5],
            "interaction_kwargs": {"env_config": {"turn_num": 4}},
            "step_idx": 2,
        },
    )

    assert result["score"] == 0.5
    assert result["traj_score"] == 0.5
    assert result["step_reward"] == 0.5
    assert result["raw_reward"] == 0.5
    assert result["turn_num"] == 4
    assert result["num_steps"] == 3


def test_compute_score_uses_observed_episode_length_when_turn_num_missing():
    result = compute_score(
        data_source="agrimanager",
        solution_str="",
        ground_truth=None,
        extra_info={
            "turn_scores": [0.1, 0.2, 0.3],
            "step_idx": 1,
        },
    )

    assert result["score"] == pytest.approx(0.6)
    assert result["traj_score"] == pytest.approx(0.6)
    assert result["step_reward"] == pytest.approx(0.2)
    assert result["turn_num"] == 3
    assert result["num_steps"] == 3


def test_compute_score_clamps_step_idx():
    result = compute_score(
        data_source="agrimanager",
        solution_str="",
        ground_truth=None,
        extra_info={
            "turn_scores": [0.0, 1.0],
            "interaction_kwargs": {"env_config": {"turn_num": 2}},
            "step_idx": 99,
        },
    )

    assert result["step_reward"] == 1.0


def test_compute_score_supports_last_step_only_trajectory_reward():
    result_mid = compute_score(
        data_source="agrimanager",
        solution_str="",
        ground_truth=None,
        extra_info={
            "turn_scores": [0.0, 0.0, 0.8],
            "interaction_kwargs": {"env_config": {"turn_num": 3}},
            "step_idx": 1,
        },
    )
    result_last = compute_score(
        data_source="agrimanager",
        solution_str="",
        ground_truth=None,
        extra_info={
            "turn_scores": [0.0, 0.0, 0.8],
            "interaction_kwargs": {"env_config": {"turn_num": 3}},
            "step_idx": 2,
        },
    )

    assert result_mid["score"] == pytest.approx(0.8)
    assert result_mid["step_reward"] == pytest.approx(0.0)
    assert result_last["score"] == pytest.approx(0.8)
    assert result_last["step_reward"] == pytest.approx(0.8)


def test_compute_score_propagates_terminal_env_trajectory_metrics():
    extra_info = {
        "turn_scores": [0.1, 0.2, 0.3],
        "interaction_kwargs": {"env_config": {"turn_num": 3}},
        "step_idx": 0,
        "interaction_metrics": [
            {"turn_metrics": {"wso": 1.0}, "trajectory_metrics": {}},
            {"turn_metrics": {"wso": 2.0}, "trajectory_metrics": {}},
            {
                "turn_metrics": {"wso": 3.0},
                "trajectory_metrics": {
                    "final_wso": 3.0,
                    "invalid_action_rate": 1.0 / 3.0,
                    "invalid_steps": 1,
                    "total_steps": 3,
                },
            },
        ],
    }

    result = compute_score(
        data_source="agrimanager",
        solution_str="",
        ground_truth=None,
        extra_info=extra_info,
    )

    assert result["target_yield"] == pytest.approx(3.0)
    assert result["invalid_action_rate"] == pytest.approx(1.0 / 3.0)
    assert result["total_steps"] == 3


def test_compute_score_canonicalizes_cross_simulator_metrics():
    dssat = compute_score(
        data_source="gym_dssat/demo",
        solution_str="",
        ground_truth=None,
        extra_info={
            "turn_scores": [0.0, 10.0],
            "interaction_kwargs": {
                "env_config": {
                    "env_name": "gym_dssat",
                    "trajectory_group_labels": {"simulator": "gym_dssat"},
                }
            },
            "interaction_metrics": [
                {"trajectory_metrics": {}},
                {
                    "trajectory_metrics": {
                        "yield_kgha": 9000.0,
                        "total_fert": 140.0,
                        "total_irrig": 80.0,
                        "invalid_action_rate": 0.25,
                        "total_steps": 2,
                    }
                },
            ],
        },
    )
    cycles = compute_score(
        data_source="cycles_gym/demo",
        solution_str="",
        ground_truth=None,
        extra_info={
            "turn_scores": [0.0, 8.0],
            "interaction_kwargs": {
                "env_config": {
                    "env_name": "cycles_gym",
                    "trajectory_group_labels": {"simulator": "cycles_gym"},
                }
            },
            "interaction_metrics": [
                {"trajectory_metrics": {}},
                {
                    "trajectory_metrics": {
                        "grain_yield": 8000.0,
                        "total_n_kg_ha": 90.0,
                        "invalid_action_rate": 0.0,
                        "total_steps": 2,
                    }
                },
            ],
        },
    )

    assert dssat["target_yield"] == pytest.approx(9000.0)
    assert dssat["total_n_kg_ha"] == pytest.approx(140.0)
    assert dssat["total_irrig_mm"] == pytest.approx(80.0)
    assert dssat["group_label/simulator"] == "gym_dssat"
    assert cycles["target_yield"] == pytest.approx(8000.0)
    assert cycles["total_n_kg_ha"] == pytest.approx(90.0)
    assert cycles["group_label/simulator"] == "cycles_gym"


def test_compute_score_does_not_treat_cycles_episode_reward_as_yield():
    result = compute_score(
        data_source="cycles_gym/demo",
        solution_str="",
        ground_truth=None,
        extra_info={
            "turn_scores": [0.1, 0.1],
            "interaction_kwargs": {
                "env_config": {
                    "env_name": "cycles_gym",
                    "trajectory_group_labels": {"simulator": "cycles_gym"},
                }
            },
            "interaction_metrics": [
                {"trajectory_metrics": {}},
                {
                    "trajectory_metrics": {
                        "episode_reward": 0.2,
                        "total_n_kg_ha": 30.0,
                    }
                },
            ],
        },
    )

    assert math.isnan(result["target_yield"])
    assert result["total_n_kg_ha"] == pytest.approx(30.0)


def test_compute_score_preserves_env_metadata_labels():
    result = compute_score(
        data_source="agrimanager",
        solution_str="",
        ground_truth=None,
        extra_info={
            "turn_scores": [0.1],
            "interaction_kwargs": {
                "env_config": {
                    "crop_name": "wheat",
                    "trajectory_group_labels": {"crop": "wheat", "split": "ood"},
                }
            },
            "step_idx": 0,
        },
    )

    assert result["crop_name"] == "wheat"
    assert result["group_label/crop"] == "wheat"
    assert result["group_label/split"] == "ood"


def test_compute_score_propagates_named_validation_labels():
    result = compute_score(
        data_source="agrimanager",
        solution_str="",
        ground_truth=None,
        extra_info={
            "turn_scores": [0.1],
            "interaction_kwargs": {
                "env_config": {
                    "trajectory_group_labels": {
                        "dataset_role": "validation",
                        "validation_set": "drought",
                        "weather_regime": "drought",
                        "reward_formulation": "yield_max",
                        "schema_tuple": "wofost_s1_lnpkw_yield",
                        "callback_family": "t32_unified",
                    }
                }
            },
            "step_idx": 0,
        },
    )

    assert result["group_label/dataset_role"] == "validation"
    assert result["group_label/validation_set"] == "drought"
    assert result["group_label/weather_regime"] == "drought"
    assert result["group_label/reward_formulation"] == "yield_max"
    assert result["group_label/schema_tuple"] == "wofost_s1_lnpkw_yield"
    assert result["group_label/callback_family"] == "t32_unified"


def test_compute_score_propagates_nutrient_stewardship_diagnostics():
    result = compute_score(
        data_source="agrimanager",
        solution_str="",
        ground_truth=None,
        extra_info={
            "turn_scores": [0.0, 0.0, 0.6],
            "interaction_kwargs": {
                "env_config": {
                    "env_name": "wofost_gym",
                    "objective_id": "nutrient_stewardship",
                    "turn_num": 3,
                }
            },
            "interaction_metrics": [
                {"trajectory_metrics": {}},
                {"trajectory_metrics": {}},
                {
                    "trajectory_metrics": {
                        "final_wso": 5100.0,
                        "y_ref": 6000.0,
                        "y_ratio": 0.85,
                        "total_n_kg_ha": 160.0,
                        "total_p_kg_ha": 140.0,
                        "total_k_kg_ha": 120.0,
                        "total_irrig_mm": 80.0,
                        "terminal_navail": 30.0,
                        "terminal_pavail": 24.0,
                        "terminal_kavail": 18.0,
                        "reward_yield_term": 0.85,
                        "reward_yield_floor_penalty": 0.0,
                        "reward_application_penalty": 0.0,
                        "reward_terminal_low_penalty": 0.0,
                        "reward_terminal_high_penalty": 0.0,
                    }
                },
            ],
        },
    )

    assert result["final_wso"] == pytest.approx(5100.0)
    assert result["target_yield"] == pytest.approx(5100.0)
    assert result["y_ref"] == pytest.approx(6000.0)
    assert result["y_ratio"] == pytest.approx(0.85)
    assert result["total_p_kg_ha"] == pytest.approx(140.0)
    assert result["terminal_kavail"] == pytest.approx(18.0)
    assert result["reward_yield_term"] == pytest.approx(0.85)
    assert result["group_label/simulator"] == "wofost_gym"
    assert result["group_label/objective_id"] == "nutrient_stewardship"


def test_add_validation_env_metrics_logs_propagated_trajectory_metrics():
    metric_dict = {}
    env_infos = {
        "__num_turns__": [1, 1],
        "reward": [7.0, 9.0],
        "final_wso": [100.0, 120.0],
        "invalid_action_rate": [0.0, 0.5],
        "invalid_steps": [0, 2],
        "total_steps": [10, 12],
        "group_label/weather_regime": ["wet", "hot"],
        "score": [7.0, 9.0],
        "step_reward": [0.2, 0.4],
        "raw_reward": [7.0, 9.0],
    }

    add_validation_env_metrics(metric_dict, env_infos)

    assert metric_dict["val-env/reward/mean"] == pytest.approx(8.0)
    assert metric_dict["val-env/final_wso/mean"] == pytest.approx(110.0)
    assert metric_dict["val-env/invalid_action_rate/mean"] == pytest.approx(0.25)
    assert metric_dict["val-env/invalid_steps/mean"] == pytest.approx(1.0)
    assert metric_dict["val-env/total_steps/mean"] == pytest.approx(11.0)
    assert metric_dict["val-env-by-weather_regime/wet/final_wso/mean"] == pytest.approx(100.0)
    assert metric_dict["val-env-by-weather_regime/hot/final_wso/mean"] == pytest.approx(120.0)
    assert "val-env/__num_turns__/mean" not in metric_dict
    assert "val-env/score/mean" not in metric_dict
    assert "val-env/step_reward/mean" not in metric_dict
    assert "val-env/raw_reward/mean" not in metric_dict


def test_add_validation_env_metrics_skips_missing_cross_simulator_values():
    metric_dict = {}
    env_infos = {
        "target_yield": [100.0, float("nan"), 300.0],
        "total_irrig_mm": [20.0, float("nan"), float("nan")],
        "invalid_action_rate": [0.0, 0.5, 0.25],
        "group_label/simulator": ["wofost_gym", "gym_dssat", "cycles_gym"],
    }

    add_validation_env_metrics(metric_dict, env_infos)

    assert metric_dict["val-env/target_yield/mean"] == pytest.approx(200.0)
    assert metric_dict["val-env/total_irrig_mm/mean"] == pytest.approx(20.0)
    assert metric_dict["val-env-by-simulator/wofost_gym/target_yield/mean"] == pytest.approx(100.0)
    assert "val-env-by-simulator/gym_dssat/target_yield/mean" not in metric_dict
    assert metric_dict["val-env-by-simulator/cycles_gym/target_yield/mean"] == pytest.approx(300.0)


def test_filter_finite_validation_infos_drops_nan_optional_metrics():
    infos = {
        "score": [1.0, 2.0],
        "target_yield": [100.0, 200.0],
        "final_wso": [100.0, float("nan")],
        "group_label/simulator": ["wofost_gym", "gym_dssat"],
        "raw_payload": [{"x": 1}, {"x": 2}],
    }

    filtered = filter_finite_validation_infos(infos)

    assert filtered["score"] == infos["score"]
    assert filtered["target_yield"] == infos["target_yield"]
    assert filtered["group_label/simulator"] == infos["group_label/simulator"]
    assert "final_wso" not in filtered
    assert "raw_payload" not in filtered


def test_add_env_metrics_discovers_numeric_trajectory_metrics_without_hardcoding():
    metric_dict = {}
    env_infos = {
        "__num_turns__": [1, 1],
        "final_wso": [100.0, 120.0],
        "invalid_action_rate": [0.0, 0.5],
        "custom_new_metric": [3.0, 5.0],
        "score": [7.0, 9.0],
        "step_reward": [0.2, 0.4],
    }

    add_env_metrics(metric_dict, env_infos, prefix="train-env", include_grouped=False)

    assert metric_dict["train-env/final_wso/mean"] == pytest.approx(110.0)
    assert metric_dict["train-env/invalid_action_rate/mean"] == pytest.approx(0.25)
    assert metric_dict["train-env/custom_new_metric/mean"] == pytest.approx(4.0)
    assert "train-env/__num_turns__/mean" not in metric_dict
    assert "train-env/score/mean" not in metric_dict
    assert "train-env/step_reward/mean" not in metric_dict


def test_add_env_metrics_skips_nan_cross_simulator_train_values():
    metric_dict = {}
    env_infos = {
        "target_yield": [100.0, 200.0, 300.0],
        "final_wso": [100.0, float("nan"), float("nan")],
        "yield_kgha": [float("nan"), 200.0, float("nan")],
        "invalid_action_rate": [0.0, 0.5, 0.25],
    }

    add_env_metrics(metric_dict, env_infos, prefix="train-env", include_grouped=False)

    assert metric_dict["train-env/target_yield/mean"] == pytest.approx(200.0)
    assert metric_dict["train-env/final_wso/mean"] == pytest.approx(100.0)
    assert metric_dict["train-env/yield_kgha/mean"] == pytest.approx(200.0)
    assert metric_dict["train-env/invalid_action_rate/mean"] == pytest.approx(0.25)
