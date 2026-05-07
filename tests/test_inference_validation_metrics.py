import pytest

from agrimanager.rollout.inference.inference_rollout import (
    _extract_trial_metrics,
    _extract_validation_metric_dict,
)


def test_inference_validation_metrics_use_training_axis_prefixes():
    results = [
        {
            "env_config": {
                "env_name": "wofost_gym",
                "objective_id": "profit_max",
                "scenario_id": "id-0",
                "validation_set": "id",
                "trajectory_group_labels": {
                    "dataset_role": "validation",
                    "validation_set": "id",
                    "weather_regime": "id",
                },
            },
            "turns": [
                {
                    "reward": 1.0,
                    "trajectory_metrics": {
                        "target_yield": 100.0,
                        "final_wso": 100.0,
                        "invalid_action_rate": 0.0,
                        "total_steps": 24.0,
                    },
                }
            ],
        },
        {
            "env_config": {
                "env_name": "wofost_gym",
                "objective_id": "profit_max",
                "scenario_id": "drought-0",
                "validation_set": "drought",
                "trajectory_group_labels": {
                    "dataset_role": "validation",
                    "validation_set": "drought",
                    "weather_regime": "drought",
                },
            },
            "turns": [
                {
                    "reward": 2.0,
                    "trajectory_metrics": {
                        "target_yield": 200.0,
                        "final_wso": 200.0,
                        "invalid_action_rate": 0.5,
                        "total_steps": 24.0,
                    },
                }
            ],
        },
    ]

    metrics = _extract_validation_metric_dict(results, validation_axis="weather_regime")

    assert metrics["val-env-weather_regime/all/target_yield/mean"] == pytest.approx(150.0)
    assert metrics["val-env-weather_regime/id/target_yield/mean"] == pytest.approx(100.0)
    assert metrics["val-env-weather_regime/drought/target_yield/mean"] == pytest.approx(200.0)
    assert metrics["val-env-weather_regime/all/reward/mean"] == pytest.approx(1.5)
    assert not any(key.startswith("val-env-by-") for key in metrics)


def test_trial_metrics_prefer_objective_reward_over_target_yield():
    metrics = _extract_trial_metrics(
        [
            {
                "env_config": {"env_name": "wofost_gym", "crop_name": "chickpea"},
                "turns": [
                    {
                        "reward": -1.0,
                        "trajectory_metrics": {
                            "objective_reward": -2.0,
                            "target_yield": 1000.0,
                            "final_wso": 1000.0,
                            "invalid_action_rate": 0.0,
                            "total_steps": 24.0,
                        },
                    }
                ],
            },
            {
                "env_config": {"env_name": "wofost_gym", "crop_name": "chickpea"},
                "turns": [
                    {
                        "reward": 1.0,
                        "trajectory_metrics": {
                            "objective_reward": 0.5,
                            "target_yield": 2000.0,
                            "final_wso": 2000.0,
                            "invalid_action_rate": 0.0,
                            "total_steps": 24.0,
                        },
                    }
                ],
            },
        ]
    )

    assert metrics["primary_metric"]["key"] == "objective_reward"
    assert metrics["primary_metric"]["mean"] == pytest.approx(-0.75)
    assert metrics["objective_reward_mean"] == pytest.approx(-0.75)
