import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from agrimanager.env.base import BaseNNEnvAdapter, create_nn_env_adapter
from agrimanager.nn_ppo.common import evaluate_model_on_dataset, run_episode
from agrimanager.nn_ppo.train import (
    _resolve_checkpoint_episode_frequency,
    _resolve_checkpoint_frequency,
    _resolve_vec_env_settings,
    _resolve_validation_frequency,
    _resolve_resume_checkpoint,
    _resolve_target_train_episodes,
    _resolve_total_timesteps,
)


class DummyNumericEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, env_config=None):
        self.env_config = dict(env_config or {})
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Box(low=-10.0, high=10.0, shape=(2,), dtype=np.float32)
        self._step = 0

    def reset(self, *, seed=None, options=None):
        del seed, options
        self._step = 0
        return np.zeros(2, dtype=np.float32), {"env_config": self.env_config}

    def step(self, action):
        self._step += 1
        done = self._step >= 1
        info = {
            "env_config": self.env_config,
            "turn_metrics": {"custom_step_metric": 1.0},
            "trajectory_metrics": {
                "custom_score": 7.0,
                "ignored_bool": True,
            },
            "group_label/difficulty": "easy",
        }
        return np.ones(2, dtype=np.float32), 1.0, done, False, info


class DummyAdapter(BaseNNEnvAdapter):
    def make_env(self, env_config):
        return DummyNumericEnv(env_config)


class DummyModel:
    def predict(self, observation, deterministic=False):
        del observation, deterministic
        return np.asarray([1.5, 2.5], dtype=np.float32), None


def test_create_nn_env_adapter_discovers_wofost_adapter():
    adapter = create_nn_env_adapter("wofost_gym")
    assert isinstance(adapter, BaseNNEnvAdapter)


def test_create_nn_env_adapter_discovers_cycles_adapter():
    adapter = create_nn_env_adapter("cycles_gym")
    assert isinstance(adapter, BaseNNEnvAdapter)


def test_cycles_adapter_group_labels_are_generic():
    from agrimanager.env.cycles_gym.nn_adapter import CyclesGymNNEnvAdapter

    adapter = CyclesGymNNEnvAdapter()
    labels = adapter.group_labels(
        {
            "env_id": "CropPlanningNewHollandRW-v1",
            "dataset_split": "val",
            "env_kwargs": {"start_year": 1982, "end_year": 2000},
            "trajectory_group_labels": {"scenario": "ood"},
        },
        {"group_label/custom": "heldout"},
    )

    assert labels == {
        "scenario": "ood",
        "env_id": "plan_nh_rw",
        "task": "crop_planning",
        "location": "NewHolland",
        "dataset_split": "val",
        "year_window": "1982-2000",
        "custom": "heldout",
    }


def test_wofost_variety_trait_encoder_uses_crop_variety_key():
    from agrimanager.env.wofost_gym.crop_traits_observation import CropTraitEncoder
    from agrimanager.env.wofost_gym.nn_adapter import _extract_trait_key

    encoder = CropTraitEncoder(
        traits_dir="agrimanager/env/wofost_gym/crop_traits",
        trait_schema="rice_variety_traits_v1",
    )

    assert "rice__rice_2" in encoder.crop_names
    assert "rice__rice_5" in encoder.crop_names
    assert encoder.vector_for_crop("rice__rice_2").shape == encoder.vector_for_crop(
        "rice__rice_5"
    ).shape

    env_config = {
        "crop_name": "rice",
        "agro_file": "rice_agro.yaml",
        "include_variety_traits": True,
        "agro_params": {"crop_variety": "rice_5"},
    }
    assert _extract_trait_key(env_config, "rice") == "rice__rice_5"


def test_wofost_crop_trait_encoder_uses_available_variety_key(tmp_path):
    from agrimanager.env.wofost_gym.crop_traits_observation import CropTraitEncoder
    from agrimanager.env.wofost_gym.nn_adapter import _extract_trait_key

    agro_dir = tmp_path / "env_config" / "agro"
    agro_dir.mkdir(parents=True)
    (agro_dir / "millet_agro.yaml").write_text(
        """
AgroManagement:
  CropCalendar:
    crop_name: millet
    crop_variety: millet_1
""".lstrip(),
        encoding="utf-8",
    )

    encoder = CropTraitEncoder(
        traits_dir="agrimanager/env/wofost_gym/crop_traits",
        trait_schema="traits_v1_23d",
    )

    assert "millet" not in encoder.crop_names
    assert "millet__millet_1" in encoder.crop_names

    env_config = {
        "crop_name": "millet",
        "agro_file": "millet_agro.yaml",
        "wofost_gym_path": str(tmp_path),
        "include_crop_traits": True,
    }
    assert _extract_trait_key(env_config, "millet", encoder) == "millet__millet_1"


def test_rice_variety_text_traits_hide_variety_identifier():
    traits_dir = Path("agrimanager/env/wofost_gym/crop_traits/rice_variety_traits_v1")
    text = (traits_dir / "rice__rice_5.txt").read_text(encoding="utf-8")
    card = json.loads((traits_dir / "rice__rice_5.json").read_text(encoding="utf-8"))

    assert "Variety:" not in text
    assert "rice_5" not in text
    assert card["variety"] == "rice_5"
    assert card["trait_key"] == "rice__rice_5"


def test_default_adapter_hooks_are_generic():
    adapter = DummyAdapter()
    assert adapter.episode_length_hint({"turn_num": "24"}) == 24
    assert adapter.episode_length_hint({}) is None
    assert adapter.group_labels(
        {"trajectory_group_labels": {"split": "ood"}},
        {"group_label/weather": "dry"},
    ) == {"split": "ood", "weather": "dry"}
    assert adapter.serialize_action(np.asarray([1, 2])) == [1, 2]


def test_epoch_budget_and_auto_timestep_cap_use_scenario_passes():
    adapter = DummyAdapter()
    env_configs = [{"turn_num": 3}, {"turn_num": 5}]
    target_episodes = _resolve_target_train_episodes(
        2,
        len(env_configs),
        sample_with_replacement=False,
        num_envs=2,
        shard_train_scenarios=True,
    )

    assert target_episodes == 4
    assert _resolve_total_timesteps(
        {"total_timesteps": None, "num_steps": 4},
        adapter,
        env_configs,
        total_train_episodes=target_episodes,
        num_envs=2,
    ) == 28


def test_validation_frequency_can_be_derived_from_epoch_budget():
    adapter = DummyAdapter()
    env_configs = [{"turn_num": 3}, {"turn_num": 5}]

    assert _resolve_validation_frequency(
        {"frequency": None, "evals_per_epoch": 2},
        {"checkpoint_frequency": 100},
        adapter,
        env_configs,
        train_epochs=2,
        num_envs=2,
    ) == 6

    assert _resolve_validation_frequency(
        {"frequency": 11, "evals_per_epoch": 2},
        {"checkpoint_frequency": 100},
        adapter,
        env_configs,
        train_epochs=2,
        num_envs=2,
    ) == 11


def test_checkpoint_frequency_uses_completed_episodes_for_epoch_budget():
    assert _resolve_checkpoint_frequency(
        {"checkpoint_mode": "epoch"},
    ) == 0

    assert _resolve_checkpoint_episode_frequency(
        {"checkpoint_mode": "epoch"},
        train_epochs=2,
        num_train_scenarios=7,
    ) == 7

    assert _resolve_checkpoint_frequency(
        {"checkpoint_frequency": 11},
    ) == 11

    assert _resolve_checkpoint_episode_frequency(
        {"checkpoint_frequency": 11},
        train_epochs=2,
        num_train_scenarios=7,
    ) == 0

    assert _resolve_checkpoint_frequency(
        {"checkpoint_mode": "none"},
    ) == 0

    assert _resolve_checkpoint_episode_frequency(
        {"checkpoint_mode": "none"},
        train_epochs=2,
        num_train_scenarios=7,
    ) == 0

    with pytest.raises(ValueError, match="checkpoint_mode=timestep"):
        _resolve_checkpoint_frequency({"checkpoint_mode": "timestep"})

    with pytest.raises(ValueError, match="checkpoint_mode=epoch"):
        _resolve_checkpoint_episode_frequency(
            {"checkpoint_mode": "epoch"},
            train_epochs=None,
            num_train_scenarios=7,
        )


def test_vec_env_settings_accept_dummy_and_subproc_aliases():
    assert _resolve_vec_env_settings({}) == ("subproc", "fork")
    assert _resolve_vec_env_settings({"vec_env": "sync"}) == ("dummy", None)
    assert _resolve_vec_env_settings(
        {"vec_env": "subprocess"}
    ) == ("subproc", "fork")
    assert _resolve_vec_env_settings(
        {"vec_env": "subprocess", "subproc_start_method": "spawn"}
    ) == ("subproc", "spawn")
    assert _resolve_vec_env_settings(
        {"vec_env": "subproc", "subproc_start_method": "null"}
    ) == ("subproc", None)

    with pytest.raises(ValueError, match="runtime.vec_env"):
        _resolve_vec_env_settings({"vec_env": "threaded"})


def test_auto_latest_resume_selects_latest_complete_checkpoint(tmp_path):
    checkpoint_root = tmp_path / "checkpoints"
    for step in (10, 20):
        checkpoint = checkpoint_root / f"step_{step:08d}"
        checkpoint.mkdir(parents=True)
        (checkpoint / "agent.zip").write_text("agent")
        (checkpoint / "vecnormalize.pkl").write_text("stats")
        (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": step}))
        (checkpoint / "config.yaml").write_text("agent: {}")

    incomplete = checkpoint_root / "step_00000030"
    incomplete.mkdir()
    (incomplete / "agent.zip").write_text("agent")

    assert _resolve_resume_checkpoint(tmp_path, {"mode": "auto_latest"}) == (
        checkpoint_root / "step_00000020"
    )


def test_run_episode_serializes_generic_actions():
    env = DummyNumericEnv({"trajectory_group_labels": {"split": "id"}})
    result, _ = run_episode(env, DummyModel(), adapter=DummyAdapter())

    assert result["turns"][0]["action"] is None
    assert result["turns"][1]["action"] == pytest.approx([1.5, 2.5])
    assert "action_id" not in result["turns"][1]


def test_evaluate_model_uses_generic_metrics_and_group_labels():
    results, env_infos, metric_dict = evaluate_model_on_dataset(
        DummyModel(),
        DummyAdapter(),
        [{"turn_num": 1, "trajectory_group_labels": {"split": "id"}}],
    )

    assert results[0]["group_labels"] == {"split": "id", "difficulty": "easy"}
    assert env_infos["custom_score"] == [7.0]
    assert "ignored_bool" not in env_infos
    assert metric_dict["val-env/custom_score/mean"] == pytest.approx(7.0)
    assert metric_dict["val-env-by-split/id/custom_score/mean"] == pytest.approx(7.0)
    assert metric_dict["val-env-by-difficulty/easy/custom_score/mean"] == pytest.approx(7.0)


def test_evaluate_model_can_log_one_validation_axis():
    _, _, metric_dict = evaluate_model_on_dataset(
        DummyModel(),
        DummyAdapter(),
        [{"turn_num": 1, "trajectory_group_labels": {"weather_regime": "cold", "split": "val_cold"}}],
        validation_axis="weather_regime",
    )

    assert metric_dict["val-env-weather_regime/all/custom_score/mean"] == pytest.approx(7.0)
    assert metric_dict["val-env-weather_regime/cold/custom_score/mean"] == pytest.approx(7.0)
    assert not any(key.startswith("val-env-by-") for key in metric_dict)
