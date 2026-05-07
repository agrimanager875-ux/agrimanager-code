from __future__ import annotations

import importlib.util
import sys
from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest
import yaml

from agrimanager.env.create_dataset import generate as generate_dataset
from agrimanager.env.base import load_env_configs_from_parquet
from agrimanager.env.wofost_gym.create_dataset import WOFOSTArtifactDatasetBuilder
from agrimanager.env.wofost_gym.weather_pool import ensure_pool


def _load_module(module_name: str, file_path: Path):
    module_dir = str(file_path.parent)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _write_pool_split(
    base_dir: Path,
    split: str,
    crop_rows: dict[str, list[tuple[int, float, float]]],
) -> None:
    split_dir = base_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    for crop, rows in crop_rows.items():
        df = pd.DataFrame(rows, columns=["year", "latitude", "longitude"])
        df.to_parquet(split_dir / f"{crop}.parquet", index=False)


class _FakeEnv:
    def __init__(self, env_config: dict[str, object]):
        self._env_config = env_config

    def reset(self):
        crop_name = self._env_config["crop_name"]
        year = self._env_config["agro_params"]["year"]
        if self._env_config.get("llm_mode", True):
            return f"observation:{crop_name}:{year}", {}
        return [float(year), 1.0, 2.0], {}

    def system_prompt(self):
        crop_name = self._env_config["crop_name"]
        return f"system:{crop_name}"

    def close(self):
        return None


@pytest.fixture
def local_weather_pool(tmp_path: Path) -> Path:
    crop_rows = {
        "wheat": [
            (2001, 51.10, 5.10),
            (2002, 51.20, 5.20),
            (2003, 51.30, 5.30),
            (2004, 51.40, 5.40),
        ],
        "maize": [
            (2011, 41.10, 6.10),
            (2012, 41.20, 6.20),
            (2013, 41.30, 6.30),
            (2014, 41.40, 6.40),
        ],
        "rice": [
            (2021, 38.10, 7.10),
            (2022, 38.20, 7.20),
            (2023, 38.30, 7.30),
            (2024, 38.40, 7.40),
        ],
        "potato": [
            (1991, 48.10, 4.10),
            (1992, 48.20, 4.20),
            (1993, 48.30, 4.30),
            (1994, 48.40, 4.40),
        ],
    }
    pool_dir = tmp_path / "weather_pool"
    for split in ("train", "val", "test"):
        _write_pool_split(pool_dir, split, crop_rows)
    (pool_dir / "meteo_cache").mkdir(parents=True, exist_ok=True)
    return pool_dir


@pytest.fixture
def artifact_config(tmp_path: Path, local_weather_pool: Path) -> Path:
    config_path = tmp_path / "datasets" / "demo_artifact.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "env_name": "wofost_gym",
        "dataset_id": "demo_artifact",
        "source": {
            "kind": "weather_pool",
            "path": str(local_weather_pool),
        },
        "sampling": {
            "generation_seed": 123,
            "splits": {
                "train": {
                    "crops": ["wheat", "maize"],
                    "num_samples": 4,
                },
                "val": {
                    "crop_budgets": {
                        "wheat": 1,
                        "maize": 2,
                        "potato": 1,
                    }
                },
                "test": {
                    "crops": ["potato", "wheat"],
                    "num_samples": 4,
                },
            },
        },
        "env": {
            "env_id": "lnpkw-v0",
            "wofost_gym_path": "../AgriManagerExternal/WOFOSTGym",
            "save_folder": "/tmp/test_wofost_dataset_contract/",
            "llm_mode": True,
            "intvn_interval": 10,
            "turn_num": 24,
            "scale_action_amounts_by_interval": True,
            "objective_id": "profit_max",
            "include_crop_traits": True,
            "crop_traits_dir": "agrimanager/env/wofost_gym/crop_traits",
            "trait_schema": "traits_v1_23d",
            "require_think": True,
            "thinking_mode": "grounding_decision",
            "think_tag": "tool_call",
        },
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


@pytest.fixture
def patched_wofost_prompt_env(monkeypatch: pytest.MonkeyPatch):
    import agrimanager.env.wofost_gym.create_dataset as wofost_create_dataset

    monkeypatch.setattr(
        wofost_create_dataset,
        "create_environment",
        lambda env_name, env_config: (_FakeEnv(env_config), {}),
    )


def test_build_dataset_smoke(
    tmp_path: Path,
    artifact_config: Path,
    local_weather_pool: Path,
    patched_wofost_prompt_env,
) -> None:
    output_dir = tmp_path / "data"
    generate_dataset(str(artifact_config), str(output_dir), num_workers=1)

    dataset_dir = output_dir / "demo_artifact"
    assert (dataset_dir / "train.parquet").is_file()
    assert (dataset_dir / "val.parquet").is_file()
    assert (dataset_dir / "test.parquet").is_file()

    env_configs, env_name, _ = load_env_configs_from_parquet(dataset_dir / "val.parquet")
    assert env_name == "wofost_gym"
    assert len(env_configs) == 4
    assert env_configs[0]["dataset_id"] == "demo_artifact"
    assert env_configs[0]["dataset_split"] == "val"
    assert env_configs[0]["scenario_id"]
    assert isinstance(env_configs[0]["seed"], int)
    assert env_configs[0]["weather_cache_dir"] == str((local_weather_pool / "meteo_cache").resolve())


def test_artifact_builder_injects_calibrated_y_ref_map(
    tmp_path: Path,
    local_weather_pool: Path,
    patched_wofost_prompt_env,
) -> None:
    config = {
        "env_name": "wofost_gym",
        "dataset_id": "demo_y_ref_map",
        "source": {
            "kind": "weather_pool",
            "path": str(local_weather_pool),
        },
        "sampling": {
            "generation_seed": 123,
            "splits": {
                "train": {
                    "crops": ["maize"],
                    "num_samples": 1,
                },
            },
        },
        "env": {
            "env_id": "lnpkw-v0",
            "wofost_gym_path": "../AgriManagerExternal/WOFOSTGym",
            "save_folder": "/tmp/test_wofost_dataset_contract/",
            "llm_mode": True,
            "intvn_interval": 10,
            "turn_num": 24,
            "scale_action_amounts_by_interval": True,
            "objective_id": "nutrient_stewardship",
            "reward_params": {"tau_y": 0.8},
        },
    }
    builder = WOFOSTArtifactDatasetBuilder(deepcopy(config), str(tmp_path / "cache"))
    scenario_id = builder.build_split_records("train")[0]["scenario_id"]
    y_ref_map = tmp_path / "calibrated_y_ref.json"
    y_ref_map.write_text(f'{{"{scenario_id}": 7123.5}}\n', encoding="utf-8")

    config["env"]["y_ref_map_path"] = str(y_ref_map)
    config_path = tmp_path / "datasets" / "demo_y_ref_map.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    output_dir = tmp_path / "data"
    generate_dataset(str(config_path), str(output_dir), num_workers=1)
    env_configs, _, _ = load_env_configs_from_parquet(output_dir / "demo_y_ref_map" / "train.parquet")

    assert env_configs[0]["y_ref"] == pytest.approx(7123.5)
    assert env_configs[0]["reward_params"]["y_ref"] == pytest.approx(7123.5)


def test_artifact_builder_supports_crop_variety_budgets(
    tmp_path: Path,
    local_weather_pool: Path,
    patched_wofost_prompt_env,
) -> None:
    config_path = tmp_path / "datasets" / "rice_variety_demo.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "env_name": "wofost_gym",
        "dataset_id": "rice_variety_demo",
        "source": {
            "kind": "weather_pool",
            "path": str(local_weather_pool),
        },
        "sampling": {
            "generation_seed": 123,
            "variety_splits": {
                "rice": {
                    "id": ["rice_1"],
                    "ood": ["rice_2"],
                }
            },
            "splits": {
                "train": {
                    "crop_variety_budgets": {
                        "rice": {
                            "rice_1": 2,
                            "rice_2": 2,
                        }
                    }
                },
                "val": {
                    "crop_variety_budgets": {
                        "rice": {
                            "rice_1": 1,
                            "rice_2": 1,
                        }
                    }
                },
            },
        },
        "env": {
            "env_id": "lnpkw-v0",
            "wofost_gym_path": "../AgriManagerExternal/WOFOSTGym",
            "llm_mode": False,
            "intvn_interval": 10,
            "turn_num": 24,
            "scale_action_amounts_by_interval": True,
            "objective_id": "profit_max",
            "include_crop_traits": True,
            "include_variety_traits": True,
            "crop_traits_dir": "agrimanager/env/wofost_gym/crop_traits",
            "trait_schema": "rice_variety_traits_v1",
        },
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    output_dir = tmp_path / "data"
    generate_dataset(str(config_path), str(output_dir), num_workers=1)

    env_configs, env_name, _ = load_env_configs_from_parquet(
        output_dir / "rice_variety_demo" / "train.parquet"
    )
    assert env_name == "wofost_gym"
    assert len(env_configs) == 4
    assert {cfg["crop_variety"] for cfg in env_configs} == {"rice_1", "rice_2"}
    assert {cfg["agro_params"]["crop_variety"] for cfg in env_configs} == {"rice_1", "rice_2"}
    assert {cfg["trajectory_group_labels"]["variety_split"] for cfg in env_configs} == {
        "id",
        "ood",
    }
    assert len({cfg["scenario_id"] for cfg in env_configs}) == 4

    by_weather = {}
    for cfg in env_configs:
        key = (
            cfg["agro_params"]["year"],
            cfg["agro_params"]["latitude"],
            cfg["agro_params"]["longitude"],
        )
        by_weather.setdefault(key, set()).add(cfg["crop_variety"])
    assert all(varieties == {"rice_1", "rice_2"} for varieties in by_weather.values())


def test_artifact_builder_defaults_to_single_worker_when_unspecified(
    tmp_path: Path,
    artifact_config: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agrimanager.env.base.create_dataset as base_create_dataset

    def fail_pool(*args, **kwargs):
        raise AssertionError("Artifact-first dataset build should default to a single process.")

    monkeypatch.setattr(base_create_dataset.mp, "Pool", fail_pool)

    output_dir = tmp_path / "data"
    generate_dataset(str(artifact_config), str(output_dir))

    assert (output_dir / "demo_artifact" / "train.parquet").is_file()


def test_artifact_builder_honors_explicit_num_workers_override(
    tmp_path: Path,
    artifact_config: Path,
) -> None:
    from agrimanager.env.wofost_gym.create_dataset import WOFOSTArtifactDatasetBuilder

    config = yaml.safe_load(artifact_config.read_text(encoding="utf-8"))
    config["num_workers"] = 32

    builder = WOFOSTArtifactDatasetBuilder(config, str(tmp_path / "data"))

    assert builder._effective_num_workers() == 32


def test_build_is_deterministic_and_split_specific(
    tmp_path: Path,
    artifact_config: Path,
    patched_wofost_prompt_env,
) -> None:
    output_one = tmp_path / "build_one"
    output_two = tmp_path / "build_two"
    generate_dataset(str(artifact_config), str(output_one), num_workers=1)
    generate_dataset(str(artifact_config), str(output_two), num_workers=1)

    val_one, _, _ = load_env_configs_from_parquet(output_one / "demo_artifact" / "val.parquet")
    val_two, _, _ = load_env_configs_from_parquet(output_two / "demo_artifact" / "val.parquet")
    train_one, _, _ = load_env_configs_from_parquet(output_one / "demo_artifact" / "train.parquet")

    assert [cfg["scenario_id"] for cfg in val_one] == [cfg["scenario_id"] for cfg in val_two]
    assert [cfg["seed"] for cfg in val_one] == [cfg["seed"] for cfg in val_two]
    assert [cfg["scenario_id"] for cfg in val_one] != [cfg["scenario_id"] for cfg in train_one]


def test_build_dataset_smoke_keeps_placeholder_prompt_text_in_numeric_mode(
    tmp_path: Path,
    artifact_config: Path,
    patched_wofost_prompt_env,
) -> None:
    config = yaml.safe_load(artifact_config.read_text(encoding="utf-8"))
    config["dataset_id"] = "demo_artifact_numeric_obs"
    config["env"]["llm_mode"] = False

    config_path = tmp_path / "datasets" / "demo_artifact_numeric_obs.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    output_dir = tmp_path / "data"
    generate_dataset(str(config_path), str(output_dir), num_workers=1)

    df = pd.read_parquet(output_dir / "demo_artifact_numeric_obs" / "train.parquet")
    prompt = df.iloc[0]["prompt"]
    assert isinstance(prompt[0]["content"], str)
    assert isinstance(prompt[1]["content"], str)
    assert prompt[1]["content"].startswith("Placeholder dataset prompt for WOFOST-Gym")


def test_artifact_builder_uses_env_config_defaults(
    tmp_path: Path,
    local_weather_pool: Path,
    patched_wofost_prompt_env,
) -> None:
    config_path = tmp_path / "datasets" / "defaults_from_env_config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "env_name": "wofost_gym",
        "dataset_id": "defaults_from_env_config",
        "source": {
            "kind": "weather_pool",
            "path": str(local_weather_pool),
        },
        "sampling": {
            "generation_seed": 123,
            "splits": {
                "val": {
                    "crops": ["wheat"],
                    "num_samples": 1,
                },
            },
        },
        "env": {},
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    output_dir = tmp_path / "data"
    generate_dataset(str(config_path), str(output_dir), num_workers=1)

    env_configs, env_name, _ = load_env_configs_from_parquet(
        output_dir / "defaults_from_env_config" / "val.parquet"
    )
    assert env_name == "wofost_gym"
    assert len(env_configs) == 1

    env_config = env_configs[0]
    assert env_config["env_id"] == "lnpkw-v0"
    assert env_config["llm_mode"] is True
    assert env_config["turn_num"] == 241
    assert env_config["intvn_interval"] == 1
    assert env_config["thinking_mode"] == "grounding_decision"
    assert env_config["think_tag"] == "tool_call"
    assert "wofost_gym_path" in env_config
    assert "save_folder" not in env_config
    assert env_config["weather_cache_dir"] == str((local_weather_pool / "meteo_cache").resolve())


def test_artifact_builder_supports_split_env_overrides(
    tmp_path: Path,
    local_weather_pool: Path,
    patched_wofost_prompt_env,
) -> None:
    config_path = tmp_path / "datasets" / "split_env_overrides.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "env_name": "wofost_gym",
        "dataset_id": "split_env_overrides",
        "source": {
            "kind": "weather_pool",
            "path": str(local_weather_pool),
        },
        "sampling": {
            "generation_seed": 123,
            "splits": {
                "train_lnpkw": {
                    "role": "train",
                    "crops": ["maize"],
                    "num_samples": 1,
                    "labels": {
                        "action_menu": "lnpkw",
                    },
                    "env": {
                        "env_id": "lnpkw-v0",
                    },
                },
                "val_ln": {
                    "role": "validation",
                    "validation_set": "ln",
                    "crops": ["maize"],
                    "num_samples": 1,
                    "labels": {
                        "action_menu": "ln",
                        "action_schema": "ln",
                        "prompt_condition": "clean",
                    },
                    "env": {
                        "env_id": "ln-v0",
                    },
                },
            },
        },
        "env": {
            "wofost_gym_path": "../AgriManagerExternal/WOFOSTGym",
            "llm_mode": True,
            "intvn_interval": 10,
            "turn_num": 24,
            "scale_action_amounts_by_interval": True,
            "objective_id": "profit_max",
            "include_crop_traits": False,
        },
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    output_dir = tmp_path / "data"
    generate_dataset(str(config_path), str(output_dir), num_workers=1)
    dataset_dir = output_dir / "split_env_overrides"

    train_configs, _, _ = load_env_configs_from_parquet(
        dataset_dir / "train_lnpkw.parquet"
    )
    ln_configs, _, _ = load_env_configs_from_parquet(
        dataset_dir / "val_ln.parquet"
    )

    train_cfg = train_configs[0]
    ln_cfg = ln_configs[0]
    assert train_cfg["env_id"] == "lnpkw-v0"
    assert train_cfg["trajectory_group_labels"]["action_menu"] == "lnpkw"

    assert ln_cfg["env_id"] == "ln-v0"
    assert ln_cfg["validation_set"] == "ln"
    assert ln_cfg["trajectory_group_labels"]["action_menu"] == "ln"
    assert ln_cfg["trajectory_group_labels"]["action_schema"] == "ln"
    assert ln_cfg["trajectory_group_labels"]["prompt_condition"] == "clean"
    assert ln_cfg["trajectory_group_labels"]["env_id"] == "ln-v0"


def test_artifact_builder_reuses_scenario_sets_with_shared_generation_seed(
    tmp_path: Path,
    local_weather_pool: Path,
    patched_wofost_prompt_env,
) -> None:
    config_path = tmp_path / "datasets" / "scenario_sets.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "env_name": "wofost_gym",
        "dataset_id": "scenario_sets",
        "validation_axis": "action_menu",
        "source": {
            "kind": "weather_pool",
            "path": str(local_weather_pool),
        },
        "sampling": {
            "generation_seed": 2201,
            "scenario_sets": {
                "train_base": {
                    "source_split": "train",
                    "crops": ["maize"],
                    "num_samples": 2,
                },
                "val_base": {
                    "source_split": "val",
                    "crops": ["maize"],
                    "num_samples": 2,
                },
            },
            "splits": {
                "train_lnpkw": {
                    "role": "train",
                    "scenario_set": "train_base",
                    "labels": {"action_menu": "lnpkw"},
                    "env": {"env_id": "lnpkw-v0"},
                },
                "train_lnpk": {
                    "role": "train",
                    "scenario_set": "train_base",
                    "labels": {"action_menu": "lnpk"},
                    "env": {"env_id": "lnpk-v0"},
                },
                "val_lnpkw": {
                    "role": "validation",
                    "validation_set": "lnpkw",
                    "scenario_set": "val_base",
                    "labels": {"action_menu": "lnpkw"},
                    "env": {"env_id": "lnpkw-v0"},
                },
                "val_lnpk": {
                    "role": "validation",
                    "validation_set": "lnpk",
                    "scenario_set": "val_base",
                    "labels": {"action_menu": "lnpk"},
                    "env": {"env_id": "lnpk-v0"},
                },
            },
        },
        "env": {
            "wofost_gym_path": "../AgriManagerExternal/WOFOSTGym",
            "llm_mode": True,
            "intvn_interval": 10,
            "turn_num": 24,
            "scale_action_amounts_by_interval": True,
            "objective_id": "profit_max",
            "include_crop_traits": False,
        },
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    output_dir = tmp_path / "data"
    generate_dataset(str(config_path), str(output_dir), num_workers=1)
    dataset_dir = output_dir / "scenario_sets"

    def weather_keys(split_name: str) -> list[tuple[int, float, float]]:
        env_configs, _, _ = load_env_configs_from_parquet(
            dataset_dir / f"{split_name}.parquet"
        )
        return [
            (
                cfg["agro_params"]["year"],
                cfg["agro_params"]["latitude"],
                cfg["agro_params"]["longitude"],
            )
            for cfg in env_configs
        ]

    assert weather_keys("train_lnpkw") == weather_keys("train_lnpk")
    assert weather_keys("val_lnpkw") == weather_keys("val_lnpk")

    lnpkw_configs, _, _ = load_env_configs_from_parquet(
        dataset_dir / "val_lnpkw.parquet"
    )
    lnpk_configs, _, _ = load_env_configs_from_parquet(
        dataset_dir / "val_lnpk.parquet"
    )
    assert [cfg["scenario_id"] for cfg in lnpkw_configs] != [
        cfg["scenario_id"] for cfg in lnpk_configs
    ]
    assert {cfg["validation_set"] for cfg in lnpkw_configs} == {"lnpkw"}
    assert {cfg["validation_set"] for cfg in lnpk_configs} == {"lnpk"}


def test_artifact_builder_supports_custom_validation_splits(
    tmp_path: Path,
    local_weather_pool: Path,
    patched_wofost_prompt_env,
) -> None:
    _write_pool_split(
        local_weather_pool,
        "val_drought",
        {
            "wheat": [
                (2031, 52.10, 4.10),
                (2032, 52.20, 4.20),
            ],
        },
    )
    config_path = tmp_path / "datasets" / "custom_validation_splits.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "env_name": "wofost_gym",
        "dataset_id": "custom_validation_splits",
        "source": {
            "kind": "weather_pool",
            "path": str(local_weather_pool),
        },
        "sampling": {
            "generation_seed": 123,
            "splits": {
                "train": {
                    "crops": ["wheat"],
                    "num_samples": 1,
                },
                "val": {
                    "crops": ["wheat"],
                    "num_samples": 1,
                },
                "val_drought": {
                    "crops": ["wheat"],
                    "num_samples": 2,
                },
            },
        },
        "env": {
            "env_id": "lnpkw-v0",
            "wofost_gym_path": "../AgriManagerExternal/WOFOSTGym",
            "llm_mode": True,
            "intvn_interval": 10,
            "turn_num": 24,
            "scale_action_amounts_by_interval": True,
            "objective_id": "profit_max",
            "include_crop_traits": False,
        },
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    output_dir = tmp_path / "data"
    generate_dataset(str(config_path), str(output_dir), num_workers=1)

    dataset_dir = output_dir / "custom_validation_splits"
    assert (dataset_dir / "train.parquet").is_file()
    assert (dataset_dir / "val.parquet").is_file()
    assert (dataset_dir / "val_drought.parquet").is_file()

    env_configs, env_name, _ = load_env_configs_from_parquet(
        dataset_dir / "val_drought.parquet"
    )
    assert env_name == "wofost_gym"
    assert len(env_configs) == 2
    assert {cfg["dataset_split"] for cfg in env_configs} == {"val_drought"}
    assert {cfg["trajectory_group_labels"]["split"] for cfg in env_configs} == {
        "val_drought"
    }
    assert {cfg["trajectory_group_labels"]["weather_regime"] for cfg in env_configs} == {
        "drought"
    }
    parquet_records = pd.read_parquet(dataset_dir / "val_drought.parquet").to_dict("records")
    assert {record["extra_info"]["split"] for record in parquet_records} == {"val_drought"}

    combined_env_configs, combined_env_name, _ = load_env_configs_from_parquet(
        [dataset_dir / "val.parquet", dataset_dir / "val_drought.parquet"]
    )
    assert combined_env_name == "wofost_gym"
    assert len(combined_env_configs) == 3
    assert {cfg["dataset_split"] for cfg in combined_env_configs} == {
        "val",
        "val_drought",
    }


def test_pool_cache_downloads_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from agrimanager.env.wofost_gym import weather_pool

    target_dir = tmp_path / "downloaded_pool"
    calls: list[tuple[str, str, str]] = []

    def fake_download(repo_id: str, local_dir: str | Path, revision: str = "main") -> Path:
        calls.append((repo_id, str(local_dir), revision))
        _write_pool_split(Path(local_dir), "train", {"wheat": [(2001, 51.1, 5.1)]})
        _write_pool_split(Path(local_dir), "val", {"wheat": [(2001, 51.1, 5.1)]})
        _write_pool_split(Path(local_dir), "test", {"wheat": [(2001, 51.1, 5.1)]})
        return Path(local_dir)

    monkeypatch.setattr(weather_pool, "download_pool", fake_download)

    first = ensure_pool("namespace/pool", revision="rev-a", local_dir=target_dir)
    second = ensure_pool("namespace/pool", revision="rev-a", local_dir=target_dir)

    assert first == second == target_dir
    assert len(calls) == 1


def test_runtime_loaders_share_same_artifact_scenarios(
    tmp_path: Path,
    artifact_config: Path,
    patched_wofost_prompt_env,
) -> None:
    output_dir = tmp_path / "artifact_data"
    generate_dataset(str(artifact_config), str(output_dir), num_workers=1)
    val_file = output_dir / "demo_artifact" / "val.parquet"

    llm_module = _load_module(
        "test_llm_infer_module",
        Path("agrimanager/rollout/inference/inference_rollout.py").resolve(),
    )
    nn_infer_module = _load_module(
        "test_nn_infer_module",
        Path("integrations/wofost_gym/inference/wofost_rl_rollout.py").resolve(),
    )
    nn_train_module = _load_module(
        "test_nn_train_module",
        Path("integrations/wofost_gym/train/train_ppo_parquet.py").resolve(),
    )

    llm_configs, _ = llm_module.load_dataset(str(val_file))
    nn_infer_configs, _ = nn_infer_module.load_dataset(str(val_file))
    reset_configs, defaults = nn_train_module.load_parquet_configs(str(val_file))

    assert [cfg["scenario_id"] for cfg in llm_configs] == [
        cfg["scenario_id"] for cfg in nn_infer_configs
    ]
    assert [cfg["scenario_id"] for cfg in llm_configs] == [
        cfg["scenario_id"] for cfg in reset_configs
    ]
    assert defaults["dataset_split"] == "val"

def test_primary_configs_and_scripts_are_artifact_first() -> None:
    public_entrypoints = [
        Path("entrypoints/dataset/build.sh"),
        Path("entrypoints/train/train.sh"),
        Path("entrypoints/eval/eval.sh"),
        Path("entrypoints/train/nn_train.sh"),
        Path("entrypoints/eval/nn_eval.sh"),
        Path("entrypoints/tools/merge.sh"),
        Path("entrypoints/tools/vllm_launch.sh"),
    ]
    for path in public_entrypoints:
        assert path.is_file(), path
    assert Path("install.sh").is_file()

    legacy_public_paths = [
        Path("scripts"),
        Path("scripts/dataset/generate_dataset.sh"),
        Path("scripts/train/train.sh"),
        Path("scripts/inference/infer.sh"),
        Path("scripts/install.sh"),
        Path("setup/install.sh"),
        Path("scripts/train/merge.sh"),
        Path("scripts/server/bash_launch.sh"),
        Path("scripts/examples/wofost_4id_2ood_test_smoke.sh"),
        Path("scripts/examples"),
        Path("scripts/dataset/gym_dssat"),
        Path("scripts/external/gym_dssat"),
        Path("scripts/inference/analysis"),
        Path("tools/checkpoint/merge.sh"),
        Path("tools/server/vllm_launch.sh"),
        Path("entrypoints/llm"),
        Path("entrypoints/nn"),
        Path("entrypoints/train/llm.sh"),
        Path("entrypoints/infer"),
        Path("entrypoints/infer/llm.sh"),
        Path("entrypoints/train/nn.sh"),
        Path("entrypoints/infer/nn.sh"),
        Path("entrypoints/train/wofostgym.sh"),
        Path("entrypoints/infer/wofostgym.sh"),
        Path("entrypoints/infer/infer.sh"),
        Path("entrypoints/infer/wofost_gym_nn_infer.sh"),
        Path("entrypoints/delta_submit"),
        Path("entrypoints/delta_submit/run_experiment.slurm"),
        Path("entrypoints/delta_submit/start_interactive.sh"),
        Path("entrypoints/delta_submit/submit.sh"),
        Path("integrations/wofost_gym/train/nn_train.sh"),
        Path("integrations/wofost_gym/inference/nn_infer.sh"),
        Path("scripts/experiments"),
        Path("experiments/legacy_wofost_weather_generalization/run_llm.sh"),
        Path("experiments/legacy_wofost_weather_generalization/run_nn.sh"),
        Path("experiments/legacy_wofost_weather_generalization/sbatch_llm.slurm"),
        Path("experiments/legacy_wofost_weather_generalization/sbatch_nn.slurm"),
        Path("entrypoints/train/wofost_gym_nn_train.sh"),
        Path("entrypoints/eval/wofost_gym_nn_eval.sh"),
        Path("entrypoints/train/config/wofost_gym_nn.yaml"),
        Path("entrypoints/eval/config/wofost_gym_nn.yaml"),
        Path("entrypoints/train/nn_ppo_train.sh"),
        Path("entrypoints/eval/nn_ppo_eval.sh"),
        Path("entrypoints/train/config/nn_ppo.yaml"),
        Path("entrypoints/eval/config/nn_ppo.yaml"),
        Path("smoke_tests/wofost_gym/run_nn_ppo_train.sh"),
        Path("smoke_tests/wofost_gym/run_nn_ppo_eval.sh"),
    ]
    for path in legacy_public_paths:
        assert not path.exists(), path

    primary_configs = [
        Path("entrypoints/eval/config/default.yaml"),
        Path("entrypoints/eval/config/nn.yaml"),
        Path("entrypoints/train/config/nn.yaml"),
    ]
    for path in primary_configs:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        assert "env" not in cfg, path
        assert "dataset" not in cfg, path
        if path.name in {"default.yaml", "nn.yaml"}:
            assert "data" in cfg, path

    grpo_cfg = yaml.safe_load(
        Path("entrypoints/train/config/agri_grpo.yaml").read_text(encoding="utf-8")
    )
    assert "custom_cls" not in (grpo_cfg.get("data") or {})

    scripts_to_check = [
        Path("experiments/legacy_wofost_weather_generalization/run_wofost_generalization_weather_wheat_llm_think_train.sh"),
        Path("experiments/legacy_wofost_weather_generalization/run_wofost_generalization_weather_wheat_llm_no_think_train.sh"),
        Path("experiments/legacy_wofost_weather_generalization/run_wofost_generalization_weather_wheat_nn_train_n1.sh"),
    ]
    forbidden_snippets = [
        "dataset.name",
        "data.weather_pool",
        "agri_grpo_pool",
        "default_pool.yaml",
        "pool_inference.sh",
        "random_inference.sh",
        "no_action_inference.sh",
        "--env-override",
    ]
    for path in scripts_to_check:
        text = path.read_text(encoding="utf-8")
        for snippet in forbidden_snippets:
            assert snippet not in text, f"{path} still contains {snippet!r}"
