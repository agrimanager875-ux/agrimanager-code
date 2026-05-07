from torch.utils.data import Dataset

from agrimanager.adapter.trainer.validation_sets import (
    EnvConfigOverrideDataset,
    NamedValidationDataset,
    annotate_env_config_with_validation_set,
    apply_env_config_overrides_to_sample,
    normalize_val_sets,
)
from agrimanager.adapter.trainer.trainer import (
    _add_axis_validation_metrics,
    _add_named_validation_metrics,
)


class _TinyDataset(Dataset):
    def __init__(self, env_name: str):
        self.env_name = env_name

    def __len__(self):
        return 1

    def __getitem__(self, index):
        assert index == 0
        return {
            "extra_info": {
                "interaction_kwargs": {
                    "env_config": {
                        "env_name": self.env_name,
                        "trajectory_group_labels": {"simulator": self.env_name},
                    }
                }
            }
        }


def test_normalize_val_sets_accepts_paths_and_lists():
    val_sets = normalize_val_sets(
        {
            "id": "val_id.parquet",
            "drought": ["val_drought.parquet"],
        }
    )

    assert val_sets == {
        "id": ["val_id.parquet"],
        "drought": ["val_drought.parquet"],
    }


def test_annotate_env_config_with_validation_set_preserves_labels():
    annotated = annotate_env_config_with_validation_set(
        {"trajectory_group_labels": {"simulator": "wofost_gym"}},
        "drought",
    )

    assert annotated["dataset_role"] == "validation"
    assert annotated["validation_set"] == "drought"
    assert annotated["trajectory_group_labels"]["simulator"] == "wofost_gym"
    assert annotated["trajectory_group_labels"]["validation_set"] == "drought"


def test_named_validation_dataset_annotates_samples_in_memory():
    dataset = NamedValidationDataset(
        {
            "id": _TinyDataset("wofost_gym"),
            "drought": _TinyDataset("wofost_gym"),
        }
    )

    first = dataset[0]
    second = dataset[1]

    first_env = first["extra_info"]["interaction_kwargs"]["env_config"]
    second_env = second["extra_info"]["interaction_kwargs"]["env_config"]
    assert first_env["validation_set"] == "id"
    assert first_env["trajectory_group_labels"]["validation_set"] == "id"
    assert second_env["validation_set"] == "drought"
    assert second_env["trajectory_group_labels"]["validation_set"] == "drought"


def test_env_config_overrides_apply_to_sample_without_mutating_source():
    sample = _TinyDataset("wofost_gym")[0]

    overridden = apply_env_config_overrides_to_sample(
        sample,
        {"require_think": True, "thinking_mode": "think"},
    )

    original_env = sample["extra_info"]["interaction_kwargs"]["env_config"]
    overridden_env = overridden["extra_info"]["interaction_kwargs"]["env_config"]
    assert "require_think" not in original_env
    assert overridden_env["require_think"] is True
    assert overridden_env["thinking_mode"] == "think"


def test_env_config_override_dataset_wraps_rows():
    dataset = EnvConfigOverrideDataset(
        _TinyDataset("wofost_gym"),
        {"require_think": False},
    )

    env_config = dataset[0]["extra_info"]["interaction_kwargs"]["env_config"]

    assert env_config["require_think"] is False


def test_named_validation_metrics_emit_all_and_per_set_env_metrics():
    metric_dict = {}

    has_named = _add_named_validation_metrics(
        metric_dict,
        sample_uids=["id-0", "drought-0"],
        env_infos={
            "reward": [1.0, 2.0],
            "target_yield": [100.0, 200.0],
            "group_label/validation_set": ["id", "drought"],
        },
    )

    assert has_named is True
    assert metric_dict["val-env/all/target_yield/mean"] == 150.0
    assert metric_dict["val-env/id/target_yield/mean"] == 100.0
    assert metric_dict["val-env/drought/target_yield/mean"] == 200.0


def test_axis_validation_metrics_emit_only_primary_axis_metrics():
    metric_dict = {}

    has_axis = _add_axis_validation_metrics(
        metric_dict,
        sample_uids=["id-0", "drought-0"],
        env_infos={
            "reward": [1.0, 2.0],
            "target_yield": [100.0, 200.0],
            "group_label/weather_regime": ["id", "drought"],
            "group_label/dataset_split": ["val_id", "val_drought"],
        },
        validation_axis="weather_regime",
    )

    assert has_axis is True
    assert metric_dict["val-env-weather_regime/all/target_yield/mean"] == 150.0
    assert metric_dict["val-env-weather_regime/id/target_yield/mean"] == 100.0
    assert metric_dict["val-env-weather_regime/drought/target_yield/mean"] == 200.0
    assert not any(key.startswith("val-env-by-") for key in metric_dict)


def test_crop_axis_validation_metrics_emit_per_crop_metrics():
    metric_dict = {}

    has_axis = _add_axis_validation_metrics(
        metric_dict,
        sample_uids=["cotton-0", "barley-0"],
        env_infos={
            "reward": [1.0, 2.0],
            "target_yield": [100.0, 200.0],
            "group_label/crop": ["cotton", "barley"],
            "group_label/crop_regime": ["id", "heldout"],
        },
        validation_axis="crop",
    )

    assert has_axis is True
    assert metric_dict["val-env-crop/cotton/target_yield/mean"] == 100.0
    assert metric_dict["val-env-crop/barley/target_yield/mean"] == 200.0
    assert any(key.startswith("val-core-crop/cotton/") for key in metric_dict)
    assert any(key.startswith("val-core-crop/barley/") for key in metric_dict)
