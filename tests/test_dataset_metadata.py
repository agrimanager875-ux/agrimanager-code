from pathlib import Path

from agrimanager.env.base.dataset_metadata import (
    apply_split_metadata_to_env_config,
    normalize_split_metadata,
    write_dataset_manifest,
)


def test_legacy_validation_split_infers_role_and_validation_set():
    metadata = normalize_split_metadata(
        "val_drought",
        {},
        inferred_labels={"weather_regime": "drought"},
    )

    assert metadata["role"] == "validation"
    assert metadata["validation_set"] == "drought"
    assert metadata["group_labels"]["dataset_split"] == "val_drought"
    assert metadata["group_labels"]["dataset_role"] == "validation"
    assert metadata["group_labels"]["validation_set"] == "drought"
    assert metadata["group_labels"]["weather_regime"] == "drought"


def test_explicit_labels_override_inferred_labels():
    metadata = normalize_split_metadata(
        "val_drought",
        {"labels": {"weather_regime": "dry_spell"}},
        inferred_labels={"weather_regime": "drought"},
    )

    assert metadata["labels"]["weather_regime"] == "dry_spell"
    assert metadata["group_labels"]["weather_regime"] == "dry_spell"


def test_train_split_never_gets_validation_set_by_default():
    metadata = normalize_split_metadata(
        "train",
        {},
        dataset_labels={"validation_set": "yield_max"},
    )

    assert metadata["role"] == "train"
    assert metadata["validation_set"] is None
    assert "validation_set" not in metadata["group_labels"]


def test_apply_split_metadata_to_env_config_sets_group_labels():
    env_config = {"trajectory_group_labels": {"simulator": "wofost_gym"}}

    updated = apply_split_metadata_to_env_config(
        env_config,
        "val_id",
        {"labels": {"weather_regime": "id"}},
    )

    assert updated["dataset_split"] == "val_id"
    assert updated["dataset_role"] == "validation"
    assert updated["validation_set"] == "id"
    assert updated["trajectory_group_labels"]["simulator"] == "wofost_gym"
    assert updated["trajectory_group_labels"]["validation_set"] == "id"
    assert updated["trajectory_group_labels"]["weather_regime"] == "id"


def test_write_dataset_manifest_records_validation_sets(tmp_path: Path):
    config_path = tmp_path / "dataset.yaml"
    config_path.write_text("env_name: wofost_gym\ndataset_id: demo\n", encoding="utf-8")
    save_dir = tmp_path / "demo"
    save_dir.mkdir()

    manifest_path = write_dataset_manifest(
        save_dir,
        {
            "dataset_id": "demo",
            "env_name": "wofost_gym",
            "validation_axis": "weather_regime",
            "_config_path": str(config_path),
        },
        {"train": 2, "val_drought": 3},
        split_metadata={
            "train": normalize_split_metadata("train", {}),
            "val_drought": normalize_split_metadata(
                "val_drought",
                {"labels": {"weather_regime": "drought"}},
            ),
        },
    )

    text = manifest_path.read_text(encoding="utf-8")
    assert '"schema_version": 1' in text
    assert '"validation_axis": "weather_regime"' in text
    assert '"num_rows": 3' in text
    assert '"validation_set": "drought"' in text
    assert '"weather_regime": "drought"' in text
