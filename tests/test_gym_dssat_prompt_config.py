from pathlib import Path

import yaml

from agrimanager.env.gym_dssat.create_dataset import DSSATDatasetGenerator
from agrimanager.env.gym_dssat.env_config import DSSATEnvConfig
from agrimanager.env.gym_dssat.prompt import DSSATPromptGenerator
from agrimanager.env.gym_dssat.prompt_cotton import CottonPromptGenerator
from agrimanager.env.gym_dssat.prompt_rice import RicePromptGenerator


def test_dssat_env_config_exports_runtime_options(tmp_path: Path):
    cfg = DSSATEnvConfig(
        env_id="maize-test",
        save_folder=str(tmp_path),
        require_think=True,
        decision_interval=10,
        num_seasons=2,
    )

    data = cfg.to_dict()

    assert data["env_name"] == "gym_dssat"
    assert data["require_think"] is True
    assert data["decision_interval"] == 10
    assert data["num_seasons"] == 2


def test_dssat_prompt_requires_exact_answer_tags():
    prompt = DSSATPromptGenerator(crop_name="maize")

    assert prompt.parse_action_response("<answer>Take no action.</answer>") == 0
    assert prompt.parse_action_response("extra <answer>Take no action.</answer>") is None
    assert prompt.parse_action_response("<answer>Apply 20 kg/ha nitrogen fertilizer.</answer>") == 2


def test_dssat_prompt_parses_combined_fertilizer_irrigation_action():
    prompt = DSSATPromptGenerator(
        crop_name="maize",
        num_fert=4,
        num_irrig=4,
        fert_amount=10.0,
        irrig_amount=10.0,
    )

    assert prompt.parse_action_response(
        "<answer>Irrigate with 20 mm of water and apply 20 kg/ha nitrogen fertilizer.</answer>"
    ) == {"anfer": 20.0, "amir": 20.0}
    assert prompt.parse_action_response(
        "<answer>Apply 20 kg/ha nitrogen fertilizer and irrigate with 30 mm of water.</answer>"
    ) == {"anfer": 20.0, "amir": 30.0}


def test_crop_prompt_generators_use_crop_specific_bounds_and_traits():
    rice = RicePromptGenerator(include_crop_traits=True)
    cotton = CottonPromptGenerator(include_crop_traits=True)

    assert rice.crop_name == "rice"
    assert rice.irrig_max == 80.0
    assert "DSSAT" not in rice.get_system_prompt()
    assert "<crop traits>" in rice.get_turn_prompt({"dap": 1})
    assert "Crop Name: rice" in rice.get_turn_prompt({"dap": 1})

    assert cotton.crop_name == "cotton"
    assert cotton.fert_max == 25.0
    assert "DSSAT" not in cotton.get_system_prompt()
    assert "<crop traits>" in cotton.get_turn_prompt({"dap": 1})
    assert "Crop Name: cotton" in cotton.get_turn_prompt({"dap": 1})


def test_weather_generalization_config_uses_distinct_random_weather_seeds(tmp_path: Path):
    config_path = Path(
        "experiments/legacy_gym_dssat_weather_generalization/config/weather_maize_llm_think.yaml"
    )
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    builder = DSSATDatasetGenerator(cfg, str(tmp_path))
    train_records = builder.build_split_records("train")
    test_records = builder.build_split_records("test")

    assert len(train_records) == cfg["train_seeds"]["count"]
    assert len(test_records) == cfg["test_seeds"]["count"]
    assert {r["env_params"]["seed"] for r in train_records}.isdisjoint(
        {r["env_params"]["seed"] for r in test_records}
    )
    assert all(r["env_params"]["random_weather"] is True for r in train_records[:10])
