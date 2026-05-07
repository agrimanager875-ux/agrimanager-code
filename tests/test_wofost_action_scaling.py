from types import SimpleNamespace

import numpy as np
import pytest

from agrimanager.env.wofost_gym.env import _scale_action_amounts_by_interval
from agrimanager.env.wofost_gym.env import compute_nutrient_stewardship_reward
from agrimanager.env.wofost_gym.env import compute_profit_max_reward
from agrimanager.env.wofost_gym.env import compute_water_stewardship_reward
from agrimanager.env.wofost_gym.env import compute_yield_max_reward
from agrimanager.env.wofost_gym.create_dataset import WOFOSTDatasetGenerator
from agrimanager.env.wofost_gym.create_dataset import _drop_implicit_save_folder
from agrimanager.env.wofost_gym.env_config import WOFOSTEnvConfig
from agrimanager.env.wofost_gym.prompt import WOFOSTPromptGenerator


def test_interval_action_scaling_flag_defaults_off():
    config = WOFOSTEnvConfig(intvn_interval=10)

    assert config.scale_action_amounts_by_interval is False


def test_objective_id_is_default_reward_definition():
    config = WOFOSTEnvConfig()
    serialized = _drop_implicit_save_folder(dict(config.to_dict()), {})

    assert config.objective_id == "profit_max"
    assert config.env_reward is None
    assert "env_reward" not in serialized


def test_interval_action_scaling_flag_can_be_enabled():
    config = WOFOSTEnvConfig(intvn_interval=10, scale_action_amounts_by_interval=True)

    assert config.scale_action_amounts_by_interval is True


def test_dataset_generator_propagates_interval_action_scaling_flag(tmp_path):
    generator = WOFOSTDatasetGenerator(
        {
            "env_name": "wofost_gym",
            "dataset_id": "test",
            "intvn_interval": 10,
            "scale_action_amounts_by_interval": True,
        },
        str(tmp_path),
    )

    assert generator.base_config["scale_action_amounts_by_interval"] is True


def test_dataset_generator_propagates_reward_objective_fields(tmp_path):
    generator = WOFOSTDatasetGenerator(
        {
            "env_name": "wofost_gym",
            "dataset_id": "test",
            "objective_id": "nutrient_stewardship",
            "prompt_objective_id": "yield_max",
            "reward_params": {"tau_Y": 0.8},
            "y_ref": 5000.0,
        },
        str(tmp_path),
    )

    assert generator.base_config["objective_id"] == "nutrient_stewardship"
    assert generator.base_config["prompt_objective_id"] == "yield_max"
    assert generator.base_config["reward_params"] == {"tau_Y": 0.8}
    assert generator.base_config["y_ref"] == pytest.approx(5000.0)


def test_scale_action_amounts_by_interval_preserves_action_count_semantics():
    npk_args = SimpleNamespace(num_fert=4, num_irrig=4, fert_amount=2.0, irrig_amount=0.5)

    _scale_action_amounts_by_interval(npk_args, intvn_interval=10)

    assert npk_args.num_fert == 4
    assert npk_args.num_irrig == 4
    assert npk_args.fert_amount == pytest.approx(20.0)
    assert npk_args.irrig_amount == pytest.approx(5.0)


def test_prompt_uses_interval_scaled_action_amounts():
    prompt_generator = WOFOSTPromptGenerator(
        num_fert=4,
        num_irrig=4,
        fert_amount=20.0,
        irrig_amount=5.0,
        intervention_interval=10,
        output_vars=["DAYS", "DVS"],
    )

    prompt = prompt_generator.get_turn_prompt(np.array([1.0, 0.0]))

    assert "actions are taken every 10 days" in prompt
    assert "Apply nitrogen fertilizer (20.0, 40.0, 60.0, or 80.0 kg/ha)" in prompt
    assert "Irrigate with 5.0, 10.0, 15.0, or 20.0 cm of water" in prompt
    assert "Example: <answer>Apply 20.0 kg/ha nitrogen fertilizer.</answer>" in prompt
    assert prompt_generator.parse_action_response(
        "<answer>Apply 80 kg/ha nitrogen fertilizer.</answer>"
    ) == 4


def test_maize_prompt_uses_approximate_season_length():
    prompt_generator = WOFOSTPromptGenerator(
        crop_name="maize",
        season_length=152,
        intervention_interval=10,
        output_vars=["DAYS", "DVS"],
    )

    prompt = prompt_generator.get_turn_prompt(np.array([10.0, 0.0]))

    assert "The planned growing window spans around 180 days" in prompt
    assert "The planned growing window spans 152 days" not in prompt


def test_prompt_uses_reward_objective_text():
    prompt_generator = WOFOSTPromptGenerator(
        num_fert=4,
        num_irrig=4,
        fert_amount=20.0,
        irrig_amount=5.0,
        intervention_interval=10,
        output_vars=["DAYS", "DVS", "NAVAIL", "PAVAIL", "KAVAIL"],
        objective_id="nutrient_stewardship",
    )

    system_prompt = prompt_generator.get_system_prompt()
    turn_prompt = prompt_generator.get_turn_prompt(np.array([1.0, 0.0, 20.0, 20.0, 20.0]))

    assert "avoiding N/P/K over-application" not in system_prompt
    assert '<management objective id="nutrient_stewardship">' in turn_prompt
    assert "terminal nutrient depletion" in turn_prompt
    assert "terminal soil nutrients in agronomic ranges" in turn_prompt


def test_nutrient_stewardship_reward_prefers_balanced_input_policy():
    params = {"y_ref": 6000.0}
    balanced = compute_nutrient_stewardship_reward(
        {
            "final_wso": 5100.0,
            "total_n_kg_ha": 160.0,
            "total_p_kg_ha": 160.0,
            "total_k_kg_ha": 160.0,
            "terminal_navail": 30.0,
            "terminal_pavail": 30.0,
            "terminal_kavail": 30.0,
        },
        params,
    )
    high_input = compute_nutrient_stewardship_reward(
        {
            "final_wso": 6000.0,
            "total_n_kg_ha": 320.0,
            "total_p_kg_ha": 320.0,
            "total_k_kg_ha": 320.0,
            "terminal_navail": 50.0,
            "terminal_pavail": 50.0,
            "terminal_kavail": 50.0,
        },
        params,
    )
    zero_input = compute_nutrient_stewardship_reward(
        {
            "final_wso": 600.0,
            "total_n_kg_ha": 0.0,
            "total_p_kg_ha": 0.0,
            "total_k_kg_ha": 0.0,
            "terminal_navail": 0.0,
            "terminal_pavail": 0.0,
            "terminal_kavail": 0.0,
        },
        params,
    )

    assert balanced["objective_reward"] > high_input["objective_reward"]
    assert high_input["objective_reward"] > zero_input["objective_reward"]
    assert high_input["reward_application_penalty"] > 0.0
    assert zero_input["reward_yield_floor_penalty"] > 0.0


def test_reward_objectives_share_calibrated_terminal_scale():
    metrics = {
        "final_wso": 5000.0,
        "y_ref": 6000.0,
        "total_n_kg_ha": 120.0,
        "total_p_kg_ha": 40.0,
        "total_k_kg_ha": 40.0,
        "total_irrig_mm": 60.0,
        "terminal_navail": 20.0,
        "terminal_pavail": 20.0,
        "terminal_kavail": 20.0,
    }

    yield_reward = compute_yield_max_reward(metrics)
    profit_reward = compute_profit_max_reward(metrics)
    water_reward = compute_water_stewardship_reward(metrics)

    assert yield_reward["objective_reward"] == pytest.approx(5000.0 / 6000.0)
    assert profit_reward["profit_ge_kg_ha"] < metrics["final_wso"]
    assert profit_reward["input_cost_ge_kg_ha"] > 0.0
    assert water_reward["reward_water_use_penalty"] > 0.0
    assert water_reward["reward_water_budget_penalty"] == pytest.approx(0.0)


def test_action_menu_prompt_restricts_actions_for_lnw_mode():
    prompt_generator = WOFOSTPromptGenerator(
        num_fert=4,
        num_irrig=4,
        fert_amount=20.0,
        irrig_amount=5.0,
        intervention_interval=10,
        output_vars=["DAYS", "DVS"],
        available_action_kinds=WOFOSTPromptGenerator.action_menu_from_env_id("lnw-v0"),
    )

    prompt = prompt_generator.get_turn_prompt(np.array([1.0, 0.0]))

    assert "Apply nitrogen fertilizer" in prompt
    assert "Irrigate with 5.0, 10.0, 15.0, or 20.0 cm of water" in prompt
    assert "Apply phosphorus fertilizer" not in prompt
    assert "Apply potassium fertilizer" not in prompt
    assert "Unavailable actions: phosphorus fertilizer or potassium fertilizer." in prompt
    assert prompt_generator.parse_action_response(
        "<answer>Irrigate with 10 cm of water.</answer>"
    ) == 6
    assert prompt_generator.describe_action(6) == (
        "<answer>Irrigate with 10.0 cm of water.</answer>"
    )


def test_action_menu_parser_rejects_prompt_unavailable_actions():
    prompt_generator = WOFOSTPromptGenerator(
        num_fert=4,
        num_irrig=4,
        fert_amount=20.0,
        irrig_amount=5.0,
        output_vars=["DAYS", "DVS"],
        available_action_kinds=WOFOSTPromptGenerator.action_menu_from_env_id("lw-v0"),
    )

    assert prompt_generator.parse_action_response(
        "<answer>Apply 20 kg/ha nitrogen fertilizer.</answer>"
    ) is None
    assert prompt_generator.parse_action_response(
        "<answer>Irrigate with 5 cm of water.</answer>"
    ) == 1


def test_dataset_generator_propagates_prompt_action_schema_override(tmp_path):
    generator = WOFOSTDatasetGenerator(
        {
            "env_name": "wofost_gym",
            "dataset_id": "test",
            "env_id": "ln-v0",
            "prompt_action_schema_env_id": "lnpkw-v0",
        },
        str(tmp_path),
    )

    assert generator.base_config["env_id"] == "ln-v0"
    assert generator.base_config["prompt_action_schema_env_id"] == "lnpkw-v0"
