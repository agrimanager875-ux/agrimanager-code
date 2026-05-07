import numpy as np

from agrimanager.env.base.objective_prompt import build_management_objective_block
from agrimanager.env.cycles_gym.create_dataset import CyclesDatasetGenerator
from agrimanager.env.cycles_gym.env_config import CyclesEnvConfig
from agrimanager.env.cycles_gym.prompt import CornPromptGenerator, build_prompt_generator
from agrimanager.env.gym_dssat.create_dataset import DSSATDatasetGenerator
from agrimanager.env.gym_dssat.env_config import DSSATEnvConfig
from agrimanager.env.gym_dssat.prompt import DSSATPromptGenerator
from agrimanager.env.wofost_gym.create_dataset import WOFOSTDatasetGenerator
from agrimanager.env.wofost_gym.prompt import WOFOSTPromptGenerator


def _assert_prompt_section_order(prompt: str) -> None:
    objective_idx = prompt.index("<management objective")
    observation_idx = prompt.index("<current observation>")
    actions_idx = prompt.index("Available actions")

    assert objective_idx < observation_idx < actions_idx


def test_profit_objective_block_prunes_costs_by_action_menu():
    full_menu = build_management_objective_block(
        "profit_max",
        {},
        available_inputs=("n", "p", "k", "irrig"),
    )
    assert '<management objective id="profit_max">' in full_menu
    assert "3.5 * total_N_kg_ha" in full_menu
    assert "5.5 * total_P_kg_ha" in full_menu
    assert "2.25 * total_K_kg_ha" in full_menu
    assert "0.05 * total_irrig_mm" in full_menu

    dssat_menu = build_management_objective_block(
        "profit_max",
        {},
        available_inputs=("n", "irrig"),
        yield_label="final_yield_kg_ha",
    )
    assert "3.5 * total_N_kg_ha" in dssat_menu
    assert "0.05 * total_irrig_mm" in dssat_menu
    assert "total_P_kg_ha" not in dssat_menu
    assert "total_K_kg_ha" not in dssat_menu

    cycle_menu = build_management_objective_block(
        "profit_max",
        {},
        available_inputs=("n",),
        yield_label="final_yield_kg_ha",
    )
    assert "3.5 * total_N_kg_ha" in cycle_menu
    assert "total_irrig_mm" not in cycle_menu


def test_yield_objective_block_has_no_input_costs():
    block = build_management_objective_block(
        "yield_max",
        {},
        available_inputs=("n", "p", "k", "irrig"),
    )

    assert '<management objective id="yield_max">' in block
    assert "score = final_WSO_kg_ha" in block
    assert "costs" not in block.lower()
    assert "total_N_kg_ha" not in block
    assert "total_irrig_mm" not in block


def test_prompt_objective_text_overrides_default_template():
    block = build_management_objective_block(
        "nutrient_stewardship",
        {},
        available_inputs=("n", "p", "k"),
        objective_text="Custom nutrient objective.",
    )

    assert block == (
        '<management objective id="nutrient_stewardship">\n'
        "Custom nutrient objective.\n"
        "</management objective>"
    )


def test_wofost_profit_objective_prompt_includes_action_costs():
    prompt_generator = WOFOSTPromptGenerator(
        num_fert=2,
        num_irrig=2,
        fert_amount=20.0,
        irrig_amount=5.0,
        intervention_interval=10,
        output_vars=["DAYS", "DVS", "WSO"],
        action_components=["n", "irrig"],
        include_profit_context=True,
        profit_context_params={"cost_n": 3.5, "cost_water": 0.05},
    )

    system_prompt = prompt_generator.get_system_prompt()
    turn_prompt = prompt_generator.get_turn_prompt(np.array([1.0, 0.0, 0.0]))

    assert system_prompt == (
        "You are an agricultural management expert. Optimize the management objective specified in the user message. "
        "Use only the available actions and respond in the required format."
    )
    assert "WOFOST-Gym" not in system_prompt
    assert "profit_wso_kg_ha" not in system_prompt
    assert "total_N_kg_ha" not in system_prompt
    assert "<profit context>" not in turn_prompt
    assert '<management objective id="profit_max">' in turn_prompt
    assert "3.5 * total_N_kg_ha" in turn_prompt
    assert "0.5 * total_irrig_cm" in turn_prompt
    assert "total_irrig_mm" not in turn_prompt
    assert "total_P_kg_ha" not in turn_prompt
    assert "Nitrogen: 1 kg/ha N costs 3.5 kg/ha WSO-equivalent." in turn_prompt
    assert "Irrigation: 1 cm water costs 0.5 kg/ha WSO-equivalent." in turn_prompt
    assert "20 kg/ha N costs 70" not in turn_prompt
    assert "Storage organ dry matter (kg/ha; harvestable yield biomass proxy): 0" in turn_prompt
    _assert_prompt_section_order(turn_prompt)


def test_dssat_profit_objective_prompt_includes_scale_and_examples():
    prompt_generator = DSSATPromptGenerator(
        crop_name="maize",
        output_vars=["dap", "cumsumfert", "swfac", "vstage", "grnwt", "xlai"],
        include_profit_context=True,
        profit_context_params={"cost_n": 3.5, "cost_water": 0.05},
    )

    system_prompt = prompt_generator.get_system_prompt()
    turn_prompt = prompt_generator.get_turn_prompt(
        {"dap": 14, "cumsumfert": 0, "swfac": 0.2, "vstage": 2, "grnwt": 0, "xlai": 0.8}
    )

    assert "DSSAT" not in system_prompt
    assert "profit_wso_kg_ha" not in system_prompt
    assert "NET PROFIT" not in system_prompt
    assert '<management objective id="profit_max">' in turn_prompt
    assert "<profit context>" not in turn_prompt
    assert "<crop traits>" not in turn_prompt
    assert "3.5 * total_N_kg_ha" in turn_prompt
    assert "0.05 * total_irrig_mm" in turn_prompt
    assert "total_P_kg_ha" not in turn_prompt
    assert "Grain dry matter (kg/ha; harvestable yield biomass): 0" in turn_prompt
    assert "Nitrogen: 1 kg/ha N costs 3.5 kg/ha WSO-equivalent." in turn_prompt
    assert "Irrigation: 1 mm water costs 0.05 kg/ha WSO-equivalent." in turn_prompt
    _assert_prompt_section_order(turn_prompt)


def test_cycles_corn_profit_objective_has_nitrogen_only_formula():
    prompt_generator = CornPromptGenerator(
        n_actions=3,
        max_n=30.0,
        obs_names=["DOY", "CUM. BIOMASS", "N TO DATE"],
        include_profit_context=True,
        profit_context_params={"cost_n": 3.5, "cost_water": 0.05},
    )

    system_prompt = prompt_generator.get_system_prompt()
    turn_prompt = prompt_generator.get_turn_prompt([120, 120, 0])

    assert "CycleGym" not in system_prompt
    assert "profit_wso_kg_ha" not in system_prompt
    assert "total_N_kg_ha" not in system_prompt
    assert "<profit context>" not in turn_prompt
    assert '<management objective id="profit_max">' in turn_prompt
    assert "profit_wso_kg_ha =" in turn_prompt
    assert "3.5 * total_N_kg_ha" in turn_prompt
    assert "total_irrig_mm" not in turn_prompt
    assert "CUM. BIOMASS (Cumulative crop biomass state): 120" in turn_prompt
    assert "Nitrogen: 1 kg/ha N costs 3.5 kg/ha WSO-equivalent." in turn_prompt
    assert "30 kg/ha N costs 105" not in turn_prompt
    _assert_prompt_section_order(turn_prompt)


def test_cycles_corn_think_parser_uses_tool_call_tag():
    prompt_generator = CornPromptGenerator(
        n_actions=3,
        max_n=30.0,
        obs_names=["DOY", "N TO DATE"],
        require_think=True,
    )

    valid = "<tool_call>N appears limiting.</tool_call> <answer>Apply 30 kg/ha nitrogen fertilizer.</answer>"
    old_tag = "<think>N appears limiting.</think> <answer>Apply 30 kg/ha nitrogen fertilizer.</answer>"

    assert prompt_generator.think_tag == "tool_call"
    assert prompt_generator.parse_action_response(valid) == 2
    assert prompt_generator.parse_action_response(old_tag) is None


def test_cropgrowth_think_tag_defaults_to_tool_call():
    wofost_prompt = WOFOSTPromptGenerator(output_vars=["DAYS", "DVS"], require_think=True)
    dssat_prompt = DSSATPromptGenerator(output_vars=["dap"], require_think=True)

    class _CornEnv:
        n_actions = 3
        maxN = 30.0
        observer = None

    corn_prompt = build_prompt_generator(
        _CornEnv(),
        env_id="CornShortRockSpringsFW-v1",
        require_think=True,
    )

    assert wofost_prompt.think_tag == "tool_call"
    assert dssat_prompt.think_tag == "tool_call"
    assert corn_prompt.think_tag == "tool_call"


def test_cycle_crop_planning_think_tag_default_is_unchanged():
    class _ActionSpace:
        nvec = [2, 3]

    class _PlanningEnv:
        rotation_crops = ["corn", "soybean"]
        action_space = _ActionSpace()
        observer = None

    prompt = build_prompt_generator(
        _PlanningEnv(),
        env_id="CropPlanningNewHollandRW-v1",
        require_think=True,
        include_crop_traits=False,
    )

    assert prompt.think_tag == "think"


def test_objective_config_is_exported_and_propagated(tmp_path):
    cycles_cfg = CyclesEnvConfig(
        include_profit_context=True,
        profit_context_params={"cost_n": 3.5},
        reward_params={"cost_water": 0.05},
    ).to_dict()
    assert cycles_cfg["objective_id"] == "profit_max"
    assert cycles_cfg["include_profit_context"] is True
    assert cycles_cfg["profit_context_params"] == {"cost_n": 3.5}
    assert cycles_cfg["reward_params"] == {"cost_water": 0.05}

    dssat_cfg = DSSATEnvConfig(
        save_folder=str(tmp_path),
        include_profit_context=True,
        profit_context_params={"cost_n": 3.5},
        reward_params={"cost_water": 0.05},
    ).to_dict()

    assert dssat_cfg["objective_id"] == "profit_max"
    assert dssat_cfg["think_tag"] == "tool_call"
    assert dssat_cfg["include_profit_context"] is True
    assert dssat_cfg["include_crop_traits"] is False
    assert dssat_cfg["profit_context_params"] == {"cost_n": 3.5}
    assert dssat_cfg["reward_params"] == {"cost_water": 0.05}

    wofost_generator = WOFOSTDatasetGenerator(
        {
            "env_name": "wofost_gym",
            "dataset_id": "test",
            "include_profit_context": True,
            "profit_context_params": {"cost_n": 3.5},
        },
        str(tmp_path),
    )
    assert wofost_generator.base_config["objective_id"] == "profit_max"
    assert wofost_generator.base_config["include_profit_context"] is True
    assert wofost_generator.base_config["profit_context_params"] == {"cost_n": 3.5}

    cycles_generator = CyclesDatasetGenerator(
        {
            "env_name": "cycles_gym",
            "dataset_id": "test",
            "include_profit_context": True,
            "profit_context_params": {"cost_n": 3.5},
            "reward_params": {"cost_water": 0.05},
            "base_config": {"env_id": "CornShortRockSpringsFW-v1"},
        },
        str(tmp_path),
    )
    assert cycles_generator.base_config["objective_id"] == "profit_max"
    assert cycles_generator.base_config["include_profit_context"] is True
    assert cycles_generator.base_config["profit_context_params"] == {"cost_n": 3.5}
    assert cycles_generator.base_config["reward_params"] == {"cost_water": 0.05}

    dssat_generator = DSSATDatasetGenerator(
        {
            "env_name": "gym_dssat",
            "dataset_id": "test",
            "include_profit_context": True,
            "profit_context_params": {"cost_n": 3.5},
            "reward_params": {"cost_water": 0.05},
        },
        str(tmp_path),
    )
    assert dssat_generator.base_config["objective_id"] == "profit_max"
    assert dssat_generator.base_config["include_profit_context"] is True
    assert dssat_generator.base_config["profit_context_params"] == {"cost_n": 3.5}
    assert dssat_generator.base_config["reward_params"] == {"cost_water": 0.05}
