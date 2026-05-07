import math

import pytest

from agrimanager.env.cycles_gym.env import CyclesEnv
from agrimanager.env.cycles_gym.env_config import CyclesEnvConfig
from agrimanager.env.gym_dssat.env import DSSATEnv
from agrimanager.env.gym_dssat.env_config import DSSATEnvConfig


class _DSSATPrompt:
    num_fert = 1
    num_irrig = 1
    fert_amount = 10
    irrig_amount = 5

    def parse_action_response(self, response):
        if response == "combined":
            return {"anfer": 10.0, "amir": 5.0}
        return 1 if response == "valid" else None

    def get_turn_prompt(self, obs, season_num=1, num_seasons=1):
        return "next dssat prompt"


class _DSSATWorker:
    _season_num = 0

    def __init__(self, *, done=False, grnwt=0.0):
        self.done = done
        self.grnwt = grnwt
        self.actions = []

    def step(self, action):
        self.actions.append(dict(action))
        obs = {
            "dap": 10,
            "xlai": 1.0,
            "grnwt": self.grnwt,
            "topwt": 2.0,
            "nstres": 0.0,
            "swfac": 0.0,
        }
        return obs, 0.0, self.done, {}


def _make_dssat_env(*, done=False, grnwt=0.0):
    env = object.__new__(DSSATEnv)
    env.prompt_generator = _DSSATPrompt()
    env.worker = _DSSATWorker(done=done, grnwt=grnwt)
    env.llm_mode = True
    env.enable_pests = False
    env.decision_interval = 1
    env._yield_only_terminal_reward = True
    env._profit_terminal_reward = False
    env.reward_params = {}
    env.valid_action_bonus = 0.1
    env._total_steps = 0
    env._invalid_steps = 0
    env._cumulative_fert = 0.0
    env._cumulative_irrig = 0.0
    env._fert_application_count = 0
    env._irrig_application_count = 0
    env._max_xlai = 0.0
    env._final_grnwt = 0.0
    env._current_dap = 0
    env._current_season = 1
    env._num_seasons = 1
    env._crop_name_cached = "maize"
    return env


class _CyclesPrompt:
    def parse_action_response(self, response):
        return 2 if response == "valid" else None

    def get_turn_prompt(self, obs, context=None):
        return "next cycles prompt"


class _CyclesObserver:
    obs_names = ["DOY", "N TO DATE"]


class _CyclesNativeEnv:
    action_space = None

    def __init__(self, *, done=False, info=None, obs=None):
        self.observer = _CyclesObserver()
        self.done = done
        self.info = info or {}
        self.obs = obs if obs is not None else [0.0, 0.0]
        self.last_action = None

    def step(self, action):
        self.last_action = action
        return list(self.obs), 0.0, self.done, dict(self.info)


def _make_cycles_env(*, done=False, grain_yield=None, total_n=0.0):
    info = {}
    if grain_yield is not None:
        info["trajectory_metrics"] = {"grain_yield": grain_yield}

    env = object.__new__(CyclesEnv)
    env.env = _CyclesNativeEnv(done=done, info=info, obs=[10.0, total_n])
    env.prompt_generator = _CyclesPrompt()
    env.env_id = "CornShortRockSpringsFW-v1"
    env.llm_mode = True
    env.turn_num = 10
    env._total_steps = 0
    env._invalid_steps = 0
    env._episode_reward = 0.0
    env._last_target_yield = float("nan")
    env._last_turn_metrics = {}
    env._yield_only_terminal_reward = True
    env._profit_terminal_reward = False
    env.reward_params = {}
    env.valid_action_bonus = 0.1
    return env


def test_dssat_valid_format_gets_bonus_before_terminal_yield():
    env = _make_dssat_env(done=False)
    _, reward, done, info = env.step("valid")

    assert done is False
    assert reward == pytest.approx(0.1)
    assert info["turn_metrics"]["reward"] == pytest.approx(0.1)
    assert env._invalid_steps == 0

    terminal_env = _make_dssat_env(done=True, grnwt=9000.0)
    _, terminal_reward, done, info = terminal_env.step("valid")

    assert done is True
    assert terminal_reward == pytest.approx(9.1)
    assert info["trajectory_metrics"]["target_yield"] == pytest.approx(9000.0)


def test_dssat_invalid_format_gets_no_bonus():
    env = _make_dssat_env(done=False)
    _, reward, _, _ = env.step("invalid")

    assert reward == pytest.approx(0.0)
    assert env._invalid_steps == 1


def test_dssat_profit_terminal_reward_uses_yield_and_inputs():
    env = _make_dssat_env(done=True, grnwt=9000.0)
    env._yield_only_terminal_reward = False
    env._profit_terminal_reward = True
    env.reward_params = {"cost_n": 3.5, "cost_water": 0.05}

    _, reward, done, info = env.step("valid")

    assert done is True
    assert reward == pytest.approx((9000.0 - 3.5 * 10.0) / 1000.0 + 0.1)
    assert info["turn_metrics"]["profit_ge_kg_ha"] == pytest.approx(8965.0)
    assert info["turn_metrics"]["nutrient_cost_ge_kg_ha"] == pytest.approx(35.0)
    assert info["trajectory_metrics"]["profit_ge_kg_ha"] == pytest.approx(8965.0)


def test_dssat_llm_combined_action_reaches_worker_and_metrics():
    env = _make_dssat_env(done=False)

    _, reward, done, info = env.step("combined")

    assert done is False
    assert reward == pytest.approx(0.1)
    assert env.worker.actions[-1] == {"anfer": 10.0, "amir": 5.0}
    assert info["action_applied"] == {"anfer": 10.0, "amir": 5.0}
    assert info["turn_metrics"]["anfer_applied"] == pytest.approx(10.0)
    assert info["turn_metrics"]["amir_applied"] == pytest.approx(5.0)
    assert info["turn_metrics"]["action_type"] == "fert+irrig"
    assert env._cumulative_fert == pytest.approx(10.0)
    assert env._cumulative_irrig == pytest.approx(5.0)


def test_cycles_valid_format_gets_bonus_before_terminal_yield():
    env = _make_cycles_env(done=False)
    _, reward, done, info = env.step("valid")

    assert done is False
    assert reward == pytest.approx(0.1)
    assert info["turn_metrics"]["reward"] == pytest.approx(0.1)
    assert info["invalid_action"] is False

    terminal_env = _make_cycles_env(done=True, grain_yield=8.0)
    _, terminal_reward, done, info = terminal_env.step("valid")

    assert done is True
    assert terminal_reward == pytest.approx(8.1)
    assert info["trajectory_metrics"]["target_yield"] == pytest.approx(8000.0)
    assert info["trajectory_metrics"]["grain_yield_t_ha"] == pytest.approx(8.0)


def test_cycles_invalid_format_gets_no_bonus():
    env = _make_cycles_env(done=False)
    _, reward, _, info = env.step("invalid")

    assert reward == pytest.approx(0.0)
    assert info["invalid_action"] is True
    assert env._invalid_steps == 1


def test_cycles_profit_terminal_reward_uses_yield_and_nitrogen():
    env = _make_cycles_env(done=True, grain_yield=8.0, total_n=90.0)
    env._yield_only_terminal_reward = False
    env._profit_terminal_reward = True
    env.reward_params = {"cost_n": 3.5}

    _, reward, done, info = env.step("valid")

    assert done is True
    assert reward == pytest.approx((8000.0 - 3.5 * 90.0) / 1000.0 + 0.1)
    assert info["turn_metrics"]["profit_ge_kg_ha"] == pytest.approx(7685.0)
    assert info["turn_metrics"]["nutrient_cost_ge_kg_ha"] == pytest.approx(315.0)
    assert info["trajectory_metrics"]["profit_ge_kg_ha"] == pytest.approx(7685.0)


def test_format_reward_defaults_match_wofost_bonus():
    assert DSSATEnvConfig().valid_action_bonus == pytest.approx(0.1)
    assert CyclesEnvConfig().valid_action_bonus == pytest.approx(0.1)
    assert DSSATEnvConfig(valid_action_bonus=None).valid_action_bonus == pytest.approx(0.1)
    assert CyclesEnvConfig(valid_action_bonus=None).valid_action_bonus == pytest.approx(0.1)
    assert math.isfinite(DSSATEnvConfig().to_dict()["valid_action_bonus"])
    assert math.isfinite(CyclesEnvConfig().to_dict()["valid_action_bonus"])
