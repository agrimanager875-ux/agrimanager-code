import numpy as np

from agrimanager.rollout.inference.ag_heuristic_rollout import (
    _normalize_mode,
    _run_single_trial,
    select_ag_heuristic_action,
)


class _DiscreteSpace:
    def __init__(self, action: int = 3):
        self.action = action

    def sample(self):
        return self.action


class _FakeWOFOSTPrompt:
    available_action_kinds = ("n", "p", "k", "irrig")
    fert_amount = 10.0
    irrig_amount = 1.0
    num_fert = 4
    num_irrig = 4

    def _action_id(self, kind, level):
        offsets = {"n": 1, "p": 5, "k": 9, "irrig": 13}
        return offsets[kind] + level - 1


class _FakeWOFOSTEnv:
    prompt_generator = _FakeWOFOSTPrompt()

    def __init__(self, crop_name, observation, objective_id="profit_max"):
        self.actions = []
        self.steps = 0
        self.config = {
            "crop_name": crop_name,
            "env_name": "wofost_gym",
            "objective_id": objective_id,
        }
        self._observation = observation

    def reset(self):
        self.steps = 0
        return "prompt", {"observation": dict(self._observation)}

    def step(self, action):
        self.steps += 1
        action_id = int(np.asarray(action).item())
        self.actions.append(action_id)
        done = self.steps >= 1
        info = {
            "observation": dict(self._observation),
            "executed_action_id": action_id,
            "invalid_action": False,
            "turn_metrics": {"wso": float(action_id), "reward": float(action_id)},
        }
        if done:
            info["trajectory_metrics"] = {
                "objective_reward": float(action_id),
                "invalid_action_rate": 0.0,
                "total_steps": float(self.steps),
            }
        return "next prompt", float(action_id), done, info


class _FakeDSSATEnv:
    decision_interval = 7

    def __init__(self):
        self.config = {"env_name": "gym_dssat"}


class _FakeCyclesEnv:
    def __init__(self):
        self.config = {"env_name": "cycles_gym"}
        self.called = False

    def _fallback_crop_planning_action(self):
        self.called = True
        return [1, 6]


def test_ag_heuristic_aliases():
    assert _normalize_mode("ag-heuristic") == "ag_heuristic"
    assert _normalize_mode("no-action") == "no_action"


def test_ag_heuristic_noops_before_dvs_threshold():
    env = _FakeWOFOSTEnv(
        "maize",
        {
            "DVS": 0.19,
            "SM": 0.45,
            "RAIN": 0.0,
            "TOTN": 0.0,
            "TOTP": 0.0,
            "TOTK": 0.0,
        },
    )
    _run_single_trial(
        envs=[env],
        env_configs=[{}],
        action_spaces=[_DiscreteSpace(action=3)],
        turn_num=1,
        mode="ag_heuristic",
    )

    assert env.actions == [0]


def test_ag_heuristic_dispatches_dssat_expert_schedule_with_interval():
    env = _FakeDSSATEnv()

    action = select_ag_heuristic_action(env, _DiscreteSpace(), {"dap": 16})
    assert action == {"anfer": 0.0, "amir": 25.0, "pesticide": 0.0}

    action = select_ag_heuristic_action(env, _DiscreteSpace(), {"dap": 39})
    assert action == {"anfer": 27.0, "amir": 0.0, "pesticide": 0.0}

    action = select_ag_heuristic_action(env, _DiscreteSpace(), {"dap": 76})
    assert action == {"anfer": 54.0, "amir": 25.0, "pesticide": 0.0}

    action = select_ag_heuristic_action(env, _DiscreteSpace(), {"dap": 80})
    assert action == {"anfer": 0.0, "amir": 0.0, "pesticide": 0.0}


def test_ag_heuristic_dispatches_cycles_crop_planning_heuristic():
    env = _FakeCyclesEnv()
    action = select_ag_heuristic_action(env, _DiscreteSpace(), {})

    assert action == [1, 6]
    assert env.called


def test_ag_heuristic_early_stage_prioritizes_p_then_k_then_n():
    env = _FakeWOFOSTEnv(
        "maize",
        {
            "DVS": 0.25,
            "SM": 0.45,
            "RAIN": 0.0,
            "TOTN": 0.0,
            "TOTP": 0.0,
            "TOTK": 0.0,
        },
    )
    _run_single_trial(
        envs=[env],
        env_configs=[{}],
        action_spaces=[_DiscreteSpace(action=3)],
        turn_num=1,
        mode="ag_heuristic",
    )
    assert env.actions == [6]

    env = _FakeWOFOSTEnv(
        "maize",
        {
            "DVS": 0.25,
            "SM": 0.45,
            "RAIN": 0.0,
            "TOTN": 0.0,
            "TOTP": 20.0,
            "TOTK": 0.0,
        },
    )
    _run_single_trial(
        envs=[env],
        env_configs=[{}],
        action_spaces=[_DiscreteSpace(action=3)],
        turn_num=1,
        mode="ag_heuristic",
    )
    assert env.actions == [10]


def test_ag_heuristic_mid_stage_uses_high_n_before_irrigation():
    env = _FakeWOFOSTEnv(
        "maize",
        {
            "DVS": 0.50,
            "SM": 0.20,
            "RAIN": 0.0,
            "TOTN": 0.0,
            "TOTP": 20.0,
            "TOTK": 20.0,
        },
    )
    _run_single_trial(
        envs=[env],
        env_configs=[{}],
        action_spaces=[_DiscreteSpace(action=3)],
        turn_num=1,
        mode="ag_heuristic",
    )

    assert env.actions == [3]


def test_ag_heuristic_late_yield_small_n_and_terminal_noop():
    env = _FakeWOFOSTEnv(
        "maize",
        {
            "DVS": 1.00,
            "SM": 0.50,
            "RAIN": 0.0,
            "TOTN": 120.0,
            "TOTP": 20.0,
            "TOTK": 20.0,
        },
        objective_id="yield_max",
    )
    _run_single_trial(
        envs=[env],
        env_configs=[{}],
        action_spaces=[_DiscreteSpace(action=3)],
        turn_num=1,
        mode="ag_heuristic",
    )
    assert env.actions == [1]

    env = _FakeWOFOSTEnv(
        "maize",
        {
            "DVS": 1.20,
            "SM": 0.10,
            "RAIN": 0.0,
            "TOTN": 0.0,
            "TOTP": 0.0,
            "TOTK": 0.0,
        },
        objective_id="yield_max",
    )
    _run_single_trial(
        envs=[env],
        env_configs=[{}],
        action_spaces=[_DiscreteSpace(action=3)],
        turn_num=1,
        mode="ag_heuristic",
    )
    assert env.actions == [0]