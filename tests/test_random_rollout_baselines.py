import numpy as np

from agrimanager.rollout.inference.random_rollout import (
    _flatten_numeric,
    _normalize_mode,
    _run_single_trial,
    _zero_action,
)


class _DiscreteSpace:
    def __init__(self, action: int = 3):
        self.action = action

    def sample(self):
        return self.action


class _MultiDiscreteSpace:
    nvec = np.array([2, 3, 4])


class _FakeEnv:
    def __init__(self):
        self.actions = []
        self.steps = 0

    def reset(self):
        self.steps = 0
        return "prompt", {"observation": {"step": 0}}

    def step(self, action):
        self.steps += 1
        action_id = int(action)
        self.actions.append(action_id)
        done = self.steps >= 2
        info = {
            "observation": {"step": self.steps},
            "executed_action_id": action_id,
            "invalid_action": False,
            "turn_metrics": {"wso": float(action_id), "reward": float(action_id)},
        }
        if done:
            info["trajectory_metrics"] = {
                "target_yield": float(sum(self.actions)),
                "invalid_action_rate": 0.0,
                "total_steps": float(self.steps),
            }
        return "next prompt", float(action_id), done, info


def test_no_action_uses_native_zero_action():
    assert _normalize_mode("no-action") == "no_action"
    np.testing.assert_array_equal(_zero_action(_MultiDiscreteSpace()), np.array([0, 0, 0]))

    env = _FakeEnv()
    results = _run_single_trial(
        envs=[env],
        env_configs=[{}],
        action_spaces=[_DiscreteSpace(action=3)],
        turn_num=2,
        mode="no_action",
    )

    assert env.actions == [0, 0]
    assert results[0]["turns"][-1]["trajectory_metrics"]["target_yield"] == 0.0


def test_random_action_samples_action_space():
    env = _FakeEnv()
    results = _run_single_trial(
        envs=[env],
        env_configs=[{}],
        action_spaces=[_DiscreteSpace(action=3)],
        turn_num=2,
        mode="random",
    )

    assert env.actions == [3, 3]
    assert results[0]["turns"][-1]["trajectory_metrics"]["target_yield"] == 6.0


def test_wandb_payload_flattening_keeps_only_numeric_leaves():
    payload = _flatten_numeric(
        "baseline",
        {
            "primary_metric": {"key": "objective_reward", "mean": -1.5},
            "invalid_action_rate_mean": 0.0,
            "baseline_mode": "random",
            "ok": True,
            "bad_value": float("nan"),
        },
    )

    assert payload == {
        "baseline/primary_metric/mean": -1.5,
        "baseline/invalid_action_rate_mean": 0.0,
        "baseline/ok": 1,
    }
