from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "agrimanager" / "adapter" / "trainer" / "stepwise_ppo.py"
MODULE_SPEC = spec_from_file_location("stepwise_ppo_module", MODULE_PATH)
assert MODULE_SPEC is not None
assert MODULE_SPEC.loader is not None
STEPWISE_ADVANTAGE = module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(STEPWISE_ADVANTAGE)


def test_extract_first_valid_token_scalar_uses_action_start_position():
    values = torch.tensor(
        [
            [10.0, 90.0, 0.0],
            [7.0, 70.0, 700.0],
            [5.0, 6.0, 7.0],
        ]
    )
    response_mask = torch.tensor(
        [
            [1, 1, 0],
            [1, 1, 1],
            [0, 0, 0],
        ],
        dtype=torch.float32,
    )

    state_values = STEPWISE_ADVANTAGE.extract_first_valid_token_scalar(values, response_mask)

    assert torch.equal(state_values, torch.tensor([10.0, 7.0, 0.0]))


def test_compute_stepwise_gae_advantage_cross_turn_uses_next_turn_start_value():
    token_level_rewards = torch.tensor(
        [
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 5.0],
        ]
    )
    values = torch.tensor(
        [
            [10.0, 900.0, 0.0],
            [7.0, 700.0, 777.0],
        ]
    )
    response_mask = torch.tensor(
        [
            [1, 1, 0],
            [1, 1, 1],
        ],
        dtype=torch.float32,
    )
    trajectory_id = ["traj-0", "traj-0"]
    step_idx = [0, 1]

    advantages, returns = STEPWISE_ADVANTAGE.compute_stepwise_gae_advantage_cross_turn(
        token_level_rewards=token_level_rewards,
        values=values,
        response_mask=response_mask,
        trajectory_id=trajectory_id,
        step_idx=step_idx,
        gamma=1.0,
        lam=1.0,
        whiten_advantages=False,
    )

    expected_advantages = torch.tensor(
        [
            [-3.0, -3.0, 0.0],
            [-2.0, -2.0, -2.0],
        ]
    )
    expected_returns = torch.tensor(
        [
            [7.0, 7.0, 0.0],
            [5.0, 5.0, 5.0],
        ]
    )

    assert torch.equal(advantages, expected_advantages)
    assert torch.equal(returns, expected_returns)


def test_compute_stepwise_gae_advantage_cross_turn_can_whiten_state_advantages():
    token_level_rewards = torch.tensor(
        [
            [1.0, 0.0],
            [4.0, 0.0],
        ]
    )
    values = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 0.0],
        ]
    )
    response_mask = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
        ]
    )

    advantages, returns = STEPWISE_ADVANTAGE.compute_stepwise_gae_advantage_cross_turn(
        token_level_rewards=token_level_rewards,
        values=values,
        response_mask=response_mask,
        trajectory_id=["traj-a", "traj-b"],
        step_idx=[0, 0],
        gamma=1.0,
        lam=1.0,
        whiten_advantages=True,
    )

    valid_advantages = advantages[response_mask.bool()]
    assert torch.allclose(valid_advantages.mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(valid_advantages.std(unbiased=True), torch.tensor(1.0), atol=1e-6)
    assert torch.equal(
        returns,
        torch.tensor(
            [
                [1.0, 0.0],
                [4.0, 0.0],
            ]
        ),
    )
