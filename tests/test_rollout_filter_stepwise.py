from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import numpy as np
import torch
from verl import DataProto


REPO_ROOT = Path(__file__).resolve().parents[1]

ROLLOUT_FILTER_PATH = REPO_ROOT / "agrimanager" / "adapter" / "trainer" / "rollout_filter.py"
ROLLOUT_FILTER_SPEC = spec_from_file_location("rollout_filter_module", ROLLOUT_FILTER_PATH)
assert ROLLOUT_FILTER_SPEC is not None
assert ROLLOUT_FILTER_SPEC.loader is not None
ROLLOUT_FILTER = module_from_spec(ROLLOUT_FILTER_SPEC)
sys.modules[ROLLOUT_FILTER_SPEC.name] = ROLLOUT_FILTER
ROLLOUT_FILTER_SPEC.loader.exec_module(ROLLOUT_FILTER)

STEPWISE_ADVANTAGE_PATH = REPO_ROOT / "agrimanager" / "adapter" / "trainer" / "stepwise_ppo.py"
STEPWISE_ADVANTAGE_SPEC = spec_from_file_location("stepwise_ppo_module", STEPWISE_ADVANTAGE_PATH)
assert STEPWISE_ADVANTAGE_SPEC is not None
assert STEPWISE_ADVANTAGE_SPEC.loader is not None
STEPWISE_ADVANTAGE = module_from_spec(STEPWISE_ADVANTAGE_SPEC)
sys.modules[STEPWISE_ADVANTAGE_SPEC.name] = STEPWISE_ADVANTAGE
STEPWISE_ADVANTAGE_SPEC.loader.exec_module(STEPWISE_ADVANTAGE)


def test_reward_variance_filter_keeps_full_stepwise_trajectories_for_ppo():
    batch = DataProto.from_dict(
        tensors={
            "responses": torch.zeros((8, 2), dtype=torch.long),
            "response_mask": torch.ones((8, 2), dtype=torch.float32),
        },
        non_tensors={
            "uid": np.array(["uid-hi"] * 4 + ["uid-lo"] * 4, dtype=object),
            "trajectory_id": np.array(
                [
                    "hi-traj-0",
                    "hi-traj-0",
                    "hi-traj-1",
                    "hi-traj-1",
                    "lo-traj-0",
                    "lo-traj-0",
                    "lo-traj-1",
                    "lo-traj-1",
                ],
                dtype=object,
            ),
            "step_idx": np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64),
        },
    )
    reward_tensor = torch.zeros((8, 2), dtype=torch.float32)
    reward_extra_infos_dict = {
        "traj_score": [0.0, 0.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5],
    }
    config = ROLLOUT_FILTER.RewardVarianceFilterConfig(
        value=0.9,
        include_zero=False,
        top_p_prob_mode="linear",
        selection_eps=0.0,
    )

    filtered_batch, metrics = ROLLOUT_FILTER.filter_batch_by_reward_variance(
        batch=batch,
        reward_tensor=reward_tensor,
        reward_extra_infos_dict=reward_extra_infos_dict,
        config=config,
    )

    assert len(filtered_batch) == 4
    assert list(filtered_batch.non_tensor_batch["uid"]) == ["uid-hi"] * 4
    assert list(filtered_batch.non_tensor_batch["trajectory_id"]) == [
        "hi-traj-0",
        "hi-traj-0",
        "hi-traj-1",
        "hi-traj-1",
    ]
    assert metrics["rollout/filter_kept_count"] == 1.0
    assert metrics["rollout/filter_kept_ratio"] == 0.5

    token_level_rewards = torch.tensor(
        [
            [0.0, 2.0],
            [0.0, 5.0],
            [0.0, 1.0],
            [0.0, 3.0],
        ],
        dtype=torch.float32,
    )
    values = torch.tensor(
        [
            [10.0, 100.0],
            [7.0, 70.0],
            [6.0, 60.0],
            [4.0, 40.0],
        ],
        dtype=torch.float32,
    )

    advantages, returns = STEPWISE_ADVANTAGE.compute_stepwise_gae_advantage_cross_turn(
        token_level_rewards=token_level_rewards,
        values=values,
        response_mask=filtered_batch.batch["response_mask"],
        trajectory_id=filtered_batch.non_tensor_batch["trajectory_id"],
        step_idx=filtered_batch.non_tensor_batch["step_idx"],
        gamma=1.0,
        lam=1.0,
        whiten_advantages=False,
    )

    expected_advantages = torch.tensor(
        [
            [-3.0, -3.0],
            [-2.0, -2.0],
            [-2.0, -2.0],
            [-1.0, -1.0],
        ],
        dtype=torch.float32,
    )
    expected_returns = torch.tensor(
        [
            [7.0, 7.0],
            [5.0, 5.0],
            [4.0, 4.0],
            [3.0, 3.0],
        ],
        dtype=torch.float32,
    )

    assert torch.equal(advantages, expected_advantages)
    assert torch.equal(returns, expected_returns)
