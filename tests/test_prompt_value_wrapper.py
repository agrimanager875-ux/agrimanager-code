from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import numpy as np
import torch
from verl import DataProto
from verl.trainer.ppo.core_algos import AdvantageEstimator

from agrimanager.adapter.trainer.trainer import AgriTrainer, compute_advantage


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "agrimanager" / "adapter" / "trainer" / "stepwise_ppo.py"
MODULE_SPEC = spec_from_file_location("stepwise_ppo_module", MODULE_PATH)
assert MODULE_SPEC is not None
assert MODULE_SPEC.loader is not None
PROMPT_VALUE_WRAPPER = module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = PROMPT_VALUE_WRAPPER
MODULE_SPEC.loader.exec_module(PROMPT_VALUE_WRAPPER)


def test_build_prompt_value_batch_keeps_prompt_and_first_response_token():
    batch = DataProto.from_dict(
        tensors={
            "input_ids": torch.tensor(
                [
                    [0, 0, 11, 12, 21, 22, 0],
                    [31, 32, 33, 34, 41, 0, 0],
                ],
                dtype=torch.long,
            ),
            "responses": torch.tensor(
                [
                    [21, 22, 0],
                    [41, 0, 0],
                ],
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                [
                    [0, 0, 1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1, 0, 0],
                ],
                dtype=torch.long,
            ),
            "position_ids": torch.tensor(
                [
                    [0, 0, 0, 1, 2, 3, 0],
                    [0, 1, 2, 3, 4, 0, 0],
                ],
                dtype=torch.long,
            ),
            "response_mask": torch.tensor(
                [
                    [1, 1, 0],
                    [1, 0, 0],
                ],
                dtype=torch.float32,
            ),
        }
    )

    wrapped = PROMPT_VALUE_WRAPPER.build_prompt_value_batch(
        batch,
        values=torch.tensor([10.0, 20.0]),
        returns=torch.tensor([7.0, 8.0]),
    )

    assert torch.equal(
        wrapped.batch["input_ids"],
        torch.tensor(
            [
                [0, 0, 11, 12, 21],
                [31, 32, 33, 34, 41],
            ],
            dtype=torch.long,
        ),
    )
    assert torch.equal(wrapped.batch["responses"], torch.tensor([[21], [41]], dtype=torch.long))
    assert torch.equal(wrapped.batch["attention_mask"], torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]]))
    assert torch.equal(wrapped.batch["position_ids"], torch.tensor([[0, 0, 0, 1, 2], [0, 1, 2, 3, 4]]))
    assert torch.equal(wrapped.batch["response_mask"], torch.tensor([[1.0], [1.0]]))
    assert torch.equal(wrapped.batch["values"], torch.tensor([[10.0], [20.0]]))
    assert torch.equal(wrapped.batch["returns"], torch.tensor([[7.0], [8.0]]))


def test_compute_advantage_accepts_prompt_only_state_values():
    data = DataProto.from_dict(
        tensors={
            "token_level_rewards": torch.tensor(
                [
                    [0.0, 2.0, 0.0],
                    [0.0, 0.0, 5.0],
                ]
            ),
            "response_mask": torch.tensor(
                [
                    [1.0, 1.0, 0.0],
                    [1.0, 1.0, 1.0],
                ]
            ),
            "state_values": torch.tensor([10.0, 7.0]),
        },
        non_tensors={
            "trajectory_id": np.array(["traj-0", "traj-0"], dtype=object),
            "step_idx": np.array([0, 1], dtype=np.int64),
        },
    )

    result = compute_advantage(
        data=data,
        adv_estimator=AdvantageEstimator.GAE,
        gamma=1.0,
        lam=1.0,
        stepwise_config={
            "enable": True,
            "mode": "per_step",
            "adv_estimator": "gae",
            "gamma": 1.0,
            "lam": 1.0,
            "whiten_advantages": False,
        },
    )

    assert torch.equal(
        result.batch["advantages"],
        torch.tensor(
            [
                [-3.0, -3.0, 0.0],
                [-2.0, -2.0, -2.0],
            ]
        ),
    )
    assert torch.equal(
        result.batch["returns"],
        torch.tensor(
            [
                [7.0, 7.0, 0.0],
                [5.0, 5.0, 5.0],
            ]
        ),
    )
    assert torch.equal(result.batch["state_returns"], torch.tensor([7.0, 5.0]))


def test_compute_advantage_uses_stepwise_default_lam_of_point_97():
    data = DataProto.from_dict(
        tensors={
            "token_level_rewards": torch.tensor(
                [
                    [0.0, 2.0, 0.0],
                    [0.0, 0.0, 5.0],
                ]
            ),
            "response_mask": torch.tensor(
                [
                    [1.0, 1.0, 0.0],
                    [1.0, 1.0, 1.0],
                ]
            ),
            "state_values": torch.tensor([10.0, 7.0]),
        },
        non_tensors={
            "trajectory_id": np.array(["traj-0", "traj-0"], dtype=object),
            "step_idx": np.array([0, 1], dtype=np.int64),
        },
    )

    result = compute_advantage(
        data=data,
        adv_estimator=AdvantageEstimator.GAE,
        gamma=0.5,
        lam=0.5,
        stepwise_config={
            "enable": True,
            "mode": "per_step",
            "adv_estimator": "gae",
            "whiten_advantages": False,
        },
    )

    assert torch.allclose(
        result.batch["advantages"],
        torch.tensor(
            [
                [-2.94, -2.94, 0.0],
                [-2.0, -2.0, -2.0],
            ]
        ),
        atol=1e-6,
    )
    assert torch.allclose(result.batch["state_returns"], torch.tensor([7.06, 5.0]), atol=1e-6)


def test_compute_advantage_allows_stepwise_gamma_and_lam_override():
    data = DataProto.from_dict(
        tensors={
            "token_level_rewards": torch.tensor(
                [
                    [0.0, 2.0, 0.0],
                    [0.0, 0.0, 5.0],
                ]
            ),
            "response_mask": torch.tensor(
                [
                    [1.0, 1.0, 0.0],
                    [1.0, 1.0, 1.0],
                ]
            ),
            "state_values": torch.tensor([10.0, 7.0]),
        },
        non_tensors={
            "trajectory_id": np.array(["traj-0", "traj-0"], dtype=object),
            "step_idx": np.array([0, 1], dtype=np.int64),
        },
    )

    result = compute_advantage(
        data=data,
        adv_estimator=AdvantageEstimator.GAE,
        gamma=1.0,
        lam=1.0,
        stepwise_config={
            "enable": True,
            "mode": "per_step",
            "adv_estimator": "gae",
            "gamma": 0.5,
            "lam": 0.5,
            "whiten_advantages": False,
        },
    )

    assert torch.allclose(
        result.batch["advantages"],
        torch.tensor(
            [
                [-5.0, -5.0, 0.0],
                [-2.0, -2.0, -2.0],
            ]
        ),
        atol=1e-6,
    )
    assert torch.allclose(result.batch["state_returns"], torch.tensor([5.0, 5.0]), atol=1e-6)


def test_prompt_only_value_metrics_use_scalar_state_values():
    batch = DataProto.from_dict(
        tensors={
            "state_values": torch.tensor([10.0, 7.0]),
            "state_returns": torch.tensor([7.0, 5.0]),
        }
    )

    metrics = AgriTrainer._compute_prompt_only_value_metrics(batch)

    assert metrics["critic/values/mean"] == 8.5
    assert metrics["critic/values/max"] == 10.0
    assert metrics["critic/values/min"] == 7.0
    assert "critic/vf_explained_var" in metrics
