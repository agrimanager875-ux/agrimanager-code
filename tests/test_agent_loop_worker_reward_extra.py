import math
from types import SimpleNamespace

import numpy as np

from agrimanager.adapter.agent_loop.worker import (
    _rectangularize_dataproto_non_tensor_batches,
    _rectangularize_reward_extra_infos,
)


class _FakeDataProto:
    def __init__(self, non_tensor_batch, meta_info=None):
        self.non_tensor_batch = non_tensor_batch
        self.meta_info = meta_info or {}

    def __len__(self):
        first_values = next(iter(self.non_tensor_batch.values()))
        return len(first_values)


def test_rectangularize_reward_extra_infos_fills_mixed_simulator_missing_keys():
    outputs = [
        SimpleNamespace(
            extra_fields={
                "reward_extra_info": {
                    "target_yield": 100.0,
                    "final_wso": 100.0,
                    "group_label/simulator": "wofost_gym",
                }
            }
        ),
        SimpleNamespace(
            extra_fields={
                "reward_extra_info": {
                    "target_yield": 200.0,
                    "yield_kgha": 200.0,
                    "group_label/simulator": "gym_dssat",
                    "crop_name": "maize",
                }
            }
        ),
    ]

    _rectangularize_reward_extra_infos(outputs)

    first = outputs[0].extra_fields["reward_extra_info"]
    second = outputs[1].extra_fields["reward_extra_info"]

    assert list(first) == list(second)
    assert first["target_yield"] == 100.0
    assert second["target_yield"] == 200.0
    assert first["final_wso"] == 100.0
    assert math.isnan(second["final_wso"])
    assert math.isnan(first["yield_kgha"])
    assert second["yield_kgha"] == 200.0
    assert first["crop_name"] == ""
    assert second["crop_name"] == "maize"


def test_rectangularize_dataproto_non_tensor_batches_before_concat():
    outputs = [
        _FakeDataProto(
            {
                "target_yield": np.array([100.0]),
                "final_wso": np.array([100.0]),
                "group_label/simulator": np.array(["wofost_gym"], dtype=object),
            },
            meta_info={"reward_extra_keys": ["target_yield", "final_wso", "group_label/simulator"]},
        ),
        _FakeDataProto(
            {
                "target_yield": np.array([200.0]),
                "yield_kgha": np.array([200.0]),
                "group_label/simulator": np.array(["gym_dssat"], dtype=object),
                "crop_name": np.array(["maize"], dtype=object),
            },
            meta_info={"reward_extra_keys": ["target_yield", "yield_kgha", "group_label/simulator", "crop_name"]},
        ),
    ]

    _rectangularize_dataproto_non_tensor_batches(outputs)

    assert set(outputs[0].non_tensor_batch) == set(outputs[1].non_tensor_batch)
    assert math.isnan(outputs[1].non_tensor_batch["final_wso"][0])
    assert math.isnan(outputs[0].non_tensor_batch["yield_kgha"][0])
    assert outputs[0].non_tensor_batch["crop_name"][0] == ""
    assert outputs[0].meta_info["reward_extra_keys"] == outputs[1].meta_info["reward_extra_keys"]
