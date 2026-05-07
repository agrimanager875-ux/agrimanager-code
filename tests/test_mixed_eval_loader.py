from pathlib import Path

import pandas as pd

from agrimanager.rollout.inference.inference_rollout import load_dataset


def _write_dataset(path: Path, env_name: str, dataset_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "data_source": f"{env_name}/{dataset_id}",
                "agent_name": "agri_tool_agent",
                "prompt": [{"role": "user", "content": "placeholder"}],
                "reward_model": {"style": "rule", "ground_truth": None},
                "extra_info": {
                    "interaction_kwargs": {
                        "name": "agri",
                        "env_config": {
                            "env_name": env_name,
                            "turn_num": 1,
                            "trajectory_group_labels": {"simulator": env_name},
                        },
                    }
                },
            }
        ]
    ).to_parquet(path)


def test_eval_loader_accepts_mixed_inference_file_list(tmp_path: Path) -> None:
    wofost_path = tmp_path / "wofost" / "test.parquet"
    dssat_path = tmp_path / "dssat" / "test.parquet"
    cycles_path = tmp_path / "cycles" / "test.parquet"
    _write_dataset(wofost_path, "wofost_gym", "demo_wofost")
    _write_dataset(dssat_path, "gym_dssat", "demo_dssat")
    _write_dataset(cycles_path, "cycles_gym", "demo_cycles")

    configs, env_name = load_dataset([str(wofost_path), str(dssat_path), str(cycles_path)])

    assert env_name == "__mixed__"
    assert [cfg["env_name"] for cfg in configs] == ["wofost_gym", "gym_dssat", "cycles_gym"]
    assert [cfg["trajectory_group_labels"]["simulator"] for cfg in configs] == [
        "wofost_gym",
        "gym_dssat",
        "cycles_gym",
    ]
