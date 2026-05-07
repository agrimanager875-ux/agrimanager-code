from pathlib import Path

from agrimanager.env.base.create_dataset import BaseDatasetGenerator
from agrimanager.env.wofost_gym.create_dataset import WOFOSTDatasetGenerator


class _DummyDatasetGenerator(BaseDatasetGenerator):
    def build_split_records(self, split: str):
        return []


def test_base_dataset_generator_uses_shared_placeholder_prompt(tmp_path: Path) -> None:
    generator = _DummyDatasetGenerator(
        {
            "env_name": "dummy_env",
            "dataset_id": "demo",
        },
        str(tmp_path),
    )

    prompt = generator._build_initial_prompt({"seed": 7})

    assert prompt == [
        {
            "role": "system",
            "content": "You are an agricultural management expert.",
        },
        {
            "role": "user",
            "content": (
                "Placeholder dataset prompt for dummy_env. "
                "Rollout and training derive the live system prompt and current "
                "observation from the environment when interaction starts, so "
                "dataset build avoids calling env.reset()."
            ),
        },
    ]


def test_wofost_dataset_generator_reuses_shared_placeholder_prompt_with_context(
    tmp_path: Path,
) -> None:
    generator = WOFOSTDatasetGenerator(
        {
            "env_name": "wofost_gym",
            "dataset_id": "demo_dataset",
        },
        str(tmp_path / "data"),
    )

    prompt = generator._render_prompt(
        {
            "crop_name": "wheat",
            "scenario_id": "deadbeefcafebabe",
            "agro_params": {
                "year": 2001,
                "latitude": 51.1,
                "longitude": 5.1,
            },
        }
    )

    assert prompt[0]["content"] == "You are an agricultural management expert."
    assert prompt[1]["content"].startswith(
        "Placeholder dataset prompt for WOFOST-Gym "
        "(crop=wheat, year=2001, lat=51.10, lon=5.10, scenario_id=deadbeefcafebabe)."
    )
