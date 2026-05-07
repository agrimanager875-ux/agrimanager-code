from __future__ import annotations

import json

import agrimanager.env.base.live_prompt_capture as live_prompt_capture


class DummyPromptEnv:
    def __init__(self):
        self.closed = False

    def system_prompt(self):
        return "system prompt"

    def reset(self):
        return "reset user prompt", {"phase": "reset"}

    def step(self, action):
        return f"next prompt after {action}", 1.5, True, {"phase": "step"}

    def close(self):
        self.closed = True


def test_capture_live_prompt_records_system_and_first_user_prompt(monkeypatch):
    created = []

    def fake_create_environment(env_name, env_config):
        created.append((env_name, env_config))
        return DummyPromptEnv(), object()

    monkeypatch.setattr(live_prompt_capture, "create_environment", fake_create_environment)

    capture = live_prompt_capture.capture_live_prompt(
        "dummy_env",
        {
            "env_id": "dummy-v0",
            "scenario_id": "scenario-1",
            "validation_set": "cold",
            "llm_mode": False,
        },
        label="cold scenario",
        sample_idx=3,
        steps=1,
        step_action=0,
    )

    assert created[0][0] == "dummy_env"
    assert created[0][1]["llm_mode"] is True
    assert capture["label"] == "cold scenario"
    assert capture["system_prompt"] == "system prompt"
    assert capture["metadata"]["env_id"] == "dummy-v0"
    assert capture["metadata"]["scenario_id"] == "scenario-1"
    assert capture["metadata"]["validation_set"] == "cold"
    assert capture["metadata"]["sample_idx"] == 3
    assert capture["turns"][0]["source"] == "reset"
    assert capture["turns"][0]["user_prompt"] == "reset user prompt"
    assert capture["turns"][1]["source"] == "step(action=0)"
    assert capture["turns"][1]["user_prompt"] == "next prompt after 0"
    assert capture["turns"][1]["reward"] == 1.5
    assert capture["turns"][1]["done"] is True


def test_write_live_prompt_artifacts(tmp_path):
    captures = [
        {
            "label": "id",
            "env_name": "wofost_gym",
            "metadata": {"env_id": "lnpkw-v0", "validation_set": "id"},
            "system_prompt": "system",
            "turns": [{"turn_index": 0, "source": "reset", "user_prompt": "user"}],
        }
    ]

    json_path, md_path = live_prompt_capture.write_live_prompt_artifacts(
        tmp_path,
        captures,
        title="Prompt Capture",
        description="Generated from live environment reset.",
    )

    assert json.loads(json_path.read_text(encoding="utf-8"))[0]["label"] == "id"
    markdown = md_path.read_text(encoding="utf-8")
    assert "# Prompt Capture" in markdown
    assert "Generated from live environment reset." in markdown
    assert "## id" in markdown
    assert "### System Prompt" in markdown
    assert "### User Prompt: Turn 0 (reset)" in markdown
    assert "```text\nsystem\n```" in markdown
    assert "```text\nuser\n```" in markdown


def test_capture_live_prompts_from_parquet_loops_over_selected_indices(monkeypatch, tmp_path):
    def fake_load_env_configs_from_parquet(dataset_file, repo_root=None):
        return (
            [
                {"env_id": "dummy-0", "scenario_id": "scenario-0"},
                {"env_id": "dummy-1", "scenario_id": "scenario-1"},
            ],
            "dummy_env",
            tmp_path / "val.parquet",
        )

    def fake_create_environment(env_name, env_config):
        return DummyPromptEnv(), object()

    monkeypatch.setattr(
        live_prompt_capture,
        "load_env_configs_from_parquet",
        fake_load_env_configs_from_parquet,
    )
    monkeypatch.setattr(live_prompt_capture, "create_environment", fake_create_environment)

    captures = live_prompt_capture.capture_live_prompts_from_parquet(
        tmp_path / "val.parquet",
        sample_indices=[1],
        label_prefix="cold",
    )

    assert len(captures) == 1
    assert captures[0]["label"] == "cold_1"
    assert captures[0]["env_name"] == "dummy_env"
    assert captures[0]["metadata"]["sample_idx"] == 1
    assert captures[0]["metadata"]["scenario_id"] == "scenario-1"
    assert captures[0]["dataset_file"] == str(tmp_path / "val.parquet")
