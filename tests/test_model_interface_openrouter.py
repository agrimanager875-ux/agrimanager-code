from pathlib import Path

from agrimanager.model_interface.model_factory import create_model


class _FakeMessage:
    content = "ok"
    reasoning = "because"


class _FakeChoice:
    message = _FakeMessage()


class _FakeResponse:
    choices = [_FakeChoice()]


class _FakeCompletions:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return _FakeResponse()


class _FakeChat:
    def __init__(self):
        self.completions = _FakeCompletions()


class _FakeClient:
    def __init__(self):
        self.chat = _FakeChat()


def test_openrouter_factory_routes_chat_completion_params(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    config_path = tmp_path / "openrouter.yaml"
    config_path.write_text(
        "\n".join(
            [
                'model_provider: "openrouter"',
                'model_name: "moonshotai/kimi-k2.6"',
                "num_workers: 1",
                'reasoning_effort: "low"',
                "provider:",
                '  sort: "latency"',
            ]
        ),
        encoding="utf-8",
    )

    model = create_model(config_path)
    fake_client = _FakeClient()
    model.client = fake_client

    messages = [[{"role": "user", "content": "2+2?"}]]
    responses = model.generate(
        messages,
        temperature=0.2,
        max_tokens=12,
        return_metadata=True,
        extra_body={"seed": 7},
    )

    assert responses == [{"content": "ok", "reasoning": "because"}]

    request = fake_client.chat.completions.kwargs
    assert request["model"] == "moonshotai/kimi-k2.6"
    assert request["messages"] == messages[0]
    assert request["temperature"] == 0.2
    assert request["top_p"] == 1.0
    assert request["max_tokens"] == 12
    assert request["extra_body"]["seed"] == 7
    assert request["extra_body"]["provider"] == {"sort": "latency"}
    assert request["extra_body"]["reasoning"] == {"effort": "low"}
