import config as config
from rag.generation.generation import (
    AnthropicGenerator,
    OllamaGenerator,
    build_prompt,
    get_generator,
)


CHUNKS = [
    {
        "source": "table.txt",
        "content": "North Q3 revenue was 128000.",
    }
]


def test_build_prompt_contains_query_and_context():
    prompt = build_prompt("What was North Q3 revenue?", CHUNKS)

    assert "North Q3 revenue was 128000." in prompt
    assert "What was North Q3 revenue?" in prompt


def test_get_generator_defaults_to_ollama(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "ollama")

    assert isinstance(get_generator(), OllamaGenerator)


def test_get_generator_supports_anthropic(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "anthropic")

    assert isinstance(get_generator(), AnthropicGenerator)


def test_get_generator_rejects_unknown_provider():
    try:
        get_generator("unknown")
    except ValueError as exc:
        assert "Unknown LLM provider" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_ollama_generator(monkeypatch):
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"response": "North Q3 revenue was 128000."}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured["kwargs"] = kwargs
        return FakeResponse()

    monkeypatch.setattr("rag.generation.generation.requests.post", fake_post)
    monkeypatch.setattr(config, "OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.setattr(config, "OLLAMA_MODEL", "qwen2.5:0.5b")

    answer = OllamaGenerator().generate(
        "What was North Q3 revenue?",
        CHUNKS,
    )

    assert answer == "North Q3 revenue was 128000."
    assert captured["url"] == "http://localhost:11434/api/generate"
    assert captured["kwargs"]["json"]["model"] == "qwen2.5:0.5b"
    assert captured["kwargs"]["json"]["stream"] is False


def test_anthropic_requires_api_key(monkeypatch):
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", None)

    try:
        AnthropicGenerator().generate("question", CHUNKS)
    except RuntimeError as exc:
        assert "ANTHROPIC_API_KEY" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError")