import pytest

import config as config
from rag.generation.generation import (
    AnthropicGenerator,
    OllamaGenerator,
    build_prompt,
    generate_answer,
    get_generator,
)


CHUNKS = [
    {
        "source": "table.txt",
        "content": "North Q3 revenue was 128000.",
    },
    {
        "source": "notes.txt",
        "content": "North revenue increased compared with Q2.",
    },
]


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def test_build_prompt_contains_context_sources_and_query():
    prompt = build_prompt(
        "What was North Q3 revenue?",
        CHUNKS,
    )

    assert "Context:" in prompt
    assert "[Source: table.txt]" in prompt
    assert "North Q3 revenue was 128000." in prompt
    assert "[Source: notes.txt]" in prompt
    assert "North revenue increased compared with Q2." in prompt
    assert "Question: What was North Q3 revenue?" in prompt
    assert prompt.endswith("Answer:")


def test_build_prompt_preserves_chunk_order():
    chunks = [
        {"source": "first.txt", "content": "First context."},
        {"source": "second.txt", "content": "Second context."},
    ]

    prompt = build_prompt("Question?", chunks)

    assert prompt.index("First context.") < prompt.index("Second context.")


def test_build_prompt_handles_empty_chunks():
    prompt = build_prompt("What happened?", [])

    assert "Context:" in prompt
    assert "Question: What happened?" in prompt
    assert prompt.endswith("Answer:")


# ---------------------------------------------------------------------------
# Provider selection
# ---------------------------------------------------------------------------


def test_get_generator_defaults_to_configured_provider(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "ollama")

    assert isinstance(get_generator(), OllamaGenerator)


def test_get_generator_supports_ollama():
    assert isinstance(get_generator("ollama"), OllamaGenerator)


def test_get_generator_supports_anthropic():
    assert isinstance(get_generator("anthropic"), AnthropicGenerator)


def test_get_generator_explicit_provider_overrides_config(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "ollama")

    assert isinstance(get_generator("anthropic"), AnthropicGenerator)


def test_get_generator_rejects_unknown_provider():
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        get_generator("unknown")


# ---------------------------------------------------------------------------
# Ollama provider
# ---------------------------------------------------------------------------


def test_ollama_generator_sends_expected_request(monkeypatch):
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

    monkeypatch.setattr(
        "rag.generation.generation.requests.post",
        fake_post,
    )
    monkeypatch.setattr(
        config,
        "OLLAMA_BASE_URL",
        "http://localhost:11434",
    )
    monkeypatch.setattr(
        config,
        "OLLAMA_MODEL",
        "qwen2.5:0.5b",
    )
    monkeypatch.setattr(
        config,
        "LLM_MAX_TOKENS",
        256,
    )
    monkeypatch.setattr(
        config,
        "LLM_TIMEOUT",
        30,
    )

    answer = OllamaGenerator().generate(
        "What was North Q3 revenue?",
        CHUNKS,
    )

    assert answer == "North Q3 revenue was 128000."

    assert captured["url"] == (
        "http://localhost:11434/api/generate"
    )

    payload = captured["kwargs"]["json"]

    assert payload["model"] == "qwen2.5:0.5b"
    assert payload["stream"] is False
    assert payload["options"]["num_predict"] == 256
    assert "What was North Q3 revenue?" in payload["prompt"]

    assert captured["kwargs"]["timeout"] == 30


def test_ollama_generator_propagates_http_error(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            raise RuntimeError("Ollama request failed")

    monkeypatch.setattr(
        "rag.generation.generation.requests.post",
        lambda *args, **kwargs: FakeResponse(),
    )

    with pytest.raises(RuntimeError, match="Ollama request failed"):
        OllamaGenerator().generate(
            "question",
            CHUNKS,
        )


def test_ollama_generator_uses_response_text(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "response": "The answer comes from the retrieved context."
            }

    monkeypatch.setattr(
        "rag.generation.generation.requests.post",
        lambda *args, **kwargs: FakeResponse(),
    )

    answer = OllamaGenerator().generate(
        "question",
        CHUNKS,
    )

    assert answer == "The answer comes from the retrieved context."


# ---------------------------------------------------------------------------
# Anthropic provider
# ---------------------------------------------------------------------------


def test_anthropic_requires_api_key(monkeypatch):
    monkeypatch.setattr(
        config,
        "ANTHROPIC_API_KEY",
        None,
    )

    with pytest.raises(
        RuntimeError,
        match="ANTHROPIC_API_KEY is not set",
    ):
        AnthropicGenerator().generate(
            "question",
            CHUNKS,
        )


def test_anthropic_generator_sends_expected_request(monkeypatch):
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "content": [
                    {
                        "text": "North Q3 revenue was 128000."
                    }
                ]
            }

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured["kwargs"] = kwargs
        return FakeResponse()

    monkeypatch.setattr(
        "rag.generation.generation.requests.post",
        fake_post,
    )
    monkeypatch.setattr(
        config,
        "ANTHROPIC_API_KEY",
        "test-api-key",
    )
    monkeypatch.setattr(
        config,
        "ANTHROPIC_MODEL",
        "test-model",
    )
    monkeypatch.setattr(
        config,
        "LLM_MAX_TOKENS",
        512,
    )
    monkeypatch.setattr(
        config,
        "LLM_TIMEOUT",
        45,
    )

    answer = AnthropicGenerator().generate(
        "What was North Q3 revenue?",
        CHUNKS,
    )

    assert answer == "North Q3 revenue was 128000."

    assert captured["url"] == (
        "https://api.anthropic.com/v1/messages"
    )

    headers = captured["kwargs"]["headers"]
    assert headers["x-api-key"] == "test-api-key"
    assert headers["anthropic-version"] == "2023-06-01"
    assert headers["content-type"] == "application/json"

    payload = captured["kwargs"]["json"]

    assert payload["model"] == "test-model"
    assert payload["max_tokens"] == 512
    assert len(payload["messages"]) == 1
    assert payload["messages"][0]["role"] == "user"
    assert "What was North Q3 revenue?" in payload["messages"][0]["content"]

    assert captured["kwargs"]["timeout"] == 45


def test_anthropic_generator_propagates_http_error(monkeypatch):
    monkeypatch.setattr(
        config,
        "ANTHROPIC_API_KEY",
        "test-api-key",
    )

    class FakeResponse:
        def raise_for_status(self):
            raise RuntimeError("Anthropic request failed")

    monkeypatch.setattr(
        "rag.generation.generation.requests.post",
        lambda *args, **kwargs: FakeResponse(),
    )

    with pytest.raises(RuntimeError, match="Anthropic request failed"):
        AnthropicGenerator().generate(
            "question",
            CHUNKS,
        )


# ---------------------------------------------------------------------------
# Public generation interface
# ---------------------------------------------------------------------------


def test_generate_answer_uses_configured_generator(monkeypatch):
    class FakeGenerator:
        def generate(self, query, chunks):
            assert query == "What was North Q3 revenue?"
            assert chunks == CHUNKS
            return "North Q3 revenue was 128000."

    monkeypatch.setattr(
        "rag.generation.generation.get_generator",
        lambda: FakeGenerator(),
    )

    answer = generate_answer(
        "What was North Q3 revenue?",
        CHUNKS,
    )

    assert answer == "North Q3 revenue was 128000."