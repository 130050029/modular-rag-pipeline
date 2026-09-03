import config
import pytest

from rag.query.decomposition import (
    OllamaQueryDecomposer,
    build_decomposition_prompt,
)


def test_build_decomposition_prompt_contains_query():
    prompt = build_decomposition_prompt(
        "What are the remote work rules and who is eligible?"
    )

    assert "What are the remote work rules and who is eligible?" in prompt
    assert "one question per line" in prompt
    assert "independent questions" in prompt


def test_decomposer_parses_one_question_per_line(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "response": (
                    "What are the remote work rules?\n"
                    "Who is eligible for remote work?"
                )
            }

    monkeypatch.setattr(
        "rag.query.decomposition.requests.post",
        lambda *args, **kwargs: FakeResponse(),
    )

    result = OllamaQueryDecomposer().decompose(
        "What are the remote work rules and who is eligible?"
    )

    assert result == [
        "What are the remote work rules?",
        "Who is eligible for remote work?",
    ]


def test_decomposer_strips_blank_lines_and_whitespace(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "response": (
                    "\n"
                    "  What are the remote work rules?  \n"
                    "\n"
                    " Who is eligible for remote work? \n"
                    "\n"
                )
            }

    monkeypatch.setattr(
        "rag.query.decomposition.requests.post",
        lambda *args, **kwargs: FakeResponse(),
    )

    result = OllamaQueryDecomposer().decompose("remote work")

    assert result == [
        "What are the remote work rules?",
        "Who is eligible for remote work?",
    ]


def test_decomposer_limits_number_of_queries(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "response": (
                    "question one\n"
                    "question two\n"
                    "question three\n"
                    "question four\n"
                    "question five"
                )
            }

    monkeypatch.setattr(
        "rag.query.decomposition.requests.post",
        lambda *args, **kwargs: FakeResponse(),
    )

    monkeypatch.setattr(
        config,
        "QUERY_DECOMPOSITION_MAX_QUERIES",
        3,
    )

    result = OllamaQueryDecomposer().decompose("complex question")

    assert result == [
        "question one",
        "question two",
        "question three",
    ]


def test_decomposer_rejects_empty_response(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "response": " \n\n "
            }

    monkeypatch.setattr(
        "rag.query.decomposition.requests.post",
        lambda *args, **kwargs: FakeResponse(),
    )

    with pytest.raises(
        RuntimeError,
        match="Query decomposer returned an empty response",
    ):
        OllamaQueryDecomposer().decompose("complex question")


def test_decomposer_propagates_http_failure(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            raise RuntimeError("ollama unavailable")

    monkeypatch.setattr(
        "rag.query.decomposition.requests.post",
        lambda *args, **kwargs: FakeResponse(),
    )

    with pytest.raises(
        RuntimeError,
        match="ollama unavailable",
    ):
        OllamaQueryDecomposer().decompose("complex question")


def test_decomposer_uses_configured_ollama_settings(monkeypatch):
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "response": "first question\nsecond question"
            }

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured["kwargs"] = kwargs
        return FakeResponse()

    monkeypatch.setattr(
        "rag.query.decomposition.requests.post",
        fake_post,
    )

    monkeypatch.setattr(
        config,
        "OLLAMA_BASE_URL",
        "http://test-ollama:11434",
    )
    monkeypatch.setattr(
        config,
        "OLLAMA_MODEL",
        "test-model",
    )
    monkeypatch.setattr(
        config,
        "LLM_MAX_TOKENS",
        128,
    )
    monkeypatch.setattr(
        config,
        "LLM_TIMEOUT",
        7,
    )

    OllamaQueryDecomposer().decompose("complex question")

    assert captured["url"] == (
        "http://test-ollama:11434/api/generate"
    )

    assert captured["kwargs"]["json"]["model"] == "test-model"
    assert captured["kwargs"]["json"]["stream"] is False
    assert captured["kwargs"]["json"]["options"] == {
        "num_predict": 128,
    }
    assert captured["kwargs"]["timeout"] == 7