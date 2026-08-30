import config
import pytest

from rag.query.processing import (
    DefaultQueryProcessor,
    get_query_processor
)

from rag.query.rewriting import build_rewrite_prompt, OllamaQueryRewriter

def test_build_rewrite_prompt_contains_query():
    prompt = build_rewrite_prompt("What is the remote work policy?")

    assert "What is the remote work policy?" in prompt
    assert "standalone search query" in prompt
    assert "Return only the rewritten query" in prompt


def test_ollama_query_rewriter(monkeypatch):
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "response": "remote work policy"
            }

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured["kwargs"] = kwargs
        return FakeResponse()

    monkeypatch.setattr(
        "rag.query.processing.requests.post",
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

    rewriter = OllamaQueryRewriter()

    result = rewriter.rewrite(
        "What is the remote work policy?"
    )

    assert result == "remote work policy"
    assert captured["url"] == "http://localhost:11434/api/generate"
    assert captured["kwargs"]["json"]["model"] == "qwen2.5:0.5b"
    assert captured["kwargs"]["json"]["stream"] is False


def test_ollama_query_rewriter_rejects_empty_response(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"response": "   "}

    monkeypatch.setattr(
        "rag.query.processing.requests.post",
        lambda *args, **kwargs: FakeResponse(),
    )

    with pytest.raises(
        RuntimeError,
        match="empty response",
    ):
        OllamaQueryRewriter().rewrite("some query")