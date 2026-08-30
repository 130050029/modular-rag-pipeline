import pytest

from rag.query.expansion import build_expansion_prompt, OllamaQueryExpander

def test_build_expansion_prompt_contains_query():
    prompt = build_expansion_prompt("What is the remote work policy?")

    assert "What is the remote work policy?" in prompt
    assert "one query per line" in prompt


def test_ollama_expander_returns_multiple_queries(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "response": (
                    "remote work policy\n"
                    "working from home rules\n"
                    "employee remote work guidelines"
                )
            }

    monkeypatch.setattr(
        "rag.query.expansion.requests.post",
        lambda *args, **kwargs: FakeResponse(),
    )

    result = OllamaQueryExpander().expand(
        "What is the remote work policy?"
    )

    assert result == [
        "remote work policy",
        "working from home rules",
        "employee remote work guidelines",
    ]

def test_ollama_expander_rejects_empty_response(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"response": "   "}

    monkeypatch.setattr(
        "rag.query.expansion.requests.post",
        lambda *args, **kwargs: FakeResponse(),
    )

    with pytest.raises(RuntimeError, match="empty response"):
        OllamaQueryExpander().expand("remote work")