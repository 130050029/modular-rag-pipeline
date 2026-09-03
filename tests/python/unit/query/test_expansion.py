import pytest
import config

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

def test_expander_removes_blank_lines_and_duplicates(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "response": (
                    "remote work policy\n"
                    "\n"
                    "working from home rules\n"
                    "remote work policy\n"
                    " employee remote work guidelines "
                )
            }

    monkeypatch.setattr(
        "rag.query.expansion.requests.post",
        lambda *args, **kwargs: FakeResponse(),
    )

    result = OllamaQueryExpander().expand("remote work")

    assert result == [
        "remote work policy",
        "working from home rules",
        "employee remote work guidelines",
    ]

def test_expander_limits_number_of_queries(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "response": (
                    "query one\n"
                    "query two\n"
                    "query three\n"
                    "query four\n"
                    "query five"
                )
            }

    monkeypatch.setattr(
        "rag.query.expansion.requests.post",
        lambda *args, **kwargs: FakeResponse(),
    )

    monkeypatch.setattr(
        config,
        "QUERY_EXPANSION_MAX_QUERIES",
        3,
    )

    result = OllamaQueryExpander().expand("original query")

    assert result == [
        "query one",
        "query two",
        "query three",
    ]

def test_expander_deduplicates_before_applying_limit(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "response": (
                    "query one\n"
                    "query one\n"
                    "query two\n"
                    "query three\n"
                )
            }

    monkeypatch.setattr(
        "rag.query.expansion.requests.post",
        lambda *args, **kwargs: FakeResponse(),
    )

    monkeypatch.setattr(
        config,
        "QUERY_EXPANSION_MAX_QUERIES",
        3,
    )

    result = OllamaQueryExpander().expand("original query")

    assert result == [
        "query one",
        "query two",
        "query three",
    ]