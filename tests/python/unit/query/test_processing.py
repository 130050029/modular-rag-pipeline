import config
import pytest

from rag.query.processing import (
    DefaultQueryProcessor,
    get_query_processor,
)


def test_processor_returns_original_query_when_no_strategies():
    processor = DefaultQueryProcessor()

    assert processor.process("original question") == [
        "original question"
    ]


def test_processor_uses_rewriter():
    class FakeRewriter:
        def rewrite(self, query):
            assert query == "What is the remote work policy?"
            return "remote work policy"

    processor = DefaultQueryProcessor(
        rewriter=FakeRewriter(),
    )

    assert processor.process(
        "What is the remote work policy?"
    ) == ["remote work policy"]


def test_processor_rewrites_before_expanding():
    calls = []

    class FakeRewriter:
        def rewrite(self, query):
            calls.append(("rewrite", query))
            return "rewritten query"

    class FakeExpander:
        def expand(self, query):
            calls.append(("expand", query))
            return ["expanded query 1", "expanded query 2"]

    processor = DefaultQueryProcessor(
        rewriter=FakeRewriter(),
        expander=FakeExpander(),
    )

    result = processor.process("original question")

    assert result == [
        "expanded query 1",
        "expanded query 2",
    ]

    assert calls == [
        ("rewrite", "original question"),
        ("expand", "rewritten query"),
    ]


def test_query_rewriting_is_disabled_by_default(
    monkeypatch,
):
    monkeypatch.setattr(
        config,
        "QUERY_REWRITE_ENABLED",
        False,
    )

    processor = get_query_processor()

    assert processor.process(
        "What is the capital of France?"
    ) == [
        "What is the capital of France?"
    ]


def test_query_rewriting_uses_configured_rewriter(
    monkeypatch,
):
    class FakeRewriter:
        def rewrite(self, query):
            assert query == "original user question"
            return "rewritten retrieval query"

    monkeypatch.setattr(
        "rag.query.processing.OllamaQueryRewriter",
        FakeRewriter,
    )

    monkeypatch.setattr(
        config,
        "QUERY_REWRITE_ENABLED",
        True,
    )

    processor = get_query_processor()

    assert processor.process(
        "original user question"
    ) == [
        "rewritten retrieval query"
    ]


def test_query_rewriting_failure_propagates(
    monkeypatch,
):
    class FakeRewriter:
        def rewrite(self, query):
            raise RuntimeError("rewriter failed")

    monkeypatch.setattr(
        "rag.query.processing.OllamaQueryRewriter",
        FakeRewriter,
    )

    monkeypatch.setattr(
        config,
        "QUERY_REWRITE_ENABLED",
        True,
    )

    processor = get_query_processor()

    with pytest.raises(RuntimeError, match="rewriter failed"):
        processor.process("original user question")