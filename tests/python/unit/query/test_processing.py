import pytest

from rag.query.processing import (
    DefaultQueryProcessor,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class FakeRouter:
    def __init__(self, complex_query):
        self.complex_query = complex_query
        self.calls = []

    def is_complex(self, query):
        self.calls.append(query)
        return self.complex_query


# ---------------------------------------------------------------------------
# Default / passthrough behavior
# ---------------------------------------------------------------------------


def test_processor_returns_original_query_when_no_strategies():
    router = FakeRouter(complex_query=False)

    processor = DefaultQueryProcessor(
        complexity_router=router,
    )

    assert processor.process("original question") == [
        "original question"
    ]

    assert router.calls == [
        "original question"
    ]


def test_processor_requires_complexity_router():
    with pytest.raises(TypeError):
        DefaultQueryProcessor()


# ---------------------------------------------------------------------------
# Simple query routing
# ---------------------------------------------------------------------------


def test_simple_query_uses_rewriter():
    router = FakeRouter(complex_query=False)

    class FakeRewriter:
        def rewrite(self, query):
            assert query == "What is the remote work policy?"
            return "remote work policy"

    class FakeDecomposer:
        def decompose(self, query):
            raise AssertionError(
                "Decomposer must not be called for a simple query"
            )

    processor = DefaultQueryProcessor(
        complexity_router=router,
        rewriter=FakeRewriter(),
        decomposer=FakeDecomposer(),
    )

    assert processor.process(
        "What is the remote work policy?"
    ) == [
        "remote work policy"
    ]


def test_simple_query_skips_decomposer():
    router = FakeRouter(complex_query=False)

    class FakeRewriter:
        def rewrite(self, query):
            return "rewritten query"

    class FakeDecomposer:
        def decompose(self, query):
            raise AssertionError(
                "Decomposer must not be called for a simple query"
            )

    processor = DefaultQueryProcessor(
        complexity_router=router,
        rewriter=FakeRewriter(),
        decomposer=FakeDecomposer(),
    )

    assert processor.process("simple question") == [
        "rewritten query"
    ]


def test_simple_query_without_rewriter_returns_original_query():
    router = FakeRouter(complex_query=False)

    class FakeDecomposer:
        def decompose(self, query):
            raise AssertionError(
                "Decomposer must not be called for a simple query"
            )

    processor = DefaultQueryProcessor(
        complexity_router=router,
        decomposer=FakeDecomposer(),
    )

    assert processor.process("simple question") == [
        "simple question"
    ]


# ---------------------------------------------------------------------------
# Complex query routing
# ---------------------------------------------------------------------------


def test_complex_query_uses_decomposer():
    router = FakeRouter(complex_query=True)

    class FakeRewriter:
        def rewrite(self, query):
            raise AssertionError(
                "Rewriter must not be called for a complex query"
            )

    class FakeDecomposer:
        def decompose(self, query):
            assert query == "complex question"
            return [
                "sub-question one",
                "sub-question two",
            ]

    processor = DefaultQueryProcessor(
        complexity_router=router,
        rewriter=FakeRewriter(),
        decomposer=FakeDecomposer(),
    )

    assert processor.process("complex question") == [
        "sub-question one",
        "sub-question two",
    ]


def test_complex_query_skips_rewriter():
    router = FakeRouter(complex_query=True)

    class FakeRewriter:
        def rewrite(self, query):
            raise AssertionError(
                "Rewriter must not be called for a complex query"
            )

    class FakeDecomposer:
        def decompose(self, query):
            return [
                "sub-question one",
            ]

    processor = DefaultQueryProcessor(
        complexity_router=router,
        rewriter=FakeRewriter(),
        decomposer=FakeDecomposer(),
    )

    assert processor.process("complex question") == [
        "sub-question one"
    ]


def test_complex_query_without_decomposer_returns_original_query():
    router = FakeRouter(complex_query=True)

    class FakeRewriter:
        def rewrite(self, query):
            raise AssertionError(
                "Rewriter must not be called for a complex query"
            )

    processor = DefaultQueryProcessor(
        complexity_router=router,
        rewriter=FakeRewriter(),
    )

    assert processor.process("complex question") == [
        "complex question"
    ]


# ---------------------------------------------------------------------------
# Rewrite
# ---------------------------------------------------------------------------


def test_processor_propagates_rewriter_failure():
    router = FakeRouter(complex_query=False)

    class FakeRewriter:
        def rewrite(self, query):
            raise RuntimeError("rewriter failed")

    processor = DefaultQueryProcessor(
        complexity_router=router,
        rewriter=FakeRewriter(),
    )

    with pytest.raises(RuntimeError, match="rewriter failed"):
        processor.process("original question")


# ---------------------------------------------------------------------------
# Expansion
# ---------------------------------------------------------------------------


def test_processor_expands_query():
    router = FakeRouter(complex_query=False)

    class FakeExpander:
        def expand(self, query):
            assert query == "original question"
            return [
                "expanded question one",
                "expanded question two",
            ]

    processor = DefaultQueryProcessor(
        complexity_router=router,
        expander=FakeExpander(),
    )

    assert processor.process("original question") == [
        "expanded question one",
        "expanded question two",
    ]


def test_processor_propagates_expander_failure():
    router = FakeRouter(complex_query=False)

    class FakeExpander:
        def expand(self, query):
            raise RuntimeError("expander failed")

    processor = DefaultQueryProcessor(
        complexity_router=router,
        expander=FakeExpander(),
    )

    with pytest.raises(RuntimeError, match="expander failed"):
        processor.process("original question")


# ---------------------------------------------------------------------------
# Rewrite -> Expansion
# ---------------------------------------------------------------------------


def test_simple_query_rewrites_before_expanding():
    router = FakeRouter(complex_query=False)
    calls = []

    class FakeRewriter:
        def rewrite(self, query):
            calls.append(("rewrite", query))
            return "rewritten query"

    class FakeExpander:
        def expand(self, query):
            calls.append(("expand", query))
            return [
                "expanded query one",
                "expanded query two",
            ]

    processor = DefaultQueryProcessor(
        complexity_router=router,
        rewriter=FakeRewriter(),
        expander=FakeExpander(),
    )

    result = processor.process("original question")

    assert result == [
        "expanded query one",
        "expanded query two",
    ]

    assert calls == [
        ("rewrite", "original question"),
        ("expand", "rewritten query"),
    ]


# ---------------------------------------------------------------------------
# Decomposition
# ---------------------------------------------------------------------------


def test_processor_decomposes_query():
    router = FakeRouter(complex_query=True)

    class FakeDecomposer:
        def decompose(self, query):
            assert query == "complex question"
            return [
                "sub-question one",
                "sub-question two",
            ]

    processor = DefaultQueryProcessor(
        complexity_router=router,
        decomposer=FakeDecomposer(),
    )

    assert processor.process("complex question") == [
        "sub-question one",
        "sub-question two",
    ]


def test_processor_propagates_decomposer_failure():
    router = FakeRouter(complex_query=True)

    class FakeDecomposer:
        def decompose(self, query):
            raise RuntimeError("decomposer failed")

    processor = DefaultQueryProcessor(
        complexity_router=router,
        decomposer=FakeDecomposer(),
    )

    with pytest.raises(RuntimeError, match="decomposer failed"):
        processor.process("complex question")


# ---------------------------------------------------------------------------
# Decomposition -> Expansion
# ---------------------------------------------------------------------------


def test_processor_decomposes_before_expanding():
    router = FakeRouter(complex_query=True)
    calls = []

    class FakeDecomposer:
        def decompose(self, query):
            calls.append(("decompose", query))
            return [
                "sub-question one",
                "sub-question two",
            ]

    class FakeExpander:
        def expand(self, query):
            calls.append(("expand", query))
            return [
                f"{query} variant 1",
                f"{query} variant 2",
            ]

    processor = DefaultQueryProcessor(
        complexity_router=router,
        decomposer=FakeDecomposer(),
        expander=FakeExpander(),
    )

    result = processor.process("complex question")

    assert result == [
        "sub-question one variant 1",
        "sub-question one variant 2",
        "sub-question two variant 1",
        "sub-question two variant 2",
    ]

    assert calls == [
        ("decompose", "complex question"),
        ("expand", "sub-question one"),
        ("expand", "sub-question two"),
    ]


def test_processor_expands_each_decomposed_query():
    router = FakeRouter(complex_query=True)
    expanded_inputs = []

    class FakeDecomposer:
        def decompose(self, query):
            return [
                "question A",
                "question B",
                "question C",
            ]

    class FakeExpander:
        def expand(self, query):
            expanded_inputs.append(query)
            return [f"{query} expanded"]

    processor = DefaultQueryProcessor(
        complexity_router=router,
        decomposer=FakeDecomposer(),
        expander=FakeExpander(),
    )

    result = processor.process("complex question")

    assert expanded_inputs == [
        "question A",
        "question B",
        "question C",
    ]

    assert result == [
        "question A expanded",
        "question B expanded",
        "question C expanded",
    ]

# ---------------------------------------------------------------------------
# Routing behavior
# ---------------------------------------------------------------------------


def test_simple_query_routes_to_rewriter():
    router = FakeRouter(complex_query=False)
    calls = []

    class FakeRewriter:
        def rewrite(self, query):
            calls.append(("rewrite", query))
            return "rewritten query"

    class FakeDecomposer:
        def decompose(self, query):
            calls.append(("decompose", query))
            raise AssertionError(
                "Decomposer must not be called for a simple query"
            )

    processor = DefaultQueryProcessor(
        complexity_router=router,
        rewriter=FakeRewriter(),
        decomposer=FakeDecomposer(),
    )

    result = processor.process("original question")

    assert result == [
        "rewritten query"
    ]

    assert calls == [
        ("rewrite", "original question"),
    ]

    assert router.calls == [
        "original question",
    ]


def test_complex_query_routes_to_decomposer():
    router = FakeRouter(complex_query=True)
    calls = []

    class FakeRewriter:
        def rewrite(self, query):
            calls.append(("rewrite", query))
            raise AssertionError(
                "Rewriter must not be called for a complex query"
            )

    class FakeDecomposer:
        def decompose(self, query):
            calls.append(("decompose", query))
            return [
                "sub-question one",
                "sub-question two",
            ]

    processor = DefaultQueryProcessor(
        complexity_router=router,
        rewriter=FakeRewriter(),
        decomposer=FakeDecomposer(),
    )

    result = processor.process("complex question")

    assert result == [
        "sub-question one",
        "sub-question two",
    ]

    assert calls == [
        ("decompose", "complex question"),
    ]

    assert router.calls == [
        "complex question",
    ]


def test_simple_query_routes_rewrite_then_expansion():
    router = FakeRouter(complex_query=False)
    calls = []

    class FakeRewriter:
        def rewrite(self, query):
            calls.append(("rewrite", query))
            return "rewritten query"

    class FakeDecomposer:
        def decompose(self, query):
            calls.append(("decompose", query))
            raise AssertionError(
                "Decomposer must not be called for a simple query"
            )

    class FakeExpander:
        def expand(self, query):
            calls.append(("expand", query))
            return [
                "expanded one",
                "expanded two",
            ]

    processor = DefaultQueryProcessor(
        complexity_router=router,
        rewriter=FakeRewriter(),
        decomposer=FakeDecomposer(),
        expander=FakeExpander(),
    )

    result = processor.process("original question")

    assert result == [
        "expanded one",
        "expanded two",
    ]

    assert calls == [
        ("rewrite", "original question"),
        ("expand", "rewritten query"),
    ]


def test_complex_query_routes_decompose_then_expansion():
    router = FakeRouter(complex_query=True)
    calls = []

    class FakeRewriter:
        def rewrite(self, query):
            calls.append(("rewrite", query))
            raise AssertionError(
                "Rewriter must not be called for a complex query"
            )

    class FakeDecomposer:
        def decompose(self, query):
            calls.append(("decompose", query))
            return [
                "sub-question one",
                "sub-question two",
            ]

    class FakeExpander:
        def expand(self, query):
            calls.append(("expand", query))
            return [
                f"{query} variant",
            ]

    processor = DefaultQueryProcessor(
        complexity_router=router,
        rewriter=FakeRewriter(),
        decomposer=FakeDecomposer(),
        expander=FakeExpander(),
    )

    result = processor.process("complex question")

    assert result == [
        "sub-question one variant",
        "sub-question two variant",
    ]

    assert calls == [
        ("decompose", "complex question"),
        ("expand", "sub-question one"),
        ("expand", "sub-question two"),
    ]


def test_router_is_called_before_any_processing_strategy():
    router = FakeRouter(complex_query=False)
    calls = []

    class FakeRewriter:
        def rewrite(self, query):
            calls.append(("rewrite", query))
            return "rewritten"

    class FakeExpander:
        def expand(self, query):
            calls.append(("expand", query))
            return ["expanded"]

    processor = DefaultQueryProcessor(
        complexity_router=router,
        rewriter=FakeRewriter(),
        expander=FakeExpander(),
    )

    result = processor.process("original")

    assert result == [
        "expanded"
    ]

    assert router.calls == [
        "original"
    ]

    assert calls == [
        ("rewrite", "original"),
        ("expand", "rewritten"),
    ]


def test_router_failure_stops_processing():
    class FakeRouter:
        def is_complex(self, query):
            raise RuntimeError("router failed")

    class FakeRewriter:
        def rewrite(self, query):
            raise AssertionError(
                "Rewriter must not run when routing fails"
            )

    class FakeDecomposer:
        def decompose(self, query):
            raise AssertionError(
                "Decomposer must not run when routing fails"
            )

    class FakeExpander:
        def expand(self, query):
            raise AssertionError(
                "Expander must not run when routing fails"
            )

    processor = DefaultQueryProcessor(
        complexity_router=FakeRouter(),
        rewriter=FakeRewriter(),
        decomposer=FakeDecomposer(),
        expander=FakeExpander(),
    )

    with pytest.raises(RuntimeError, match="router failed"):
        processor.process("some question")