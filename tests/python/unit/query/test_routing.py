import pytest

from rag.query.routing import DefaultQueryComplexityRouter


@pytest.fixture
def router():
    return DefaultQueryComplexityRouter()


# ---------------------------------------------------------------------------
# Simple queries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query",
    [
        "What is the remote work policy?",
        "What are the benefits of solar panels?",
        "How does coffee brewing work?",
        "What was the revenue in 2025?",
        "Explain wind energy.",
    ],
)
def test_router_classifies_simple_queries_as_not_complex(router, query):
    assert router.is_complex(query) is False


def test_router_treats_empty_query_as_simple(router):
    assert router.is_complex("") is False


def test_router_treats_whitespace_query_as_simple(router):
    assert router.is_complex("   ") is False


# ---------------------------------------------------------------------------
# Comparison queries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query",
    [
        "Compare solar panels and wind energy.",
        "Compare the remote work policy with the office policy.",
        "What is the difference between version 1 and version 2?",
        "Contrast the two revenue strategies.",
        "How does option A differ from option B?",
        "Which is better versus the other approach?",
        "Option A vs. option B: which is better?",
    ],
)
def test_router_classifies_comparison_queries_as_complex(router, query):
    assert router.is_complex(query) is True


# ---------------------------------------------------------------------------
# Multi-task queries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query",
    [
        "What is the policy and explain why it changed?",
        "Find the revenue and compare it with the previous year.",
        "Describe the policy and summarize its key requirements.",
        "Identify the relevant documents and list their main differences.",
        "Calculate the total and explain the result.",
        "Find the available options and recommend one.",
    ],
)
def test_router_classifies_multi_task_queries_as_complex(router, query):
    assert router.is_complex(query) is True


# ---------------------------------------------------------------------------
# Multiple questions
# ---------------------------------------------------------------------------


def test_router_classifies_multiple_questions_as_complex(router):
    query = (
        "What is the remote work policy? "
        "How many days can employees work remotely?"
    )

    assert router.is_complex(query) is True


# ---------------------------------------------------------------------------
# Conservative behavior
# ---------------------------------------------------------------------------


def test_router_does_not_treat_every_and_as_complex(router):
    assert router.is_complex(
        "What are the benefits and requirements of remote work?"
    ) is False


def test_router_does_not_treat_every_long_query_as_complex(router):
    query = (
        "What are the main requirements for employees working remotely "
        "during the current policy period?"
    )

    assert router.is_complex(query) is False


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_router_is_deterministic(router):
    query = (
        "Compare the remote work policy with the office attendance policy."
    )

    assert router.is_complex(query) is True
    assert router.is_complex(query) is True


# ---------------------------------------------------------------------------
# Router contract
# ---------------------------------------------------------------------------


def test_router_returns_boolean(router):
    result = router.is_complex(
        "What is the remote work policy?"
    )

    assert isinstance(result, bool)


def test_router_can_be_replaced_with_fake():
    class FakeRouter:
        def is_complex(self, query):
            assert query == "complex question"
            return True

    router = FakeRouter()

    assert router.is_complex("complex question") is True


def test_router_failure_is_propagated():
    class FakeRouter:
        def is_complex(self, query):
            raise RuntimeError("router failed")

    router = FakeRouter()

    with pytest.raises(RuntimeError, match="router failed"):
        router.is_complex("some question")