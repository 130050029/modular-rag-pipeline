import pytest

from rag.retrieval.rank_fusion import reciprocal_rank_fusion


def test_rrf_uses_rank_not_raw_scores():
    dense = [("a", 0.99), ("b", 0.50)]
    sparse = [("b", -100.0), ("a", -0.01)]

    fused = reciprocal_rank_fusion(dense, sparse, k=60)

    # a: rank 1 + rank 2
    # b: rank 2 + rank 1
    # Therefore they must tie regardless of their original scores.
    assert dict(fused)["a"] == dict(fused)["b"]


def test_rrf_rewards_consensus():
    dense = [
        ("a", 0.99),
        ("b", 0.50),
        ("c", 0.10),
    ]

    sparse = [
        ("b", 1.0),
        ("c", 0.5),
        ("a", 0.1),
    ]

    fused = reciprocal_rank_fusion(
        dense,
        sparse,
        k=60,
    )

    # b is ranked highly by both retrievers and therefore wins.
    assert fused[0][0] == "b"


def test_rrf_ignores_duplicate_id_within_one_retriever():
    dense = [
        ("a", 0.99),
        ("a", 0.98),
        ("b", 0.50),
    ]

    sparse = [
        ("b", 1.0),
    ]

    fused = reciprocal_rank_fusion(
        dense,
        sparse,
        k=60,
    )

    expected_a = 1 / 61
    expected_b = 1 / 61 + 1 / 62

    assert dict(fused)["a"] == expected_a
    assert dict(fused)["b"] == expected_b
    assert fused[0][0] == "b"


def test_rrf_duplicate_does_not_consume_rank():
    fused = reciprocal_rank_fusion(
        [
            ("a", 1.0),
            ("a", 0.9),
            ("b", 0.8),
        ],
        k=60,
    )

    assert dict(fused)["a"] == 1 / 61
    assert dict(fused)["b"] == 1 / 62


def test_rrf_ties_use_chunk_id_as_deterministic_tiebreaker():
    fused = reciprocal_rank_fusion(
        [("b", 1.0), ("a", 0.5)],
        [("a", 1.0), ("b", 0.5)],
        k=60,
    )

    assert dict(fused)["a"] == dict(fused)["b"]
    assert [chunk_id for chunk_id, _ in fused] == ["a", "b"]


def test_rrf_empty_lists_are_valid():
    assert reciprocal_rank_fusion([], []) == []


def test_rrf_top_k_limits_results():
    fused = reciprocal_rank_fusion(
        [
            ("a", 1.0),
            ("b", 0.9),
            ("c", 0.8),
        ],
        top_k=2,
    )

    assert len(fused) == 2
    assert {chunk_id for chunk_id, _ in fused} <= {"a", "b", "c"}


def test_rrf_top_k_zero_returns_empty():
    fused = reciprocal_rank_fusion(
        [("a", 1.0)],
        top_k=0,
    )

    assert fused == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"k": 0},
        {"k": -1},
        {"top_k": -1},
    ],
)
def test_rrf_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([], **kwargs)