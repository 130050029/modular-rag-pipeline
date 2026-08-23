from rag.retrieval.rank_fusion import reciprocal_rank_fusion


def test_rrf_uses_rank_not_raw_scores():
    dense = [("a", 0.99), ("b", 0.50)]
    sparse = [("b", -100.0), ("a", -0.01)]

    fused = reciprocal_rank_fusion(dense, sparse, k=60)

    # Both documents appear once in each list, so the document that is first
    # in one list and second in the other gets the same score as its reverse.
    assert fused[0][1] == fused[1][1]


def test_rrf_rewards_consensus():
    dense = [("a", 0.99), ("b", 0.50), ("c", 0.10)]
    sparse = [("b", 1.0), ("a", 0.1), ("c", 0.01)]

    fused = reciprocal_rank_fusion(dense, sparse, k=60)

    assert [chunk_id for chunk_id, _ in fused] == ["a", "b", "c"]


def test_rrf_ignores_duplicate_id_within_one_retriever():
    dense = [("a", 0.99), ("a", 0.98), ("b", 0.50)]
    sparse = [("b", 1.0)]

    fused = reciprocal_rank_fusion(dense, sparse, k=60)

    expected_a = 1 / 61
    expected_b = 1 / 61 + 1 / 62

    assert dict(fused)["a"] == expected_a
    assert dict(fused)["b"] == expected_b
    assert fused[0][0] == "b"


def test_rrf_ties_are_deterministic():
    first = reciprocal_rank_fusion([("b", 1.0), ("a", 0.5)], k=60)
    second = reciprocal_rank_fusion([("b", 1.0), ("a", 0.5)], k=60)

    assert first == second


def test_rrf_top_k():
    fused = reciprocal_rank_fusion(
        [("a", 1.0), ("b", 0.9), ("c", 0.8)],
        k=60,
        top_k=2,
    )

    assert len(fused) == 2


def test_rrf_rejects_invalid_parameters():
    try:
        reciprocal_rank_fusion([], k=0)
        assert False, "expected ValueError"
    except ValueError:
        pass

    try:
        reciprocal_rank_fusion([], top_k=-1)
        assert False, "expected ValueError"
    except ValueError:
        pass
