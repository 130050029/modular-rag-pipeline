from rag.evaluation.metrics import (
    evidence_recall_at_k,
    precision_at_k,
    reciprocal_rank,
)


def test_evidence_recall_at_k():
    assert evidence_recall_at_k(
        [True, False, True],
        3,
    ) == 2 / 3


def test_evidence_recall_zero_evidence():
    assert evidence_recall_at_k([], 0) == 0.0


def test_precision_at_k():
    assert precision_at_k(
        [True, False, True, False],
        3,
    ) == 2 / 3


def test_precision_empty():
    assert precision_at_k([], 5) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(
        [False, False, True],
    ) == 1 / 3


def test_reciprocal_rank_no_hit():
    assert reciprocal_rank(
        [False, False],
    ) == 0.0
