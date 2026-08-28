
from rag.evaluation.metrics import (
    evidence_recall_at_k,
    mean,
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


def test_evidence_recall_all_evidence_covered():
    assert evidence_recall_at_k(
        [True, True, True],
        3,
    ) == 1.0


def test_evidence_recall_no_evidence_covered():
    assert evidence_recall_at_k(
        [False, False],
        2,
    ) == 0.0


def test_precision_at_k():
    assert precision_at_k(
        [True, False, True, False],
        3,
    ) == 2 / 3


def test_precision_k_larger_than_available_results():
    assert precision_at_k(
        [True, False, True],
        10,
    ) == 2 / 3


def test_precision_empty():
    assert precision_at_k([], 5) == 0.0


def test_precision_zero_k():
    assert precision_at_k(
        [True, False],
        0,
    ) == 0.0


def test_precision_negative_k():
    assert precision_at_k(
        [True, False],
        -1,
    ) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(
        [False, False, True],
    ) == 1 / 3


def test_reciprocal_rank_first_result():
    assert reciprocal_rank(
        [True, False, False],
    ) == 1.0


def test_reciprocal_rank_no_hit():
    assert reciprocal_rank(
        [False, False],
    ) == 0.0


def test_mean():
    assert mean([1.0, 2.0, 3.0]) == 2.0


def test_mean_empty():
    assert mean([]) == 0.0


def test_mean_handles_single_value():
    assert mean([0.75]) == 0.75
