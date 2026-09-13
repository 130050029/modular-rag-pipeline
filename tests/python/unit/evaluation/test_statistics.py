import pytest

from rag.evaluation.statistics import (
    binary_confidence_interval,
    paired_binary_test,
)


def test_binary_confidence_interval_returns_observed_estimate():
    result = binary_confidence_interval(
        [1.0, 1.0, 1.0, 0.0, 0.0],
    )

    assert result.estimate == 0.6
    assert 0.0 <= result.lower <= result.estimate
    assert result.estimate <= result.upper <= 1.0
    assert result.confidence_level == 0.95


def test_binary_confidence_interval_all_successes():
    result = binary_confidence_interval(
        [1.0, 1.0, 1.0, 1.0],
    )

    assert result.estimate == 1.0
    assert result.lower < 1.0
    assert result.upper == 1.0


def test_binary_confidence_interval_all_failures():
    result = binary_confidence_interval(
        [0.0, 0.0, 0.0, 0.0],
    )

    assert result.estimate == 0.0
    assert result.lower == 0.0
    assert result.upper > 0.0


def test_binary_confidence_interval_rejects_empty_input():
    with pytest.raises(
        ValueError,
        match="empty list",
    ):
        binary_confidence_interval([])


def test_binary_confidence_interval_rejects_non_binary_values():
    with pytest.raises(
        ValueError,
        match="only 0.0 and 1.0",
    ):
        binary_confidence_interval(
            [1.0, 0.5, 0.0],
        )


def test_binary_confidence_interval_rejects_invalid_confidence_level():
    with pytest.raises(
        ValueError,
        match="between 0.0 and 1.0",
    ):
        binary_confidence_interval(
            [1.0, 0.0],
            confidence_level=1.5,
        )


def test_paired_binary_test_counts_discordant_pairs():
    first = [
        1.0,
        1.0,
        0.0,
        0.0,
        1.0,
    ]

    second = [
        1.0,
        0.0,
        1.0,
        0.0,
        0.0,
    ]

    result = paired_binary_test(
        first,
        second,
    )

    assert result.first_only == 2
    assert result.second_only == 1
    assert 0.0 <= result.p_value <= 1.0


def test_paired_binary_test_identical_results_have_no_difference():
    values = [
        1.0,
        0.0,
        1.0,
        1.0,
        0.0,
    ]

    result = paired_binary_test(
        values,
        values,
    )

    assert result.first_only == 0
    assert result.second_only == 0
    assert result.p_value == 1.0


def test_paired_binary_test_rejects_different_lengths():
    with pytest.raises(
        ValueError,
        match="equal length",
    ):
        paired_binary_test(
            [1.0, 0.0],
            [1.0],
        )


def test_paired_binary_test_rejects_empty_input():
    with pytest.raises(
        ValueError,
        match="non-empty",
    ):
        paired_binary_test(
            [],
            [],
        )


def test_paired_binary_test_rejects_non_binary_values():
    with pytest.raises(
        ValueError,
        match="only 0.0 and 1.0",
    ):
        paired_binary_test(
            [1.0, 0.5],
            [1.0, 0.0],
        )