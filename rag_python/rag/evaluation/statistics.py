"""Statistical utilities for RAG evaluation results."""

from dataclasses import dataclass

from scipy.stats import binomtest


@dataclass(frozen=True)
class ConfidenceInterval:
    """Confidence interval around a binary metric."""

    estimate: float
    lower: float
    upper: float
    confidence_level: float


@dataclass(frozen=True)
class PairedBinaryTestResult:
    """Result of a paired binary comparison."""

    first_only: int
    second_only: int
    p_value: float


def binary_confidence_interval(
    values: list[float],
    confidence_level: float = 0.95,
) -> ConfidenceInterval:
    """
    Calculate an exact binomial confidence interval for a binary metric.

    `values` must contain only 0.0 and 1.0 values.

    The estimate is the observed proportion. The interval uses the exact
    Clopper-Pearson binomial confidence interval.
    """

    if not values:
        raise ValueError(
            "Cannot calculate a confidence interval from an empty list."
        )

    if not 0.0 < confidence_level < 1.0:
        raise ValueError(
            "confidence_level must be between 0.0 and 1.0."
        )

    if any(value not in (0.0, 1.0) for value in values):
        raise ValueError(
            "binary_confidence_interval requires values containing "
            "only 0.0 and 1.0."
        )

    successes = int(sum(values))
    total = len(values)

    result = binomtest(
        successes,
        total,
    )

    interval = result.proportion_ci(
        confidence_level=confidence_level,
        method="exact",
    )

    return ConfidenceInterval(
        estimate=successes / total,
        lower=interval.low,
        upper=interval.high,
        confidence_level=confidence_level,
    )


def paired_binary_test(
    first: list[float],
    second: list[float],
) -> PairedBinaryTestResult:
    """
    Compare two binary metrics evaluated on the same cases.

    This implements McNemar's exact test using only discordant pairs.

    `first` and `second` must:
      - contain the same number of observations
      - contain only 0.0 and 1.0 values

    `first_only` counts cases where the first system succeeds and the
    second fails.

    `second_only` counts cases where the second system succeeds and the
    first fails.
    """

    if not first or not second:
        raise ValueError(
            "paired_binary_test requires two non-empty lists."
        )

    if len(first) != len(second):
        raise ValueError(
            "paired_binary_test requires lists of equal length."
        )

    if any(value not in (0.0, 1.0) for value in first):
        raise ValueError(
            "First input must contain only 0.0 and 1.0."
        )

    if any(value not in (0.0, 1.0) for value in second):
        raise ValueError(
            "Second input must contain only 0.0 and 1.0."
        )

    first_only = sum(
        first_value == 1.0 and second_value == 0.0
        for first_value, second_value in zip(first, second)
    )

    second_only = sum(
        first_value == 0.0 and second_value == 1.0
        for first_value, second_value in zip(first, second)
    )

    discordant = first_only + second_only

    if discordant == 0:
        p_value = 1.0
    else:
        # Under the null hypothesis, each discordant case is equally
        # likely to favor either system.
        #
        # Exact two-sided McNemar test.
        smaller = min(first_only, second_only)

        p_value = 2.0 * sum(
            _binomial_probability(
                discordant,
                k,
            )
            for k in range(smaller + 1)
        )

        p_value = min(p_value, 1.0)

    return PairedBinaryTestResult(
        first_only=first_only,
        second_only=second_only,
        p_value=p_value,
    )


def _binomial_probability(
    n: int,
    k: int,
) -> float:
    """Return P(X=k) for X ~ Binomial(n, 0.5)."""

    from math import comb

    return comb(n, k) / (2 ** n)