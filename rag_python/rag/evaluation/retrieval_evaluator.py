"""Evaluation orchestration for golden-set evaluation."""

from dataclasses import dataclass

from rag.evaluation.evidence_matching import evidence_is_covered
from rag.evaluation.metrics import (
    evidence_recall_at_k,
    mean,
    precision_at_k,
    reciprocal_rank,
)


@dataclass(frozen=True)
class EvaluationResult:
    """Evaluation result for one golden-set case."""

    case_id: str
    answerable: bool
    metrics: dict[str, float]


class RetrievalEvaluator:
    """Evaluate retrieved chunks against golden-set entries.

    This evaluator does not perform retrieval. It receives retrieval
    results from the caller and evaluates their evidence coverage.

    The current implementation is intentionally deterministic and
    focuses only on retrieval quality. Generation-related evaluation
    will be added separately in later Phase-D slices.
    """

    def __init__(self, top_k_values: tuple[int, ...] = (1, 3, 5)):
        self.top_k_values = top_k_values

    def evaluate(
        self,
        entry: dict,
        results: list[dict],
    ) -> EvaluationResult:
        """Evaluate one golden-set entry against retrieved results."""

        evidence = entry.get("evidence", [])

        coverage = [
            [
                evidence_is_covered(
                    item["text"],
                    result.get("content", ""),
                )
                for item in evidence
            ]
            for result in results
        ]

        relevance = [any(item) for item in coverage]

        metrics = {}

        for k in self.top_k_values:
            covered = [False] * len(evidence)

            for chunk in coverage[:k]:
                covered = [
                    old or new
                    for old, new in zip(covered, chunk)
                ]

            metrics[f"recall@{k}"] = evidence_recall_at_k(
                covered,
                len(evidence),
            )

            metrics[f"precision@{k}"] = precision_at_k(
                relevance,
                k,
            )

        metrics["rr"] = reciprocal_rank(relevance)

        return EvaluationResult(
            case_id=entry["id"],
            answerable=entry.get("answerable", True),
            metrics=metrics,
        )

    def aggregate(
        self,
        results: list[EvaluationResult],
    ) -> dict[str, float]:
        """Aggregate retrieval metrics across answerable cases."""

        answerable = [
            result
            for result in results
            if result.answerable
        ]

        aggregate = {}

        for k in self.top_k_values:
            aggregate[f"recall@{k}"] = mean([
                result.metrics[f"recall@{k}"]
                for result in answerable
            ])

            aggregate[f"precision@{k}"] = mean([
                result.metrics[f"precision@{k}"]
                for result in answerable
            ])

        aggregate["mrr"] = mean([
            result.metrics["rr"]
            for result in answerable
        ])

        return aggregate
