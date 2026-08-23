"""Retrieval metrics used by the golden-set evaluator."""


def evidence_recall_at_k(
    evidence_covered: list[bool],
    total_evidence: int,
) -> float:
    """Fraction of required evidence items covered by top-k retrieval."""
    if total_evidence == 0:
        return 0.0
    return sum(evidence_covered) / total_evidence


def precision_at_k(relevance: list[bool], k: int) -> float:
    """Fraction of the first k retrieved chunks that are relevant."""
    if k <= 0:
        return 0.0

    top_k = relevance[:k]
    if not top_k:
        return 0.0

    return sum(top_k) / len(top_k)


def reciprocal_rank(relevance: list[bool]) -> float:
    """Reciprocal rank of the first relevant retrieved chunk."""
    for rank, is_relevant in enumerate(relevance, start=1):
        if is_relevant:
            return 1.0 / rank
    return 0.0


def mean(values: list[float]) -> float:
    """Arithmetic mean with a safe empty-list result."""
    return sum(values) / len(values) if values else 0.0