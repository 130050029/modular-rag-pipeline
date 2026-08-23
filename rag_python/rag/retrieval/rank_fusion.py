"""
rank_fusion.py -- Reciprocal Rank Fusion (RRF).

RRF combines independently ranked result lists without comparing their raw
scores. That matters for hybrid retrieval because dense similarity and BM25
scores are on different, backend-specific scales.

For a result at 1-indexed *unique* rank r in a list, its contribution is:

    1 / (k + r)

A result appearing in multiple lists accumulates the contributions from each
list. If a malformed ranked list contains the same chunk more than once, only
its first occurrence is considered and duplicate occurrences do not consume a
rank position. This makes the fusion robust to duplicate candidates emitted by
an upstream retriever.
"""

from collections.abc import Sequence


def reciprocal_rank_fusion(
    *ranked_lists: Sequence[tuple[str, float]],
    k: int = 60,
    top_k: int | None = None,
) -> list[tuple[str, float]]:
    """Fuse ranked result lists using Reciprocal Rank Fusion.

    Parameters
    ----------
    ranked_lists:
        Each list must be ordered best-first and contain ``(chunk_id, score)``
        tuples. The original score is intentionally ignored; only the rank of
        each *unique* chunk is used. Empty lists are valid.
    k:
        Positive smoothing constant from the RRF formula. Larger values make
        rank differences matter less.
    top_k:
        Optional maximum number of fused results. ``None`` returns all unique
        chunk IDs.

    Returns
    -------
    list[tuple[str, float]]
        ``(chunk_id, rrf_score)`` ordered by descending RRF score, with
        ``chunk_id`` as a deterministic tie-breaker.

    Notes
    -----
    RRF operates on rank, not the raw score produced by the underlying
    retriever. Therefore dense similarity and BM25 scores never need to be
    normalized before they are fused.

    Duplicate IDs inside one retriever are ignored completely after their
    first occurrence. In particular, they do not consume a rank position:

        [a, a, b] -> a is rank 1, b is rank 2

    This is useful because duplicate candidates are not additional evidence
    from the same retriever and should not unfairly push later unique results
    down the RRF ranking.
    """
    if k < 1:
        raise ValueError("RRF k must be >= 1")
    if top_k is not None and top_k < 0:
        raise ValueError("top_k must be >= 0 or None")

    scores: dict[str, float] = {}

    for ranked_list in ranked_lists:
        seen_in_list: set[str] = set()
        unique_rank = 0

        for chunk_id, _original_score in ranked_list:
            # A duplicate candidate from the same retriever is not additional
            # evidence. Ignore it and, importantly, do not advance the rank
            # assigned to subsequent unique candidates.
            if chunk_id in seen_in_list:
                continue

            seen_in_list.add(chunk_id)
            unique_rank += 1
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (
                k + unique_rank
            )

    fused = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return fused[:top_k] if top_k is not None else fused
