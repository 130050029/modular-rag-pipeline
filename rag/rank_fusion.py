"""
rank_fusion.py -- Reciprocal Rank Fusion (RRF): merges multiple independently
ranked result lists (e.g. dense/embedding search + sparse/BM25 search) into
one combined ranking.

Deliberately uses ONLY rank position, never the raw scores -- BM25 scores
and cosine similarity scores live on completely incomparable scales (the
same class of mistake fixed earlier this session for FAISS's L2-vs-inner-
product metric bug: adding two differently-scaled numbers together and
pretending the result means something). Rank position is comparable across
any two ranking systems; raw scores are not.

Formula: for each list a chunk appears in, it contributes 1/(k + rank) to
its total score (rank is 1-indexed: the top result contributes the most).
A chunk that ranks decently in BOTH lists usually beats a chunk that's #1
in only one -- rewarding consensus over a single unconfirmed strong signal.
"""


def reciprocal_rank_fusion(*ranked_lists, k: int = 60, top_k: int | None = None):
    """Each ranked_list is a list of (chunk_id, score) tuples, already
    sorted best-first (score itself is ignored -- only position matters).
    Returns a list of (chunk_id, rrf_score) sorted by rrf_score descending."""
    scores: dict[str, float] = {}

    for ranked_list in ranked_lists:
        for rank, (chunk_id, _original_score) in enumerate(ranked_list, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)

    fused = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    if top_k is not None:
        fused = fused[:top_k]
    return fused