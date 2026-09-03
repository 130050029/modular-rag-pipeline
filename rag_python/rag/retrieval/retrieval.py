"""Configurable dense, sparse, or hybrid retrieval with optional reranking."""

from config import TOP_K
from rag.embeddings import embed_query
from rag.query.processing import get_query_processor
from rag.retrieval.rank_fusion import (
    multi_query_fusion,
    reciprocal_rank_fusion,
)
from rag.retrieval.reranking import rerank
from rag.storage.db import (
    get_chunks_with_parent_by_ids,
    keyword_search,
)
from rag.storage.indexing import vector_index


VALID_SEARCH_MODES = {"dense", "sparse", "hybrid"}


def _get_search_mode() -> str:
    """Return the configured retrieval mode, validating it first."""
    from config import SEARCH_MODE

    mode = SEARCH_MODE.lower()

    if mode not in VALID_SEARCH_MODES:
        raise ValueError(
            f"Unsupported SEARCH_MODE={mode!r}. "
            f"Expected one of: {sorted(VALID_SEARCH_MODES)}"
        )

    return mode


def _retrieve_single_query(
    query: str,
    top_k: int,
    *,
    rerank_enabled: bool,
    rerank_candidate_k: int,
    rrf_k: int,
    dense_candidate_k: int,
    sparse_candidate_k: int,
) -> list[dict]:
    """Retrieve results for one already-processed query.

    Query processing is intentionally handled by retrieve(). This function
    performs only the actual dense, sparse, or hybrid retrieval pass.
    """
    mode = _get_search_mode()

    candidate_k = (
        max(top_k, rerank_candidate_k)
        if rerank_enabled
        else top_k
    )

    if mode == "dense":
        hits = vector_index.search(
            embed_query(query),
            candidate_k,
        )
        score_type = "dense"

    elif mode == "sparse":
        hits = keyword_search(
            query,
            candidate_k,
        )
        score_type = "sparse"

    else:
        dense_hits = vector_index.search(
            embed_query(query),
            max(candidate_k, dense_candidate_k),
        )

        sparse_hits = keyword_search(
            query,
            max(candidate_k, sparse_candidate_k),
        )

        hits = reciprocal_rank_fusion(
            dense_hits,
            sparse_hits,
            k=rrf_k,
            top_k=candidate_k,
        )
        score_type = "rrf"

    if not hits:
        return []

    chunk_ids = [chunk_id for chunk_id, _ in hits]
    chunks = get_chunks_with_parent_by_ids(chunk_ids)

    results = []

    for rank, (chunk_id, score) in enumerate(hits, start=1):
        chunk = chunks.get(chunk_id)

        if not chunk:
            continue

        results.append(
            {
                "chunk_id": chunk_id,
                "parent_chunk_id": chunk["parent_chunk_id"],
                "content": chunk["content"],
                "source": chunk["source"],
                "score": score,
                "score_type": score_type,
                "rank": rank,
            }
        )

    if rerank_enabled:
        return rerank(query, results, top_k)

    return results[:top_k]


def _retrieve_processed_queries(
    processed_queries: list[str],
    top_k: int,
    *,
    rerank_enabled: bool,
    rerank_candidate_k: int,
    rrf_k: int,
    dense_candidate_k: int,
    sparse_candidate_k: int,
) -> list[dict]:
    """Retrieve and fuse results for multiple processed queries."""
    query_results = [
        _retrieve_single_query(
            query,
            top_k,
            rerank_enabled=rerank_enabled,
            rerank_candidate_k=rerank_candidate_k,
            rrf_k=rrf_k,
            dense_candidate_k=dense_candidate_k,
            sparse_candidate_k=sparse_candidate_k,
        )
        for query in processed_queries
    ]

    if not any(query_results):
        return []

    fused = multi_query_fusion(
        [
            [
                (result["chunk_id"], result["score"])
                for result in results
            ]
            for results in query_results
        ],
        k=rrf_k,
        top_k=top_k,
    )

    if not fused:
        return []

    chunks = get_chunks_with_parent_by_ids([chunk_id for chunk_id, _ in fused])

    final_results = []

    for rank, (chunk_id, score) in enumerate(fused, start=1):
        chunk = chunks.get(chunk_id)

        if not chunk:
            continue

        final_results.append(
            {
                "chunk_id": chunk_id,
                "parent_chunk_id": chunk["parent_chunk_id"],
                "content": chunk["content"],
                "source": chunk["source"],
                "score": float(score),
                "score_type": "multi_query",
                "rank": rank,
            }
        )

    return final_results[:top_k]


def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    """Process a user query, retrieve candidates, and return top-K results."""
    if top_k <= 0:
        return []

    from config import (
        DENSE_CANDIDATE_K,
        RERANK_CANDIDATE_K,
        RERANK_ENABLED,
        RRF_K,
        SPARSE_CANDIDATE_K,
    )

    processed_queries = get_query_processor().process(query)

    if not processed_queries:
        return []

    # Preserve the existing single-query retrieval contract.
    #
    # dense  -> score_type="dense"
    # sparse -> score_type="sparse"
    # hybrid -> score_type="rrf"
    if len(processed_queries) == 1:
        return _retrieve_single_query(
            processed_queries[0],
            top_k,
            rerank_enabled=RERANK_ENABLED,
            rerank_candidate_k=RERANK_CANDIDATE_K,
            rrf_k=RRF_K,
            dense_candidate_k=DENSE_CANDIDATE_K,
            sparse_candidate_k=SPARSE_CANDIDATE_K,
        )

    # Multiple processed queries are retrieved independently and then
    # combined by the dedicated multi-query fusion strategy.
    return _retrieve_processed_queries(
        processed_queries,
        top_k,
        rerank_enabled=RERANK_ENABLED,
        rerank_candidate_k=RERANK_CANDIDATE_K,
        rrf_k=RRF_K,
        dense_candidate_k=DENSE_CANDIDATE_K,
        sparse_candidate_k=SPARSE_CANDIDATE_K,
    )