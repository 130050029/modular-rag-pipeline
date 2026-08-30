"""Configurable dense, sparse, or hybrid retrieval with optional reranking."""

from rag.embeddings import embed_query
from rag.storage.indexing import vector_index
from rag.storage.db import get_chunks_with_parent_by_ids, keyword_search
from rag.retrieval.rank_fusion import reciprocal_rank_fusion
from rag.retrieval.reranking import rerank
from config import TOP_K

from rag.query.processing import get_query_processor

VALID_SEARCH_MODES = {"dense", "sparse", "hybrid"}

def _get_search_mode() -> str:
    from config import SEARCH_MODE

    mode = SEARCH_MODE.lower()

    if mode not in VALID_SEARCH_MODES:
        raise ValueError(
            f"Unsupported SEARCH_MODE={mode!r}. "
            f"Expected one of: {sorted(VALID_SEARCH_MODES)}"
        )

    return mode


def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    if top_k <= 0:
        return []

    from config import (
        RERANK_ENABLED,
        RERANK_CANDIDATE_K,
        RRF_K,
        DENSE_CANDIDATE_K,
        SPARSE_CANDIDATE_K,
    )

    processed_queries = get_query_processor().process(query)
    
    if not processed_queries:
        return []

    query = processed_queries[0]

    mode = _get_search_mode()
    candidate_k = max(top_k, RERANK_CANDIDATE_K) if RERANK_ENABLED else top_k

    if mode == "dense":
        hits = vector_index.search(embed_query(query), candidate_k)
        score_type = "dense"

    elif mode == "sparse":
        hits = keyword_search(query, candidate_k)
        score_type = "sparse"

    else:
        dense_hits = vector_index.search(
            embed_query(query),
            max(candidate_k, DENSE_CANDIDATE_K),
        )
        sparse_hits = keyword_search(
            query,
            max(candidate_k, SPARSE_CANDIDATE_K),
        )

        hits = reciprocal_rank_fusion(
            dense_hits,
            sparse_hits,
            k=RRF_K,
            top_k=candidate_k,
        )
        score_type = "rrf"

    if not hits:
        return []

    data = get_chunks_with_parent_by_ids(
        [chunk_id for chunk_id, _ in hits]
    )

    results = []

    for rank, (chunk_id, score) in enumerate(hits, start=1):
        chunk = data.get(chunk_id)

        if not chunk:
            continue

        results.append({
            "chunk_id": chunk_id,
            "parent_chunk_id": chunk["parent_chunk_id"],
            "content": chunk["content"],
            "source": chunk["source"],
            "score": score,
            "score_type": score_type,
            "rank": rank,
        })

    if RERANK_ENABLED:
        return rerank(query, results, top_k)

    return results[:top_k]