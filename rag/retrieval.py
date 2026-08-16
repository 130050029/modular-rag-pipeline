"""
retrieval.py -- embed the query, search (dense vector + optionally sparse
BM25/keyword, fused via Reciprocal Rank Fusion), then fetch each match's
PARENT content to actually hand to the LLM ("small-to-retrieve,
large-to-generate").

Hybrid search (config.HYBRID_SEARCH_ENABLED, on by default): runs dense
(embedding) search and sparse (keyword) search independently, each
contributing config.DENSE_CANDIDATE_K / SPARSE_CANDIDATE_K candidates, then
merges the two ranked lists via rag.rank_fusion.reciprocal_rank_fusion --
see that module's docstring for why raw scores are never compared directly
across the two search types.
"""

from rag.embeddings import embed_query
from rag.storage.indexing import vector_index
from rag.storage.db import get_chunks_with_parent_by_ids, keyword_search
from rag.rank_fusion import reciprocal_rank_fusion
from config import TOP_K


def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    from config import HYBRID_SEARCH_ENABLED, RRF_K, DENSE_CANDIDATE_K, SPARSE_CANDIDATE_K

    query_vector = embed_query(query)

    if HYBRID_SEARCH_ENABLED:
        dense_hits = vector_index.search(query_vector, DENSE_CANDIDATE_K)
        sparse_hits = keyword_search(query, SPARSE_CANDIDATE_K)
        hits = reciprocal_rank_fusion(dense_hits, sparse_hits, k=RRF_K, top_k=top_k)
    else:
        hits = vector_index.search(query_vector, top_k)   # list of (small_chunk_id, score)

    if not hits:
        return []

    chunk_id_to_data = get_chunks_with_parent_by_ids([chunk_id for chunk_id, _ in hits])

    results = []
    for chunk_id, score in hits:
        data = chunk_id_to_data.get(chunk_id)
        if data:
            results.append({"content": data["content"], "source": data["source"], "score": score})
    return results