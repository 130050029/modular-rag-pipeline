"""
retrieval.py -- embed the query, search the vector index (which only ever
contains SMALL, non-duplicate chunks), then fetch each match's PARENT
content to actually hand to the LLM ("small-to-retrieve, large-to-generate").
"""

from rag.embeddings import embed_query
from rag.storage.indexing import vector_index
from rag.storage.db import get_chunks_with_parent_by_ids
from config import TOP_K


def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    query_vector = embed_query(query)
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