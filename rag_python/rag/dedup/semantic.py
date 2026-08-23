"""
semantic_dedup.py -- CHUNK-level semantic duplicate detection via embedding
similarity. Catches "different wording, same meaning" -- the case neither
exact hashing nor MinHash/LSH can catch, since the actual words differ.

Runs AFTER a chunk has already been embedded (we need the vector to compare
similarity), and BEFORE that vector gets added to the index -- if it's a
near-duplicate of something already indexed, we link it for lineage but
skip adding a second, redundant vector.
"""

from config import SEMANTIC_DUP_COSINE_THRESHOLD
from rag.storage.indexing import vector_index


def find_semantic_duplicate(vector) -> tuple[str, float] | None:
    """vector: a single embedding, shape (1, dim). Returns (chunk_id, score)
    of the nearest existing neighbor if it's above threshold, else None."""
    if vector_index.size == 0:
        return None

    hits = vector_index.search(vector, top_k=1)
    if not hits:
        return None

    chunk_id, score = hits[0]
    if score >= SEMANTIC_DUP_COSINE_THRESHOLD:
        return chunk_id, score
    return None