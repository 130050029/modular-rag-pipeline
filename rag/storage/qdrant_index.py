"""
qdrant_index.py -- Qdrant-backed vector index, implementing the SAME public
interface as VectorIndex (rag/storage/indexing.py's FAISS wrapper): add(),
remove(), search(), size, save(), load_from_disk() -- so it can be swapped
in as a drop-in replacement via config.VECTOR_BACKEND, with zero changes
needed in rag/ingestion/pipeline.py, rag/retrieval.py, or
rag/dedup/semantic.py (they only ever call these same methods on whatever
`vector_index` object rag.storage.indexing exposes -- see the factory at
the bottom of that file).

Meaningfully SIMPLER than the FAISS wrapper in two ways, both verified
directly against a real (embedded-mode) Qdrant client before relying on
either:
  1. No int_id <-> chunk_id mapping needed at all -- Qdrant accepts our
     UUID chunk_ids directly as point IDs.
  2. TRUE removal, not tombstoning -- Qdrant's delete() genuinely removes
     the point; reclaiming that space is handled internally by Qdrant
     itself, not something we need to reason about here (unlike FAISS's
     HNSW, which required the tombstone workaround in indexing.py).

save()/load_from_disk() are no-ops here (load_from_disk always returns
True) -- Qdrant persists itself continuously as data is written (as long
as its storage directory is durable -- see docker-compose.yml's bind
mount), so there's nothing for OUR code to serialize. This is also why
server.py's lifespan needs no Qdrant-specific branching at all:
load_from_disk() returning True immediately skips the expensive
rebuild-from-database path, exactly as intended, with the exact same
lifespan code that handles the FAISS backend.

CONSTRAINT VERIFIED DIRECTLY (not assumed): Qdrant point IDs must be either
unsigned integers or genuine UUID strings -- an arbitrary string like
"chunk-a" raises "Point id ... is not a valid UUID". Our real pipeline
always uses actual str(uuid.uuid4()) values as chunk_ids (see
rag/ingestion/pipeline.py), so this is satisfied naturally in practice --
just don't hand-write a non-UUID chunk_id anywhere, including in tests.
"""

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


class QdrantVectorIndex:
    def __init__(self):
        # Read fresh here, NOT at module top -- this is the exact bug class
        # already fixed for DB_BACKEND, NEAR_DUP_BACKEND, PDF_EXTRACTION_METHOD,
        # and FAISS_INDEX_PATH earlier this session, caught here too: a
        # module-level `from config import QDRANT_COLLECTION` would bind the
        # value that existed the FIRST time this module was ever imported,
        # permanently, so a test's monkeypatch.setattr("config.QDRANT_COLLECTION", ...)
        # would silently have no effect -- QdrantVectorIndex would keep
        # reading and writing the real configured collection instead of an
        # isolated test one. Confirmed this happened for real: a test run
        # wrote into the actual production "rag_chunks" collection.
        from config import QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION, EMBEDDING_DIM

        self._client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self._collection = QDRANT_COLLECTION
        existing = [c.name for c in self._client.get_collections().collections]
        if self._collection not in existing:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            )

    def add(self, vectors, chunk_ids: list[str]):
        vectors_list = np.asarray(vectors, dtype="float32").tolist()
        points = [
            PointStruct(id=chunk_id, vector=vector)
            for chunk_id, vector in zip(chunk_ids, vectors_list)
        ]
        self._client.upsert(collection_name=self._collection, points=points)

    def remove(self, chunk_ids: list[str]):
        if chunk_ids:
            self._client.delete(collection_name=self._collection, points_selector=chunk_ids)

    def search(self, query_vector, top_k: int):
        # Callers pass a (1, dim) array (matching the shape FAISS expects) --
        # flatten to a plain list for Qdrant's query API.
        vector = np.asarray(query_vector, dtype="float32").reshape(-1).tolist()
        results = self._client.query_points(collection_name=self._collection, query=vector, limit=top_k)
        return [(str(point.id), float(point.score)) for point in results.points]

    @property
    def size(self):
        return self._client.count(collection_name=self._collection).count

    def save(self, path_prefix: str):
        pass   # no-op -- Qdrant persists continuously on its own

    def load_from_disk(self, path_prefix: str) -> bool:
        return True   # no-op success -- signals "nothing to rebuild" to lifespan