"""
indexing.py -- wraps FAISS. Supports two index types (config.INDEX_TYPE):

  "flat" -- IndexFlatIP, exact brute-force search. True removal via
            remove_ids() -- a removed vector is genuinely gone.
  "hnsw" -- IndexHNSWFlat, approximate graph-based search (default).
            Much better query scaling at real scale, but FAISS's HNSW does
            NOT support remove_ids() at all (verified directly -- it raises
            "remove_ids not implemented for this type of index"). Soft-delete
            therefore works differently for HNSW: remove() deletes the
            chunk_id from OUR OWN bookkeeping dict, not from FAISS's graph.
            search() already looks up each raw FAISS result through that
            same dict and skips anything missing -- so a "removed" vector
            becomes permanently unreachable through search, even though it
            physically still sits in the HNSW graph forever. This is exactly
            why HNSW deployments need periodic full rebuilds in a way exact
            IndexFlatIP+remove_ids() never did: tombstoned vectors
            accumulate, wasting memory and slightly slowing graph traversal,
            with no way to reclaim that space short of rebuilding.

Because tombstoned HNSW results can appear among a query's raw top-k
results and then get filtered out, search() over-fetches (asks FAISS for
more than top_k) to still have a good chance of returning top_k genuine
results after filtering.

Also supports persistence (save/load_from_disk) -- without this, every
server restart re-runs the embedding model over every indexable chunk in the
entire corpus, which at real scale (millions of chunks) can turn a restart
into a multi-hour operation instead of an instant one. See server.py's
lifespan for how this gets used: try loading first, only rebuild from the
database (the expensive path) if no persisted index exists yet.
"""

import json
import os
import faiss
import numpy as np
from config import EMBEDDING_DIM, INDEX_TYPE, HNSW_M, HNSW_EF_CONSTRUCTION, HNSW_EF_SEARCH

# Extra safety alongside the KMP_DUPLICATE_LIB_OK env var set in server.py/
# conftest.py -- forcing FAISS to a single thread removes the specific
# multi-threaded code path where the OpenMP conflict with torch tends to
# actually crash.
faiss.omp_set_num_threads(1)

# How many extra candidates to request from FAISS beyond top_k, to leave
# room for tombstoned (soft-deleted) HNSW results getting filtered out
# afterward. A heuristic, not exact -- see module docstring.
_OVERFETCH_MULTIPLIER = 3
_OVERFETCH_MIN_EXTRA = 10


class VectorIndex:
    def __init__(self):
        self._index = faiss.IndexIDMap2(build_base_index())
        self._chunk_id_to_int: dict[str, int] = {}
        self._int_to_chunk_id: dict[int, str] = {}
        self._next_id = 0

    def add(self, vectors, chunk_ids: list[str]):
        ids = []
        for cid in chunk_ids:
            int_id = self._next_id
            self._next_id += 1
            self._chunk_id_to_int[cid] = int_id
            self._int_to_chunk_id[int_id] = cid
            ids.append(int_id)
        self._index.add_with_ids(np.array(vectors, dtype="float32"), np.array(ids, dtype="int64"))

    def remove(self, chunk_ids: list[str]):
        """Used when a document is marked stale (soft-delete). Tries a real
        FAISS-level removal first (works for "flat"); falls back to
        removing only from our own bookkeeping dict if the underlying index
        doesn't support it (true for "hnsw" -- see module docstring)."""
        int_ids = [self._chunk_id_to_int.pop(cid) for cid in chunk_ids if cid in self._chunk_id_to_int]
        if int_ids:
            try:
                self._index.remove_ids(np.array(int_ids, dtype="int64"))
            except RuntimeError:
                pass   # index type doesn't support true removal -- tombstone only (below)
        for i in int_ids:
            self._int_to_chunk_id.pop(i, None)

    def search(self, query_vector, top_k: int):
        if self._index.ntotal == 0:
            return []

        # Over-fetch: some raw results may be tombstoned (removed from our
        # mapping but still physically in an HNSW graph) and get filtered
        # out below, so ask for more than top_k to compensate.
        raw_k = min(top_k * _OVERFETCH_MULTIPLIER + _OVERFETCH_MIN_EXTRA, self._index.ntotal)

        scores, ids = self._index.search(np.array(query_vector, dtype="float32"), raw_k)
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue
            chunk_id = self._int_to_chunk_id.get(int(idx))
            if chunk_id:
                results.append((chunk_id, float(score)))
            if len(results) >= top_k:
                break
        return results

    @property
    def size(self):
        # Deliberately NOT self._index.ntotal -- for HNSW that count
        # includes tombstoned (soft-deleted) vectors that FAISS never
        # actually forgot. len() of our own live mapping is the true count
        # of what's actually reachable through search().
        return len(self._chunk_id_to_int)

    def save(self, path_prefix: str):
        """Writes <path_prefix>.faiss (the actual FAISS index) and
        <path_prefix>.meta.json (our chunk_id<->int_id mapping and next_id
        counter -- FAISS itself knows nothing about these, they're purely
        our own bookkeeping)."""
        os.makedirs(os.path.dirname(path_prefix) or ".", exist_ok=True)
        faiss.write_index(self._index, f"{path_prefix}.faiss")
        with open(f"{path_prefix}.meta.json", "w") as f:
            json.dump(
                {
                    "next_id": self._next_id,
                    "int_to_chunk_id": {str(k): v for k, v in self._int_to_chunk_id.items()},
                },
                f,
            )

    def load_from_disk(self, path_prefix: str) -> bool:
        """Mutates THIS instance in place, rather than constructing and
        returning a new one -- every module that already did
        `from rag.storage.indexing import vector_index` holds a reference
        to this exact object, and reassigning a new object elsewhere would
        silently leave those references pointing at the old, empty index
        (the same class of bug hit earlier with monkeypatched module-level
        config values).

        Returns True if a persisted index was found and loaded, False if
        none exists yet -- callers (see server.py's lifespan) should fall
        back to a full rebuild in that case.
        """
        faiss_path = f"{path_prefix}.faiss"
        meta_path = f"{path_prefix}.meta.json"
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            return False

        self._index = faiss.read_index(faiss_path)
        with open(meta_path) as f:
            meta = json.load(f)
        self._next_id = meta["next_id"]
        self._int_to_chunk_id = {int(k): v for k, v in meta["int_to_chunk_id"].items()}
        self._chunk_id_to_int = {v: int(k) for k, v in self._int_to_chunk_id.items()}
        return True


def build_base_index():
    if INDEX_TYPE == "flat":
        return faiss.IndexFlatIP(EMBEDDING_DIM)
    if INDEX_TYPE == "hnsw":
        # CRITICAL: faiss.IndexHNSWFlat defaults to L2 distance if metric
        # isn't specified explicitly -- unlike IndexFlatIP, which is inner
        # product by name/construction. Confirmed directly: without this,
        # HNSW returns L2-scale "scores" (small numbers, e.g. ~0.18 for
        # very similar vectors) while the rest of this codebase assumes
        # cosine-similarity-scale scores (large numbers close to 1.0 for
        # similar vectors, e.g. IndexFlatIP's ~0.90 for the same pair).
        # That mismatch silently broke rag/dedup/semantic.py's
        # `score >= SEMANTIC_DUP_COSINE_THRESHOLD` check: a LARGE L2
        # distance (genuinely DISSIMILAR chunks) could exceed 0.95 and get
        # misread as "very similar", causing unrelated documents to be
        # incorrectly flagged as semantic duplicates of whatever was
        # ingested first and silently excluded from the vector index.
        index = faiss.IndexHNSWFlat(EMBEDDING_DIM, HNSW_M, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
        index.hnsw.efSearch = HNSW_EF_SEARCH
        return index
    raise ValueError(f"Unknown INDEX_TYPE: {INDEX_TYPE}")


def _build_vector_index():
    """Selects the vector index implementation based on config.VECTOR_BACKEND
    -- "faiss" (default, in-process) or "qdrant" (separate service). Both
    implement the identical interface (add/remove/search/size/save/
    load_from_disk), so rag/ingestion/pipeline.py, rag/retrieval.py, and
    rag/dedup/semantic.py never need to know or care which one is active --
    they only ever call methods on whatever `vector_index` ends up being."""
    from config import VECTOR_BACKEND
    if VECTOR_BACKEND == "qdrant":
        from rag.storage.qdrant_index import QdrantVectorIndex
        return QdrantVectorIndex()
    return VectorIndex()


vector_index = _build_vector_index()