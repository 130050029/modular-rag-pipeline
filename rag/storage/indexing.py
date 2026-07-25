"""
indexing.py -- wraps FAISS. Upgraded to support REMOVAL, which soft-delete
requires: when a document is marked stale, its vectors must actually leave
the index, or a re-ingested/updated doc's old content keeps getting retrieved
forever alongside the new version.

FAISS's plain IndexFlatIP doesn't support removal or arbitrary IDs -- it only
knows positional order. IndexIDMap2 wraps a base index and lets us assign our
own integer IDs, which can later be removed with remove_ids(). Since our
chunk_ids are UUID strings (not ints), we keep a chunk_id <-> int_id mapping
here.

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
from config import EMBEDDING_DIM, INDEX_TYPE

# Extra safety alongside the KMP_DUPLICATE_LIB_OK env var set in server.py/
# conftest.py -- forcing FAISS to a single thread removes the specific
# multi-threaded code path where the OpenMP conflict with torch tends to
# actually crash.
faiss.omp_set_num_threads(1)


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
        """Used when a document is marked stale (soft-delete) -- actually
        removes its vectors from the searchable index."""
        int_ids = [self._chunk_id_to_int.pop(cid) for cid in chunk_ids if cid in self._chunk_id_to_int]
        if int_ids:
            self._index.remove_ids(np.array(int_ids, dtype="int64"))
        for i in int_ids:
            self._int_to_chunk_id.pop(i, None)

    def search(self, query_vector, top_k: int):
        if self._index.ntotal == 0:
            return []
        scores, ids = self._index.search(np.array(query_vector, dtype="float32"), min(top_k, self._index.ntotal))
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue
            chunk_id = self._int_to_chunk_id.get(int(idx))
            if chunk_id:
                results.append((chunk_id, float(score)))
        return results

    @property
    def size(self):
        return self._index.ntotal

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
    raise ValueError(f"Unknown INDEX_TYPE: {INDEX_TYPE}")


vector_index = VectorIndex()