"""
near.py -- document-level NEAR-duplicate detection via MinHash + LSH.

Two backends, selected via config.NEAR_DUP_BACKEND:
  "memory" (default) -- datasketch's MinHashLSH with its default in-process
                          storage. Fast, zero setup, but lost on restart and
                          not shared across multiple app replicas.
  "redis"             -- the SAME MinHashLSH class, using datasketch's own
                          built-in Redis storage layer
                          (storage_config={"type": "redis", ...}) rather
                          than a hand-rolled banding implementation. This
                          persists across restarts and is shared
                          consistently across every app replica pointed at
                          the same Redis instance -- and reuses datasketch's
                          own optimal band/row parameter search internally,
                          the same one the "memory" backend benefits from,
                          rather than an approximated fixed banding scheme.

IMPORTANT (verified against a real Redis instance, not assumed): the Redis
storage_config's "basename" MUST be a fixed, explicit value -- datasketch
generates a random one if omitted, which means two separate processes would
each silently get their own private, disconnected index, defeating the
entire point of using Redis. config.NEAR_DUP_REDIS_BASENAME below exists
specifically to avoid this.

The public interface (check_near_duplicate, register_document) is
unchanged regardless of backend. The active backend is read fresh from
config on every call (not cached at import time), so tests (and any
runtime config change) can reliably override it.
"""

from datasketch import MinHash, MinHashLSH
from config import MINHASH_NUM_PERM, MINHASH_SHINGLE_SIZE, NEAR_DUP_JACCARD_THRESHOLD


def _shingles(text: str, k: int = MINHASH_SHINGLE_SIZE) -> set[str]:
    words = text.split()
    if len(words) < k:
        return {" ".join(words)}
    return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}


def compute_minhash(text: str) -> MinHash:
    m = MinHash(num_perm=MINHASH_NUM_PERM)
    for shingle in _shingles(text):
        m.update(shingle.encode("utf-8"))
    return m


# ---------------------------------------------------------------------------
# Backend: in-memory (datasketch's default in-process storage)
# ---------------------------------------------------------------------------
class _MemoryBackend:
    def __init__(self):
        self._lsh = MinHashLSH(threshold=NEAR_DUP_JACCARD_THRESHOLD, num_perm=MINHASH_NUM_PERM)

    def check(self, minhash: MinHash):
        matches = self._lsh.query(minhash)
        return matches[0] if matches else None

    def register(self, doc_id: str, minhash: MinHash):
        self._lsh.insert(doc_id, minhash)


_memory_backend = _MemoryBackend()


# ---------------------------------------------------------------------------
# Backend: Redis (datasketch's built-in Redis storage layer)
# ---------------------------------------------------------------------------
class _RedisBackend:
    def __init__(self):
        from config import REDIS_HOST, REDIS_PORT, REDIS_DB, NEAR_DUP_REDIS_BASENAME

        self._lsh = MinHashLSH(
            threshold=NEAR_DUP_JACCARD_THRESHOLD,
            num_perm=MINHASH_NUM_PERM,
            storage_config={
                "type": "redis",
                "basename": NEAR_DUP_REDIS_BASENAME,   # fixed -- see module docstring
                "redis": {"host": REDIS_HOST, "port": REDIS_PORT, "db": REDIS_DB},
            },
        )

    def check(self, minhash: MinHash):
        matches = self._lsh.query(minhash)
        return matches[0] if matches else None

    def register(self, doc_id: str, minhash: MinHash):
        self._lsh.insert(doc_id, minhash)


_redis_backend = None   # lazily constructed -- see _get_redis_backend()


def _get_redis_backend() -> _RedisBackend:
    global _redis_backend
    if _redis_backend is None:
        _redis_backend = _RedisBackend()
    return _redis_backend


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------
def check_near_duplicate(text: str) -> str | None:
    """Returns the doc_id of a near-duplicate already registered, or None."""
    from config import NEAR_DUP_BACKEND

    minhash = compute_minhash(text)
    backend = _get_redis_backend() if NEAR_DUP_BACKEND == "redis" else _memory_backend
    return backend.check(minhash)


def register_document(doc_id: str, text: str):
    """Adds this document's MinHash signature to the active backend's index
    so future documents can be checked against it."""
    from config import NEAR_DUP_BACKEND

    minhash = compute_minhash(text)
    backend = _get_redis_backend() if NEAR_DUP_BACKEND == "redis" else _memory_backend
    backend.register(doc_id, minhash)