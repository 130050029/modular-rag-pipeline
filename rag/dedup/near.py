"""
near_dedup.py -- document-level NEAR-duplicate detection via MinHash + LSH.

Catches "same content, different wording/formatting/minor edits" cases that
exact hashing (dedup.py) can't, without the O(n^2) cost of comparing every
new document against every existing one.

How it works (recap of what we walked through):
  1. Shingle the document into overlapping word n-grams.
  2. MinHash: compress the shingle set into a small, fixed-size signature
     such that similar documents produce similar signatures.
  3. LSH: bucket documents by bands of their signature, so only documents
     sharing at least one full band even get compared -- turning "compare
     against everyone" into "compare against a tiny candidate set."

Uses the `datasketch` library rather than a hand-rolled implementation --
this is exactly what you'd reach for in real code (pip install datasketch).

LIMITATION (toy-project scale): the LSH index lives in memory only and is
rebuilt from scratch on every server restart (see rebuild() in server.py).
At real scale this would be a persistent, possibly distributed structure.
"""

from datasketch import MinHash, MinHashLSH
from config import MINHASH_NUM_PERM, MINHASH_SHINGLE_SIZE, NEAR_DUP_JACCARD_THRESHOLD

_lsh = MinHashLSH(threshold=NEAR_DUP_JACCARD_THRESHOLD, num_perm=MINHASH_NUM_PERM)


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


def check_near_duplicate(text: str) -> str | None:
    """Returns the doc_id of a near-duplicate already in the LSH index, or None."""
    m = compute_minhash(text)
    matches = _lsh.query(m)
    return matches[0] if matches else None


def register_document(doc_id: str, text: str):
    """Add this document's MinHash signature to the LSH index so future
    documents can be checked against it."""
    m = compute_minhash(text)
    _lsh.insert(doc_id, m)