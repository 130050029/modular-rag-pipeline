"""
dedup.py -- exact-match hashing utility only.

The actual "is this an exact duplicate" DECISION now lives in ingestion.py's
version-check logic (comparing doc_hash against get_latest_document_by_source()),
since exact-duplicate detection and versioning turned out to be the same
check: same source_uri + same hash = exact duplicate; same source_uri +
different hash = new version. This file just provides the hashing function
both dedup.py's old logic and ingestion.py rely on.

See near_dedup.py for MinHash/LSH near-duplicates, and semantic_dedup.py for
embedding-based semantic duplicates.
"""

import hashlib


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()