"""
pipeline.py -- orchestrates the full ingestion pipeline:

    0. GLOBAL exact-content duplicate? (doc_hash, any filename) -> skip entirely, nothing inserted
    1. Has THIS source_uri been ingested before, under a different hash?
         -> NEW VERSION: mark old version's chunks stale, remove their
            vectors from the index, then continue ingesting the new content
    2. Near-duplicate document? (MinHash/LSH)          -> record, skip chunking
    3. Split into parent + small chunks (fixed or semantic strategy)
    4. Per small chunk:
         a. Exact chunk duplicate (content_hash)         -> link, skip embedding
         b. Compute embedding_text (table-aware via tables.py)
         c. Semantic duplicate (embedding similarity)     -> link, skip indexing
         d. Otherwise: store + add vector to index
"""

import uuid

from rag.dedup.exact import hash_text
from rag.dedup.near import check_near_duplicate, register_document
from rag.dedup.semantic import find_semantic_duplicate
from rag.ingestion.chunking import chunk_document
from rag.ingestion.tables import get_embedding_text
from rag.embeddings import embed_texts
from rag.storage.db import (
    insert_document,
    insert_parent_chunk,
    insert_small_chunk,
    get_chunk_by_content_hash,
    get_document_by_content_hash,
    get_latest_document_by_source,
    mark_document_stale,
)
from rag.storage.indexing import vector_index


def ingest_document(filename: str, text: str, source_uri: str = None) -> dict:
    source_uri = source_uri or filename
    doc_hash = hash_text(text)

    # --- 0. GLOBAL exact-content duplicate check (cheapest, filename-independent) ---
    # Runs before anything else: byte-identical content skips entirely, no
    # documents-table insert, no MinHash computation. This is the check that
    # was missing -- without it, identical content under a different filename
    # (e.g. SQuAD's repeated contexts under synthetic per-passage filenames)
    # fell through to the more expensive near-dup path and still inserted a
    # documents row, which doesn't scale well if a large fraction of ingested
    # content is exact repeats.
    global_exact_match = get_document_by_content_hash(doc_hash)
    if global_exact_match:
        return {"status": "skipped_exact_duplicate", "doc_id": global_exact_match}

    # --- 1. Version check against the latest version of THIS source_uri ---
    # By the time we get here, the global check above already ruled out an
    # identical-hash match, so if `existing` exists at all, its hash MUST
    # differ -- this is unconditionally a new version, not a duplicate.
    existing = get_latest_document_by_source(source_uri)
    version = 1
    if existing:
        stale_chunk_ids = mark_document_stale(existing["doc_id"])
        vector_index.remove(stale_chunk_ids)
        version = existing["version"] + 1

    # --- 2. Near-duplicate document-level (MinHash/LSH) ----------------------
    near_dup_doc_id = check_near_duplicate(text)

    doc_id = str(uuid.uuid4())
    insert_document(doc_id, filename, source_uri, doc_hash, version=version, near_dup_of_doc_id=near_dup_doc_id)
    register_document(doc_id, text)

    if near_dup_doc_id:
        return {"status": "skipped_near_duplicate", "doc_id": doc_id, "near_dup_of": near_dup_doc_id, "version": version}

    # --- 3. Chunk into parent + small -----------------------------------------
    parent_groups = chunk_document(text)
    if not parent_groups:
        return {"status": "ingested_no_content", "doc_id": doc_id, "chunks": 0, "version": version}

    total_small_chunks = 0
    duplicate_small_chunks = 0

    for position, group in enumerate(parent_groups):
        parent_chunk_id = str(uuid.uuid4())
        insert_parent_chunk(parent_chunk_id, doc_id, position, group["parent_text"], hash_text(group["parent_text"]))

        small_texts = group["small_texts"]
        if not small_texts:
            continue

        # --- 4a. Exact chunk-level duplicate check (content_hash) ------------
        pending = []
        records = []
        for i, small_text in enumerate(small_texts):
            content_hash = hash_text(small_text)
            exact_dup_chunk_id = get_chunk_by_content_hash(content_hash)
            records.append({"text": small_text, "hash": content_hash, "dup_of": exact_dup_chunk_id})
            if exact_dup_chunk_id is None:
                pending.append(i)

        # --- 4b. Table-aware embedding text, THEN embed --------------------
        embed_inputs = [get_embedding_text(records[i]["text"]) for i in pending]
        vectors = embed_texts(embed_inputs) if embed_inputs else []
        vector_by_index = dict(zip(pending, vectors))
        embed_text_by_index = dict(zip(pending, embed_inputs))

        for i, rec in enumerate(records):
            small_chunk_id = str(uuid.uuid4())
            total_small_chunks += 1

            if rec["dup_of"] is not None:
                insert_small_chunk(small_chunk_id, doc_id, parent_chunk_id, i,
                                    rec["text"], rec["text"], rec["hash"], duplicate_of_chunk_id=rec["dup_of"])
                duplicate_small_chunks += 1
                continue

            vector = vector_by_index[i].reshape(1, -1)
            embedding_text = embed_text_by_index[i]

            # --- 4c. Semantic duplicate check --------------------------------
            semantic_match = find_semantic_duplicate(vector)
            if semantic_match:
                matched_chunk_id, _score = semantic_match
                insert_small_chunk(small_chunk_id, doc_id, parent_chunk_id, i,
                                    rec["text"], embedding_text, rec["hash"], duplicate_of_chunk_id=matched_chunk_id)
                duplicate_small_chunks += 1
                continue

            # --- 4d. Genuinely new: store + index ----------------------------
            insert_small_chunk(small_chunk_id, doc_id, parent_chunk_id, i, rec["text"], embedding_text, rec["hash"])
            vector_index.add(vector, [small_chunk_id])

    return {
        "status": "ingested",
        "doc_id": doc_id,
        "version": version,
        "chunks": total_small_chunks,
        "duplicate_chunks_skipped": duplicate_small_chunks,
    }