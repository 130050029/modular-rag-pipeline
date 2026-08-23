"""
pipeline.py -- orchestrates the full ingestion pipeline:

    0. GLOBAL exact-content duplicate?
         -> skip entirely

    1. Has THIS source_uri been ingested before under a different hash?
         -> NEW VERSION:
            mark old version stale
            remove old live vectors
            continue ingesting new content

    2. Near-duplicate document?
         -> record relationship and skip chunking

    3. Split into parent + small chunks

    4. Per small chunk:
         a. Exact chunk duplicate?
                -> store lineage
                -> do not independently index

         b. Compute embedding_text

         c. Semantic duplicate?
                -> STORE the chunk
                -> make it searchable by BM25
                -> DO NOT add it to vector index

         d. Otherwise:
                -> store
                -> add vector to vector index

The important distinction is that semantic deduplication is now
vector-index-specific rather than a global "this chunk does not exist"
decision.

A semantically similar chunk can still contain lexical information that is
important to sparse retrieval, especially for document versions containing
different dates, numbers, limits, names, or exact terminology.
"""

import uuid

from rag.dedup.exact import hash_text
from rag.dedup.near import (
    check_near_duplicate,
    register_document,
)
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


def _save_index_if_mutated(mutated: bool):
    if not mutated:
        return

    from config import FAISS_INDEX_PATH

    vector_index.save(FAISS_INDEX_PATH)


def ingest_document(
    filename: str,
    text: str,
    source_uri: str = None,
) -> dict:
    source_uri = source_uri or filename

    doc_hash = hash_text(text)

    index_mutated = False

    # -----------------------------------------------------------------------
    # 0. GLOBAL exact-content duplicate
    # -----------------------------------------------------------------------
    global_exact_match = get_document_by_content_hash(
        doc_hash
    )

    if global_exact_match:
        return {
            "status": "skipped_exact_duplicate",
            "doc_id": global_exact_match,
        }

    # -----------------------------------------------------------------------
    # 1. Version check
    # -----------------------------------------------------------------------
    existing = get_latest_document_by_source(
        source_uri
    )

    version = 1

    if existing:
        stale_chunk_ids = mark_document_stale(
            existing["doc_id"]
        )

        vector_index.remove(
            stale_chunk_ids
        )

        index_mutated = True

        version = existing["version"] + 1

    # -----------------------------------------------------------------------
    # 2. Near-duplicate document check
    # -----------------------------------------------------------------------
    near_dup_doc_id = check_near_duplicate(
        text
    )

    doc_id = str(uuid.uuid4())

    insert_document(
        doc_id,
        filename,
        source_uri,
        doc_hash,
        version=version,
        near_dup_of_doc_id=near_dup_doc_id,
    )

    register_document(
        doc_id,
        text,
    )

    if near_dup_doc_id:
        _save_index_if_mutated(
            index_mutated
        )

        return {
            "status": "skipped_near_duplicate",
            "doc_id": doc_id,
            "near_dup_of": near_dup_doc_id,
            "version": version,
        }

    # -----------------------------------------------------------------------
    # 3. Chunk into parent + small chunks
    # -----------------------------------------------------------------------
    parent_groups = chunk_document(
        text
    )

    if not parent_groups:
        _save_index_if_mutated(
            index_mutated
        )

        return {
            "status": "ingested_no_content",
            "doc_id": doc_id,
            "chunks": 0,
            "version": version,
        }

    total_small_chunks = 0
    duplicate_small_chunks = 0

    for position, group in enumerate(
        parent_groups
    ):
        parent_chunk_id = str(
            uuid.uuid4()
        )

        insert_parent_chunk(
            parent_chunk_id,
            doc_id,
            position,
            group["parent_text"],
            hash_text(
                group["parent_text"]
            ),
        )

        small_texts = group["small_texts"]

        if not small_texts:
            continue

        # -------------------------------------------------------------------
        # 4a. Exact chunk-level duplicate check
        # -------------------------------------------------------------------
        pending = []
        records = []

        for i, small_text in enumerate(
            small_texts
        ):
            content_hash = hash_text(
                small_text
            )

            exact_dup_chunk_id = (
                get_chunk_by_content_hash(
                    content_hash
                )
            )

            records.append(
                {
                    "text": small_text,
                    "hash": content_hash,
                    "dup_of": exact_dup_chunk_id,
                }
            )

            if exact_dup_chunk_id is None:
                pending.append(i)

        # -------------------------------------------------------------------
        # 4b. Compute embedding inputs
        # -------------------------------------------------------------------
        embed_inputs = [
            get_embedding_text(
                records[i]["text"]
            )
            for i in pending
        ]

        vectors = (
            embed_texts(embed_inputs)
            if embed_inputs
            else []
        )

        vector_by_index = dict(
            zip(
                pending,
                vectors,
            )
        )

        embed_text_by_index = dict(
            zip(
                pending,
                embed_inputs,
            )
        )

        # -------------------------------------------------------------------
        # 4c/4d. Store and optionally vector-index each small chunk
        # -------------------------------------------------------------------
        for i, rec in enumerate(
            records
        ):
            small_chunk_id = str(
                uuid.uuid4()
            )

            total_small_chunks += 1

            # ---------------------------------------------------------------
            # Exact duplicate:
            #
            # Store lineage, but do not add another searchable copy.
            # ---------------------------------------------------------------
            if rec["dup_of"] is not None:
                insert_small_chunk(
                    small_chunk_id,
                    doc_id,
                    parent_chunk_id,
                    i,
                    rec["text"],
                    rec["text"],
                    rec["hash"],
                    duplicate_of_chunk_id=rec[
                        "dup_of"
                    ],
                    duplicate_reason="exact",
                )

                duplicate_small_chunks += 1

                continue

            vector = vector_by_index[i].reshape(
                1,
                -1,
            )

            embedding_text = (
                embed_text_by_index[i]
            )

            # ---------------------------------------------------------------
            # Semantic duplicate:
            #
            # IMPORTANT:
            #   - store the chunk
            #   - make it available to sparse retrieval
            #   - DO NOT add its vector to FAISS/Qdrant
            #
            # This prevents semantic deduplication from hiding lexical
            # information such as version numbers, dates, or exact values.
            # ---------------------------------------------------------------
            semantic_match = (
                find_semantic_duplicate(
                    vector
                )
            )

            if semantic_match:
                matched_chunk_id, _score = (
                    semantic_match
                )

                insert_small_chunk(
                    small_chunk_id,
                    doc_id,
                    parent_chunk_id,
                    i,
                    rec["text"],
                    embedding_text,
                    rec["hash"],
                    duplicate_of_chunk_id=(
                        matched_chunk_id
                    ),
                    duplicate_reason="semantic",
                )

                duplicate_small_chunks += 1

                # Deliberately DO NOT:
                #
                #     vector_index.add(...)
                #
                # The chunk remains in SQLite/Postgres and in BM25/FTS.
                continue

            # ---------------------------------------------------------------
            # Genuinely new chunk:
            # store + vector index
            # ---------------------------------------------------------------
            insert_small_chunk(
                small_chunk_id,
                doc_id,
                parent_chunk_id,
                i,
                rec["text"],
                embedding_text,
                rec["hash"],
            )

            vector_index.add(
                vector,
                [small_chunk_id],
            )

            index_mutated = True

    _save_index_if_mutated(
        index_mutated
    )

    return {
        "status": "ingested",
        "doc_id": doc_id,
        "version": version,
        "chunks": total_small_chunks,
        "duplicate_chunks_skipped": duplicate_small_chunks,
    }