"""
debug_retrieval.py -- inspect Dense, Sparse, and Hybrid retrieval.

Examples:

    python debug_retrieval.py "2025 remote work"

    python debug_retrieval.py --query-ids \
        qv2_017 qv2_018 qv2_019 qv2_024 qv2_025 qv2_026
"""

import argparse
import json
import os
import tempfile

import rag_python.config as config

_tmp_dir = tempfile.mkdtemp(prefix="rag_debug_")

config.DB_PATH = os.path.join(_tmp_dir, "debug.db")
config.FAISS_INDEX_PATH = os.path.join(_tmp_dir, "debug_index")
config.DB_BACKEND = "sqlite"
config.NEAR_DUP_BACKEND = "memory"
config.VECTOR_BACKEND = "faiss"

from rag_python.rag.storage.db import init_db
from rag_python.rag.ingestion.extractors import extract_text
from rag_python.rag.ingestion.pipeline import ingest_document
from rag_python.rag.retrieval.retrieval import retrieve


MANUAL_FILES_DIR = "data/manual_test_files"
GOLDEN_SET_PATH = "data/golden_set/queries_v2.json"

TOP_K = 10
MODES = ("dense", "sparse", "hybrid")


def ingest_files():
    count = 0

    for filename in sorted(os.listdir(MANUAL_FILES_DIR)):
        path = os.path.join(MANUAL_FILES_DIR, filename)
        if not os.path.isfile(path):
            continue

        try:
            with open(path, "rb") as f:
                text = extract_text(filename, f.read())
        except Exception as exc:
            print(f"  {filename}: extraction failed ({exc})")
            continue

        result = ingest_document(filename, text)

        if result["status"] == "ingested":
            count += 1
        else:
            print(f"  {filename}: {result['status']}")

    return count


def load_queries(ids):
    with open(GOLDEN_SET_PATH, encoding="utf-8") as f:
        entries = json.load(f)

    by_id = {entry["id"]: entry["query"] for entry in entries}

    missing = [query_id for query_id in ids if query_id not in by_id]
    if missing:
        raise ValueError(f"Unknown query IDs: {', '.join(missing)}")

    return [(query_id, by_id[query_id]) for query_id in ids]


def debug_query(query_id, query):
    print(f"\n{'#' * 90}")
    print(f"{query_id}: {query}")
    print(f"{'#' * 90}")

    for mode in MODES:
        old_mode = config.SEARCH_MODE
        config.SEARCH_MODE = mode

        try:
            results = retrieve(query, top_k=TOP_K)
        finally:
            config.SEARCH_MODE = old_mode

        print(f"\n{mode.upper()}")

        if not results:
            print("  No results.")
            continue

        for rank, result in enumerate(results, 1):
            source = (
                result.get("source")
                or result.get("filename")
                or result.get("file_name")
                or "<unknown>"
            )
            score = result.get("score")
            print(f"  {rank:>2}. {source:<30} {score=}")


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument("query", nargs="*")
    group.add_argument("--query-ids", nargs="+")

    args = parser.parse_args()

    if args.query_ids:
        queries = load_queries(args.query_ids)
    else:
        queries = [("<adhoc>", " ".join(args.query))]

    print("RETRIEVAL DEBUGGER")
    print(f"Modes: {', '.join(MODES)}")
    print(f"Isolated environment: {_tmp_dir}")

    init_db()

    print(f"\nIngesting {MANUAL_FILES_DIR}/ ...")
    count = ingest_files()
    print(f"Ingested {count} documents.")

    for query_id, query in queries:
        debug_query(query_id, query)


if __name__ == "__main__":
    main()