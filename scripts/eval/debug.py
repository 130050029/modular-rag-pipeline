"""
debug_retrieval.py -- inspect generation behavior for selected queries.

Examples:

    python debug_retrieval.py "2025 remote work"

    python debug_retrieval.py --query-ids \
        qv2_017 qv2_018 qv2_019 qv2_024 qv2_025 qv2_026
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = ROOT / "rag_python"
sys.path.insert(0, str(PYTHON_ROOT))

import argparse
import json
import os
import tempfile

import config as config

_tmp_dir = tempfile.mkdtemp(prefix="rag_debug_")

config.DB_PATH = os.path.join(_tmp_dir, "debug.db")
config.FAISS_INDEX_PATH = os.path.join(_tmp_dir, "debug_index")
config.DB_BACKEND = "sqlite"
config.NEAR_DUP_BACKEND = "memory"
config.VECTOR_BACKEND = "faiss"

from rag.storage.db import init_db
from rag.ingestion.extractors import extract_text
from rag.ingestion.pipeline import ingest_document
from rag.retrieval.retrieval import retrieve
from rag.generation.generation import generate_answer


MANUAL_FILES_DIR = "data/manual_test_files"
GOLDEN_SET_PATH = "data/golden_set/queries_v2.json"

TOP_K = 10
MODES = ("dense", "sparse", "hybrid")

MODELS = ("ollama", "nvidia")

GENERATION_K_VALUES = (1, 3, 5)


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

def generate_for_model(
    query,
    context_results,
    model,
):
    old_provider = config.LLM_PROVIDER
    config.LLM_PROVIDER = model

    try:
        return generate_answer(
            query,
            context_results,
        )
    finally:
        config.LLM_PROVIDER = old_provider

def load_queries(ids):
    with open(GOLDEN_SET_PATH, encoding="utf-8") as f:
        entries = json.load(f)

    by_id = {
        entry["id"]: (
            entry["query"],
            entry.get("answerable", True),
        )
        for entry in entries
    }

    missing = [query_id for query_id in ids if query_id not in by_id]

    if missing:
        raise ValueError(
            f"Unknown query IDs: {', '.join(missing)}"
        )

    return [
        (query_id, *by_id[query_id])
        for query_id in ids
    ]


def debug_query(query_id, query, answerable):
    print(f"\n{'#' * 90}")
    print(f"{query_id}: {query}")
    print(f"Answerable: {answerable}")
    print(f"{'#' * 90}")


    for model in MODELS:
        print(f"\n{'=' * 90}")
        print(f"{model.upper()} GENERATION")
        for mode in MODES:
            old_mode = config.SEARCH_MODE
            config.SEARCH_MODE = mode

            try:
                results = retrieve(
                    query,
                    top_k=TOP_K,
                )
            finally:
                config.SEARCH_MODE = old_mode

            print(f"\n{'-' * 90}")
            print(f"{mode.upper()} GENERATION")

            if not results:
                print("  No retrieved context.")
                continue

            for k in GENERATION_K_VALUES:
                context_results = results[:k]

                try:
                    # answer = generate_answer(
                    #     query,
                    #     context_results,
                    # )

                    answer = generate_for_model(
                        query,
                        context_results,
                        model
                    )

                    
                except Exception as exc:
                    print(f"\n  K={k}")
                    print(f"  Generation failed: {exc}")
                    continue

                print(f"\n  K={k}")
                print("  Answer:")

                for line in answer.splitlines():
                    print(f"    {line}")


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument("query", nargs="*")
    group.add_argument("--query-ids", nargs="+")

    args = parser.parse_args()

    if args.query_ids:
        queries = load_queries(args.query_ids)
    else:
        queries = [
            ("<adhoc>", " ".join(args.query), True)
        ]

    print("GENERATION DEBUGGER")
    print(f"Modes: {', '.join(MODES)}")
    print(f"Generation K values: {', '.join(map(str, GENERATION_K_VALUES))}")
    print(f"Isolated environment: {_tmp_dir}")

    init_db()

    print(f"\nIngesting {MANUAL_FILES_DIR}/ ...")
    count = ingest_files()
    print(f"Ingested {count} documents.")

    for query_id, query, answerable in queries:
        debug_query(
            query_id,
            query,
            answerable,
        )


if __name__ == "__main__":
    main()
