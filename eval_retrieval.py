"""
eval_retrieval.py -- standalone retrieval-quality evaluation script.

Deliberately NOT part of the pytest suite or CI. Retrieval quality
(Recall@k, MRR) is a MEASUREMENT on a scale, not a pass/fail correctness
check the way pytest tests are -- wiring this into CI would mean blocking
every push over small, expected fluctuations from a chunking tweak or
embedding model change, which aren't "bugs" to fail a build over. This is
meant to be run deliberately, by a human, whenever you want to check
whether a pipeline change helped or hurt retrieval quality -- an offline
admin activity, not an automated gate.

Ingests every file in data/manual_test_files/ into an ISOLATED, temporary
database and vector index -- never touches your real data/rag.db or
persisted FAISS index -- so results are reproducible and don't depend on
whatever else you've separately ingested via /seed or /upload. All backend
selections are forced to their simplest defaults (sqlite/memory/faiss)
regardless of what's set in your shell environment, for the same reason.

Run:
    python eval_retrieval.py
"""

import json
import os
import tempfile

import config

# Force isolated, simple backends BEFORE importing anything that reads
# these values at construction time -- rag.storage.indexing's vector_index
# singleton in particular is built once, at import time, so these
# overrides MUST happen before any `from rag....` import below.
_tmp_dir = tempfile.mkdtemp(prefix="rag_eval_")
config.DB_PATH = os.path.join(_tmp_dir, "eval.db")
config.FAISS_INDEX_PATH = os.path.join(_tmp_dir, "eval_index")
config.DB_BACKEND = "sqlite"
config.NEAR_DUP_BACKEND = "memory"
config.VECTOR_BACKEND = "faiss"

from rag.storage.db import init_db
from rag.ingestion.extractors import extract_text
from rag.ingestion.pipeline import ingest_document
from rag.retrieval import retrieve

MANUAL_FILES_DIR = "data/manual_test_files"
GOLDEN_SET_PATH = "data/golden_set/queries.json"
TOP_K = 5


def ingest_manual_test_files() -> int:
    count = 0
    for filename in sorted(os.listdir(MANUAL_FILES_DIR)):
        path = os.path.join(MANUAL_FILES_DIR, filename)
        if not os.path.isfile(path):
            continue
        with open(path, "rb") as f:
            raw = f.read()
        try:
            text = extract_text(filename, raw)
        except Exception as e:
            print(f"  Skipping {filename}: extraction failed ({e})")
            continue
        result = ingest_document(filename, text)
        if result["status"] == "ingested":
            count += 1
        else:
            print(f"  {filename}: {result['status']}")
    return count


def evaluate():
    with open(GOLDEN_SET_PATH) as f:
        golden_set = json.load(f)

    hits = 0
    reciprocal_ranks = []
    rows = []

    for entry in golden_set:
        query = entry["query"]
        expected_source = entry["expected_source"]

        results = retrieve(query, top_k=TOP_K)
        sources = [r["source"] for r in results]

        rank = None
        for i, source in enumerate(sources, start=1):
            if source == expected_source:
                rank = i
                break

        found = rank is not None
        if found:
            hits += 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

        rows.append({
            "query": query,
            "expected_source": expected_source,
            "found": found,
            "rank": rank,
            "top_sources": sources,
        })

    recall_at_k = hits / len(golden_set) if golden_set else 0.0
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
    return rows, recall_at_k, mrr


def print_report(rows, recall_at_k, mrr):
    print()
    print("=" * 78)
    print(f"{'Query':<45}{'Found?':<8}{'Rank':<6}")
    print("-" * 78)
    for row in rows:
        query_display = (row["query"][:42] + "...") if len(row["query"]) > 45 else row["query"]
        found_display = "yes" if row["found"] else "NO"
        rank_display = str(row["rank"]) if row["rank"] else "-"
        print(f"{query_display:<45}{found_display:<8}{rank_display:<6}")
        if not row["found"]:
            print(f"    expected: {row['expected_source']}, got: {row['top_sources']}")
    print("-" * 78)
    print(f"Recall@{TOP_K}: {recall_at_k:.2%}   MRR: {mrr:.3f}")
    print("=" * 78)
    print()
    print(f"(Isolated eval environment was: {_tmp_dir} -- safe to delete)")


def main():
    print(f"Isolated evaluation environment: {_tmp_dir}")
    init_db()

    print(f"Ingesting files from {MANUAL_FILES_DIR}/ ...")
    count = ingest_manual_test_files()
    print(f"Ingested {count} documents.")

    print("Running golden set evaluation ...")
    rows, recall_at_k, mrr = evaluate()
    print_report(rows, recall_at_k, mrr)


if __name__ == "__main__":
    main()