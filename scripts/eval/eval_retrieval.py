"""
eval_retrieval.py -- standalone retrieval-quality evaluation.

Runs dense, sparse, and hybrid retrieval against the same isolated corpus
and golden set so the modes can be compared directly.

Run:

    python eval_retrieval.py
"""

import json
import os
import tempfile

import rag_python.config as config

_tmp_dir = tempfile.mkdtemp(prefix="rag_eval_")

config.DB_PATH = os.path.join(_tmp_dir, "eval.db")
config.FAISS_INDEX_PATH = os.path.join(_tmp_dir, "eval_index")
config.DB_BACKEND = "sqlite"
config.NEAR_DUP_BACKEND = "memory"
config.VECTOR_BACKEND = "faiss"

from rag_python.rag.storage.db import init_db
from rag_python.rag.ingestion.extractors import extract_text
from rag_python.rag.ingestion.pipeline import ingest_document
from rag_python.rag.retrieval.retrieval import retrieve
from rag_python.rag.evaluation.evidence_matching import evidence_is_covered
from rag_python.rag.evaluation.metrics import (
    evidence_recall_at_k,
    mean,
    precision_at_k,
    reciprocal_rank,
)


MANUAL_FILES_DIR = "data/manual_test_files"
GOLDEN_SET_PATH = "data/golden_set/queries_v2.json"

MODES = ("dense", "sparse", "hybrid")
TOP_K_VALUES = (1, 3, 5)
TOP_K = max(TOP_K_VALUES)


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


def load_golden_set():
    with open(GOLDEN_SET_PATH, encoding="utf-8") as f:
        return json.load(f)


def evaluate_entry(entry, mode):
    old_mode = config.SEARCH_MODE
    config.SEARCH_MODE = mode

    try:
        results = retrieve(entry["query"], top_k=TOP_K)
    finally:
        config.SEARCH_MODE = old_mode

    evidence = entry.get("evidence", [])

    coverage = [
        [
            evidence_is_covered(
                item["text"],
                result.get("content", ""),
            )
            for item in evidence
        ]
        for result in results
    ]

    relevance = [any(item) for item in coverage]
    metrics = {}

    for k in TOP_K_VALUES:
        covered = [False] * len(evidence)

        for chunk in coverage[:k]:
            covered = [
                old or new
                for old, new in zip(covered, chunk)
            ]

        metrics[f"recall@{k}"] = evidence_recall_at_k(
            covered,
            len(evidence),
        )

        metrics[f"precision@{k}"] = precision_at_k(
            relevance,
            k,
        )

    metrics["rr"] = reciprocal_rank(relevance)

    return {
        "id": entry["id"],
        "answerable": entry.get("answerable", True),
        "metrics": metrics,
    }


def evaluate_mode(golden_set, mode):
    rows = [
        evaluate_entry(entry, mode)
        for entry in golden_set
    ]

    answerable = [
        row for row in rows
        if row["answerable"]
    ]

    aggregate = {}

    for k in TOP_K_VALUES:
        aggregate[f"recall@{k}"] = mean([
            row["metrics"][f"recall@{k}"]
            for row in answerable
        ])

        aggregate[f"precision@{k}"] = mean([
            row["metrics"][f"precision@{k}"]
            for row in answerable
        ])

    aggregate["mrr"] = mean([
        row["metrics"]["rr"]
        for row in answerable
    ])

    return rows, aggregate


def print_mode_results(results):
    print()
    print("RETRIEVAL MODE COMPARISON")
    print("=" * 108)

    print(
        f"{'Mode':<10}"
        f"{'R@1':<10}"
        f"{'R@3':<10}"
        f"{'R@5':<10}"
        f"{'P@1':<10}"
        f"{'P@3':<10}"
        f"{'P@5':<10}"
        f"{'MRR':<10}"
    )

    print("-" * 108)

    for mode, metrics in results.items():
        print(
            f"{mode:<10}"
            f"{metrics['recall@1']:.2%}    "
            f"{metrics['recall@3']:.2%}    "
            f"{metrics['recall@5']:.2%}    "
            f"{metrics['precision@1']:.2%}    "
            f"{metrics['precision@3']:.2%}    "
            f"{metrics['precision@5']:.2%}    "
            f"{metrics['mrr']:.3f}"
        )

    print("=" * 108)


def main():
    print(f"Evaluation environment: {_tmp_dir}")

    init_db()

    print(f"Ingesting {MANUAL_FILES_DIR}/ ...")
    count = ingest_files()
    print(f"Ingested {count} documents.")

    print(f"Evaluating {GOLDEN_SET_PATH} ...")
    golden_set = load_golden_set()

    results = {}

    for mode in MODES:
        print(f"  Running {mode}...")
        _, aggregate = evaluate_mode(
            golden_set,
            mode,
        )
        results[mode] = aggregate

    print_mode_results(results)


if __name__ == "__main__":
    main()