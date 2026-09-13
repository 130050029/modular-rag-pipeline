"""
evaluate.py -- standalone RAG quality evaluation.

Runs dense, sparse, and hybrid retrieval against the same isolated corpus
and golden set, then evaluates retrieval and generation quality at each
retrieval context size.

Run:

    python evaluate.py
"""

import os

# Prevent Hugging Face tokenizers from enabling parallelism before a fork.
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = ROOT / "rag_python"
sys.path.insert(0, str(PYTHON_ROOT))

import json
import tempfile

import config as config


_tmp_dir = tempfile.mkdtemp(prefix="rag_eval_")

config.DB_PATH = os.path.join(_tmp_dir, "eval.db")
config.FAISS_INDEX_PATH = os.path.join(_tmp_dir, "eval_index")
config.DB_BACKEND = "sqlite"
config.NEAR_DUP_BACKEND = "memory"
config.VECTOR_BACKEND = "faiss"


from rag.storage.db import init_db
from rag.ingestion.extractors import extract_text
from rag.ingestion.pipeline import ingest_document
from rag.retrieval.retrieval import retrieve
from rag.generation.generation import generate_answer
from rag.evaluation.retrieval_evaluator import RetrievalEvaluator
from rag.evaluation.generation_evaluator import GenerationEvaluator
from rag.evaluation.outlines_llm_judge import OutlinesLLMJudge
from rag.evaluation.statistics import paired_binary_test


MANUAL_FILES_DIR = "data/manual_test_files"
GOLDEN_SET_PATH = "data/golden_set/queries_v2.json"


# Retrieval modes to compare.
MODES = ("dense", "sparse")

# Generation models to evaluate.
MODELS = ("ollama",)

TOP_K_VALUES = (1, 3, 5)
TOP_K = max(TOP_K_VALUES)

SIGNIFICANCE_LEVEL = 0.05


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


def retrieve_for_mode(entry, mode):
    old_mode = config.SEARCH_MODE
    config.SEARCH_MODE = mode

    try:
        return retrieve(
            entry["query"],
            top_k=TOP_K,
        )
    finally:
        config.SEARCH_MODE = old_mode


def evaluate_retrieval_mode(
    golden_set,
    mode,
    retrieval_evaluator,
):
    """Evaluate retrieval quality once for a retrieval mode."""

    retrieval_results = []

    for entry in golden_set:
        retrieved_results = retrieve_for_mode(
            entry,
            mode,
        )

        retrieval_result = retrieval_evaluator.evaluate(
            entry,
            retrieved_results,
        )

        retrieval_results.append(
            {
                "entry": entry,
                "retrieved_results": retrieved_results,
                "retrieval_result": retrieval_result,
            }
        )

    retrieval_aggregate = retrieval_evaluator.aggregate(
        [
            item["retrieval_result"]
            for item in retrieval_results
        ]
    )

    return retrieval_results, retrieval_aggregate


def evaluate_generation_mode(
    retrieval_results,
    model,
    generation_evaluator,
):
    """Evaluate one generation model using previously retrieved results."""

    generation_results = {
        k: []
        for k in TOP_K_VALUES
    }

    for item in retrieval_results:
        entry = item["entry"]
        retrieved_results = item["retrieved_results"]

        for k in TOP_K_VALUES:
            context_results = retrieved_results[:k]

            answer = generate_for_model(
                entry["query"],
                context_results,
                model,
            )

            generation_result = generation_evaluator.evaluate(
                entry,
                answer,
                context_results,
            )

            generation_results[k].append(
                generation_result
            )

    generation_aggregate = {
        k: generation_evaluator.aggregate(
            generation_results[k],
        )
        for k in TOP_K_VALUES
    }

    return generation_results, generation_aggregate


def _metric_value(result, metric_name):
    """
    Extract one metric from an evaluator result.

    Evaluator result objects expose their per-question metrics through
    the `metrics` mapping.
    """

    try:
        return float(result.metrics[metric_name])
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Could not extract metric '{metric_name}' "
            f"from evaluator result."
        ) from exc


def _binary_retrieval_outcome(
    result,
    k,
):
    """
    Convert per-question retrieval recall into a binary outcome.

    McNemar's test requires binary paired observations.

    A retrieval case is considered a success only when all expected
    evidence was retrieved, i.e. recall@K == 1.0.
    """

    recall = _metric_value(
        result,
        f"recall@{k}",
    )

    return 1.0 if recall == 1.0 else 0.0


def _binary_generation_outcome(
    result,
    metric,
):
    """
    Extract a binary generation metric for paired comparison.

    Generation binary metrics are already represented as 0.0 / 1.0
    by GenerationEvaluator.
    """

    value = _metric_value(
        result,
        metric,
    )

    if value not in (0.0, 1.0):
        raise ValueError(
            f"Generation metric '{metric}' must be binary for "
            f"paired significance testing, got {value}."
        )

    return value


def print_retrieval_results(results):
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


def print_retrieval_significance(
    retrieval_rows_by_mode,
):
    """
    Compare retrieval modes using exact paired McNemar tests.

    The test is run separately for each K because recall@1, recall@3,
    and recall@5 represent different binary success definitions.

    A success means recall@K == 1.0 for that question.
    """

    modes = list(retrieval_rows_by_mode.keys())

    if len(modes) < 2:
        return

    first_mode = modes[0]

    for second_mode in modes[1:]:
        print()
        print("RETRIEVAL STATISTICAL SIGNIFICANCE")
        print("=" * 108)
        print(
            "Exact paired McNemar test on per-question binary "
            "full-recall outcomes"
        )
        print("=" * 108)
        print()
        print(f"{first_mode} vs {second_mode}")
        print("-" * 108)

        print(
            f"{'Metric':<12}"
            f"{first_mode:<12}"
            f"{second_mode:<12}"
            f"{'First only':<14}"
            f"{'Second only':<15}"
            f"{'p-value':<12}"
            f"{'Significant?':<15}"
        )

        print("-" * 108)

        first_rows = retrieval_rows_by_mode[first_mode]
        second_rows = retrieval_rows_by_mode[second_mode]

        for k in TOP_K_VALUES:
            first_outcomes = [
                _binary_retrieval_outcome(
                    item["retrieval_result"],
                    k,
                )
                for item in first_rows
            ]

            second_outcomes = [
                _binary_retrieval_outcome(
                    item["retrieval_result"],
                    k,
                )
                for item in second_rows
            ]

            result = paired_binary_test(
                first_outcomes,
                second_outcomes,
            )

            first_estimate = sum(first_outcomes) / len(first_outcomes)
            second_estimate = sum(second_outcomes) / len(second_outcomes)

            significant = result.p_value < SIGNIFICANCE_LEVEL

            print(
                f"{f'recall@{k}':<12}"
                f"{first_estimate:<12.2%}"
                f"{second_estimate:<12.2%}"
                f"{result.first_only:<14}"
                f"{result.second_only:<15}"
                f"{result.p_value:<12.4f}"
                f"{'YES' if significant else 'NO':<15}"
            )

        print("=" * 108)


def print_generation_results(results):
    print()
    print("GENERATION MODEL + RETRIEVAL COMPARISON")
    print("=" * 108)

    print(
        f"{'Model':<12}"
        f"{'Mode':<10}"
        f"{'K':<8}"
        f"{'Correctness':<15}"
        f"{'Groundedness':<15}"
        f"{'Citation':<15}"
        f"{'Abstention':<15}"
        f"{'Semantic Similarity':<20}"
        f"{'Judge Correct':<15}"
        f"{'Judge Grounded':<15}"
        f"{'Judge Relevant':<15}"
        f"{'Judge Complete':<15}"
    )

    print("-" * 108)

    for model, metrics_by_mode in results.items():
        for mode, metrics_by_k in metrics_by_mode.items():
            for k, metrics in metrics_by_k.items():
                print(
                    f"{model:<12}"
                    f"{mode:<10}"
                    f"{k:<8}"
                    f"{metrics['correctness']:.2%}         "
                    f"{metrics['groundedness']:.2%}         "
                    f"{metrics['citation']:.2%}         "
                    f"{metrics['abstention']:.2%}         "
                    f"{metrics['semantic_similarity']:.2%}              "
                    f"{metrics['judge_correctness']:.2%}         "
                    f"{metrics['judge_groundedness']:.2%}         "
                    f"{metrics['judge_relevance']:.2%}         "
                    f"{metrics['judge_completeness']:.2%}"
                )

    print("=" * 108)


def print_generation_significance(
    generation_rows_by_model_mode,
):
    """
    Compare generation configurations using paired McNemar tests.

    Comparisons are performed on the same golden-set questions.

    For each K, the following binary metrics are tested:
      - correctness
      - groundedness
      - citation
      - abstention

    When multiple generation models or retrieval modes exist, the first
    configuration is compared against every subsequent configuration.
    """

    configurations = list(
        generation_rows_by_model_mode.keys()
    )

    if len(configurations) < 2:
        return

    metric_names = (
        "correctness",
        "groundedness",
        "citation",
        "abstention",
    )

    for index, first_configuration in enumerate(configurations[:-1]):
        for second_configuration in configurations[index + 1:]:
            print()
            print("GENERATION STATISTICAL SIGNIFICANCE")
            print("=" * 108)
            print(
                "Exact paired McNemar test on per-question "
                "binary generation outcomes"
            )
            print("=" * 108)
            print()
            print(
                f"{first_configuration[0]} + "
                f"{first_configuration[1]} vs "
                f"{second_configuration[0]} + "
                f"{second_configuration[1]}"
            )
            print("-" * 108)

            print(
                f"{'K':<6}"
                f"{'Metric':<16}"
                f"{'First':<12}"
                f"{'Second':<12}"
                f"{'First only':<14}"
                f"{'Second only':<15}"
                f"{'p-value':<12}"
                f"{'Significant?':<15}"
            )

            print("-" * 108)

            first_rows_by_k = generation_rows_by_model_mode[
                first_configuration
            ]

            second_rows_by_k = generation_rows_by_model_mode[
                second_configuration
            ]

            for k in TOP_K_VALUES:
                first_rows = first_rows_by_k[k]
                second_rows = second_rows_by_k[k]

                for metric in metric_names:
                    first_outcomes = [
                        _binary_generation_outcome(
                            result,
                            metric,
                        )
                        for result in first_rows
                    ]

                    second_outcomes = [
                        _binary_generation_outcome(
                            result,
                            metric,
                        )
                        for result in second_rows
                    ]

                    result = paired_binary_test(
                        first_outcomes,
                        second_outcomes,
                    )

                    first_estimate = (
                        sum(first_outcomes)
                        / len(first_outcomes)
                    )

                    second_estimate = (
                        sum(second_outcomes)
                        / len(second_outcomes)
                    )

                    significant = (
                        result.p_value < SIGNIFICANCE_LEVEL
                    )

                    print(
                        f"{k:<6}"
                        f"{metric:<16}"
                        f"{first_estimate:<12.2%}"
                        f"{second_estimate:<12.2%}"
                        f"{result.first_only:<14}"
                        f"{result.second_only:<15}"
                        f"{result.p_value:<12.4f}"
                        f"{'YES' if significant else 'NO':<15}"
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

    retrieval_evaluator = RetrievalEvaluator(
        top_k_values=TOP_K_VALUES,
    )

    generation_evaluator = GenerationEvaluator(
        llm_judge=OutlinesLLMJudge(),
    )

    retrieval_results = {}

    generation_results = {
        model: {}
        for model in MODELS
    }

    retrieval_rows_by_mode = {}

    # ---------------------------------------------------------------
    # Retrieval evaluation
    #
    # Retrieval is independent of the generation model.
    # Run it once per retrieval mode.
    # ---------------------------------------------------------------

    for mode in MODES:
        print(f"  Running retrieval: {mode}...")

        (
            retrieval_rows,
            retrieval_aggregate,
        ) = evaluate_retrieval_mode(
            golden_set,
            mode,
            retrieval_evaluator,
        )

        retrieval_rows_by_mode[mode] = retrieval_rows
        retrieval_results[mode] = retrieval_aggregate

    # ---------------------------------------------------------------
    # Generation evaluation
    #
    # Reuse the exact same retrieval results for every generation model.
    # ---------------------------------------------------------------

    generation_rows_by_model_mode = {}

    for model in MODELS:
        for mode in MODES:
            print(
                f"  Running generation: "
                f"{model} + {mode}..."
            )

            (
                generation_rows,
                generation_aggregate,
            ) = evaluate_generation_mode(
                retrieval_rows_by_mode[mode],
                model,
                generation_evaluator,
            )

            generation_results[model][mode] = (
                generation_aggregate
            )

            generation_rows_by_model_mode[
                (model, mode)
            ] = generation_rows

    # ---------------------------------------------------------------
    # Standard evaluation results
    # ---------------------------------------------------------------

    print_retrieval_results(
        retrieval_results,
    )

    print_generation_results(
        generation_results,
    )

    # ---------------------------------------------------------------
    # Statistical significance
    #
    # Retrieval:
    #   dense vs sparse, separately for recall@1/@3/@5.
    #
    # Generation:
    #   same golden-set questions, separately for each K and each
    #   binary generation metric.
    # ---------------------------------------------------------------

    print_retrieval_significance(
        retrieval_rows_by_mode,
    )

    print_generation_significance(
        generation_rows_by_model_mode,
    )


if __name__ == "__main__":
    main()