import numpy as np

from rag.evaluation.generation_evaluator import (
    GenerationEvaluationResult,
    GenerationEvaluator,
)


def test_evaluate_returns_generation_result():
    evaluator = GenerationEvaluator()

    entry = {
        "id": "q001",
        "answerable": True,
        "expected_sources": ["sample.txt"],
    }

    answer = (
        "Revenue increased by 20%. "
        "[source: sample.txt]"
    )

    retrieved_results = [
        {
            "source": "sample.txt",
            "content": "Revenue increased by 20%.",
        }
    ]

    result = evaluator.evaluate(
        entry,
        answer,
        retrieved_results,
    )

    assert isinstance(result, GenerationEvaluationResult)
    assert result.case_id == "q001"
    assert result.answerable is True


def test_valid_citation_passes():
    evaluator = GenerationEvaluator()

    entry = {
        "id": "q002",
        "answerable": True,
        "expected_sources": ["sample.txt"],
    }

    answer = "Revenue increased. [source: sample.txt]"

    retrieved_results = [
        {
            "source": "sample.txt",
            "content": "Revenue increased by 20%.",
        }
    ]

    result = evaluator.evaluate(
        entry,
        answer,
        retrieved_results,
    )

    assert result.metrics["citation"] == 1.0


def test_missing_citation_fails_for_answerable_case():
    evaluator = GenerationEvaluator()

    entry = {
        "id": "q003",
        "answerable": True,
        "expected_sources": ["sample.txt"],
    }

    answer = "Revenue increased by 20%."

    retrieved_results = [
        {
            "source": "sample.txt",
            "content": "Revenue increased by 20%.",
        }
    ]

    result = evaluator.evaluate(
        entry,
        answer,
        retrieved_results,
    )

    assert result.metrics["citation"] == 0.0


def test_citation_from_unretrieved_source_fails():
    evaluator = GenerationEvaluator()

    entry = {
        "id": "q004",
        "answerable": True,
        "expected_sources": ["sample.txt"],
    }

    answer = "Revenue increased. [source: other.txt]"

    retrieved_results = [
        {
            "source": "sample.txt",
            "content": "Revenue increased by 20%.",
        }
    ]

    result = evaluator.evaluate(
        entry,
        answer,
        retrieved_results,
    )

    assert result.metrics["citation"] == 0.0


def test_unanswerable_answer_with_no_citation_passes_citation_check():
    evaluator = GenerationEvaluator()

    entry = {
        "id": "q005",
        "answerable": False,
        "expected_sources": [],
    }

    answer = "I don't know based on the available information."

    result = evaluator.evaluate(
        entry,
        answer,
        [],
    )

    assert result.metrics["citation"] == 1.0


def test_unanswerable_answer_with_citation_fails_citation_check():
    evaluator = GenerationEvaluator()

    entry = {
        "id": "q006",
        "answerable": False,
        "expected_sources": [],
    }

    answer = "I don't know. sample.txt"

    retrieved_results = [
        {"source": "sample.txt"},
    ]

    result = evaluator.evaluate(
        entry,
        answer,
        retrieved_results,
    )

    assert result.metrics["citation"] == 0.0


def test_unanswerable_answer_that_abstains_passes():
    evaluator = GenerationEvaluator()

    entry = {
        "id": "q007",
        "answerable": False,
        "expected_sources": [],
    }

    answer = "I don't know based on the available information."

    result = evaluator.evaluate(
        entry,
        answer,
        [],
    )

    assert result.metrics["abstention"] == 1.0


def test_unanswerable_answer_with_factual_claim_fails_abstention():
    evaluator = GenerationEvaluator()

    entry = {
        "id": "q008",
        "answerable": False,
        "expected_sources": [],
    }

    answer = "The answer is 42."

    result = evaluator.evaluate(
        entry,
        answer,
        [],
    )

    assert result.metrics["abstention"] == 0.0


def test_answerable_case_passes_abstention_check():
    evaluator = GenerationEvaluator()

    entry = {
        "id": "q009",
        "answerable": True,
        "expected_sources": ["sample.txt"],
    }

    answer = "The answer is supported. [source: sample.txt]"

    result = evaluator.evaluate(
        entry,
        answer,
        [
            {
                "source": "sample.txt",
                "content": "The answer is supported.",
            }
        ],
    )

    assert result.metrics["abstention"] == 1.0


def test_aggregate_uses_answerable_cases_for_citation():
    evaluator = GenerationEvaluator()

    results = [
        GenerationEvaluationResult(
            case_id="q001",
            answerable=True,
            metrics={
                "correctness": 1.0,
                "citation": 1.0,
                "abstention": 1.0,
            },
        ),
        GenerationEvaluationResult(
            case_id="q002",
            answerable=True,
            metrics={
                "correctness": 1.0,
                "citation": 0.0,
                "abstention": 1.0,
            },
        ),
        GenerationEvaluationResult(
            case_id="q003",
            answerable=False,
            metrics={
                "correctness": 1.0,
                "citation": 0.0,
                "abstention": 1.0,
            },
        ),
    ]

    aggregate = evaluator.aggregate(results)

    assert aggregate["citation"] == 0.5


def test_aggregate_uses_unanswerable_cases_for_abstention():
    evaluator = GenerationEvaluator()

    results = [
        GenerationEvaluationResult(
            case_id="q001",
            answerable=True,
            metrics={
                "correctness": 1.0,
                "citation": 1.0,
                "abstention": 1.0,
            },
        ),
        GenerationEvaluationResult(
            case_id="q002",
            answerable=False,
            metrics={
                "correctness": 1.0,
                "citation": 1.0,
                "abstention": 1.0,
            },
        ),
        GenerationEvaluationResult(
            case_id="q003",
            answerable=False,
            metrics={
                "correctness": 1.0,
                "citation": 1.0,
                "abstention": 0.0,
            },
        ),
    ]

    aggregate = evaluator.aggregate(results)

    assert aggregate["abstention"] == 0.5


def test_aggregate_returns_zero_when_no_applicable_cases():
    evaluator = GenerationEvaluator()

    results = [
        GenerationEvaluationResult(
            case_id="q001",
            answerable=True,
            metrics={
                "citation": 1.0,
                "abstention": 1.0,
            },
        )
    ]

    aggregate = evaluator.aggregate(results)

    assert aggregate["citation"] == 1.0
    assert aggregate["abstention"] == 0.0


def test_correct_answer_matches_reference():
    evaluator = GenerationEvaluator()

    entry = {
        "id": "q001",
        "answerable": True,
        "reference_answer": (
            "The Amazon rainforest is home to an estimated "
            "ten percent of all known species on Earth."
        ),
    }

    answer = (
        "The Amazon rainforest is home to an estimated "
        "ten percent of all known species on Earth."
    )

    result = evaluator.evaluate(entry, answer, [])

    assert result.metrics["correctness"] == 1.0


def test_wrong_answer_fails_correctness(monkeypatch):
    evaluator = GenerationEvaluator()


    entry = {
        "id": "q001",
        "answerable": True,
        "reference_answer": (
            "The Amazon rainforest is home to an estimated "
            "ten percent of all known species on Earth."
        ),
    }

    answer = "The Amazon rainforest contains about one percent of all species."

    result = evaluator.evaluate(entry, answer, [])

    assert result.metrics["correctness"] == 0.0


def test_incomplete_answer_fails_correctness(monkeypatch):
    evaluator = GenerationEvaluator()


    entry = {
        "id": "q001",
        "answerable": True,
        "reference_answer": (
            "The Amazon rainforest is home to an estimated "
            "ten percent of all known species on Earth."
        ),
    }

    answer = "The Amazon rainforest is home to many species."

    result = evaluator.evaluate(entry, answer, [])

    assert result.metrics["correctness"] == 0.0


def test_unanswerable_case_is_not_penalized_for_correctness():
    evaluator = GenerationEvaluator()

    entry = {
        "id": "q006",
        "answerable": False,
        "reference_answer": "",
    }

    answer = "I don't have enough information from the provided context."

    result = evaluator.evaluate(entry, answer, [])

    assert result.metrics["correctness"] == 1.0


def test_grounded_answer_passes_groundedness_check():
    evaluator = GenerationEvaluator()

    entry = {
        "id": "q001",
        "answerable": True,
        "evidence": [
            {
                "id": "E1",
                "source": "sample.txt",
                "text": "The system uses SQLite for local storage.",
            }
        ],
    }

    answer = "The system uses SQLite for local storage."

    retrieved_results = [
        {
            "source": "sample.txt",
            "content": "The system uses SQLite for local storage.",
        }
    ]

    result = evaluator.evaluate(
        entry,
        answer,
        retrieved_results,
    )

    assert result.metrics["groundedness"] == 1.0


def test_unsupported_answer_fails_groundedness_check():
    evaluator = GenerationEvaluator()

    entry = {
        "id": "q002",
        "answerable": True,
        "evidence": [
            {
                "id": "E1",
                "source": "sample.txt",
                "text": "The system uses SQLite for local storage.",
            }
        ],
    }

    answer = "The system uses PostgreSQL for production storage."

    retrieved_results = [
        {
            "source": "sample.txt",
            "content": "The system uses SQLite for local storage.",
        }
    ]

    result = evaluator.evaluate(
        entry,
        answer,
        retrieved_results,
    )

    assert result.metrics["groundedness"] == 0.0


def test_groundedness_fails_when_evidence_is_not_retrieved():
    evaluator = GenerationEvaluator()

    entry = {
        "id": "q003",
        "answerable": True,
        "evidence": [
            {
                "id": "E1",
                "source": "sample.txt",
                "text": "The system uses SQLite for local storage.",
            }
        ],
    }

    answer = "The system uses SQLite for local storage."

    retrieved_results = [
        {
            "source": "other.txt",
            "content": "The system uses PostgreSQL for production storage.",
        }
    ]

    result = evaluator.evaluate(
        entry,
        answer,
        retrieved_results,
    )

    assert result.metrics["groundedness"] == 0.0


def test_groundedness_fails_for_empty_answer():
    evaluator = GenerationEvaluator()

    entry = {
        "id": "q004",
        "answerable": True,
        "evidence": [
            {
                "id": "E1",
                "source": "sample.txt",
                "text": "The system uses SQLite for local storage.",
            }
        ],
    }

    answer = ""

    retrieved_results = [
        {
            "source": "sample.txt",
            "content": "The system uses SQLite for local storage.",
        }
    ]

    result = evaluator.evaluate(
        entry,
        answer,
        retrieved_results,
    )

    assert result.metrics["groundedness"] == 0.0


def test_groundedness_fails_when_no_evidence_exists():
    evaluator = GenerationEvaluator()

    entry = {
        "id": "q005",
        "answerable": True,
        "evidence": [],
    }

    answer = "The system uses SQLite for local storage."

    retrieved_results = [
        {
            "source": "sample.txt",
            "content": "The system uses SQLite for local storage.",
        }
    ]

    result = evaluator.evaluate(
        entry,
        answer,
        retrieved_results,
    )

    assert result.metrics["groundedness"] == 0.0


def test_unanswerable_groundedness_follows_abstention():
    evaluator = GenerationEvaluator()

    entry = {
        "id": "q006",
        "answerable": False,
        "expected_sources": [],
        "evidence": [],
    }

    answer = "I don't have enough information from the provided context."

    retrieved_results = []

    result = evaluator.evaluate(
        entry,
        answer,
        retrieved_results,
    )

    assert result.metrics["groundedness"] == 1.0


def test_unanswerable_non_abstaining_answer_fails_groundedness():
    evaluator = GenerationEvaluator()

    entry = {
        "id": "q007",
        "answerable": False,
        "expected_sources": [],
        "evidence": [],
    }

    answer = "The system uses SQLite."

    retrieved_results = []

    result = evaluator.evaluate(
        entry,
        answer,
        retrieved_results,
    )

    assert result.metrics["groundedness"] == 0.0


def test_groundedness_is_evaluated_against_supplied_context():
    evaluator = GenerationEvaluator()

    entry = {
        "id": "q008",
        "answerable": True,
        "evidence": [
            {
                "id": "E1",
                "source": "sample.txt",
                "text": "The project uses FAISS for vector search.",
            }
        ],
    }

    answer = "The project uses FAISS for vector search."

    retrieved_results = [
        {
            "source": "sample.txt",
            "content": "The project uses FAISS for vector search.",
        }
    ]

    result = evaluator.evaluate(
        entry,
        answer,
        retrieved_results,
    )

    assert result.metrics["groundedness"] == 1.0


# ---------------------------------------------------------------------------
# Semantic similarity
# ---------------------------------------------------------------------------


def test_semantic_similarity_is_one_for_identical_embeddings(monkeypatch):
    evaluator = GenerationEvaluator()

    monkeypatch.setattr(
        "rag.evaluation.generation_evaluator.embed_texts",
        lambda texts: np.array(
            [
                [1.0, 0.0],
                [1.0, 0.0],
            ]
        ),
    )

    entry = {
        "id": "q009",
        "answerable": True,
        "reference_answer": "Revenue increased by 20%.",
    }

    result = evaluator.evaluate(
        entry,
        "Revenue increased by 20%.",
        [],
    )

    assert result.metrics["semantic_similarity"] == 1.0


def test_semantic_similarity_handles_semantically_similar_answers(monkeypatch):
    evaluator = GenerationEvaluator()

    monkeypatch.setattr(
        "rag.evaluation.generation_evaluator.embed_texts",
        lambda texts: np.array(
            [
                [1.0, 0.0],
                [0.8, 0.6],
            ]
        ),
    )

    entry = {
        "id": "q010",
        "answerable": True,
        "reference_answer": "Revenue increased by 20%.",
    }

    result = evaluator.evaluate(
        entry,
        "Revenue grew by twenty percent.",
        [],
    )

    assert result.metrics["semantic_similarity"] == 0.8


def test_semantic_similarity_is_low_for_dissimilar_answers(monkeypatch):
    evaluator = GenerationEvaluator()

    monkeypatch.setattr(
        "rag.evaluation.generation_evaluator.embed_texts",
        lambda texts: np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        ),
    )

    entry = {
        "id": "q011",
        "answerable": True,
        "reference_answer": "Revenue increased by 20%.",
    }

    result = evaluator.evaluate(
        entry,
        "The company uses SQLite for local storage.",
        [],
    )

    assert result.metrics["semantic_similarity"] == 0.0


def test_semantic_similarity_is_zero_when_reference_or_answer_is_missing(
    monkeypatch,
):
    evaluator = GenerationEvaluator()

    def fail_if_called(texts):
        raise AssertionError("Embeddings should not be generated.")

    monkeypatch.setattr(
        "rag.evaluation.generation_evaluator.embed_texts",
        fail_if_called,
    )

    entry = {
        "id": "q012",
        "answerable": True,
        "reference_answer": "",
    }

    result = evaluator.evaluate(
        entry,
        "Some generated answer.",
        [],
    )

    assert result.metrics["semantic_similarity"] == 0.0


def test_semantic_similarity_is_not_applicable_to_unanswerable_cases(
    monkeypatch,
):
    evaluator = GenerationEvaluator()

    def fail_if_called(texts):
        raise AssertionError("Embeddings should not be generated.")

    monkeypatch.setattr(
        "rag.evaluation.generation_evaluator.embed_texts",
        fail_if_called,
    )

    entry = {
        "id": "q013",
        "answerable": False,
        "reference_answer": "",
    }

    result = evaluator.evaluate(
        entry,
        "I don't have enough information.",
        [],
    )

    assert "semantic_similarity" not in result.metrics