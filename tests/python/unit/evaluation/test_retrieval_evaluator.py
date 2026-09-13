from rag.evaluation.retrieval_evaluator import (
    EvaluationResult,
    RetrievalEvaluator,
)


def test_evaluate_returns_result_for_golden_set_entry():
    evaluator = RetrievalEvaluator()

    entry = {
        "id": "q001",
        "answerable": True,
        "evidence": [
            {
                "id": "E1",
                "source": "sample.txt",
                "text": "The company increased revenue by 20%.",
            }
        ],
    }

    results = [
        {
            "source": "sample.txt",
            "content": "The company increased revenue by 20% during the quarter.",
        }
    ]

    result = evaluator.evaluate(entry, results)

    assert isinstance(result, EvaluationResult)
    assert result.case_id == "q001"
    assert result.answerable is True


def test_evaluate_calculates_retrieval_metrics():
    evaluator = RetrievalEvaluator()

    entry = {
        "id": "q002",
        "answerable": True,
        "evidence": [
            {
                "id": "E1",
                "source": "sample.txt",
                "text": "The company increased revenue by 20%.",
            }
        ],
    }

    results = [
        {
            "source": "irrelevant.txt",
            "content": "The weather was sunny.",
        },
        {
            "source": "sample.txt",
            "content": "The company increased revenue by 20% during the quarter.",
        },
        {
            "source": "other.txt",
            "content": "The office moved downtown.",
        },
    ]

    result = evaluator.evaluate(entry, results)

    assert result.metrics["recall@1"] == 0.0
    assert result.metrics["recall@3"] == 1.0

    assert result.metrics["precision@1"] == 0.0
    assert result.metrics["precision@3"] == 1 / 3

    assert result.metrics["rr"] == 0.5


def test_evaluate_supports_multiple_evidence_items():
    evaluator = RetrievalEvaluator()

    entry = {
        "id": "q003",
        "answerable": True,
        "evidence": [
            {
                "id": "E1",
                "source": "sample.txt",
                "text": "Revenue increased by 20%.",
            },
            {
                "id": "E2",
                "source": "sample.txt",
                "text": "Profit increased by 10%.",
            },
        ],
    }

    results = [
        {
            "source": "sample.txt",
            "content": (
                "Revenue increased by 20% during the quarter."
            ),
        },
        {
            "source": "sample.txt",
            "content": (
                "Profit increased by 10% during the quarter."
            ),
        },
    ]

    result = evaluator.evaluate(entry, results)

    assert result.metrics["recall@1"] == 0.5
    assert result.metrics["recall@3"] == 1.0
    assert result.metrics["precision@1"] == 1.0
    assert result.metrics["precision@3"] == 1.0
    assert result.metrics["rr"] == 1.0


def test_evaluate_preserves_answerable_flag():
    evaluator = RetrievalEvaluator()

    entry = {
        "id": "q004",
        "answerable": False,
        "evidence": [],
    }

    result = evaluator.evaluate(entry, [])

    assert result.case_id == "q004"
    assert result.answerable is False


def test_evaluate_handles_no_retrieved_results():
    evaluator = RetrievalEvaluator()

    entry = {
        "id": "q005",
        "answerable": True,
        "evidence": [
            {
                "id": "E1",
                "source": "sample.txt",
                "text": "Useful evidence is present.",
            }
        ],
    }

    result = evaluator.evaluate(entry, [])

    assert result.metrics["recall@1"] == 0.0
    assert result.metrics["recall@3"] == 0.0
    assert result.metrics["recall@5"] == 0.0
    assert result.metrics["precision@1"] == 0.0
    assert result.metrics["precision@3"] == 0.0
    assert result.metrics["precision@5"] == 0.0
    assert result.metrics["rr"] == 0.0


def test_evaluate_supports_custom_top_k_values():
    evaluator = RetrievalEvaluator(top_k_values=(1, 2))

    entry = {
        "id": "q006",
        "answerable": True,
        "evidence": [
            {
                "id": "E1",
                "source": "sample.txt",
                "text": "Revenue was 160000.",
            }
        ],
    }

    results = [
        {
            "source": "other.txt",
            "content": "Revenue was 150000.",
        },
        {
            "source": "sample.txt",
            "content": "Revenue was 160000.",
        },
    ]

    result = evaluator.evaluate(entry, results)

    assert result.metrics["recall@1"] == 0.0
    assert result.metrics["recall@2"] == 1.0
    assert result.metrics["precision@1"] == 0.0
    assert result.metrics["precision@2"] == 0.5


def test_aggregate_ignores_unanswerable_cases():
    evaluator = RetrievalEvaluator()

    results = [
        EvaluationResult(
            case_id="q001",
            answerable=True,
            metrics={
                "recall@1": 1.0,
                "recall@3": 1.0,
                "recall@5": 1.0,
                "precision@1": 1.0,
                "precision@3": 0.5,
                "precision@5": 0.5,
                "rr": 1.0,
            },
        ),
        EvaluationResult(
            case_id="q002",
            answerable=False,
            metrics={
                "recall@1": 0.0,
                "recall@3": 0.0,
                "recall@5": 0.0,
                "precision@1": 0.0,
                "precision@3": 0.0,
                "precision@5": 0.0,
                "rr": 0.0,
            },
        ),
    ]

    aggregate = evaluator.aggregate(results)

    assert aggregate["recall@1"] == 1.0
    assert aggregate["recall@3"] == 1.0
    assert aggregate["recall@5"] == 1.0
    assert aggregate["precision@1"] == 1.0
    assert aggregate["precision@3"] == 0.5
    assert aggregate["precision@5"] == 0.5
    assert aggregate["mrr"] == 1.0
