from rag.retrieval import reranking


class FakeModel:
    def __init__(self, scores):
        self.scores = scores
        self.pairs = None

    def predict(self, pairs, show_progress_bar=False):
        self.pairs = pairs
        return self.scores


def test_rerank_orders_results_by_cross_encoder_score(monkeypatch):
    model = FakeModel([0.2, 0.9])

    monkeypatch.setattr(reranking, "_model", model)

    results = [
        {"content": "bad", "score": 1.0, "source": "a.txt"},
        {"content": "good", "score": 0.5, "source": "b.txt"},
    ]

    ranked = reranking.rerank(
        "query",
        results,
        top_k=2,
    )

    assert [result["source"] for result in ranked] == [
        "b.txt",
        "a.txt",
    ]

    assert ranked[0]["score"] == 0.9
    assert ranked[0]["retrieval_score"] == 0.5
    assert ranked[0]["score_type"] == "reranker"
    assert ranked[0]["rank"] == 1

    assert ranked[1]["rank"] == 2


def test_rerank_sends_query_content_pairs_to_model(monkeypatch):
    model = FakeModel([0.7, 0.3])
    monkeypatch.setattr(reranking, "_model", model)

    results = [
        {"content": "first", "score": 1.0},
        {"content": "second", "score": 0.5},
    ]

    reranking.rerank(
        "my query",
        results,
        top_k=2,
    )

    assert model.pairs == [
        ("my query", "first"),
        ("my query", "second"),
    ]


def test_rerank_respects_top_k(monkeypatch):
    model = FakeModel([0.9, 0.8, 0.7])
    monkeypatch.setattr(reranking, "_model", model)

    results = [
        {"content": "first", "score": 1.0},
        {"content": "second", "score": 0.9},
        {"content": "third", "score": 0.8},
    ]

    ranked = reranking.rerank(
        "query",
        results,
        top_k=2,
    )

    assert len(ranked) == 2
    assert [result["rank"] for result in ranked] == [1, 2]


def test_rerank_does_not_mutate_input_results(monkeypatch):
    model = FakeModel([0.9])
    monkeypatch.setattr(reranking, "_model", model)

    original = {
        "content": "content",
        "score": 1.0,
        "source": "a.txt",
    }

    results = [original]

    ranked = reranking.rerank(
        "query",
        results,
        top_k=1,
    )

    assert original["score"] == 1.0
    assert "retrieval_score" not in original
    assert ranked[0] is not original


def test_rerank_empty_results():
    assert reranking.rerank(
        "query",
        [],
        top_k=5,
    ) == []