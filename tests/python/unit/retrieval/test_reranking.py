from rag.retrieval import reranking


def test_rerank_orders_results(monkeypatch):
    class FakeModel:
        def predict(self, pairs, show_progress_bar=False):
            return [0.2, 0.9]

    monkeypatch.setattr(reranking, "_model", FakeModel())

    results = [
        {"content": "bad", "score": 1.0, "source": "a.txt"},
        {"content": "good", "score": 0.5, "source": "b.txt"},
    ]

    ranked = reranking.rerank("query", results, top_k=2)

    assert ranked[0]["source"] == "b.txt"
    assert ranked[0]["score"] == 0.9
    assert ranked[0]["score_type"] == "reranker"
    assert ranked[0]["retrieval_score"] == 0.5
    assert ranked[0]["rank"] == 1


def test_rerank_empty_results():
    assert reranking.rerank("query", [], top_k=5) == []