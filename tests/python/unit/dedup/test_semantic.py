import config

from rag.dedup import semantic


def test_no_duplicate_when_vector_index_is_empty(monkeypatch):
    class EmptyIndex:
        size = 0

    monkeypatch.setattr(semantic, "vector_index", EmptyIndex())

    result = semantic.find_semantic_duplicate([[0.1, 0.2]])

    assert result is None


def test_no_duplicate_when_search_returns_no_hits(monkeypatch):
    class Index:
        size = 1

        def search(self, vector, top_k):
            return []

    monkeypatch.setattr(semantic, "vector_index", Index())

    result = semantic.find_semantic_duplicate([[0.1, 0.2]])

    assert result is None


def test_no_duplicate_when_score_is_below_threshold(monkeypatch):
    class Index:
        size = 1

        def search(self, vector, top_k):
            assert top_k == 1
            return [("chunk-1", config.SEMANTIC_DUP_COSINE_THRESHOLD - 0.01)]

    monkeypatch.setattr(semantic, "vector_index", Index())

    result = semantic.find_semantic_duplicate([[0.1, 0.2]])

    assert result is None


def test_duplicate_when_score_equals_threshold(monkeypatch):
    class Index:
        size = 1

        def search(self, vector, top_k):
            return [("chunk-1", config.SEMANTIC_DUP_COSINE_THRESHOLD)]

    monkeypatch.setattr(semantic, "vector_index", Index())

    result = semantic.find_semantic_duplicate([[0.1, 0.2]])

    assert result == (
        "chunk-1",
        config.SEMANTIC_DUP_COSINE_THRESHOLD,
    )


def test_duplicate_when_score_is_above_threshold(monkeypatch):
    score = config.SEMANTIC_DUP_COSINE_THRESHOLD + 0.05

    class Index:
        size = 1

        def search(self, vector, top_k):
            return [("chunk-42", score)]

    monkeypatch.setattr(semantic, "vector_index", Index())

    result = semantic.find_semantic_duplicate([[0.1, 0.2]])

    assert result == ("chunk-42", score)


def test_returns_nearest_hit_unchanged(monkeypatch):
    score = config.SEMANTIC_DUP_COSINE_THRESHOLD + 0.02

    class Index:
        size = 3

        def search(self, vector, top_k):
            assert top_k == 1
            return [("chunk-nearest", score)]

    monkeypatch.setattr(semantic, "vector_index", Index())

    vector = [[0.3, 0.4, 0.5]]

    result = semantic.find_semantic_duplicate(vector)

    assert result == ("chunk-nearest", score)