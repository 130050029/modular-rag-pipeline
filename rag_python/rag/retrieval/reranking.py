"""Cross-encoder reranking for retrieved candidates."""

from sentence_transformers import CrossEncoder

_model = None


def _get_model():
    global _model

    if _model is None:
        from config import RERANKER_MODEL
        print(f"Loading reranker '{RERANKER_MODEL}'...")
        _model = CrossEncoder(RERANKER_MODEL)

    return _model


def rerank(query: str, results: list[dict], top_k: int) -> list[dict]:
    if not results:
        return []

    model = _get_model()
    pairs = [(query, result["content"]) for result in results]
    scores = model.predict(pairs, show_progress_bar=False)

    ranked = []

    for result, score in zip(results, scores):
        item = dict(result)
        item["retrieval_score"] = item.pop("score")
        item["score"] = float(score)
        item["score_type"] = "reranker"
        ranked.append(item)

    ranked.sort(key=lambda x: x["score"], reverse=True)

    for rank, result in enumerate(ranked[:top_k], start=1):
        result["rank"] = rank

    return ranked[:top_k]