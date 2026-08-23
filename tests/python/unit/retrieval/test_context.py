from rag.retrieval.context import build_context


def test_build_context_preserves_retrieval_order():
    results = [
        {"content": "First result"},
        {"content": "Second result"},
    ]

    context = build_context(results)

    assert context == "First result\n\nSecond result"


def test_build_context_removes_duplicate_chunks():
    results = [
        {"content": "Same content"},
        {"content": "Same content"},
        {"content": "Different content"},
    ]

    context = build_context(results)

    assert context == "Same content\n\nDifferent content"


def test_build_context_respects_character_limit():
    results = [
        {"content": "12345"},
        {"content": "67890"},
        {"content": "abcdef"},
    ]

    context = build_context(results, max_chars=10)

    assert context == "12345\n\n67890"


def test_build_context_skips_empty_content():
    results = [
        {"content": ""},
        {"content": "Useful content"},
        {"content": "   "},
    ]

    context = build_context(results)

    assert context == "Useful content"


def test_build_context_empty_results():
    assert build_context([]) == ""