from rag.retrieval.context import build_context


def test_build_context_preserves_retrieval_order():
    results = [
        {"content": "First result"},
        {"content": "Second result"},
        {"content": "Third result"},
    ]

    context = build_context(results)

    assert context == "First result\n\nSecond result\n\nThird result"


def test_build_context_removes_duplicate_content():
    results = [
        {"content": "Same content"},
        {"content": "Different content"},
        {"content": "Same content"},
    ]

    context = build_context(results)

    assert context == "Same content\n\nDifferent content"


def test_build_context_skips_empty_and_whitespace_content():
    results = [
        {"content": ""},
        {"content": "   "},
        {"content": "\n\t"},
        {"content": "Useful content"},
    ]

    assert build_context(results) == "Useful content"


def test_build_context_strips_content_before_packing():
    results = [
        {"content": "  First result  "},
        {"content": "\nSecond result\n"},
    ]

    assert build_context(results) == "First result\n\nSecond result"


def test_build_context_respects_character_limit():
    results = [
        {"content": "12345"},
        {"content": "67890"},
        {"content": "abcdef"},
    ]

    assert build_context(results, max_chars=10) == "12345\n\n67890"


def test_build_context_does_not_include_chunk_that_exceeds_remaining_budget():
    results = [
        {"content": "12345"},
        {"content": "678"},
    ]

    assert build_context(results, max_chars=6) == "12345"


def test_build_context_exact_limit_is_allowed():
    results = [
        {"content": "12345"},
    ]

    assert build_context(results, max_chars=5) == "12345"


def test_build_context_zero_or_negative_budget_returns_empty():
    results = [
        {"content": "Useful content"},
    ]

    assert build_context(results, max_chars=0) == ""
    assert build_context(results, max_chars=-1) == ""


def test_build_context_empty_results():
    assert build_context([]) == ""