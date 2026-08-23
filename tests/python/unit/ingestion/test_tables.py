from rag.ingestion.tables import is_table_like, get_embedding_text, split_into_segments

TABLE_TEXT = """| Name | Price |
|------|-------|
| Widget X | $40 |
| Widget Y | $55 |"""


def test_detects_table_like_text():
    assert is_table_like(TABLE_TEXT) is True


def test_plain_prose_not_detected_as_table():
    assert is_table_like("This is just a normal sentence about widgets.") is False


def test_embedding_text_is_description_for_tables():
    result = get_embedding_text(TABLE_TEXT)
    assert "Name" in result and "Price" in result
    assert result != TABLE_TEXT  # embedding text differs from what's shown to the LLM


def test_embedding_text_unchanged_for_prose():
    prose = "This is just a normal sentence about widgets."
    assert get_embedding_text(prose) == prose


def test_segments_preserve_original_document_order():
    """Regression test: an earlier version of this logic extracted all
    tables first and merged ALL surrounding prose into one blob, losing the
    table's actual position and incorrectly merging unrelated prose that
    appeared before and after it."""
    text = (
        "Intro paragraph before the table.\n\n"
        f"{TABLE_TEXT}\n\n"
        "Closing paragraph after the table."
    )
    segments = split_into_segments(text)

    assert len(segments) == 3
    assert segments[0][0] == "prose" and "Intro" in segments[0][1]
    assert segments[1][0] == "table"
    assert segments[2][0] == "prose" and "Closing" in segments[2][1]
    # The two prose sections must remain SEPARATE, not merged together
    assert "Intro" not in segments[2][1]
    assert "Closing" not in segments[0][1]


def test_plain_prose_produces_single_segment():
    segments = split_into_segments("Just a plain paragraph with no table content at all.")
    assert len(segments) == 1
    assert segments[0][0] == "prose"