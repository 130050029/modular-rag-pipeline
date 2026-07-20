from rag.ingestion.tables import is_table_like, get_embedding_text

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
