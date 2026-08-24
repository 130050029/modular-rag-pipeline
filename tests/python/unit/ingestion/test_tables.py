from rag.ingestion.tables import (
    get_embedding_text,
    is_table_like,
    split_into_segments,
)


TABLE_TEXT = """| Name | Price |
|------|-------|
| Widget X | $40 |
| Widget Y | $55 |"""


def test_detects_table_like_text():
    assert is_table_like(TABLE_TEXT)


def test_plain_prose_not_detected_as_table():
    assert not is_table_like("This is just a normal sentence about widgets.")


def test_embedding_text_is_description_for_tables():
    result = get_embedding_text(TABLE_TEXT)

    assert "Name" in result
    assert "Price" in result
    assert result != TABLE_TEXT


def test_embedding_text_unchanged_for_prose():
    prose = "This is just a normal sentence about widgets."
    assert get_embedding_text(prose) == prose


def test_segments_preserve_original_document_order():
    text = (
        "Intro paragraph before the table.\n\n"
        f"{TABLE_TEXT}\n\n"
        "Closing paragraph after the table."
    )

    segments = split_into_segments(text)

    assert len(segments) == 3
    assert segments[0][0] == "prose"
    assert "Intro" in segments[0][1]

    assert segments[1][0] == "table"

    assert segments[2][0] == "prose"
    assert "Closing" in segments[2][1]

    assert "Intro" not in segments[2][1]
    assert "Closing" not in segments[0][1]


def test_plain_prose_produces_single_segment():
    segments = split_into_segments(
        "Just a plain paragraph with no table content at all."
    )

    assert segments == [
        ("prose", "Just a plain paragraph with no table content at all.")
    ]