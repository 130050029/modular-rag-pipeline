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
    assert is_table_like(TABLE_TEXT) is True


def test_empty_text_is_not_table():
    assert is_table_like("") is False


def test_plain_prose_not_detected_as_table():
    assert is_table_like("This is a normal sentence about widgets.") is False


def test_table_with_mostly_table_lines_is_detected():
    text = """| Name | Price |
|------|-------|
| Widget X | $40 |
This is a note."""

    assert is_table_like(text) is True


def test_embedding_text_is_description_for_tables(monkeypatch):
    monkeypatch.setattr(
        "rag.ingestion.tables.TABLE_EMBEDDING_DESCRIPTION",
        "template",
    )

    result = get_embedding_text(TABLE_TEXT)

    assert "Name" in result
    assert "Price" in result
    assert "2 rows" in result
    assert result != TABLE_TEXT


def test_embedding_text_unchanged_for_prose():
    prose = "This is just a normal sentence about widgets."

    assert get_embedding_text(prose) == prose


def test_segments_preserve_document_order():
    text = (
        "Intro paragraph before the table.\n\n"
        f"{TABLE_TEXT}\n\n"
        "Closing paragraph after the table."
    )

    segments = split_into_segments(text)

    assert len(segments) == 3
    assert segments[0][0] == "prose"
    assert "Intro" in segments[0][1]

    assert segments[1] == ("table", TABLE_TEXT)

    assert segments[2][0] == "prose"
    assert "Closing" in segments[2][1]


def test_prose_before_and_after_table_stays_separate():
    text = (
        "Before the table.\n"
        f"{TABLE_TEXT}\n"
        "After the table."
    )

    segments = split_into_segments(text)

    assert len(segments) == 3
    assert "Before" in segments[0][1]
    assert "Before" not in segments[2][1]
    assert "After" in segments[2][1]
    assert "After" not in segments[0][1]


def test_plain_prose_produces_single_segment():
    text = "Just a plain paragraph with no table content at all."

    segments = split_into_segments(text)

    assert segments == [("prose", text)]


def test_multiple_tables_preserve_order():
    table1 = "| A | B |\n|---|---|\n| 1 | 2 |"
    table2 = "| C | D |\n|---|---|\n| 3 | 4 |"

    text = f"Intro\n{table1}\nMiddle\n{table2}\nEnd"

    segments = split_into_segments(text)

    assert [kind for kind, _ in segments] == [
        "prose",
        "table",
        "prose",
        "table",
        "prose",
    ]
    assert segments[1][1] == table1
    assert segments[3][1] == table2


def test_lone_pipe_line_does_not_become_table():
    text = "Normal prose\n| just a line |\nMore prose"

    segments = split_into_segments(text)

    assert all(kind == "prose" for kind, _ in segments)