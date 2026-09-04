import pytest

from rag.generation.generation import OllamaGenerator


CHUNKS = [
    {
        "source": "revenue.txt",
        "content": "North Q3 revenue was 128000.",
    },
    {
        "source": "notes.txt",
        "content": "North revenue increased compared with Q2.",
    },
]


@pytest.mark.integration
def test_ollama_generates_grounded_answer():
    generator = OllamaGenerator()

    answer = generator.generate(
        "What was North Q3 revenue?",
        CHUNKS,
    )

    assert isinstance(answer, str)
    assert answer.strip()

    assert "128000" in answer