


def test_rag_generates_grounded_answer_from_uploaded_document(
    client,
    fake_embeddings,
    ollama_available,
):
    upload_response = client.post(
        "/upload",
        files={
            "file": (
                "revenue.txt",
                (
                    b"North region Q3 revenue was 128000. "
                    b"North region revenue increased from Q2."
                ),
                "text/plain",
            )
        },
    )

    assert upload_response.status_code == 200
    assert upload_response.json()["status"] == "ingested"

    response = client.post(
        "/chat",
        json={"query": "What was North region Q3 revenue?"},
    )

    assert response.status_code == 200

    data = response.json()

    assert isinstance(data["answer"], str)
    assert data["answer"].strip()

    assert "128000" in data["answer"]

    assert data["sources"]
    assert any(
        source.get("source") == "revenue.txt"
        for source in data["sources"]
    )