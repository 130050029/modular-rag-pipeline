


def test_upload_and_chat_flow(client, fake_embeddings, ollama_available):
    upload_response = client.post(
        "/upload",
        files={
            "file": (
                "doc.txt",
                b"Paris is the capital of France.",
                "text/plain",
            )
        },
    )

    assert upload_response.status_code == 200

    upload_data = upload_response.json()

    assert upload_data["status"] == "ingested"
    assert upload_data["doc_id"]
    assert upload_data["chunks"] >= 1

    chat_response = client.post(
        "/chat",
        json={"query": "What is the capital of France?"},
    )

    assert chat_response.status_code == 200

    chat_data = chat_response.json()

    assert "answer" in chat_data
    assert "sources" in chat_data

    assert isinstance(chat_data["answer"], str)
    assert chat_data["answer"].strip()

    assert chat_data["sources"]
    assert any(
        source.get("source") == "doc.txt"
        for source in chat_data["sources"]
    )


def test_chat_with_no_documents_returns_helpful_message(client):
    response = client.post(
        "/chat",
        json={"query": "What is the capital of France?"},
    )

    assert response.status_code == 200

    data = response.json()

    assert "answer" in data
    assert "sources" in data
    assert data["sources"] == []


def test_uploaded_document_can_answer_multiple_queries(
    client,
    fake_embeddings,
    ollama_available
):
    upload_response = client.post(
        "/upload",
        files={
            "file": (
                "company.txt",
                (
                    b"The company was founded in 2010. "
                    b"Its headquarters are in Berlin."
                ),
                "text/plain",
            )
        },
    )

    assert upload_response.status_code == 200
    assert upload_response.json()["status"] == "ingested"

    founded_response = client.post(
        "/chat",
        json={"query": "When was the company founded?"},
    )

    assert founded_response.status_code == 200
    founded_data = founded_response.json()

    assert founded_data["sources"]
    assert any(
        source.get("source") == "company.txt"
        for source in founded_data["sources"]
    )
    assert "2010" in founded_data["answer"]

    headquarters_response = client.post(
        "/chat",
        json={"query": "Where is the company's headquarters?"},
    )

    assert headquarters_response.status_code == 200
    headquarters_data = headquarters_response.json()

    assert headquarters_data["sources"]
    assert any(
        source.get("source") == "company.txt"
        for source in headquarters_data["sources"]
    )
    assert "Berlin" in headquarters_data["answer"]