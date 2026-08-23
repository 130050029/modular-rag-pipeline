"""
API-level tests: use FastAPI's TestClient to hit real endpoints, including
the lifespan startup/shutdown. `with TestClient(...) as client:` is what
actually triggers the lifespan context manager -- a plain
TestClient(app).get(...) without the `with` block would skip startup
entirely and is a common mistake.
"""

import pytest
from fastapi.testclient import TestClient

import server as server


@pytest.fixture
def client(temp_db, fresh_vector_index, fresh_near_dedup_index):
    # Explicitly depends on the reset fixtures above so they run BEFORE the
    # app's lifespan startup executes (autouse ordering isn't guaranteed
    # relative to a fixture another fixture requests indirectly).
    with TestClient(server.app) as c:
        yield c


def test_upload_and_chat_flow(client, fake_embeddings):
    upload_response = client.post(
        "/upload", files={"file": ("doc.txt", b"Paris is the capital of France.", "text/plain")}
    )
    assert upload_response.status_code == 200
    assert upload_response.json()["status"] == "ingested"

    chat_response = client.post("/chat", json={"query": "What is the capital of France?"})
    assert chat_response.status_code == 200
    assert "sources" in chat_response.json()


def test_chat_with_no_documents_returns_helpful_message(client):
    response = client.post("/chat", json={"query": "anything"})
    assert response.status_code == 200
    assert response.json()["sources"] == []
