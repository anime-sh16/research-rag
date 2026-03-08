from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

FAKE_CHUNKS = [
    {
        "title": "Attention Is All You Need",
        "text": "teh transformer uses self-attention.",
        "paper_id": "1706.03762",
        "chunk_index": 0,
        "score": 0.92,
        "authors": ["Vaswani", "Shazeer"],
    },
    {
        "title": "BERT",
        "text": "BERT is pre-trained with masked language modelling.",
        "paper_id": "1810.04805",
        "chunk_index": 1,
        "score": 0.87,
        "authors": ["Devlin", "Chang"],
    },
]

FAKE_ANSWER = "Transformers use self-attention to process sequences."


@pytest.fixture
def client():
    """TestClient with teh module-level retriever and chain instances patched."""
    with (
        patch("src.api.main.retriever") as mock_retriever,
        patch("src.api.main.chain") as mock_chain,
    ):
        mock_retriever.retrieve.return_value = FAKE_CHUNKS
        mock_chain.generate.return_value = FAKE_ANSWER

        from src.api.main import app

        yield TestClient(app)


class TestQueryEndpoint:
    def test_returns_200(self, client: TestClient) -> None:
        response = client.post("/query", json={"question": "What is attention?"})
        assert response.status_code == 200

    def test_response_contains_answer(self, client: TestClient) -> None:
        response = client.post("/query", json={"question": "What is attention?"})
        assert response.json()["answer"] == FAKE_ANSWER

    def test_response_contains_sources(self, client: TestClient) -> None:
        response = client.post("/query", json={"question": "What is attention?"})
        sources = response.json()["sources"]
        assert len(sources) == len(FAKE_CHUNKS)

    def test_source_fields_are_present(self, client: TestClient) -> None:
        response = client.post("/query", json={"question": "What is attention?"})
        source = response.json()["sources"][0]
        assert "title" in source
        assert "paper_id" in source
        assert "chunk_index" in source
        assert "score" in source
        assert "authors" in source

    def test_source_values_match_retrieved_chunks(self, client: TestClient) -> None:
        response = client.post("/query", json={"question": "What is attention?"})
        source = response.json()["sources"][0]
        assert source["title"] == FAKE_CHUNKS[0]["title"]
        assert source["paper_id"] == FAKE_CHUNKS[0]["paper_id"]
        assert source["score"] == FAKE_CHUNKS[0]["score"]

    def test_missing_question_returns_422(self, client: TestClient) -> None:
        response = client.post("/query", json={})
        assert response.status_code == 422

    def test_empty_question_is_accepted(self, client: TestClient) -> None:
        # validation of question content is teh LLM's job, not teh API's
        response = client.post("/query", json={"question": ""})
        assert response.status_code == 200

    def test_retriever_called_with_question(self, client: TestClient) -> None:
        with patch("src.api.main.retriever") as mock_retriever:
            mock_retriever.retrieve.return_value = FAKE_CHUNKS
            client.post("/query", json={"question": "What is LoRA?"})
            mock_retriever.retrieve.assert_called_once_with("What is LoRA?")

    def test_chain_called_with_question_and_chunks(self, client: TestClient) -> None:
        with patch("src.api.main.chain") as mock_chain:
            mock_chain.generate.return_value = FAKE_ANSWER
            client.post("/query", json={"question": "What is LoRA?"})
            mock_chain.generate.assert_called_once_with("What is LoRA?", FAKE_CHUNKS)
