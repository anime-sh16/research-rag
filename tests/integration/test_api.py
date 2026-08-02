from unittest.mock import AsyncMock, MagicMock

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


def _mock_retriever() -> MagicMock:
    mock = MagicMock()
    mock.retrieve = AsyncMock(return_value=FAKE_CHUNKS)
    return mock


def _mock_chain() -> MagicMock:
    mock = MagicMock()
    mock.generate = AsyncMock(return_value=FAKE_ANSWER)
    return mock


@pytest.fixture
def client():
    """TestClient with app.state.retriever/chain (normally set by lifespan) stubbed
    directly — bare TestClient(app) never triggers FastAPI's lifespan, so nothing
    else would populate them."""
    from src.api.main import app

    app.state.retriever = _mock_retriever()
    app.state.chain = _mock_chain()

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

    def test_empty_question_returns_422(self, client: TestClient) -> None:
        response = client.post("/query", json={"question": ""})
        assert response.status_code == 422

    def test_whitespace_only_question_returns_422(self, client: TestClient) -> None:
        response = client.post("/query", json={"question": "   "})
        assert response.status_code == 422

    def test_too_short_question_returns_422(self, client: TestClient) -> None:
        response = client.post("/query", json={"question": "ab"})
        assert response.status_code == 422

    def test_oversized_question_returns_422(self, client: TestClient) -> None:
        response = client.post("/query", json={"question": "x" * 1001})
        assert response.status_code == 422

    def test_valid_length_question_returns_200(self, client: TestClient) -> None:
        response = client.post("/query", json={"question": "What is attention?"})
        assert response.status_code == 200

    def test_retriever_called_with_question(self) -> None:
        from src.api.main import app

        app.state.retriever = _mock_retriever()
        app.state.chain = _mock_chain()

        c = TestClient(app)
        c.post("/query", json={"question": "What is LoRA?"})
        app.state.retriever.retrieve.assert_called_once_with("What is LoRA?")

    def test_empty_retrieval_returns_fallback_answer(self, client: TestClient) -> None:
        """run_pipeline short-circuits with a fallback message when retrieval is empty."""
        from src.api.main import app

        app.state.retriever.retrieve = AsyncMock(return_value=[])
        response = client.post("/query", json={"question": "Unknown topic?"})
        assert response.status_code == 200
        assert "don't have enough context" in response.json()["answer"].lower()
        assert response.json()["sources"] == []

    def test_off_domain_query_returns_friendly_message(
        self, client: TestClient
    ) -> None:
        from src.api.main import app
        from src.retrieval.retriever import OffDomainQuery

        app.state.retriever.retrieve = AsyncMock(side_effect=OffDomainQuery())
        response = client.post(
            "/query", json={"question": "What is the weather in Tokyo?"}
        )
        assert response.status_code == 200
        assert response.json()["sources"] == []
        assert "knowledge base" in response.json()["answer"].lower()

    def test_chain_called_with_question_and_chunks(self) -> None:
        from src.api.main import app

        app.state.retriever = _mock_retriever()
        app.state.chain = _mock_chain()

        c = TestClient(app)
        c.post("/query", json={"question": "What is LoRA?"})
        app.state.chain.generate.assert_called_once_with(
            "What is LoRA?", FAKE_CHUNKS, prompt_version=None
        )

    def test_upstream_timeout_returns_503_structured(self, client: TestClient) -> None:
        from src.api.main import app

        app.state.retriever.retrieve = AsyncMock(
            side_effect=TimeoutError("upstream slow")
        )
        response = client.post("/query", json={"question": "What is attention?"})
        assert response.status_code == 503
        body = response.json()["detail"]
        assert body["error"] == "service_unavailable"
        assert "upstream slow" not in str(body)

    def test_unexpected_error_returns_500_structured(self, client: TestClient) -> None:
        from src.api.main import app

        app.state.retriever.retrieve = AsyncMock(
            side_effect=ValueError("SECRET-STACK-TRACE")
        )
        response = client.post("/query", json={"question": "What is attention?"})
        assert response.status_code == 500
        body = response.json()["detail"]
        assert body["error"] == "internal_error"
        assert "SECRET-STACK-TRACE" not in str(response.json())


class TestRateLimiting:
    def test_sixth_request_per_minute_is_rate_limited(self) -> None:
        from src.api.main import app, limiter

        limiter.enabled = True
        app.state.retriever = _mock_retriever()
        app.state.chain = _mock_chain()
        try:
            c = TestClient(app)
            statuses = [
                c.post("/query", json={"question": "What is attention?"}).status_code
                for _ in range(6)
            ]
        finally:
            limiter.enabled = False
        assert statuses[:5] == [200, 200, 200, 200, 200]
        assert statuses[5] == 429


class TestHealthEndpoint:
    def test_returns_200(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200

    def test_payload_has_status_ok(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.json()["status"] == "ok"

    def test_payload_includes_pipeline_version(self, client: TestClient) -> None:
        from src.config.config import settings

        response = client.get("/health")
        assert response.json()["pipeline_version"] == settings.pipeline_version

    def test_payload_includes_git_sha(self, client: TestClient) -> None:
        from src.config.config import settings

        response = client.get("/health")
        assert response.json()["git_sha"] == settings.git_sha
