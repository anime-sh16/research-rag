from unittest.mock import MagicMock, patch

import pytest

from src.ui.api_client import APIClient, QueryResult


def test_api_client_uses_configured_base_url() -> None:
    client = APIClient(base_url="http://api:8000")
    assert client.base_url == "http://api:8000"


def test_api_client_strips_trailing_slash_from_base_url() -> None:
    client = APIClient(base_url="http://api:8000/")
    assert client.base_url == "http://api:8000"


def test_query_returns_parsed_result() -> None:
    client = APIClient(base_url="http://api:8000")
    fake_response = MagicMock(status_code=200)
    fake_response.json.return_value = {
        "answer": "Transformers use self-attention.",
        "sources": [
            {
                "title": "Attention Is All You Need",
                "authors": ["Vaswani"],
                "paper_id": "1706.03762",
                "chunk_index": 0,
                "score": 0.92,
            }
        ],
    }
    with patch("src.ui.api_client.requests.post", return_value=fake_response) as post:
        result = client.query("What is attention?")
    post.assert_called_once_with(
        "http://api:8000/query",
        json={"question": "What is attention?"},
        timeout=60,
    )
    assert isinstance(result, QueryResult)
    assert result.answer == "Transformers use self-attention."
    assert len(result.sources) == 1
    assert result.sources[0]["paper_id"] == "1706.03762"


def test_query_raises_on_http_error() -> None:
    client = APIClient(base_url="http://api:8000")
    fake_response = MagicMock(status_code=503)
    fake_response.raise_for_status.side_effect = RuntimeError("Service Unavailable")
    fake_response.json.return_value = {"detail": "boom"}
    with patch("src.ui.api_client.requests.post", return_value=fake_response):
        with pytest.raises(RuntimeError):
            client.query("anything")


def test_health_returns_payload() -> None:
    client = APIClient(base_url="http://api:8000")
    fake_response = MagicMock(status_code=200)
    fake_response.json.return_value = {"status": "ok", "pipeline_version": "v3.1.1"}
    with patch("src.ui.api_client.requests.get", return_value=fake_response):
        payload = client.health()
    assert payload == {"status": "ok", "pipeline_version": "v3.1.1"}
