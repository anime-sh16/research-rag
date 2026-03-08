import os
from datetime import datetime
from unittest.mock import MagicMock, patch

import fitz
import pytest

from src.ingestion.arxiv_client import ArxivClient, ArxivResult


def _make_mock_raw_result(
    entry_id="https://arxiv.org/abs/1706.03762v5",
    title="Attention Is All You Need",
    published=datetime(2017, 6, 12),
    summary="We propose the Transformer architecture.",
    authors=None,
    comment="9 pages",
    primary_category="cs.CL",
    categories=["cs.CL", "cs.LG"],
) -> MagicMock:
    """Fake arxiv.Result with only the fields _parse_arxiv_result accesses."""
    mock = MagicMock()
    mock.entry_id = entry_id
    mock.title = title
    mock.published = published
    mock.summary = summary
    mock.authors = authors or []
    mock.comment = comment
    mock.primary_category = primary_category
    mock.categories = categories
    return mock


@pytest.fixture
def client() -> ArxivClient:
    with patch("src.ingestion.arxiv_client.arxiv.Client"):
        return ArxivClient()


class TestParseArxivResult:
    def test_entry_id_strips_url_prefix_and_version(self, client: ArxivClient) -> None:
        raw = _make_mock_raw_result(entry_id="https://arxiv.org/abs/1706.03762v5")
        result = client._parse_arxiv_result(raw, "transformers")
        assert result.entry_id == "1706.03762"

    def test_entry_id_without_version_suffix(self, client: ArxivClient) -> None:
        raw = _make_mock_raw_result(entry_id="https://arxiv.org/abs/1706.03762")
        result = client._parse_arxiv_result(raw, "transformers")
        assert result.entry_id == "1706.03762"

    def test_authors_extracted_by_name(self, client: ArxivClient) -> None:
        a, b = MagicMock(), MagicMock()
        a.name = "Vaswani"
        b.name = "Shazeer"
        raw = _make_mock_raw_result(authors=[a, b])
        result = client._parse_arxiv_result(raw, "transformers")
        assert result.authors == ["Vaswani", "Shazeer"]

    def test_empty_authors_returns_empty_list(self, client: ArxivClient) -> None:
        raw = _make_mock_raw_result(authors=[])
        result = client._parse_arxiv_result(raw, "transformers")
        assert result.authors == []

    def test_comment_preserved_when_present(self, client: ArxivClient) -> None:
        raw = _make_mock_raw_result(comment="9 pages, 5 figures")
        result = client._parse_arxiv_result(raw, "transformers")
        assert result.comment == "9 pages, 5 figures"

    def test_none_comment_stored_as_none(self, client: ArxivClient) -> None:
        raw = _make_mock_raw_result(comment=None)
        result = client._parse_arxiv_result(raw, "transformers")
        assert result.comment is None

    def test_returns_arxiv_result_instance(self, client: ArxivClient) -> None:
        raw = _make_mock_raw_result()
        assert isinstance(client._parse_arxiv_result(raw, "transformers"), ArxivResult)

    def test_topic_stored_from_query(self, client: ArxivClient) -> None:
        raw = _make_mock_raw_result()
        result = client._parse_arxiv_result(raw, "retrieval augmented generation")
        assert result.topic == "retrieval augmented generation"

    def test_scalar_fields_mapped_correctly(self, client: ArxivClient) -> None:
        raw = _make_mock_raw_result(
            title="BERT",
            published=datetime(2018, 10, 11),
            summary="Deep bidirectional transformers.",
            primary_category="cs.CL",
            categories=["cs.CL"],
        )
        result = client._parse_arxiv_result(raw, "transformers")
        assert result.title == "BERT"
        assert result.published == datetime(2018, 10, 11)
        assert result.summary == "Deep bidirectional transformers."
        assert result.primary_category == "cs.CL"
        assert result.categories == ["cs.CL"]


class TestGetArxivResults:
    def test_returns_list_of_arxiv_results(self, client: ArxivClient) -> None:
        client.client.results.return_value = [_make_mock_raw_result()]
        results = client.get_arxiv_results("transformers")
        assert len(results) == 1
        assert isinstance(results[0], ArxivResult)

    def test_empty_api_response_returns_empty_list(self, client: ArxivClient) -> None:
        client.client.results.return_value = []
        assert client.get_arxiv_results("transformers") == []

    def test_result_count_matches_api_output(self, client: ArxivClient) -> None:
        client.client.results.return_value = [_make_mock_raw_result()] * 3
        assert len(client.get_arxiv_results("transformers", max_results=3)) == 3


class TestDownloadAndExtraction:
    def test_download_pdf_success(self, tmp_path, client: ArxivClient) -> None:
        raw = _make_mock_raw_result()
        raw.pdf_url = "http://example.com/foo.pdf"
        # fake response object
        fake_resp = MagicMock()
        fake_resp.iter_content.return_value = [b"abc", b""]
        fake_resp.raise_for_status.return_value = None
        with patch(
            "src.ingestion.arxiv_client.requests.get", return_value=fake_resp
        ) as mock_get:
            path = client._download_pdf_locally(raw, str(tmp_path))
        assert path.startswith(str(tmp_path))
        assert path.endswith("1706.03762.pdf")
        assert path and os.path.exists(path)
        with open(path, "rb") as f:
            assert f.read() == b"abc"
        mock_get.assert_called_with("http://example.com/foo.pdf", stream=True)

    def test_extract_pdf_text_success(self, tmp_path, client: ArxivClient) -> None:
        # create a minimal PDF containing known text
        pdf_file = tmp_path / "test.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Hello world")
        doc.save(str(pdf_file))

        raw = _make_mock_raw_result()
        with patch.object(client, "_download_pdf_locally", return_value=str(pdf_file)):
            text = client._extract_pdf_text(raw)
        assert "Hello world" in text

    def test_extract_pdf_text_failure(self, client: ArxivClient) -> None:
        raw = _make_mock_raw_result()
        with patch.object(
            client, "_download_pdf_locally", side_effect=Exception("oops")
        ):
            text = client._extract_pdf_text(raw)
        assert text is None
