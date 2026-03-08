from datetime import datetime

import pytest

from src.ingestion.arxiv_client import ArxivResult
from src.ingestion.chunker import BasicChunker


@pytest.fixture
def chunker() -> BasicChunker:
    return BasicChunker()


@pytest.fixture
def full_arxiv_result() -> ArxivResult:
    return ArxivResult(
        entry_id="1234",
        title="Test Paper 1",
        topic="transformers",
        published=datetime(2022, 1, 1),
        summary="This is a test summary",
        authors=["John Doe", "Jane Doe 2"],
        comment="This is a test comment",
        primary_category="cs.RO",
        categories=["cs.RO", "cs.LG"],
    )


@pytest.fixture
def sparse_arxiv_result() -> ArxivResult:
    """ArxivResult with optional fields set to None."""
    return ArxivResult(
        entry_id="1234",
        title="Test Paper 1",
        topic=None,
        published=datetime(2022, 1, 1),
        summary="This is a test summary",
        authors=None,
        comment=None,
        primary_category="cs.RO",
        categories=None,
    )


def _make_long_result(topic: str = "cs.LG") -> ArxivResult:
    return ArxivResult(
        entry_id="5678",
        title="Long Paper",
        topic=topic,
        published=datetime(2023, 6, 1),
        summary="word " * 100,
        authors=["Alice"],
        comment=None,
        primary_category="cs.LG",
        categories=["cs.LG"],
    )


class TestChunkResult:
    def test_returns_one_chunk_for_short_text(
        self, chunker: BasicChunker, full_arxiv_result: ArxivResult
    ) -> None:
        chunks = chunker.chunk_result(full_arxiv_result)
        assert len(chunks) == 1

    def test_chunk_metadata_fields(
        self, chunker: BasicChunker, full_arxiv_result: ArxivResult
    ) -> None:
        chunk = chunker.chunk_result(full_arxiv_result)[0]
        assert chunk.chunk_id == "1234_Test_Paper_1_0"
        assert chunk.chunk_index == 0
        assert chunk.paper_id == "1234"
        assert chunk.title == "Test Paper 1"
        assert chunk.topic == "transformers"
        assert chunk.authors == ["John Doe", "Jane Doe 2"]
        assert chunk.primary_category == "cs.RO"
        assert chunk.categories == ["cs.RO", "cs.LG"]
        assert chunk.published == datetime(2022, 1, 1)
        assert chunk.comment == "This is a test comment"

    def test_topic_propagates_to_chunk(
        self, chunker: BasicChunker, full_arxiv_result: ArxivResult
    ) -> None:
        chunk = chunker.chunk_result(full_arxiv_result)[0]
        assert chunk.topic == full_arxiv_result.topic

    def test_none_topic_propagates_to_chunk(
        self, chunker: BasicChunker, sparse_arxiv_result: ArxivResult
    ) -> None:
        chunk = chunker.chunk_result(sparse_arxiv_result)[0]
        assert chunk.topic is None

    def test_source_text_includes_title_and_summary(
        self, chunker: BasicChunker, full_arxiv_result: ArxivResult
    ) -> None:
        chunk = chunker.chunk_result(full_arxiv_result)[0]
        assert "Test Paper 1" in chunk.source_text
        assert "This is a test summary" in chunk.source_text

    def test_source_text_includes_comment_when_present(
        self, chunker: BasicChunker, full_arxiv_result: ArxivResult
    ) -> None:
        chunk = chunker.chunk_result(full_arxiv_result)[0]
        assert "This is a test comment" in chunk.source_text

    def test_source_text_excludes_comment_when_absent(
        self, chunker: BasicChunker, sparse_arxiv_result: ArxivResult
    ) -> None:
        chunk = chunker.chunk_result(sparse_arxiv_result)[0]
        assert "Comment" not in chunk.source_text

    def test_none_authors_defaults_to_empty_list(
        self, chunker: BasicChunker, sparse_arxiv_result: ArxivResult
    ) -> None:
        chunk = chunker.chunk_result(sparse_arxiv_result)[0]
        assert chunk.authors == []

    def test_none_categories_defaults_to_empty_list(
        self, chunker: BasicChunker, sparse_arxiv_result: ArxivResult
    ) -> None:
        chunk = chunker.chunk_result(sparse_arxiv_result)[0]
        assert chunk.categories == []

    def test_none_comment_is_stored_as_none(
        self, chunker: BasicChunker, sparse_arxiv_result: ArxivResult
    ) -> None:
        chunk = chunker.chunk_result(sparse_arxiv_result)[0]
        assert chunk.comment is None

    def test_long_text_produces_multiple_chunks(self) -> None:
        chunker = BasicChunker(chunk_size=50, chunk_overlap=0)
        chunks = chunker.chunk_result(_make_long_result())
        assert len(chunks) > 1

    def test_chunk_indices_are_sequential(self) -> None:
        chunker = BasicChunker(chunk_size=50, chunk_overlap=0)
        chunks = chunker.chunk_result(_make_long_result())
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_chunk_ids_are_unique(self) -> None:
        chunker = BasicChunker(chunk_size=50, chunk_overlap=0)
        chunks = chunker.chunk_result(_make_long_result())
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        assert len(chunk_ids) == len(set(chunk_ids))

    def test_all_chunks_share_same_paper_id(self) -> None:
        chunker = BasicChunker(chunk_size=50, chunk_overlap=0)
        chunks = chunker.chunk_result(_make_long_result())
        assert all(chunk.paper_id == "5678" for chunk in chunks)


class TestChunkAllResults:
    def test_aggregates_chunks_from_all_papers(self, chunker: BasicChunker) -> None:
        paper_a = ArxivResult(
            entry_id="aaa",
            title="Paper A",
            topic="rag",
            published=datetime(2021, 1, 1),
            summary="Short summary for paper A.",
            authors=["Author A"],
            comment=None,
            primary_category="cs.AI",
            categories=["cs.AI"],
        )
        paper_b = ArxivResult(
            entry_id="bbb",
            title="Paper B",
            topic="transformers",
            published=datetime(2021, 2, 1),
            summary="Short summary for paper B.",
            authors=["Author B"],
            comment=None,
            primary_category="cs.CV",
            categories=["cs.CV"],
        )
        chunks = chunker.chunk_all_results([paper_a, paper_b])
        paper_ids = {chunk.paper_id for chunk in chunks}
        assert "aaa" in paper_ids
        assert "bbb" in paper_ids

    def test_empty_input_returns_empty_list(self, chunker: BasicChunker) -> None:
        assert chunker.chunk_all_results([]) == []
