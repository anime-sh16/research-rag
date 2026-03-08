from datetime import datetime
from unittest.mock import MagicMock, call, patch

import pytest

from src.ingestion.arxiv_client import ArxivResult
from src.ingestion.chunker import BasicChunker, ChunkMetaData


def _make_chunk(paper_id: str = "1234", index: int = 0) -> ChunkMetaData:
    return ChunkMetaData(
        chunk_id=f"{paper_id}_Paper_{index}",
        chunk_index=index,
        paper_id=paper_id,
        title="Test Paper",
        comment=None,
        authors=["Author A"],
        primary_category="cs.AI",
        categories=["cs.AI"],
        published=datetime(2022, 1, 1),
        source_text="Some chunk text.",
    )


@pytest.fixture
def pipeline():
    """SimpleIngestionPipeline with all external dependencies patched."""
    with (
        patch("src.ingestion.pipeline.ArxivClient") as MockArxiv,
        patch("src.ingestion.pipeline.BasicChunker") as MockChunker,
        patch("src.ingestion.pipeline.VectorStore") as MockVectorStore,
    ):
        from src.ingestion.pipeline import SimpleIngestionPipeline

        instance = SimpleIngestionPipeline(query="transformers", max_results=5)

        # expose mocks so tests can configure and inspect them
        instance._mock_arxiv = MockArxiv.return_value
        instance._mock_chunker = MockChunker.return_value
        instance._mock_vector_store = MockVectorStore.return_value

        yield instance


class TestProcess:
    def test_calls_arxiv_client_with_query_and_max_results(
        self, pipeline
    ) -> None:
        pipeline._mock_arxiv.get_arxiv_results.return_value = []
        pipeline._mock_chunker.chunk_all_results.return_value = []
        pipeline.process()
        pipeline._mock_arxiv.get_arxiv_results.assert_called_once_with(
            "transformers", max_results=5
        )

    def test_passes_arxiv_results_to_chunker(self, pipeline) -> None:
        fake_results = [MagicMock(spec=ArxivResult)]
        pipeline._mock_arxiv.get_arxiv_results.return_value = fake_results
        pipeline._mock_chunker.chunk_all_results.return_value = []
        pipeline.process()
        pipeline._mock_chunker.chunk_all_results.assert_called_once_with(fake_results)

    def test_upserts_chunks_to_vector_store(self, pipeline) -> None:
        chunks = [_make_chunk("aaa", 0), _make_chunk("aaa", 1)]
        pipeline._mock_arxiv.get_arxiv_results.return_value = []
        pipeline._mock_chunker.chunk_all_results.return_value = chunks
        pipeline.process()
        pipeline._mock_vector_store.upsert_chunks.assert_called_once_with(chunks)

    def test_returns_all_chunks(self, pipeline) -> None:
        chunks = [_make_chunk("aaa", 0), _make_chunk("bbb", 0)]
        pipeline._mock_arxiv.get_arxiv_results.return_value = []
        pipeline._mock_chunker.chunk_all_results.return_value = chunks
        result = pipeline.process()
        assert result == chunks

    def test_ensures_collection_before_upsert(self, pipeline) -> None:
        call_order = []
        pipeline._mock_vector_store.ensure_collection.side_effect = lambda **kw: call_order.append("ensure")
        pipeline._mock_vector_store.upsert_chunks.side_effect = lambda chunks: call_order.append("upsert")
        pipeline._mock_arxiv.get_arxiv_results.return_value = []
        pipeline._mock_chunker.chunk_all_results.return_value = []
        pipeline.process()
        assert call_order == ["ensure", "upsert"]

    def test_empty_results_still_calls_upsert(self, pipeline) -> None:
        pipeline._mock_arxiv.get_arxiv_results.return_value = []
        pipeline._mock_chunker.chunk_all_results.return_value = []
        pipeline.process()
        pipeline._mock_vector_store.upsert_chunks.assert_called_once_with([])
