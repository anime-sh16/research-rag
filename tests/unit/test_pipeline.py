from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.ingestion.arxiv_client import ArxivResult
from src.ingestion.chunker import ChunkMetaData
from src.ingestion.pipeline import IngestionRunSummary, TopicIngestionStats

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_arxiv_result(
    paper_id: str = "1234", topic: str = "transformers"
) -> ArxivResult:
    return ArxivResult(
        entry_id=paper_id,
        title="Test Paper",
        topic=topic,
        published=datetime(2022, 1, 1),
        summary="A test summary.",
        authors=["Author A"],
        comment=None,
        primary_category="cs.AI",
        categories=["cs.AI"],
        pdf_url=None,
        full_text=None,
    )


def _make_chunk(paper_id: str = "1234", index: int = 0) -> ChunkMetaData:
    return ChunkMetaData(
        chunk_id=f"{paper_id}_Test_Paper_{index}",
        chunk_index=index,
        topic="transformers",
        paper_id=paper_id,
        title="Test Paper",
        comment=None,
        authors=["Author A"],
        primary_category="cs.AI",
        categories=["cs.AI"],
        published=datetime(2022, 1, 1),
        source_text="Some chunk text.",
    )


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def pipeline():
    """SimpleIngestionPipeline with all external dependencies patched."""
    with (
        patch("src.ingestion.pipeline.ArxivClient") as MockArxiv,
        patch("src.ingestion.pipeline.BasicChunker") as MockChunker,
        patch("src.ingestion.pipeline.VectorStore") as MockVectorStore,
    ):
        from src.ingestion.pipeline import SimpleIngestionPipeline

        instance = SimpleIngestionPipeline(topics=["transformers"], target_papers_no=3)

        instance._mock_arxiv = MockArxiv.return_value
        instance._mock_chunker = MockChunker.return_value
        instance._mock_vector_store = MockVectorStore.return_value

        yield instance


# ---------------------------------------------------------------------------
# TestFetchPaperSingleTopic
# ---------------------------------------------------------------------------


class TestFetchPaperSingleTopic:
    def test_returns_topic_ingestion_stats(self, pipeline) -> None:
        pipeline._mock_arxiv.get_arxiv_results.return_value = []
        stats, _ = pipeline.fetch_paper_single_topic("transformers")
        assert isinstance(stats, TopicIngestionStats)

    def test_returns_selected_papers_list(self, pipeline) -> None:
        pipeline._mock_arxiv.get_arxiv_results.return_value = []
        _, papers = pipeline.fetch_paper_single_topic("transformers")
        assert isinstance(papers, list)

    def test_fetched_count_matches_api_response(self, pipeline) -> None:
        papers = [_make_arxiv_result(f"id_{i}") for i in range(5)]
        pipeline._mock_arxiv.get_arxiv_results.return_value = papers
        stats, _ = pipeline.fetch_paper_single_topic("transformers")
        assert stats.fetched == 5

    def test_selected_respects_per_topic_cap(self, pipeline) -> None:
        # pipeline has target_papers_no=3, return 10 papers
        papers = [_make_arxiv_result(f"id_{i}") for i in range(10)]
        pipeline._mock_arxiv.get_arxiv_results.return_value = papers
        stats, _ = pipeline.fetch_paper_single_topic("transformers")
        assert stats.selected == 3

    def test_duplicate_entry_id_is_skipped(self, pipeline) -> None:
        pipeline.seen_ids.add("dup_id")
        papers = [_make_arxiv_result("dup_id"), _make_arxiv_result("new_id")]
        pipeline._mock_arxiv.get_arxiv_results.return_value = papers
        stats, selected = pipeline.fetch_paper_single_topic("transformers")
        selected_ids = [p.entry_id for p in selected]
        assert "dup_id" not in selected_ids
        assert "new_id" in selected_ids
        assert stats.selected == 1
        assert stats.fetched == 2

    def test_fetch_returns_all_selected_papers(self, pipeline) -> None:
        papers = [_make_arxiv_result("aaa"), _make_arxiv_result("bbb")]
        pipeline._mock_arxiv.get_arxiv_results.return_value = papers
        _, selected = pipeline.fetch_paper_single_topic("transformers")
        assert len(selected) == 2
        assert {p.entry_id for p in selected} == {"aaa", "bbb"}

    def test_seen_ids_updated_after_fetch(self, pipeline) -> None:
        papers = [_make_arxiv_result("aaa"), _make_arxiv_result("bbb")]
        pipeline._mock_arxiv.get_arxiv_results.return_value = papers
        pipeline.fetch_paper_single_topic("transformers")
        assert "aaa" in pipeline.seen_ids
        assert "bbb" in pipeline.seen_ids

    def test_seen_ids_shared_across_topic_calls(self, pipeline) -> None:
        pipeline._mock_arxiv.get_arxiv_results.return_value = [
            _make_arxiv_result("shared_id")
        ]
        _, papers_a = pipeline.fetch_paper_single_topic("topic_a")
        _, papers_b = pipeline.fetch_paper_single_topic("topic_b")
        all_selected = papers_a + papers_b
        selected_ids = [p.entry_id for p in all_selected]
        assert selected_ids.count("shared_id") == 1

    def test_chunks_field_defaults_to_zero(self, pipeline) -> None:
        pipeline._mock_arxiv.get_arxiv_results.return_value = []
        stats, _ = pipeline.fetch_paper_single_topic("transformers")
        assert stats.chunks == 0


# ---------------------------------------------------------------------------
# TestChunkSingleTopic
# ---------------------------------------------------------------------------


class TestChunkSingleTopic:
    def test_delegates_to_chunker(self, pipeline) -> None:
        papers = [_make_arxiv_result("aaa")]
        expected = [_make_chunk("aaa")]
        pipeline._mock_chunker.chunk_all_results.return_value = expected
        result = pipeline.chunk_single_topic(papers)
        pipeline._mock_chunker.chunk_all_results.assert_called_once_with(papers)
        assert result == expected

    def test_empty_papers_returns_empty_list(self, pipeline) -> None:
        pipeline._mock_chunker.chunk_all_results.return_value = []
        result = pipeline.chunk_single_topic([])
        assert result == []


# ---------------------------------------------------------------------------
# TestProcessSingleTopic
# ---------------------------------------------------------------------------


class TestProcessSingleTopic:
    def test_returns_tuple_of_stats_and_chunks(self, pipeline) -> None:
        pipeline._mock_arxiv.get_arxiv_results.return_value = [
            _make_arxiv_result("aaa")
        ]
        pipeline._mock_chunker.chunk_all_results.return_value = [_make_chunk("aaa")]
        result = pipeline.process_single_topic("transformers")
        assert isinstance(result, tuple) and len(result) == 2
        assert isinstance(result[0], TopicIngestionStats)
        assert isinstance(result[1], list)

    def test_chunks_field_populated_on_stats(self, pipeline) -> None:
        pipeline._mock_arxiv.get_arxiv_results.return_value = [
            _make_arxiv_result("aaa")
        ]
        chunks = [_make_chunk("aaa", 0), _make_chunk("aaa", 1)]
        pipeline._mock_chunker.chunk_all_results.return_value = chunks
        stats, _ = pipeline.process_single_topic("transformers")
        assert stats.chunks == 2


# ---------------------------------------------------------------------------
# TestProcess
# ---------------------------------------------------------------------------


class TestProcess:
    @pytest.fixture
    def patched_pipeline(self, pipeline):
        """Additionally patches file I/O and logging so process() doesn't touch disk."""
        pipeline.save_chunks_to_jsonl = MagicMock()
        pipeline.save_summary_to_json = MagicMock()
        with patch("src.ingestion.pipeline.setup_ingestion_logging") as mock_logging:
            mock_log_handler = MagicMock()
            mock_logging.return_value = mock_log_handler
            pipeline._mock_logging = mock_logging
            yield pipeline

    def test_returns_ingestion_run_summary(self, patched_pipeline) -> None:
        patched_pipeline._mock_arxiv.get_arxiv_results.return_value = []
        patched_pipeline._mock_chunker.chunk_all_results.return_value = []
        result = patched_pipeline.process()
        assert isinstance(result, IngestionRunSummary)

    def test_summary_topic_count_matches_topics(self, patched_pipeline) -> None:
        patched_pipeline._mock_arxiv.get_arxiv_results.return_value = []
        patched_pipeline._mock_chunker.chunk_all_results.return_value = []
        summary = patched_pipeline.process()
        assert len(summary.topics) == len(patched_pipeline.topics)

    def test_summary_totals_are_correct(self, patched_pipeline) -> None:
        papers = [_make_arxiv_result("aaa"), _make_arxiv_result("bbb")]
        chunks = [_make_chunk("aaa"), _make_chunk("bbb")]
        patched_pipeline._mock_arxiv.get_arxiv_results.return_value = papers
        patched_pipeline._mock_chunker.chunk_all_results.return_value = chunks
        summary = patched_pipeline.process()
        assert summary.total_papers == 2
        assert summary.total_chunks == 2

    def test_upserts_chunks_per_topic(self, patched_pipeline) -> None:
        chunks = [_make_chunk("aaa", 0), _make_chunk("bbb", 0)]
        patched_pipeline._mock_arxiv.get_arxiv_results.return_value = []
        patched_pipeline._mock_chunker.chunk_all_results.return_value = chunks
        patched_pipeline.process()
        # One topic → upsert called once with that topic's chunks
        patched_pipeline._mock_vector_store.upsert_chunks.assert_called_once_with(
            chunks
        )

    def test_ensures_collection_before_upsert(self, patched_pipeline) -> None:
        call_order = []
        patched_pipeline._mock_vector_store.ensure_collection.side_effect = (
            lambda **kw: call_order.append("ensure")
        )
        patched_pipeline._mock_vector_store.upsert_chunks.side_effect = lambda c: (
            call_order.append("upsert")
        )
        patched_pipeline._mock_arxiv.get_arxiv_results.return_value = []
        patched_pipeline._mock_chunker.chunk_all_results.return_value = []
        patched_pipeline.process()
        assert call_order == ["ensure", "upsert"]

    def test_save_methods_called_once_each(self, patched_pipeline) -> None:
        patched_pipeline._mock_arxiv.get_arxiv_results.return_value = []
        patched_pipeline._mock_chunker.chunk_all_results.return_value = []
        patched_pipeline.process()
        patched_pipeline.save_chunks_to_jsonl.assert_called_once()
        patched_pipeline.save_summary_to_json.assert_called_once()
