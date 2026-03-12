import json
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.ingestion.arxiv_client import ArxivResult
from src.ingestion.chunker import ChunkMetaData
from src.ingestion.pipeline import IngestionRunSummary, TopicIngestionStats


def _make_arxiv_result(
    paper_id: str = "1234",
    topic: str = "transformers",
    primary_category: str = "cs.AI",
) -> ArxivResult:
    return ArxivResult(
        entry_id=paper_id,
        title="Test Paper",
        topic=topic,
        published=datetime(2022, 1, 1),
        summary="A test summary.",
        authors=["Author A"],
        comment=None,
        primary_category=primary_category,
        categories=[primary_category],
        pdf_url=None,
        full_text=None,
    )


def _make_chunk(
    paper_id: str = "1234", index: int = 0, topic: str = "transformers"
) -> ChunkMetaData:
    return ChunkMetaData(
        chunk_id=f"{paper_id}_Test_Paper_{index}",
        chunk_index=index,
        topic=topic,
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

        instance = SimpleIngestionPipeline(topics=["transformers"], target_papers_no=3)

        instance._mock_arxiv = MockArxiv.return_value
        instance._mock_chunker = MockChunker.return_value
        instance._mock_vector_store = MockVectorStore.return_value

        yield instance


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

    def test_off_domain_paper_is_filtered(self, pipeline) -> None:
        papers = [
            _make_arxiv_result("math_id", primary_category="math.OC"),
            _make_arxiv_result("cs_id", primary_category="cs.LG"),
        ]
        pipeline._mock_arxiv.get_arxiv_results.return_value = papers
        stats, selected = pipeline.fetch_paper_single_topic("transformers")
        selected_ids = [p.entry_id for p in selected]
        assert "math_id" not in selected_ids
        assert "cs_id" in selected_ids
        assert stats.filtered_by_category == 1
        assert stats.selected == 1

    def test_filtered_paper_not_added_to_seen_ids(self, pipeline) -> None:
        papers = [_make_arxiv_result("phys_id", primary_category="hep-ph")]
        pipeline._mock_arxiv.get_arxiv_results.return_value = papers
        pipeline.fetch_paper_single_topic("transformers")
        assert "phys_id" not in pipeline.seen_ids


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


class TestProcess:
    @pytest.fixture
    def patched_pipeline(self, pipeline):
        """Additionally patches file I/O and logging so process() doesn't touch disk."""
        pipeline._save_chunks_to_jsonl = MagicMock()
        pipeline._save_summary_to_json = MagicMock()
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

    def test_save_methods_called_with_correct_data(self, patched_pipeline) -> None:
        papers = [_make_arxiv_result("aaa")]
        chunks = [_make_chunk("aaa")]
        patched_pipeline._mock_arxiv.get_arxiv_results.return_value = papers
        patched_pipeline._mock_chunker.chunk_all_results.return_value = chunks
        patched_pipeline.process()
        chunks_call_args = patched_pipeline._save_chunks_to_jsonl.call_args
        assert chunks_call_args.args[0] == chunks
        summary_call_args = patched_pipeline._save_summary_to_json.call_args
        from src.ingestion.pipeline import IngestionRunSummary

        assert isinstance(summary_call_args.args[0], IngestionRunSummary)


def _make_chunk_dict(
    paper_id: str = "1234",
    index: int = 0,
    topic: str = "transformers",
) -> dict:
    """Return a raw dict that matches ChunkMetaData schema — for writing to JSONL."""
    return {
        "chunk_id": f"{paper_id}_Test_Paper_{index}",
        "chunk_index": index,
        "topic": topic,
        "paper_id": paper_id,
        "title": "Test Paper",
        "comment": None,
        "authors": ["Author A"],
        "primary_category": "cs.AI",
        "categories": ["cs.AI"],
        "published": "2022-01-01T00:00:00",
        "source_text": f"Chunk text for {paper_id} index {index}.",
    }


def _write_jsonl(path: Path, dicts: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for d in dicts:
            f.write(json.dumps(d) + "\n")


class TestLoadChunksFromJsonl:
    """Test the JSONL parsing in _load_chunks_from_jsonl."""

    def test_parses_all_lines_into_chunk_metadata(self, tmp_path) -> None:
        jsonl = tmp_path / "chunks.jsonl"
        _write_jsonl(jsonl, [_make_chunk_dict("aaa", 0), _make_chunk_dict("bbb", 1)])

        from src.ingestion.pipeline import SimpleIngestionPipeline

        chunks = SimpleIngestionPipeline._load_chunks_from_jsonl(jsonl)
        assert len(chunks) == 2
        assert chunks[0].paper_id == "aaa"
        assert chunks[1].paper_id == "bbb"

    def test_skips_blank_lines(self, tmp_path) -> None:
        jsonl = tmp_path / "chunks.jsonl"
        with open(jsonl, "w") as f:
            f.write(json.dumps(_make_chunk_dict("aaa")) + "\n")
            f.write("\n")
            f.write("   \n")
            f.write(json.dumps(_make_chunk_dict("bbb")) + "\n")

        from src.ingestion.pipeline import SimpleIngestionPipeline

        chunks = SimpleIngestionPipeline._load_chunks_from_jsonl(jsonl)
        assert len(chunks) == 2

    def test_raises_on_missing_file(self, tmp_path) -> None:
        from src.ingestion.pipeline import SimpleIngestionPipeline

        with pytest.raises(FileNotFoundError):
            SimpleIngestionPipeline._load_chunks_from_jsonl(tmp_path / "nope.jsonl")

    def test_preserves_source_text_content(self, tmp_path) -> None:
        d = _make_chunk_dict("paper1", 0)
        d["source_text"] = "Specific text about attention mechanisms."
        jsonl = tmp_path / "chunks.jsonl"
        _write_jsonl(jsonl, [d])

        from src.ingestion.pipeline import SimpleIngestionPipeline

        chunks = SimpleIngestionPipeline._load_chunks_from_jsonl(jsonl)
        assert chunks[0].source_text == "Specific text about attention mechanisms."


class TestProcessFromJsonl:
    """Test process_from_jsonl end-to-end logic with real JSONL parsing."""

    @pytest.fixture
    def mock_infra(self):
        with (
            patch("src.ingestion.pipeline.VectorStore") as MockVS,
            patch("src.ingestion.pipeline.setup_ingestion_logging") as MockLog,
        ):
            mock_vs = MockVS.return_value
            MockLog.return_value = MagicMock()
            yield mock_vs

    def test_groups_chunks_by_topic_correctly(self, tmp_path, mock_infra) -> None:
        """Two topics, verify stats split correctly."""
        chunks_data = [
            _make_chunk_dict("p1", 0, topic="LLM"),
            _make_chunk_dict("p1", 1, topic="LLM"),
            _make_chunk_dict("p2", 0, topic="RAG"),
        ]
        jsonl = tmp_path / "chunks.jsonl"
        _write_jsonl(jsonl, chunks_data)

        from src.ingestion.pipeline import SimpleIngestionPipeline

        summary = SimpleIngestionPipeline.process_from_jsonl(jsonl)

        stats_by_topic = {s.topic: s for s in summary.topics}
        assert stats_by_topic["LLM"].chunks == 2
        assert stats_by_topic["LLM"].selected == 1  # 1 unique paper
        assert stats_by_topic["RAG"].chunks == 1
        assert stats_by_topic["RAG"].selected == 1

    def test_counts_unique_papers_not_chunk_count(self, tmp_path, mock_infra) -> None:
        """3 chunks from same paper → 1 paper in stats."""
        chunks_data = [_make_chunk_dict("same_paper", i) for i in range(3)]
        jsonl = tmp_path / "chunks.jsonl"
        _write_jsonl(jsonl, chunks_data)

        from src.ingestion.pipeline import SimpleIngestionPipeline

        summary = SimpleIngestionPipeline.process_from_jsonl(jsonl)
        assert summary.total_papers == 1
        assert summary.total_chunks == 3

    def test_upserts_all_parsed_chunks(self, tmp_path, mock_infra) -> None:
        chunks_data = [_make_chunk_dict("a", 0), _make_chunk_dict("b", 0)]
        jsonl = tmp_path / "chunks.jsonl"
        _write_jsonl(jsonl, chunks_data)

        from src.ingestion.pipeline import SimpleIngestionPipeline

        SimpleIngestionPipeline.process_from_jsonl(jsonl)

        upserted = mock_infra.upsert_chunks.call_args[0][0]
        assert len(upserted) == 2
        assert {c.paper_id for c in upserted} == {"a", "b"}

    def test_null_topic_grouped_as_unknown(self, tmp_path, mock_infra) -> None:
        d = _make_chunk_dict("p1", 0)
        d["topic"] = None
        jsonl = tmp_path / "chunks.jsonl"
        _write_jsonl(jsonl, [d])

        from src.ingestion.pipeline import SimpleIngestionPipeline

        summary = SimpleIngestionPipeline.process_from_jsonl(jsonl)
        assert summary.topics[0].topic == "Unknown"


class TestProcessFromPdfs:
    """Test process_from_pdfs logic — PDF extraction and chunking mocked."""

    @pytest.fixture
    def mock_infra(self):
        with (
            patch("src.ingestion.pipeline.VectorStore") as MockVS,
            patch("src.ingestion.pipeline.ArxivClient") as MockArxiv,
            patch("src.ingestion.pipeline.BasicChunker") as MockChunker,
            patch("src.ingestion.pipeline.setup_ingestion_logging") as MockLog,
            patch(
                "src.ingestion.pipeline.SimpleIngestionPipeline._save_chunks_to_jsonl"
            ),
            patch(
                "src.ingestion.pipeline.SimpleIngestionPipeline._save_summary_to_json"
            ),
        ):
            mock_vs = MockVS.return_value
            MockLog.return_value = MagicMock()
            yield {
                "vector_store": mock_vs,
                "extract_pdf": MockArxiv.extract_text_from_pdf,
                "chunker": MockChunker.return_value,
            }

    def test_topic_from_subdirectory_name(self, tmp_path, mock_infra) -> None:
        """PDF in data/pdfs/LLM/paper.pdf → topic 'LLM'."""
        sub = tmp_path / "LLM"
        sub.mkdir()
        (sub / "paper.pdf").write_bytes(b"fake")

        mock_infra["extract_pdf"].return_value = "Some text"
        mock_infra["chunker"].chunk_result.return_value = [
            _make_chunk(paper_id="paper", index=0)
        ]

        from src.ingestion.pipeline import SimpleIngestionPipeline

        SimpleIngestionPipeline.process_from_pdfs(tmp_path)

        # Check the ArxivResult passed to chunker had topic="LLM"
        call_args = mock_infra["chunker"].chunk_result.call_args[0][0]
        assert call_args.topic == "LLM"

    def test_topic_is_local_for_root_level_pdf(self, tmp_path, mock_infra) -> None:
        (tmp_path / "paper.pdf").write_bytes(b"fake")

        mock_infra["extract_pdf"].return_value = "Some text"
        mock_infra["chunker"].chunk_result.return_value = [
            _make_chunk(paper_id="paper", index=0)
        ]

        from src.ingestion.pipeline import SimpleIngestionPipeline

        SimpleIngestionPipeline.process_from_pdfs(tmp_path)

        call_args = mock_infra["chunker"].chunk_result.call_args[0][0]
        assert call_args.topic == "local"

    def test_paper_id_is_filename_stem(self, tmp_path, mock_infra) -> None:
        (tmp_path / "attention_is_all_you_need.pdf").write_bytes(b"fake")

        mock_infra["extract_pdf"].return_value = "Some text"
        mock_infra["chunker"].chunk_result.return_value = [
            _make_chunk(paper_id="attention_is_all_you_need", index=0)
        ]

        from src.ingestion.pipeline import SimpleIngestionPipeline

        SimpleIngestionPipeline.process_from_pdfs(tmp_path)

        result = mock_infra["chunker"].chunk_result.call_args[0][0]
        assert result.entry_id == "attention_is_all_you_need"
        assert result.title == "attention is all you need"

    def test_skips_pdf_when_extraction_fails(self, tmp_path, mock_infra) -> None:
        (tmp_path / "good.pdf").write_bytes(b"fake")
        (tmp_path / "bad.pdf").write_bytes(b"fake")

        mock_infra["extract_pdf"].side_effect = lambda p: (
            "Text" if "good" in str(p) else None
        )
        good_chunk = _make_chunk(paper_id="good", index=0)
        mock_infra["chunker"].chunk_result.return_value = [good_chunk]

        from src.ingestion.pipeline import SimpleIngestionPipeline

        summary = SimpleIngestionPipeline.process_from_pdfs(tmp_path)

        # Chunker called only once (for the good PDF)
        assert mock_infra["chunker"].chunk_result.call_count == 1
        assert summary.total_chunks == 1
        assert summary.total_papers == 1

    def test_multi_topic_stats_are_separate(self, tmp_path, mock_infra) -> None:
        (tmp_path / "LLM").mkdir()
        (tmp_path / "LLM" / "a.pdf").write_bytes(b"fake")
        (tmp_path / "LLM" / "b.pdf").write_bytes(b"fake")
        (tmp_path / "RAG").mkdir()
        (tmp_path / "RAG" / "c.pdf").write_bytes(b"fake")

        mock_infra["extract_pdf"].return_value = "Text"

        def make_chunks_for_call(result):
            return [
                _make_chunk(paper_id=result.entry_id, index=0, topic=result.topic),
                _make_chunk(paper_id=result.entry_id, index=1, topic=result.topic),
            ]

        mock_infra["chunker"].chunk_result.side_effect = make_chunks_for_call

        from src.ingestion.pipeline import SimpleIngestionPipeline

        summary = SimpleIngestionPipeline.process_from_pdfs(tmp_path)

        stats_by_topic = {s.topic: s for s in summary.topics}
        assert stats_by_topic["LLM"].selected == 2  # 2 unique papers
        assert stats_by_topic["LLM"].chunks == 4  # 2 chunks each
        assert stats_by_topic["RAG"].selected == 1
        assert stats_by_topic["RAG"].chunks == 2
        assert summary.total_chunks == 6

    def test_raises_on_nonexistent_directory(self, tmp_path) -> None:
        with patch("src.ingestion.pipeline.setup_ingestion_logging") as MockLog:
            MockLog.return_value = MagicMock()

            from src.ingestion.pipeline import SimpleIngestionPipeline

            with pytest.raises(NotADirectoryError):
                SimpleIngestionPipeline.process_from_pdfs(tmp_path / "nope")

    def test_raises_when_no_pdfs_found(self, tmp_path) -> None:
        with patch("src.ingestion.pipeline.setup_ingestion_logging") as MockLog:
            MockLog.return_value = MagicMock()

            from src.ingestion.pipeline import SimpleIngestionPipeline

            with pytest.raises(FileNotFoundError):
                SimpleIngestionPipeline.process_from_pdfs(tmp_path)

    def test_upserts_chunks_from_all_pdfs(self, tmp_path, mock_infra) -> None:
        (tmp_path / "a.pdf").write_bytes(b"fake")
        (tmp_path / "b.pdf").write_bytes(b"fake")

        mock_infra["extract_pdf"].return_value = "Text"

        def make_chunks(result):
            return [_make_chunk(paper_id=result.entry_id, index=0)]

        mock_infra["chunker"].chunk_result.side_effect = make_chunks

        from src.ingestion.pipeline import SimpleIngestionPipeline

        SimpleIngestionPipeline.process_from_pdfs(tmp_path)

        upserted = mock_infra["vector_store"].upsert_chunks.call_args[0][0]
        assert len(upserted) == 2
        assert {c.paper_id for c in upserted} == {"a", "b"}
