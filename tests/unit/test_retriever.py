import json
from unittest.mock import MagicMock, patch

import pytest

from src.config.config import settings

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_qdrant_point(
    score: float = 0.9,
    title: str = "Paper A",
    paper_id: str = "1234",
    chunk_index: int = 0,
    authors: list | None = None,
    source_text: str = "Some text.",
) -> MagicMock:
    point = MagicMock()
    point.score = score
    point.payload = {
        "title": title,
        "paper_id": paper_id,
        "chunk_index": chunk_index,
        "authors": authors or ["Author A"],
        "source_text": source_text,
    }
    return point


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def retriever(tmp_path):
    """Retriever with all external dependencies patched."""
    mock_gemini = MagicMock()

    with (
        patch("src.retrieval.retriever.QdrantClient") as MockQdrant,
        patch("src.retrieval.retriever.genai.Client", return_value=mock_gemini),
        patch("src.retrieval.retriever.wrappers.wrap_gemini", return_value=mock_gemini),
        patch("src.retrieval.retriever.Client"),  # LangSmith Client
        patch(
            "src.retrieval.retriever.QUERY_CACHE_FILE",
            tmp_path / "query_cache.jsonl",
        ),
    ):
        from src.retrieval.retriever import Retriever

        instance = Retriever(top_k=3)
        instance._mock_qdrant = MockQdrant.return_value
        instance._mock_gemini = mock_gemini
        yield instance


# ---------------------------------------------------------------------------
# TestLoadCache
# ---------------------------------------------------------------------------


class TestLoadCache:
    def test_returns_empty_dict_when_no_cache_file(self, retriever) -> None:
        # Cache file does not exist (tmp_path is clean)
        retriever._cache = retriever._load_cache()
        assert isinstance(retriever._cache, dict)

    def test_loads_entries_from_existing_cache(self, tmp_path) -> None:
        cache_file = tmp_path / "query_cache.jsonl"
        entry = {
            "query": "test query",
            "hash": "abc123",
            "embedding": [0.1, 0.2, 0.3],
        }
        cache_file.write_text(json.dumps(entry) + "\n")

        mock_gemini = MagicMock()
        with (
            patch("src.retrieval.retriever.QdrantClient"),
            patch("src.retrieval.retriever.genai.Client", return_value=mock_gemini),
            patch(
                "src.retrieval.retriever.wrappers.wrap_gemini",
                return_value=mock_gemini,
            ),
            patch("src.retrieval.retriever.Client"),
            patch("src.retrieval.retriever.QUERY_CACHE_FILE", cache_file),
        ):
            from src.retrieval.retriever import Retriever

            r = Retriever(top_k=3)
        assert "abc123" in r._cache
        assert r._cache["abc123"] == [0.1, 0.2, 0.3]

    def test_handles_corrupt_cache_gracefully(self, tmp_path) -> None:
        cache_file = tmp_path / "query_cache.jsonl"
        cache_file.write_text("not valid json\n")

        mock_gemini = MagicMock()
        with (
            patch("src.retrieval.retriever.QdrantClient"),
            patch("src.retrieval.retriever.genai.Client", return_value=mock_gemini),
            patch(
                "src.retrieval.retriever.wrappers.wrap_gemini",
                return_value=mock_gemini,
            ),
            patch("src.retrieval.retriever.Client"),
            patch("src.retrieval.retriever.QUERY_CACHE_FILE", cache_file),
        ):
            from src.retrieval.retriever import Retriever

            r = Retriever(top_k=3)
        # Should not raise; cache is empty
        assert r._cache == {}


# ---------------------------------------------------------------------------
# TestGetQueryVector
# ---------------------------------------------------------------------------


class TestGetQueryVector:
    def test_cache_hit_skips_embed_call(self, retriever) -> None:
        import hashlib

        cached_vector = [0.5] * settings.db.embedding_dimension
        query = "test query"
        query_hash = hashlib.md5(query.encode()).hexdigest()
        retriever._cache[query_hash] = cached_vector

        result = retriever._get_query_vector(query)

        retriever._mock_gemini.models.embed_content.assert_not_called()
        assert result == cached_vector

    def test_cache_miss_triggers_embed(self, retriever) -> None:
        fake_embedding = MagicMock()
        fake_embedding.values = [0.1] * settings.db.embedding_dimension
        retriever._mock_gemini.models.embed_content.return_value.embeddings = [
            fake_embedding
        ]

        retriever._get_query_vector("fresh query")

        retriever._mock_gemini.models.embed_content.assert_called_once()

    def test_result_is_stored_in_cache_after_embed(self, retriever) -> None:
        import hashlib

        query = "brand new query"
        query_hash = hashlib.md5(query.encode()).hexdigest()
        fake_embedding = MagicMock()
        fake_embedding.values = [0.2] * settings.db.embedding_dimension
        retriever._mock_gemini.models.embed_content.return_value.embeddings = [
            fake_embedding
        ]

        retriever._get_query_vector(query)

        assert query_hash in retriever._cache


# ---------------------------------------------------------------------------
# TestRetrieve
# ---------------------------------------------------------------------------


class TestRetrieve:
    @pytest.fixture(autouse=True)
    def _patch_langsmith_run(self):
        """Prevent LangSmith tracing from making external calls."""
        with patch("src.retrieval.retriever.get_current_run_tree", return_value=None):
            yield

    def _setup_embed(self, retriever) -> None:
        fake_embedding = MagicMock()
        fake_embedding.values = [0.1] * settings.db.embedding_dimension
        retriever._mock_gemini.models.embed_content.return_value.embeddings = [
            fake_embedding
        ]

    def test_returns_list_of_dicts(self, retriever) -> None:
        self._setup_embed(retriever)
        point = _fake_qdrant_point()
        retriever._mock_qdrant.query_points.return_value.points = [point]

        result = retriever.retrieve("What is attention?")

        assert isinstance(result, list)
        assert all(isinstance(c, dict) for c in result)

    def test_chunk_fields_are_present(self, retriever) -> None:
        self._setup_embed(retriever)
        point = _fake_qdrant_point(title="Transformer", paper_id="1706")
        retriever._mock_qdrant.query_points.return_value.points = [point]

        chunks = retriever.retrieve("What is a transformer?")

        assert len(chunks) == 1
        chunk = chunks[0]
        for field in ("score", "text", "title", "paper_id", "chunk_index", "authors"):
            assert field in chunk

    def test_score_maps_correctly(self, retriever) -> None:
        self._setup_embed(retriever)
        point = _fake_qdrant_point(score=0.88)
        retriever._mock_qdrant.query_points.return_value.points = [point]

        chunks = retriever.retrieve("query")

        assert chunks[0]["score"] == pytest.approx(0.88)

    def test_empty_qdrant_response_returns_empty_list(self, retriever) -> None:
        self._setup_embed(retriever)
        retriever._mock_qdrant.query_points.return_value.points = []

        result = retriever.retrieve("obscure query")

        assert result == []

    def test_query_calls_qdrant_with_correct_limit(self, retriever) -> None:
        self._setup_embed(retriever)
        retriever._mock_qdrant.query_points.return_value.points = []

        retriever.retrieve("query")

        call_kwargs = retriever._mock_qdrant.query_points.call_args
        assert call_kwargs.kwargs.get("limit") == retriever.top_k

    def test_result_count_matches_qdrant_points(self, retriever) -> None:
        self._setup_embed(retriever)
        points = [_fake_qdrant_point(paper_id=str(i)) for i in range(3)]
        retriever._mock_qdrant.query_points.return_value.points = points

        chunks = retriever.retrieve("query")

        assert len(chunks) == 3


# ---------------------------------------------------------------------------
# TestTracingDoesNotMutateChunks
# ---------------------------------------------------------------------------


class TestTracingDoesNotMutateChunks:
    """Tracing truncation must not mutate the chunks returned to the caller."""

    def _setup_embed(self, retriever) -> None:
        fake_embedding = MagicMock()
        fake_embedding.values = [0.1] * settings.db.embedding_dimension
        retriever._mock_gemini.models.embed_content.return_value.embeddings = [
            fake_embedding
        ]

    def test_returned_chunks_have_full_text_when_tracing_active(
        self, retriever
    ) -> None:
        self._setup_embed(retriever)
        long_text = "A" * 500
        point = _fake_qdrant_point(source_text=long_text)
        retriever._mock_qdrant.query_points.return_value.points = [point]

        mock_run = MagicMock()
        with patch(
            "src.retrieval.retriever.get_current_run_tree", return_value=mock_run
        ):
            chunks = retriever.retrieve("query")

        assert len(chunks[0]["text"]) == 500
        assert chunks[0]["text"] == long_text


# ---------------------------------------------------------------------------
# TestEmbedQuery
# ---------------------------------------------------------------------------


class TestEmbedQuery:
    def test_raises_descriptive_error_on_empty_embeddings(self, retriever) -> None:
        """Empty embeddings from Gemini should raise ValueError, not IndexError."""
        retriever._mock_gemini.models.embed_content.return_value.embeddings = []

        with pytest.raises(ValueError, match="empty embeddings"):
            retriever._embed_query("test query")
