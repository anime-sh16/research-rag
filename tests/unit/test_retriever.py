import json
from unittest.mock import MagicMock, patch

import pytest

from src.config.config import settings


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


def _make_jina_response(
    candidates: list[dict], scores: list[float] | None = None
) -> dict:
    """Build a fake Jina rerank API response that returns all candidates in order."""
    if scores is None:
        scores = [0.9 - i * 0.05 for i in range(len(candidates))]
    return {
        "results": [
            {"index": i, "relevance_score": score} for i, score in enumerate(scores)
        ]
    }


@pytest.fixture
def retriever(tmp_path):
    """Retriever with all external dependencies patched out."""
    mock_gemini = MagicMock()
    mock_jina_response = MagicMock()
    mock_jina_response.json.return_value = {"results": []}

    with (
        patch("src.retrieval.retriever.QdrantClient") as MockQdrant,
        patch("src.retrieval.retriever.genai.Client", return_value=mock_gemini),
        patch("src.retrieval.retriever.wrappers.wrap_gemini", return_value=mock_gemini),
        patch("src.retrieval.retriever.Client"),  # LangSmith Client
        patch(
            "src.retrieval.retriever.requests.post",
            return_value=mock_jina_response,
        ) as MockPost,
        patch(
            "src.retrieval.retriever.QUERY_CACHE_FILE",
            tmp_path / "query_cache.jsonl",
        ),
    ):
        from src.retrieval.retriever import Retriever

        instance = Retriever(top_k=3)
        instance._mock_qdrant = MockQdrant.return_value
        instance._mock_gemini = mock_gemini
        instance._mock_post = MockPost
        instance._mock_jina_response = mock_jina_response
        yield instance


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
            patch("src.retrieval.retriever.requests.post"),
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
            patch("src.retrieval.retriever.requests.post"),
            patch("src.retrieval.retriever.QUERY_CACHE_FILE", cache_file),
        ):
            from src.retrieval.retriever import Retriever

            r = Retriever(top_k=3)
        # Should not raise; cache is empty
        assert r._cache == {}


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
        expected_vector = [0.2] * settings.db.embedding_dimension
        fake_embedding = MagicMock()
        fake_embedding.values = expected_vector
        retriever._mock_gemini.models.embed_content.return_value.embeddings = [
            fake_embedding
        ]

        retriever._get_query_vector(query)

        assert query_hash in retriever._cache
        assert retriever._cache[query_hash] == expected_vector


class TestRerank:
    """Unit tests for the Jina reranking step in isolation."""

    def test_empty_candidates_returns_empty_without_http_call(self, retriever) -> None:
        result = retriever._rerank("query", [])
        retriever._mock_post.assert_not_called()
        assert result == []

    def test_calls_jina_with_candidate_texts(self, retriever) -> None:
        candidates = [
            {"text": "chunk A", "title": "Paper A", "paper_id": "1"},
            {"text": "chunk B", "title": "Paper B", "paper_id": "2"},
        ]
        retriever._mock_jina_response.json.return_value = {
            "results": [
                {"index": 0, "relevance_score": 0.9},
                {"index": 1, "relevance_score": 0.7},
            ]
        }

        retriever._rerank("test query", candidates)

        retriever._mock_post.assert_called_once()
        call_kwargs = retriever._mock_post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs.args[1]
        assert payload["query"] == "test query"
        assert payload["documents"] == ["chunk A", "chunk B"]

    def test_reorders_candidates_by_relevance_score(self, retriever) -> None:
        """Jina result at index 1 has higher score — it must come first."""
        candidates = [
            {"text": "chunk A", "title": "Paper A", "paper_id": "1"},
            {"text": "chunk B", "title": "Paper B", "paper_id": "2"},
        ]
        retriever._mock_jina_response.json.return_value = {
            "results": [
                {"index": 1, "relevance_score": 0.95},
                {"index": 0, "relevance_score": 0.60},
            ]
        }

        result = retriever._rerank("query", candidates)

        assert result[0]["title"] == "Paper B"
        assert result[1]["title"] == "Paper A"

    def test_relevance_score_replaces_qdrant_score(self, retriever) -> None:
        """After rerank, chunk['score'] must be Jina's relevance_score, not the original."""
        candidates = [{"text": "text", "title": "T", "paper_id": "1", "score": 0.5}]
        retriever._mock_jina_response.json.return_value = {
            "results": [{"index": 0, "relevance_score": 0.82}]
        }

        result = retriever._rerank("q", candidates)

        assert result[0]["score"] == pytest.approx(0.82)

    def test_preserves_all_candidate_fields(self, retriever) -> None:
        """Rerank must not drop payload fields from the original candidate."""
        candidates = [
            {
                "text": "text",
                "title": "Paper X",
                "paper_id": "999",
                "chunk_index": 3,
                "authors": ["Author Z"],
            }
        ]
        retriever._mock_jina_response.json.return_value = {
            "results": [{"index": 0, "relevance_score": 0.7}]
        }

        result = retriever._rerank("q", candidates)

        assert result[0]["title"] == "Paper X"
        assert result[0]["paper_id"] == "999"
        assert result[0]["chunk_index"] == 3
        assert result[0]["authors"] == ["Author Z"]


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

    def _setup_rerank(self, retriever, points: list) -> None:
        """Configure Jina mock to return all points in order with decreasing scores."""
        retriever._mock_jina_response.json.return_value = _make_jina_response(points)

    def test_returns_list_of_dicts(self, retriever) -> None:
        self._setup_embed(retriever)
        point = _fake_qdrant_point()
        retriever._mock_qdrant.query_points.return_value.points = [point]
        self._setup_rerank(retriever, [point])

        result = retriever.retrieve("What is attention?")

        assert isinstance(result, list)
        assert all(isinstance(c, dict) for c in result)

    def test_chunk_fields_are_present(self, retriever) -> None:
        self._setup_embed(retriever)
        point = _fake_qdrant_point(title="Transformer", paper_id="1706")
        retriever._mock_qdrant.query_points.return_value.points = [point]
        self._setup_rerank(retriever, [point])

        chunks = retriever.retrieve("What is a transformer?")

        assert len(chunks) == 1
        chunk = chunks[0]
        for field in ("score", "text", "title", "paper_id", "chunk_index", "authors"):
            assert field in chunk

    def test_score_comes_from_jina_not_qdrant(self, retriever) -> None:
        """After reranking, score must be Jina's relevance_score, not qdrant's score."""
        self._setup_embed(retriever)
        point = _fake_qdrant_point(
            score=0.88
        )  # qdrant score — must NOT appear in output
        retriever._mock_qdrant.query_points.return_value.points = [point]
        retriever._mock_jina_response.json.return_value = {
            "results": [{"index": 0, "relevance_score": 0.75}]
        }

        chunks = retriever.retrieve("query")

        assert chunks[0]["score"] == pytest.approx(0.75)

    def test_empty_qdrant_response_returns_empty_list(self, retriever) -> None:
        self._setup_embed(retriever)
        retriever._mock_qdrant.query_points.return_value.points = []

        result = retriever.retrieve("obscure query")

        assert result == []

    def test_qdrant_called_with_prefetch_k_limit(self, retriever) -> None:
        """Qdrant must be called with prefetch_k (20) as limit, not top_k (3)."""
        self._setup_embed(retriever)
        retriever._mock_qdrant.query_points.return_value.points = []

        retriever.retrieve("query")

        call_kwargs = retriever._mock_qdrant.query_points.call_args
        assert call_kwargs.kwargs.get("limit") == retriever.prefetch_k

    def test_hybrid_search_passes_two_prefetch_entries(self, retriever) -> None:
        """Hybrid search requires exactly two Prefetch objects: dense + sparse."""
        self._setup_embed(retriever)
        retriever._mock_qdrant.query_points.return_value.points = []

        retriever.retrieve("query")

        call_kwargs = retriever._mock_qdrant.query_points.call_args
        prefetch = call_kwargs.kwargs.get("prefetch")
        assert prefetch is not None
        assert len(prefetch) == 2

    def test_hybrid_search_uses_rrf_fusion_query(self, retriever) -> None:
        """query= arg to qdrant must be a FusionQuery (RRF fusion)."""
        from qdrant_client.models import FusionQuery

        self._setup_embed(retriever)
        retriever._mock_qdrant.query_points.return_value.points = []

        retriever.retrieve("query")

        call_kwargs = retriever._mock_qdrant.query_points.call_args
        query_arg = call_kwargs.kwargs.get("query")
        assert isinstance(query_arg, FusionQuery)

    def test_result_count_matches_reranked_output(self, retriever) -> None:
        self._setup_embed(retriever)
        points = [_fake_qdrant_point(paper_id=str(i)) for i in range(3)]
        retriever._mock_qdrant.query_points.return_value.points = points
        self._setup_rerank(retriever, points)

        chunks = retriever.retrieve("query")

        assert len(chunks) == 3


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
        retriever._mock_jina_response.json.return_value = {
            "results": [{"index": 0, "relevance_score": 0.9}]
        }

        mock_run = MagicMock()
        with patch(
            "src.retrieval.retriever.get_current_run_tree", return_value=mock_run
        ):
            chunks = retriever.retrieve("query")

        assert len(chunks[0]["text"]) == 500
        assert chunks[0]["text"] == long_text


class TestEmbedQuery:
    def test_raises_descriptive_error_on_empty_embeddings(self, retriever) -> None:
        """Empty embeddings from Gemini should raise ValueError, not IndexError."""
        retriever._mock_gemini.models.embed_content.return_value.embeddings = []

        with pytest.raises(ValueError, match="empty embeddings"):
            retriever._embed_query("test query")
