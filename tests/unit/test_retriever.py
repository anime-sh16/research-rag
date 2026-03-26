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
            {"index": i, "relevance_score": score, "embedding": [0.1] * 64}
            for i, score in enumerate(scores)
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
                {"index": 0, "relevance_score": 0.9, "embedding": [0.1] * 64},
                {"index": 1, "relevance_score": 0.7, "embedding": [0.1] * 64},
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
                {"index": 1, "relevance_score": 0.95, "embedding": [0.1] * 64},
                {"index": 0, "relevance_score": 0.60, "embedding": [0.1] * 64},
            ]
        }

        result = retriever._rerank("query", candidates)

        assert result[0]["title"] == "Paper B"
        assert result[1]["title"] == "Paper A"

    def test_relevance_score_replaces_qdrant_score(self, retriever) -> None:
        """After rerank, chunk['score'] must be Jina's relevance_score, not the original."""
        candidates = [{"text": "text", "title": "T", "paper_id": "1", "score": 0.5}]
        retriever._mock_jina_response.json.return_value = {
            "results": [{"index": 0, "relevance_score": 0.82, "embedding": [0.1] * 64}]
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
            "results": [{"index": 0, "relevance_score": 0.7, "embedding": [0.1] * 64}]
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

    @pytest.fixture(autouse=True)
    def _patch_extract_subquery(self, retriever):
        """Bypass LLM-based query analysis — return original query as single sub-query."""
        retriever._extract_subquery = MagicMock(
            side_effect=lambda q: {"subquery": [{"query": q, "expansion_terms": []}]}
        )

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
            "results": [{"index": 0, "relevance_score": 0.75, "embedding": [0.1] * 64}]
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

    @pytest.fixture(autouse=True)
    def _patch_extract_subquery(self, retriever):
        """Bypass LLM-based query analysis."""
        retriever._extract_subquery = MagicMock(
            side_effect=lambda q: {"subquery": [{"query": q, "expansion_terms": []}]}
        )

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
            "results": [{"index": 0, "relevance_score": 0.9, "embedding": [0.1] * 64}]
        }

        mock_run = MagicMock()
        with patch(
            "src.retrieval.retriever.get_current_run_tree", return_value=mock_run
        ):
            chunks = retriever.retrieve("query")

        assert len(chunks[0]["text"]) == 500
        assert chunks[0]["text"] == long_text


class TestExtractSubquery:
    """Tests for the LLM-based query decomposition + expansion step."""

    @pytest.fixture(autouse=True)
    def _patch_langsmith_run(self):
        with patch("src.retrieval.retriever.get_current_run_tree", return_value=None):
            yield

    def _mock_llm_response(self, retriever, parsed_obj):
        """Configure gemini mock to return a parsed Query object."""
        response = MagicMock()
        response.parsed = parsed_obj
        response.usage_metadata = None
        retriever._mock_gemini.models.generate_content.return_value = response

    def test_returns_plain_dict_with_expected_structure(self, retriever) -> None:
        """Callers (retrieve loop) depend on dict with 'subquery' key containing list of dicts."""
        from src.retrieval.retriever import Query, Subquery

        self._mock_llm_response(
            retriever,
            Query(
                subquery=[
                    Subquery(query="q1", expansion_terms=["t1"]),
                    Subquery(query="q2", expansion_terms=["t2", "t3"]),
                ]
            ),
        )

        result = retriever._extract_subquery("multi-topic query")

        assert isinstance(result, dict)
        assert isinstance(result["subquery"], list)
        assert all(isinstance(sq, dict) for sq in result["subquery"])
        assert result["subquery"][0]["query"] == "q1"
        assert result["subquery"][1]["expansion_terms"] == ["t2", "t3"]


class TestExtractSubqueryIntegrationWithRetrieve:
    """Verify that _extract_subquery output correctly drives the retrieval loop."""

    @pytest.fixture(autouse=True)
    def _patch_langsmith_run(self):
        with patch("src.retrieval.retriever.get_current_run_tree", return_value=None):
            yield

    def _setup_embed(self, retriever) -> None:
        fake_embedding = MagicMock()
        fake_embedding.values = [0.1] * settings.db.embedding_dimension
        retriever._mock_gemini.models.embed_content.return_value.embeddings = [
            fake_embedding
        ]

    def test_multi_subquery_triggers_multiple_qdrant_searches(self, retriever) -> None:
        """Each sub-query must result in a separate Qdrant search call."""
        self._setup_embed(retriever)
        retriever._mock_qdrant.query_points.return_value.points = []
        retriever._extract_subquery = MagicMock(
            return_value={
                "subquery": [
                    {"query": "sub-query A", "expansion_terms": ["x"]},
                    {"query": "sub-query B", "expansion_terms": ["y"]},
                ]
            }
        )

        retriever.retrieve("original query")

        assert retriever._mock_qdrant.query_points.call_count == 2

    def test_expansion_terms_injected_into_bm25_text(self, retriever) -> None:
        """Expansion terms must appear in the sparse/BM25 Document text, not the dense query."""
        self._setup_embed(retriever)
        retriever._mock_qdrant.query_points.return_value.points = []
        retriever._extract_subquery = MagicMock(
            return_value={
                "subquery": [
                    {
                        "query": "QServe inference",
                        "expansion_terms": ["SmoothAttention", "W4A8KV4"],
                    },
                ]
            }
        )

        retriever.retrieve("query")

        call_kwargs = retriever._mock_qdrant.query_points.call_args.kwargs
        sparse_prefetch = call_kwargs["prefetch"][1]
        bm25_doc_text = sparse_prefetch.query.text
        assert "SmoothAttention" in bm25_doc_text
        assert "W4A8KV4" in bm25_doc_text
        assert "QServe inference" in bm25_doc_text

    def test_rerank_called_with_subquery_not_original(self, retriever) -> None:
        """Reranking must score against the sub-query, not the original user query."""
        self._setup_embed(retriever)
        point = _fake_qdrant_point()
        retriever._mock_qdrant.query_points.return_value.points = [point]
        retriever._mock_jina_response.json.return_value = {
            "results": [{"index": 0, "relevance_score": 0.8, "embedding": [0.1] * 64}]
        }
        retriever._extract_subquery = MagicMock(
            return_value={
                "subquery": [
                    {"query": "decomposed sub-query", "expansion_terms": []},
                ]
            }
        )

        retriever.retrieve("original full query")

        rerank_call = retriever._mock_post.call_args
        payload = rerank_call.kwargs.get("json") or rerank_call.args[1]
        assert payload["query"] == "decomposed sub-query"

    def test_fallback_on_extraction_failure_uses_original_query(
        self, retriever
    ) -> None:
        """If LLM call fails, retrieve must fall back to searching with original query."""
        self._setup_embed(retriever)
        retriever._mock_qdrant.query_points.return_value.points = []
        retriever._extract_subquery = MagicMock(side_effect=RuntimeError("LLM down"))

        result = retriever.retrieve("my query")

        # Should not raise, and should have searched with the original query
        assert isinstance(result, list)
        call_kwargs = retriever._mock_qdrant.query_points.call_args.kwargs
        sparse_prefetch = call_kwargs["prefetch"][1]
        assert sparse_prefetch.query.text == "my query"

    def test_prefetch_limit_scales_with_subquery_count(self, retriever) -> None:
        """Prefetch per sub-query must be prefetch_k // num_subqueries to keep total pool constant."""
        self._setup_embed(retriever)
        retriever._mock_qdrant.query_points.return_value.points = []
        retriever._extract_subquery = MagicMock(
            return_value={
                "subquery": [
                    {"query": "sub A", "expansion_terms": []},
                    {"query": "sub B", "expansion_terms": []},
                ]
            }
        )

        retriever.retrieve("multi-topic query")

        expected_limit = retriever.prefetch_k // 2
        for call in retriever._mock_qdrant.query_points.call_args_list:
            assert call.kwargs["limit"] == expected_limit


class TestRRFMerge:
    """Tests for Reciprocal Rank Fusion merging of per-subquery reranked lists."""

    def test_single_list_returns_same_order(self, retriever) -> None:
        ranked = [
            {"paper_id": "A", "chunk_index": 0, "score": 0.9, "text": "a"},
            {"paper_id": "B", "chunk_index": 0, "score": 0.7, "text": "b"},
        ]
        result = retriever._rrf_merge([ranked])
        assert result[0]["paper_id"] == "A"
        assert result[1]["paper_id"] == "B"

    def test_duplicate_across_lists_accumulates_scores(self, retriever) -> None:
        """A chunk appearing at rank 0 in both lists should score higher than one in only one list."""
        list_a = [
            {"paper_id": "A", "chunk_index": 0, "score": 0.9, "text": "shared"},
            {"paper_id": "B", "chunk_index": 0, "score": 0.8, "text": "only-a"},
        ]
        list_b = [
            {"paper_id": "A", "chunk_index": 0, "score": 0.85, "text": "shared"},
            {"paper_id": "C", "chunk_index": 0, "score": 0.7, "text": "only-b"},
        ]
        result = retriever._rrf_merge([list_a, list_b], k=60)

        # A appears in both lists at rank 0 → score = 2 * 1/(60+1) ≈ 0.0328
        # B appears once at rank 1 → score = 1/(60+2) ≈ 0.0161
        # C appears once at rank 1 → score = 1/(60+2) ≈ 0.0161
        assert result[0]["paper_id"] == "A"
        assert result[0]["score"] == pytest.approx(2 / 61, abs=1e-6)

    def test_interleaves_results_from_different_lists(self, retriever) -> None:
        """Two disjoint lists should interleave: rank-0 items first, then rank-1 items."""
        list_a = [
            {"paper_id": "A", "chunk_index": 0, "score": 0.9, "text": "a0"},
            {"paper_id": "A", "chunk_index": 1, "score": 0.8, "text": "a1"},
        ]
        list_b = [
            {"paper_id": "B", "chunk_index": 0, "score": 0.9, "text": "b0"},
            {"paper_id": "B", "chunk_index": 1, "score": 0.8, "text": "b1"},
        ]
        result = retriever._rrf_merge([list_a, list_b], k=60)

        # Rank-0 items both get 1/61, rank-1 items both get 1/62
        # So rank-0 items come first (tied), then rank-1 items (tied)
        top_two_papers = {result[0]["paper_id"], result[1]["paper_id"]}
        assert top_two_papers == {"A", "B"}
        assert result[0]["score"] == pytest.approx(result[1]["score"])

    def test_empty_lists_returns_empty(self, retriever) -> None:
        result = retriever._rrf_merge([[], []])
        assert result == []

    def test_preserves_metadata_fields(self, retriever) -> None:
        ranked = [
            {
                "paper_id": "X",
                "chunk_index": 5,
                "score": 0.9,
                "text": "content",
                "title": "Paper X",
                "authors": ["Auth"],
            },
        ]
        result = retriever._rrf_merge([ranked])
        assert result[0]["title"] == "Paper X"
        assert result[0]["authors"] == ["Auth"]
        assert result[0]["chunk_index"] == 5

    def test_three_lists_with_overlap(self, retriever) -> None:
        """Chunk appearing in all 3 lists beats chunk appearing in only 2."""
        shared = {"paper_id": "S", "chunk_index": 0, "score": 0.5, "text": "s"}
        partial = {"paper_id": "P", "chunk_index": 0, "score": 0.9, "text": "p"}
        list_a = [shared, partial]
        list_b = [
            shared,
            {"paper_id": "X", "chunk_index": 0, "score": 0.3, "text": "x"},
        ]
        list_c = [shared]

        result = retriever._rrf_merge([list_a, list_b, list_c], k=60)
        assert result[0]["paper_id"] == "S"
        # S: 3 * 1/61 ≈ 0.0492; P: 1/62 ≈ 0.0161
        assert result[0]["score"] == pytest.approx(3 / 61, abs=1e-6)


class TestSubqueryReranking:
    """Tests for the new per-subquery reranking + RRF merge flow in retrieve."""

    @pytest.fixture(autouse=True)
    def _patch_langsmith_run(self):
        with patch("src.retrieval.retriever.get_current_run_tree", return_value=None):
            yield

    def _setup_embed(self, retriever) -> None:
        fake_embedding = MagicMock()
        fake_embedding.values = [0.1] * settings.db.embedding_dimension
        retriever._mock_gemini.models.embed_content.return_value.embeddings = [
            fake_embedding
        ]

    def test_multi_subquery_reranks_per_subquery(self, retriever) -> None:
        """With 2 sub-queries, rerank should be called twice (once per sub-query)."""
        self._setup_embed(retriever)
        point_a = _fake_qdrant_point(paper_id="A", title="Paper A")
        point_b = _fake_qdrant_point(paper_id="B", title="Paper B")  # noqa
        retriever._mock_qdrant.query_points.return_value.points = [point_a]
        retriever._extract_subquery = MagicMock(
            return_value={
                "subquery": [
                    {"query": "sub-query A", "expansion_terms": []},
                    {"query": "sub-query B", "expansion_terms": []},
                ]
            }
        )
        retriever._mock_jina_response.json.return_value = {
            "results": [{"index": 0, "relevance_score": 0.8, "embedding": [0.1] * 64}]
        }

        retriever.retrieve("compare A and B")

        assert retriever._mock_post.call_count == 2
        queries_sent = [
            call.kwargs.get("json", call.args[1] if len(call.args) > 1 else {}).get(
                "query"
            )
            or call.kwargs.get("json", {}).get("query")
            for call in retriever._mock_post.call_args_list
        ]
        assert "sub-query A" in queries_sent
        assert "sub-query B" in queries_sent

    def test_single_subquery_skips_rrf(self, retriever) -> None:
        """Single sub-query should return reranked results directly, no RRF."""
        self._setup_embed(retriever)
        points = [_fake_qdrant_point(paper_id=str(i)) for i in range(3)]
        retriever._mock_qdrant.query_points.return_value.points = points
        retriever._extract_subquery = MagicMock(
            return_value={
                "subquery": [{"query": "single topic", "expansion_terms": []}]
            }
        )
        retriever._mock_jina_response.json.return_value = _make_jina_response(
            points, scores=[0.9, 0.8, 0.7]
        )

        retriever._rrf_merge = MagicMock(wraps=retriever._rrf_merge)
        chunks = retriever.retrieve("single topic query")

        retriever._rrf_merge.assert_not_called()
        assert len(chunks) <= retriever.top_k

    def test_multi_subquery_uses_rrf(self, retriever) -> None:
        """Multiple sub-queries should trigger RRF merge."""
        self._setup_embed(retriever)
        point = _fake_qdrant_point()
        retriever._mock_qdrant.query_points.return_value.points = [point]
        retriever._extract_subquery = MagicMock(
            return_value={
                "subquery": [
                    {"query": "sub A", "expansion_terms": []},
                    {"query": "sub B", "expansion_terms": []},
                ]
            }
        )
        retriever._mock_jina_response.json.return_value = {
            "results": [{"index": 0, "relevance_score": 0.8, "embedding": [0.1] * 64}]
        }

        retriever._rrf_merge = MagicMock(wraps=retriever._rrf_merge)
        retriever.retrieve("multi topic")

        retriever._rrf_merge.assert_called_once()


class TestEmbedQuery:
    def test_raises_descriptive_error_on_empty_embeddings(self, retriever) -> None:
        """Empty embeddings from Gemini should raise ValueError, not IndexError."""
        retriever._mock_gemini.models.embed_content.return_value.embeddings = []

        with pytest.raises(ValueError, match="empty embeddings"):
            retriever._embed_query("test query")
