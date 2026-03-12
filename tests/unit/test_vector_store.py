from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.config.config import settings
from src.ingestion.chunker import ChunkMetaData


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
def vector_store():
    """VectorStore with Qdrant, Gemini, and LangSmith wrappers fully patched."""
    mock_gemini = MagicMock()
    with (
        patch("src.ingestion.vector_store.QdrantClient") as MockQdrant,
        patch("src.ingestion.vector_store.genai.Client", return_value=mock_gemini),
        patch(
            "src.ingestion.vector_store.wrappers.wrap_gemini", return_value=mock_gemini
        ),
    ):
        from src.ingestion.vector_store import VectorStore

        instance = VectorStore()
        instance._mock_qdrant = MockQdrant.return_value
        instance._mock_gemini = mock_gemini
        yield instance


class TestEnsureCollection:
    def test_creates_collection_when_absent(self, vector_store) -> None:
        vector_store._mock_qdrant.collection_exists.return_value = False
        vector_store.ensure_collection("test_collection")
        vector_store._mock_qdrant.create_collection.assert_called_once()

    def test_skips_creation_when_collection_exists(self, vector_store) -> None:
        vector_store._mock_qdrant.collection_exists.return_value = True
        vector_store.ensure_collection("test_collection")
        vector_store._mock_qdrant.create_collection.assert_not_called()

    def test_checks_the_correct_collection_name(self, vector_store) -> None:
        vector_store._mock_qdrant.collection_exists.return_value = False
        vector_store.ensure_collection("my_collection")
        vector_store._mock_qdrant.collection_exists.assert_called_once_with(
            "my_collection"
        )


class TestUpsertChunks:
    def _setup_no_existing(self, vector_store) -> None:
        """Make the dedup retrieve call return no existing points."""
        vector_store._mock_qdrant.retrieve.return_value = []

    def _fake_embed_text(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * settings.db.embedding_dimension for _ in texts]

    def test_embeds_source_text_of_new_chunks(self, vector_store) -> None:
        self._setup_no_existing(vector_store)
        chunks = [_make_chunk("aaa", 0), _make_chunk("aaa", 1)]
        with patch.object(
            vector_store, "_embed_text", side_effect=self._fake_embed_text
        ) as mock_embed:
            vector_store.upsert_chunks(chunks)
        texts_embedded = mock_embed.call_args[0][0]
        assert texts_embedded == [chunk.source_text for chunk in chunks]

    def test_calls_qdrant_upsert_for_new_chunks(self, vector_store) -> None:
        self._setup_no_existing(vector_store)
        chunks = [_make_chunk("aaa", 0)]
        with patch.object(
            vector_store, "_embed_text", side_effect=self._fake_embed_text
        ):
            vector_store.upsert_chunks(chunks)
        vector_store._mock_qdrant.upsert.assert_called_once()

    def test_upsert_point_count_matches_chunk_count(self, vector_store) -> None:
        self._setup_no_existing(vector_store)
        chunks = [_make_chunk("aaa", i) for i in range(3)]
        with patch.object(
            vector_store, "_embed_text", side_effect=self._fake_embed_text
        ):
            vector_store.upsert_chunks(chunks)
        call_kwargs = vector_store._mock_qdrant.upsert.call_args
        points = call_kwargs.kwargs["points"]
        assert len(points) == 3

    def test_skips_upsert_when_all_chunks_exist(self, vector_store) -> None:
        """If all chunks are already in Qdrant, no embed or upsert should happen."""
        chunks = [_make_chunk("aaa", 0)]
        existing_record = MagicMock()
        import uuid

        existing_record.id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunks[0].chunk_id))
        vector_store._mock_qdrant.retrieve.return_value = [existing_record]

        with patch.object(vector_store, "_embed_text") as mock_embed:
            vector_store.upsert_chunks(chunks)

        mock_embed.assert_not_called()
        vector_store._mock_qdrant.upsert.assert_not_called()

    def test_dedup_check_uses_correct_collection(self, vector_store) -> None:
        self._setup_no_existing(vector_store)
        chunks = [_make_chunk("aaa", 0)]
        with patch.object(
            vector_store, "_embed_text", side_effect=self._fake_embed_text
        ):
            vector_store.upsert_chunks(chunks, collection_name="custom_col")
        retrieve_call = vector_store._mock_qdrant.retrieve.call_args
        assert retrieve_call.kwargs["collection_name"] == "custom_col"


class TestNormalizeEmbedding:
    def test_short_embedding_is_normalized_to_unit_length(self, vector_store) -> None:
        import math

        embedding = [3.0, 4.0]  # magnitude = 5.0 → normalized = [0.6, 0.8]
        result = vector_store._normalize_embedding(embedding)
        magnitude = math.sqrt(sum(x**2 for x in result))
        assert abs(magnitude - 1.0) < 1e-6

    def test_full_dim_embedding_returned_unchanged(self, vector_store) -> None:
        embedding = [0.5] * settings.db.full_embedding_dimension
        result = vector_store._normalize_embedding(embedding)
        assert result == embedding
