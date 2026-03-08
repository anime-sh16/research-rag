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
    """VectorStore with Qdrant and Gemini clients fully patched."""
    with (
        patch("src.ingestion.vector_store.QdrantClient") as MockQdrant,
        patch("src.ingestion.vector_store.genai.Client") as MockGemini,
    ):
        from src.ingestion.vector_store import VectorStore

        instance = VectorStore()
        instance._mock_qdrant = MockQdrant.return_value
        instance._mock_gemini = MockGemini.return_value
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
    def test_embeds_source_text_not_other_fields(self, vector_store) -> None:
        chunks = [_make_chunk("aaa", 0), _make_chunk("aaa", 1)]
        fake_embedding = MagicMock()
        fake_embedding.values = [0.1] * 768
        vector_store._mock_gemini.models.embed_content.return_value.embeddings = [
            fake_embedding,
            fake_embedding,
        ]
        vector_store.upsert_chunks(chunks)
        call_kwargs = vector_store._mock_gemini.models.embed_content.call_args
        texts = call_kwargs.kwargs.get("contents") or call_kwargs.args[1]
        expected = [chunk.source_text for chunk in chunks]
        assert texts == expected

    def test_calls_qdrant_upsert_once(self, vector_store) -> None:
        chunks = [_make_chunk("aaa", 0)]
        fake_embedding = MagicMock()
        fake_embedding.values = [0.1] * 768
        vector_store._mock_gemini.models.embed_content.return_value.embeddings = [
            fake_embedding
        ]
        vector_store.upsert_chunks(chunks)
        vector_store._mock_qdrant.upsert.assert_called_once()

    def test_upsert_point_count_matches_chunk_count(self, vector_store) -> None:
        chunks = [_make_chunk("aaa", i) for i in range(3)]
        fake_embedding = MagicMock()
        fake_embedding.values = [0.1] * 768
        vector_store._mock_gemini.models.embed_content.return_value.embeddings = [
            fake_embedding
        ] * 3
        vector_store.upsert_chunks(chunks)
        call_kwargs = vector_store._mock_qdrant.upsert.call_args
        points = call_kwargs.kwargs["points"]
        assert len(points) == 3


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
