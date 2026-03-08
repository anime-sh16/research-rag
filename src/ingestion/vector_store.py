import logging
import time
import uuid

import numpy as np
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from src.config.config import settings
from src.ingestion.chunker import ChunkMetaData

logger = logging.getLogger(__name__)

COLLECTION_NAME = settings.db.collection_name
VECTOR_DIM = settings.db.embedding_dimension
FULL_EMBEDDING_DIM = settings.db.full_embedding_dimension

# Rate-limit constants (tuned for 30K TPM / 1K RPD free tier)
# 50 chunks × 512 tokens = ~25,600 tokens per batch — safely under 30K TPM
_EMBED_BATCH_SIZE = 50
_INTER_BATCH_SLEEP_SECS = 2  # polite pause between successful batches


def _is_rate_limit_error(exc: BaseException) -> bool:
    return isinstance(exc, genai_errors.ClientError) and (
        getattr(exc, "status_code", None) == 429 or "429" in str(exc)
    )


class VectorStore:
    def __init__(self):
        self.qdrant_client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key.get_secret_value(),
        )
        self.gemini_client = genai.Client(
            api_key=settings.google_api_key.get_secret_value()
        )

    @retry(
        retry=retry_if_exception(_is_rate_limit_error),
        wait=wait_exponential(multiplier=60, min=60, max=480),
        stop=stop_after_attempt(5),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _embed_text(
        self,
        text: str | list[str],
        task_type: str = settings.db.ingest_task_type,
        output_dimensionality: int = VECTOR_DIM,
    ) -> list[list[float]]:
        """Call Gemini embed_content with exponential backoff on rate-limit errors."""
        embeddings = self.gemini_client.models.embed_content(
            model=settings.db.embedding_model,
            contents=text,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=output_dimensionality,
            ),
        )
        return [
            self._normalize_embedding(embedding.values)
            for embedding in embeddings.embeddings
        ]

    def _normalize_embedding(self, embedding: list[float]) -> list[float]:
        if len(embedding) != FULL_EMBEDDING_DIM:
            arr = np.array(embedding)
            embedding = (arr / np.linalg.norm(arr)).tolist()
        return embedding

    def ensure_collection(
        self,
        collection_name: str = COLLECTION_NAME,
        embedding_dimension: int = VECTOR_DIM,
    ):
        # check if collection exists, create if not
        if not self.qdrant_client.collection_exists(collection_name):
            self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=embedding_dimension, distance=Distance.COSINE
                ),
            )
            logger.info("Collection %s created.", collection_name)
        else:
            logger.info("Collection %s already exists.", collection_name)

    def upsert_chunks(
        self, chunks: list[ChunkMetaData], collection_name: str = COLLECTION_NAME
    ) -> None:
        """Embed chunks in batches and upsert to Qdrant.

        Batches are capped at _EMBED_BATCH_SIZE to stay within the 30K TPM
        free-tier limit (~50 × 512 tokens ≈ 25,600 tokens per batch).
        A short sleep between batches avoids sustained quota pressure.
        """
        # Skip chunks already in Qdrant to avoid wasting Gemini quota on re-runs
        point_ids = [
            str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id)) for chunk in chunks
        ]
        existing = self.qdrant_client.retrieve(
            collection_name=collection_name,
            ids=point_ids,
            with_payload=False,
            with_vectors=False,
        )
        existing_ids = {str(record.id) for record in existing}
        new_chunks = [
            chunk for chunk, pid in zip(chunks, point_ids) if pid not in existing_ids
        ]

        if not new_chunks:
            logger.info("All %d chunks already in Qdrant — skipping.", len(chunks))
            return
        if len(new_chunks) < len(chunks):
            logger.info(
                "Skipping %d already-ingested chunks; embedding %d new.",
                len(chunks) - len(new_chunks),
                len(new_chunks),
            )

        total = len(new_chunks)
        all_points: list[PointStruct] = []

        for batch_start in range(0, total, _EMBED_BATCH_SIZE):
            batch = new_chunks[batch_start : batch_start + _EMBED_BATCH_SIZE]
            texts = [chunk.source_text for chunk in batch]

            logger.info(
                "Embedding batch %d-%d / %d chunks.",
                batch_start + 1,
                batch_start + len(batch),
                total,
            )
            embeddings = self._embed_text(texts)

            all_points.extend(
                PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id)),
                    vector=emb_vector,
                    payload=chunk.model_dump(mode="json"),
                )
                for chunk, emb_vector in zip(batch, embeddings)
            )

            if batch_start + _EMBED_BATCH_SIZE < total:
                time.sleep(_INTER_BATCH_SLEEP_SECS)

        self.qdrant_client.upsert(collection_name=collection_name, points=all_points)
        logger.info(
            "Upserted %d points to collection '%s'.", len(all_points), collection_name
        )
