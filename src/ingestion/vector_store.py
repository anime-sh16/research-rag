import logging
import uuid

import numpy as np
from google import genai
from google.genai import types
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from src.config.config import settings
from src.ingestion.chunker import ChunkMetaData

logger = logging.getLogger(__name__)

COLLECTION_NAME = settings.db.collection_name
VECTOR_DIM = settings.db.embedding_dimension
FULL_EMBEDDING_DIM = settings.db.full_embedding_dimension


class VectorStore:
    def __init__(self):
        self.qdrant_client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key.get_secret_value(),
        )
        self.gemini_client = genai.Client(
            api_key=settings.google_api_key.get_secret_value()
        )

    def _embed_text(
        self,
        text: str | list[str],
        task_type: str = settings.db.ingest_task_type,
        output_dimensionality: int = VECTOR_DIM,
    ) -> list[list[float]]:
        embeddings = self.gemini_client.models.embed_content(
            model=settings.db.embedding_model,
            contents=text,
            config=types.EmbedContentConfig(
                task_type=task_type, output_dimensionality=output_dimensionality
            ),
        )

        embedding_values = [
            self._normalize_embedding(embedding.values)
            for embedding in embeddings.embeddings
        ]
        return embedding_values

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
        texts = [chunk.source_text for chunk in chunks]
        embeddings = self._embed_text(texts)

        points = [
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id)),
                vector=emb_vector,
                payload=chunk.model_dump(mode="json"),
            )
            for chunk, emb_vector in zip(chunks, embeddings)
        ]

        self.qdrant_client.upsert(collection_name=collection_name, points=points)
