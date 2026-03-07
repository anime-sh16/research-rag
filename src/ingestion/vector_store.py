import uuid

from google import genai
from google.genai import types
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from src.config.config import settings, db_settings
from src.ingestion.chunker import ChunkMetaData

COLLECTION_NAME = db_settings.collection_name
VECTOR_DIM = db_settings.embedding_dimension


class VectorStore:
    def __init__(self):
        self.qdrant_client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )
        self.gemini_client = genai.Client(api_key=settings.google_api_key)
    
    def _embed_text(self, text: str | list[str], task_type: str = "SEMANTIC_SIMILARITY", output_dimensionality: int = VECTOR_DIM):
        embeddings = self.gemini_client.models.embed_content(
            model="gemini-embedding-001",
            contents=text,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=output_dimensionality
            )
        )
        
        embedding_values = [self._normalize_embedding(embedding.values) for embedding in embeddings.embeddings]
        return embedding_values
    
    def _normalize_embedding(self, embedding: list[float]):
        if len(embedding) != 3072:
            import numpy as np
            embedding = np.array(embedding)
            embedding = embedding / np.linalg.norm(embedding)
            embedding = embedding.tolist()
        return embedding

    def ensure_collection(self, collection_name: str = COLLECTION_NAME, embedding_dimension: int = VECTOR_DIM):
        # check if collection exists, create if not
        if not self.qdrant_client.collection_exists(collection_name):
            self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                        size=embedding_dimension, 
                        distance=Distance.COSINE
                    ),
            )
            print(f"Collection {collection_name} created.")
        else:
            print(f"Collection {collection_name} exists.")
        

    def upsert_chunks(self, chunks: list[ChunkMetaData]):
        texts = [chunk.source_text for chunk in chunks]
        embeddings = self._embed_text(texts)

        points = []
        for chunk, emb_vector in zip(chunks, embeddings):
            points.append(
                PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id)),
                    vector=emb_vector,
                    payload=chunk.model_dump(
                        mode="json"
                    ),
                )
            )

        self.qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
