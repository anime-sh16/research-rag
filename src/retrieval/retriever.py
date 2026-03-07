import hashlib
import json
from pathlib import Path

from google import genai
from google.genai import types
from qdrant_client import QdrantClient

from src.config.config import data_settings, db_settings, settings

COLLECTION_NAME = db_settings.collection_name
VECTOR_DIM = db_settings.embedding_dimension
QUERY_CACHE_FILE = data_settings.temp_dir / "query_cache.jsonl"


class Retriever:
    def __init__(self, top_k: int = 5):
        self.qdrant_client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )
        self.gemini_client = genai.Client(api_key=settings.google_api_key)
        self.top_k = top_k
        self._cache = self._load_cache()

    def _load_cache(self) -> dict[str, list[float]]:
        """Load existing query cache from JSONL into memory as {hash: vector}."""
        cache: dict[str, list[float]] = {}
        if QUERY_CACHE_FILE.exists():
            with open(QUERY_CACHE_FILE, "r") as f:
                for line in f:
                    entry = json.loads(line.strip())
                    cache[entry["hash"]] = entry["embedding"]
        return cache

    def _save_to_cache(
        self, query: str, query_hash: str, embedding: list[float]
    ) -> None:
        """Append a new query entry to the JSONL cache file."""
        Path(QUERY_CACHE_FILE).parent.mkdir(parents=True, exist_ok=True)
        with open(QUERY_CACHE_FILE, "a") as f:
            f.write(
                json.dumps(
                    {
                        "query": query,
                        "hash": query_hash,
                        "embedding": embedding,
                    }
                )
                + "\n"
            )

    def _embed_query(self, query: str) -> list[float]:
        response = self.gemini_client.models.embed_content(
            model="gemini-embedding-001",
            contents=query,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=VECTOR_DIM,
            ),
        )
        return response.embeddings[0].values

    def _get_query_vector(self, query: str) -> list[float]:
        """Return cached embedding if available, otherwise embed and cache."""
        query_hash = hashlib.md5(query.encode()).hexdigest()

        if query_hash in self._cache:
            return self._cache[query_hash]

        embedding = self._embed_query(query)
        self._cache[query_hash] = embedding
        self._save_to_cache(query, query_hash, embedding)
        return embedding

    def retrieve(self, query: str) -> list[dict]:
        query_vector = self._get_query_vector(query)

        # query_points is the current API — .search() is removed in latest client
        results = self.qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,  # list[float] → dense nearest neighbour search
            limit=self.top_k,
            with_payload=True,
        ).points  # returns QueryResponse, .points is the list

        return [
            {
                "score": result.score,
                "text": result.payload.get("source_text"),
                "title": result.payload.get("title"),
                "paper_id": result.payload.get("paper_id"),
                "chunk_index": result.payload.get("chunk_index"),
                "authors": result.payload.get("authors"),
            }
            for result in results
        ]
