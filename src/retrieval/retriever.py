import hashlib
import json
import logging
from pathlib import Path

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from langsmith import Client, traceable, wrappers
from langsmith.run_helpers import get_current_run_tree
from qdrant_client import QdrantClient
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from src.config.config import settings

logger = logging.getLogger(__name__)


def _is_rate_limit_error(exc: BaseException) -> bool:
    return isinstance(exc, genai_errors.ClientError) and (
        getattr(exc, "status_code", None) == 429 or "429" in str(exc)
    )


COLLECTION_NAME = settings.db.collection_name
VECTOR_DIM = settings.db.embedding_dimension
QUERY_CACHE_FILE = settings.data.temp_dir / settings.data.query_cache_file


class Retriever:
    def __init__(self, top_k: int = 5):
        self.qdrant_client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key.get_secret_value(),
        )
        self.gemini_client = wrappers.wrap_gemini(
            genai.Client(api_key=settings.google_api_key.get_secret_value()),
            tracing_extra={
                "tags": ["gemini", "python"],
                "metadata": {
                    "integration": "google-genai",
                },
            },
        )
        self.langsmith_client = Client()
        self.top_k = top_k
        self._cache = self._load_cache()

    def _load_cache(self) -> dict[str, list[float]]:
        """Load existing query cache from JSONL into memory as {hash: vector}."""
        cache: dict[str, list[float]] = {}
        if not QUERY_CACHE_FILE.exists():
            return cache
        try:
            with open(QUERY_CACHE_FILE, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    cache[entry["hash"]] = entry["embedding"]
        except (OSError, json.JSONDecodeError, KeyError):
            logger.warning(
                "Query cache at %s is corrupt — starting fresh.", QUERY_CACHE_FILE
            )
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

    @retry(
        retry=retry_if_exception(_is_rate_limit_error),
        wait=wait_exponential(multiplier=60, min=60, max=480),
        stop=stop_after_attempt(5),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _embed_query(self, query: str) -> list[float]:
        response = self.gemini_client.models.embed_content(
            model=settings.db.embedding_model,
            contents=query,
            config=types.EmbedContentConfig(
                task_type=settings.db.retrieval_task_type,
                output_dimensionality=VECTOR_DIM,
            ),
        )
        return response.embeddings[0].values

    def _get_query_vector(self, query: str) -> list[float]:
        """Return cached embedding if available, otherwise embed and cache."""
        query_hash = hashlib.md5(query.encode()).hexdigest()

        if query_hash in self._cache:
            logger.debug("Cache hit for query: '%s'", query)
            return self._cache[query_hash]

        logger.debug("Cache miss — embedding query: '%s'", query)
        embedding = self._embed_query(query)
        self._cache[query_hash] = embedding
        self._save_to_cache(query, query_hash, embedding)
        return embedding

    @traceable(run_type="retriever", name="retrieval/dense_search")
    def retrieve(self, query: str) -> list[dict]:
        logger.info("Retrieving top-%d chunks for query: '%s'", self.top_k, query)
        query_vector = self._get_query_vector(query)

        # query_points is the current API — .search() is removed in latest client
        results = self.qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,  # list[float] → dense nearest neighbour search
            limit=self.top_k,
            with_payload=True,
        ).points  # returns QueryResponse, .points is the list

        chunks = [
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

        # LangSmith Tracing & Proxy Metrics
        run = get_current_run_tree()
        if run:
            if not chunks:
                run.add_tags(["empty_retrieval"])
            else:
                scores = [c["score"] for c in chunks]
                unique_papers = len(
                    set(c["paper_id"] for c in chunks if c.get("paper_id"))
                )

                # Log the raw data so you can inspect it in the UI
                run.add_metadata(
                    {
                        "chunks_returned": chunks,
                        "top_k_requested": self.top_k,
                    }
                )

                # Log proxy metrics to LangSmith
                self.langsmith_client.create_feedback(
                    run_id=run.id,
                    key="retrieval_avg_score",
                    score=sum(scores) / len(scores),
                    trace_id=run.trace_id,
                )
                self.langsmith_client.create_feedback(
                    run_id=run.id,
                    key="retrieval_score_spread",
                    score=max(scores) - min(scores),
                    trace_id=run.trace_id,
                )
                self.langsmith_client.create_feedback(
                    run_id=run.id,
                    key="source_diversity",
                    score=unique_papers,
                    trace_id=run.trace_id,
                )

        logger.info(
            "Retrieved %d chunks (scores: %s).",
            len(chunks),
            [round(c["score"], 3) for c in chunks],
        )
        return chunks
