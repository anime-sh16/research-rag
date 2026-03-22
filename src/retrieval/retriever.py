import hashlib
import json
import logging
from pathlib import Path

import httpx
import requests
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from langsmith import Client, traceable, wrappers
from langsmith.run_helpers import get_current_run_tree
from qdrant_client import QdrantClient
from qdrant_client.models import Document, Fusion, FusionQuery, Prefetch
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


def _is_timeout_error(exc: BaseException) -> bool:
    if isinstance(exc, (TimeoutError, httpx.TimeoutException)):
        return True
    if isinstance(exc, genai_errors.ServerError) and (
        getattr(exc, "status_code", None) == 504
        or "504" in str(exc)
        or "DEADLINE_EXCEEDED" in str(exc)
    ):
        return True
    return False


def _is_service_unavailable_error(exc: BaseException) -> bool:
    return isinstance(exc, genai_errors.ServerError) and (
        getattr(exc, "status_code", None) == 503 or "503" in str(exc)
    )


COLLECTION_NAME = settings.db.collection_name
VECTOR_DIM = settings.db.embedding_dimension
QUERY_CACHE_FILE = settings.data.temp_dir / settings.data.query_cache_file
_JINA_RERANK_URL = settings.jina_rerank_url


class Retriever:
    def __init__(self, top_k: int = 5):
        self.qdrant_client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key.get_secret_value(),
            timeout=30,
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
        self.jina_top_n = settings.retrieval.rerank_top_n
        self.prefetch_k = settings.retrieval.hybrid_prefetch_k
        self.mmr_lambda = settings.retrieval.mmr_lambda
        self._jina_headers = {
            "Authorization": f"Bearer {settings.jina_api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }
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
        retry=retry_if_exception(_is_service_unavailable_error),
        wait=wait_exponential(multiplier=180, min=180, max=900, exp_base=1),
        stop=stop_after_attempt(5),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    @retry(
        retry=retry_if_exception(_is_rate_limit_error),
        wait=wait_exponential(multiplier=60, min=60, max=480),
        stop=stop_after_attempt(5),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    @retry(
        retry=retry_if_exception(_is_timeout_error),
        wait=wait_exponential(multiplier=5, min=5, max=60),
        stop=stop_after_attempt(3),
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
        if not response.embeddings:
            raise ValueError(
                f"Gemini returned empty embeddings for query: {query[:100]!r}"
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

    @traceable(run_type="tool", name="retrieval/jina_rerank")
    def _rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        """Rerank candidates using Jina Reranker API, return top_k sorted by relevance."""
        if not candidates:
            return candidates

        response = requests.post(
            _JINA_RERANK_URL,
            headers=self._jina_headers,
            json={
                "model": settings.retrieval.jina_rerank_model,
                "query": query,
                "documents": [c["text"] for c in candidates],
                "top_n": self.jina_top_n,
                "return_embeddings": True,
                "return_documents": False,
            },
            timeout=60,
        )
        response.raise_for_status()

        return [
            {
                **candidates[r["index"]],
                "score": r["relevance_score"],
                "dense_embedding": r["embedding"],
            }
            for r in response.json()["results"]
        ]

    @traceable(run_type="tool", name="retrieval/mmr_selection")
    def _mmr_selection(self, candidates: list[dict], lamda: float = 0.8) -> list[dict]:
        """
        Select top-k candidates using Maximal Marginal Relevance (MMR).
        """
        # TODO: implenet the MMR selection
        pass

    @traceable(run_type="retriever", name="retrieval/hybrid_search")
    def retrieve(self, query: str) -> list[dict]:
        logger.info("Retrieving top-%d chunks for query: '%s'", self.top_k, query)
        query_vector = self._get_query_vector(query)

        # Hybrid search: dense + BM25 sparse with server-side RRF fusion
        results = self.qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                Prefetch(
                    query=query_vector,
                    using=settings.db.dense_name,
                    limit=self.prefetch_k,
                ),
                Prefetch(
                    query=Document(text=query, model=settings.db.sparse_model),
                    using=settings.db.sparse_name,
                    limit=self.prefetch_k,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=self.prefetch_k,
            with_payload=True,
        ).points

        candidates = [
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

        chunks = self._rerank(query, candidates)

        chunks = self._mmr_selection(chunks)

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
                candidate_papers = len(
                    set(c["paper_id"] for c in candidates if c.get("paper_id"))
                )

                trace_chunks = [
                    {**chunk, "text": (chunk["text"] or "")[:100]} for chunk in chunks
                ]

                run.add_metadata(
                    {
                        "chunks_returned": trace_chunks,
                        "top_k_requested": self.top_k,
                        "prefetch_k": self.prefetch_k,
                        "candidate_papers_before_rerank": candidate_papers,
                    }
                )

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
            "Retrieved %d chunks after rerank (scores: %s).",
            len(chunks),
            [round(c["score"], 3) for c in chunks],
        )
        return chunks
