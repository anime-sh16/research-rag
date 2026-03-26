import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import requests
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from langsmith import Client, traceable, wrappers
from langsmith.run_helpers import get_current_run_tree
from pydantic import BaseModel, Field
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


# Retry predicates


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


# Composable retry decorator for all Gemini API calls
_gemini_retry = lambda fn: (  # noqa: E731
    retry(
        retry=retry_if_exception(_is_service_unavailable_error),
        wait=wait_exponential(multiplier=180, min=180, max=900, exp_base=1),
        stop=stop_after_attempt(5),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )(
        retry(
            retry=retry_if_exception(_is_rate_limit_error),
            wait=wait_exponential(multiplier=60, min=60, max=480),
            stop=stop_after_attempt(5),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )(
            retry(
                retry=retry_if_exception(_is_timeout_error),
                wait=wait_exponential(multiplier=5, min=5, max=60),
                stop=stop_after_attempt(3),
                before_sleep=before_sleep_log(logger, logging.WARNING),
                reraise=True,
            )(fn)
        )
    )
)


# Module constants

COLLECTION_NAME = settings.db.collection_name
VECTOR_DIM = settings.db.embedding_dimension
QUERY_CACHE_FILE = settings.data.temp_dir / settings.data.query_cache_file
_JINA_RERANK_URL = settings.jina_rerank_url

QUERY_ANALYSIS_PROMPT = """You are a search query analyzer for an ML research paper retrieval system.

Given a user query, your job is to:
1. Determine if the query asks about MULTIPLE DISTINCT topics/papers/methods. If so, decompose into separate sub-queries — one per topic.
2. For EACH sub-query, extract expansion terms: domain-specific synonyms, abbreviations, related technical jargon, or alternative names that authors might use in their papers instead of the terms in the query.

Rules:
- If the query is about a SINGLE topic, return exactly ONE sub-query with the original query text.
- Do NOT rephrase or simplify the sub-queries. Keep them close to the original wording, just scoped to one topic each.
- Each sub-query MUST explicitly name the entity/method/model it refers to. Never use pronouns ("they", "it", "this approach") or implicit references that rely on the other sub-query for context. Each sub-query must be understandable on its own.
- Expansion terms should bridge vocabulary gaps — include terms a paper might use even if the user didn't. Think: method names, architecture components, technique aliases.
- Keep expansion terms focused (3-8 per sub-query). Quality over quantity.

Examples:
Query: "What is the role of knowledge distillation in MobileBERT?"
→ Single topic. One sub-query. Expansion: knowledge distillation, teacher-student, model compression, bottleneck layers, inverted bottleneck.

Query: "How does LoRA fine-tuning compare to prefix tuning, and what mixture-of-experts routing strategy does Switch Transformer use?"
→ Two distinct topics. Sub-query 1: LoRA vs prefix tuning for fine-tuning (expansion: low-rank adaptation, soft prompts, parameter-efficient, PEFT, adapter layers). Sub-query 2: Switch Transformer routing strategy (expansion: mixture-of-experts, MoE, top-k gating, expert capacity, load balancing, sparse activation)."""


# Pydantic schemas for structured LLM output


class Subquery(BaseModel):
    query: str = Field(..., description="The query to search for")
    expansion_terms: list[str] = Field(
        ...,
        description="The expansion terms to use for the BM25 sparse query",
    )


class Query(BaseModel):
    subquery: list[Subquery] = Field(
        ...,
        description=(
            "The subquery to search for. Each subquery has a query and a list"
            " of expansion terms. May be one or more subqueries."
        ),
    )


# Retriever


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
                "metadata": {"integration": "google-genai"},
            },
        )
        self.langsmith_client = Client()
        self.query_model = settings.retrieval.query_model
        self.top_k = top_k
        self.jina_top_n = settings.retrieval.rerank_top_n
        self.prefetch_k = settings.retrieval.hybrid_prefetch_k
        self.mmr_lambda = settings.retrieval.mmr_lambda
        self._jina_headers = {
            "Authorization": f"Bearer {settings.jina_api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }
        self._cache = self._load_cache()

    # Embedding cache

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
                json.dumps({"query": query, "hash": query_hash, "embedding": embedding})
                + "\n"
            )

    # Embedding

    @_gemini_retry
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

    # Query analysis (decomposition + expansion)

    @traceable(run_type="llm", name="retrieval/query_extraction")
    @_gemini_retry
    def _extract_subquery(self, query: str) -> dict:
        """Decompose a query into sub-queries with BM25 expansion terms.

        Returns a dict matching the Query schema:
            {"subquery": [{"query": str, "expansion_terms": [str, ...]}, ...]}
        """
        run = get_current_run_tree()
        if run:
            run.add_metadata(
                {
                    "full_prompt_sent": query,
                    "system_instruction": QUERY_ANALYSIS_PROMPT,
                    "model": self.query_model,
                }
            )

        response = self.gemini_client.models.generate_content(
            model=self.query_model,
            contents=f"Query: {query}",
            config=types.GenerateContentConfig(
                system_instruction=QUERY_ANALYSIS_PROMPT,
                temperature=settings.retrieval.temperature,
                thinking_config=types.ThinkingConfig(thinking_level="minimal"),
                http_options=types.HttpOptions(timeout=60000),
                response_mime_type="application/json",
                response_schema=Query,
            ),
        )

        if run and hasattr(response, "usage_metadata") and response.usage_metadata:
            run.add_metadata(
                {
                    "input_tokens": response.usage_metadata.prompt_token_count,
                    "output_tokens": response.usage_metadata.candidates_token_count,
                }
            )

        return response.parsed.model_dump()

    # Hybrid search

    @traceable(run_type="retriever", name="retrieval/subquery_search")
    def _search_subquery(
        self, sub_query: str, expansion_terms: list[str], prefetch_limit: int
    ) -> list[dict]:
        """Run hybrid search for a single sub-query with BM25 expansion terms."""
        query_vector = self._get_query_vector(sub_query)

        bm25_text = sub_query
        if expansion_terms:
            bm25_text = f"{sub_query} {' '.join(expansion_terms)}"

        results = self.qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                Prefetch(
                    query=query_vector,
                    using=settings.db.dense_name,
                    limit=prefetch_limit,
                ),
                Prefetch(
                    query=Document(text=bm25_text, model=settings.db.sparse_model),
                    using=settings.db.sparse_name,
                    limit=prefetch_limit,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=prefetch_limit,
            with_payload=True,
        ).points

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

    def _merge_candidates(self, candidate_lists: list[dict]) -> list[dict]:
        """Union + deduplicate candidates from multiple sub-queries, keeping highest score."""
        seen: dict[tuple, dict] = {}
        for c in candidate_lists:
            key = (c["paper_id"], c["chunk_index"])
            if key not in seen or c["score"] > seen[key]["score"]:
                seen[key] = c
        return list(seen.values())

    def _rrf_merge(self, ranked_lists: list[list[dict]], k: int = 60) -> list[dict]:
        """Merge multiple ranked lists using Reciprocal Rank Fusion.

        For each candidate, RRF score = sum over lists of 1/(k + rank),
        where rank is 1-based position in that list.
        Deduplicates by (paper_id, chunk_index).
        """
        seen: dict[tuple, dict] = {}
        for ranked_list in ranked_lists:
            for rank, c in enumerate(ranked_list):
                key = (c["paper_id"], c["chunk_index"])
                rrf_score = 1 / (k + rank + 1)
                if key in seen:
                    seen[key]["score"] += rrf_score
                else:
                    seen[key] = {**c, "score": rrf_score}

        return sorted(seen.values(), key=lambda x: x["score"], reverse=True)

    # Reranking

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
                # "return_embeddings": True,  # TODO: re-enable when implementing MMR
                "return_documents": False,
            },
            timeout=60,
        )
        response.raise_for_status()

        return [
            {
                **candidates[r["index"]],
                "score": r["relevance_score"],
                # "dense_embedding": r["embedding"],  # TODO: re-enable when implementing MMR
            }
            for r in response.json()["results"]
        ]

    # MMR

    @traceable(run_type="tool", name="retrieval/mmr_selection")
    def _mmr_selection(self, candidates: list[dict], lamda: float = 0.8) -> list[dict]:
        """Select top-k candidates using Maximal Marginal Relevance (MMR)."""
        # TODO: implement MMR selection

        for candidate in candidates:
            candidate.pop("dense_embedding", None)

        return candidates

    # -- Main entry point ----------------------------------------------------

    @traceable(run_type="retriever", name="retrieval/hybrid_search")
    def retrieve(self, query: str) -> list[dict]:
        logger.info("Retrieving top-%d chunks for query: '%s'", self.top_k, query)

        # Step 1: Query analysis — decompose + extract expansion terms
        try:
            extraction = self._extract_subquery(query)
        except Exception as e:
            logger.warning(
                "Sub-query extraction failed for query '%s': %s. Skipping.", query, e
            )
            extraction = {"subquery": [{"query": query, "expansion_terms": []}]}
        sub_queries = extraction["subquery"]
        logger.info(
            "Query decomposed into %d sub-queries: %s",
            len(sub_queries),
            [sq["query"][:80] for sq in sub_queries],
        )

        # Step 2+3: Hybrid search per sub-query + re-rank (parallel)
        # Scale prefetch per sub-query so total candidates stay ~prefetch_k
        prefetch_limit = self.prefetch_k // len(sub_queries)

        def _search_and_rerank(sq: dict) -> list[dict]:
            candidates = self._search_subquery(
                sq["query"], sq["expansion_terms"], prefetch_limit
            )
            return self._rerank(sq["query"], candidates)

        if len(sub_queries) == 1:
            reranked_lists = [_search_and_rerank(sub_queries[0])]
        else:
            with ThreadPoolExecutor(max_workers=len(sub_queries)) as pool:
                reranked_lists = list(pool.map(_search_and_rerank, sub_queries))

        # Step 4: merged across sub queries
        if len(reranked_lists) == 1:
            chunks = reranked_lists[0][: self.top_k]
        else:
            chunks = self._rrf_merge(reranked_lists)[: self.top_k]

        # Step 5: MMR diversity selection
        chunks = self._mmr_selection(chunks)

        # Step 6: LangSmith Tracing & Proxy Metrics
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
                    set(
                        c["paper_id"]
                        for rl in reranked_lists
                        for c in rl
                        if c.get("paper_id")
                    )
                )

                run.add_metadata(
                    {
                        "chunks_returned": [
                            {**chunk, "text": (chunk["text"] or "")[:100]}
                            for chunk in chunks
                        ],
                        "top_k_requested": self.top_k,
                        "prefetch_k": self.prefetch_k,
                        "candidate_papers_after_rerank_and_rrf": candidate_papers,
                        "sub_queries": [sq["query"] for sq in sub_queries],
                        "expansion_terms": {
                            sq["query"]: sq["expansion_terms"] for sq in sub_queries
                        },
                    }
                )

                for key, value in [
                    ("retrieval_avg_score", sum(scores) / len(scores)),
                    ("retrieval_score_spread", max(scores) - min(scores)),
                    ("source_diversity", unique_papers),
                ]:
                    self.langsmith_client.create_feedback(
                        run_id=run.id, key=key, score=value, trace_id=run.trace_id
                    )

        logger.info(
            "Retrieved %d chunks after rerank (scores: %s).",
            len(chunks),
            [round(c["score"], 3) for c in chunks],
        )
        return chunks
