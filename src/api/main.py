import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Annotated

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree
from pydantic import BaseModel, StringConstraints
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from src.api.errors import OFF_DOMAIN_MESSAGE, map_exception
from src.config.config import settings
from src.config.logging_config import setup_api_logging
from src.generation.chain import RAGChain
from src.retrieval.retriever import OffDomainQuery, Retriever

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_api_logging()
    # Clients are constructed here, not at import time, so importing this module
    # has no side effects and startup/shutdown of network connections is explicit.
    app.state.retriever = Retriever(
        top_k=settings.generation.top_k, http_client=httpx.AsyncClient()
    )
    app.state.chain = RAGChain(model=settings.generation.model)
    logger.info("API server started. Logging to logs/api/api.log")
    yield
    await app.state.retriever.aclose()


app = FastAPI(title=settings.api.title, lifespan=lifespan)

limiter = Limiter(
    key_func=get_remote_address,
    enabled=settings.api.rate_limit_enabled,
    headers_enabled=True,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


class QueryRequest(BaseModel):
    question: Annotated[
        str,
        StringConstraints(
            strip_whitespace=True,
            min_length=settings.api.query_min_length,
            max_length=settings.api.query_max_length,
        ),
    ]


class SourceChunk(BaseModel):
    title: str
    authors: list[str] | None
    paper_id: str
    chunk_index: int
    score: float


class HealthResponse(BaseModel):
    status: str
    pipeline_version: str
    git_sha: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]


@traceable(
    run_type="chain",
    tags=[
        f"pipeline_version:{settings.pipeline_version}",
        "retrieval_method:hybrid_rerank",
    ],
)
async def run_pipeline(question: str, prompt_version: str | None = None) -> dict:
    """Core orchestration logic, decoupled from HTTP for easier evaluation."""
    run = get_current_run_tree()

    if run:
        run.name = f"query|{settings.pipeline_version}|{datetime.now().strftime('%m%d_%H%M%S')}"

    try:
        chunks = await app.state.retriever.retrieve(question)
    except OffDomainQuery:
        if run:
            run.add_metadata(
                {
                    "summary": {
                        "query": question,
                        "chunks_retrieved": 0,
                        "papers_cited": [],
                        "answer_preview": OFF_DOMAIN_MESSAGE,
                        "retrieval_method": "hybrid_rerank",
                        "flag": "off_domain",
                    }
                }
            )
        return {"answer": OFF_DOMAIN_MESSAGE, "sources": []}

    # Handle the empty retrieval edge case gracefully
    if not chunks:
        if run:
            run.add_metadata(
                {
                    "summary": {
                        "query": question,
                        "chunks_retrieved": 0,
                        "papers_cited": [],
                        "answer_preview": "I don't have enough context to answer that.",
                        "retrieval_method": "hybrid_rerank",
                        "flag": "empty_retrieval",
                    }
                }
            )
        return {"answer": "I don't have enough context to answer that.", "sources": []}

    answer = await app.state.chain.generate(
        question, chunks, prompt_version=prompt_version
    )

    # Add a human-readable summary to the Root Span
    if run:
        scores = [c["score"] for c in chunks]
        avg_score = sum(scores) / len(scores) if scores else 0
        unique_papers = len(set(c["paper_id"] for c in chunks if c.get("paper_id")))

        flag = None
        if avg_score < 0.5:
            flag = "low_retrieval_score"
        elif unique_papers == 1 and len(chunks) > 1:
            flag = "single_source_warning"

        # Human-Readable Summary
        run.add_metadata(
            {
                "summary": {
                    "query": question,
                    "chunks_retrieved": len(chunks),
                    "papers_cited": list(
                        set(c["paper_id"] for c in chunks if c.get("paper_id"))
                    ),
                    "answer_preview": answer[:150] + "..."
                    if len(answer) > 150
                    else answer,
                    "retrieval_method": "hybrid_rerank",
                    "flag": flag,
                }
            }
        )

    return {"answer": answer, "sources": chunks}


@app.post("/query", response_model=QueryResponse)
@limiter.limit(f"{settings.api.rate_limit_per_minute}/minute")
@limiter.limit(f"{settings.api.rate_limit_per_day}/day")
async def query(
    request: Request, response: Response, body: QueryRequest
) -> QueryResponse:
    logger.info("Received query: '%s'", body.question)
    try:
        result = await run_pipeline(body.question)
    except Exception as e:
        logger.exception("Query pipeline failed for: '%s'", body.question)
        status_code, error_body = map_exception(e)
        raise HTTPException(status_code=status_code, detail=error_body.model_dump())
    logger.info("Returning answer with %d sources.", len(result["sources"]))
    sources = [
        SourceChunk(
            title=chunk["title"],
            authors=chunk["authors"],
            paper_id=chunk["paper_id"],
            chunk_index=chunk["chunk_index"],
            score=chunk["score"],
        )
        for chunk in result["sources"]
    ]
    return QueryResponse(answer=result["answer"], sources=sources)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        pipeline_version=settings.pipeline_version,
        git_sha=settings.git_sha,
    )
