import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from langsmith import traceable
from langsmith.run_helpers import get_current_run
from pydantic import BaseModel

from src.config.config import settings
from src.config.logging_config import setup_api_logging
from src.generation.chain import RAGChain
from src.retrieval.retriever import Retriever

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_api_logging()
    logger.info("API server started. Logging to logs/api/api.log")
    yield


app = FastAPI(title=settings.api.title, lifespan=lifespan)

retriever = Retriever(top_k=settings.generation.top_k)
chain = RAGChain(model=settings.generation.model)


class QueryRequest(BaseModel):
    question: str


class SourceChunk(BaseModel):
    title: str
    authors: list[str] | None
    paper_id: str
    chunk_index: int
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]


@traceable(
    run_type="chain",
    name="query_pipeline",
    tags=["pipeline_version:baseline", "query_type:general"],
)
def run_pipeline(question: str) -> dict:
    """Core orchestration logic, decoupled from HTTP for easier evaluation."""
    chunks = retriever.retrieve(question)

    # Handle the empty retrieval edge case gracefully
    if not chunks:
        return {"answer": "I don't have enough context to answer that.", "sources": []}

    answer = chain.generate(question, chunks)

    # Add a human-readable summary to the Root Span
    run = get_current_run()
    if run:
        run.add_metadata(
            {
                "summary": {
                    "chunks_retrieved": len(chunks),
                    "answer_preview": answer[:150] + "..."
                    if len(answer) > 150
                    else answer,
                }
            }
        )

    return {"answer": answer, "sources": chunks}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    logger.info("Received query: '%s'", request.question)
    try:
        result = run_pipeline(request.question)
    except Exception as e:
        logger.exception("Query pipeline failed for: '%s'", request.question)
        raise HTTPException(status_code=503, detail=str(e))
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
