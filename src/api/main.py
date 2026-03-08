import logging

from fastapi import FastAPI
from pydantic import BaseModel

from src.config.config import settings
from src.generation.chain import RAGChain
from src.retrieval.retriever import Retriever

logger = logging.getLogger(__name__)

app = FastAPI(title=settings.api.title)

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


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    logger.info("Received query: '%s'", request.question)
    chunks = retriever.retrieve(request.question)
    answer = chain.generate(request.question, chunks)
    logger.info("Returning answer with %d sources.", len(chunks))
    sources = [
        SourceChunk(
            title=chunk["title"],
            authors=chunk["authors"],
            paper_id=chunk["paper_id"],
            chunk_index=chunk["chunk_index"],
            score=chunk["score"],
        )
        for chunk in chunks
    ]
    return QueryResponse(answer=answer, sources=sources)
