from fastapi import FastAPI
from pydantic import BaseModel

from src.generation.chain import RAGChain
from src.retrieval.retriever import Retriever

app = FastAPI(title="ArXiv RAG API")

retriever = Retriever(top_k=5)
chain = RAGChain(model="gemini-2.5-flash-lite")


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
    chunks = retriever.retrieve(request.question)

    answer = chain.generate(request.question, chunks)

    # step 3: build the sources list from the retrieved chunks
    # hint: use a list comprehension over chunks, constructing a SourceChunk for each
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
