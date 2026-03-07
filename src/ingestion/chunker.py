from datetime import datetime

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel

from src.ingestion.arxiv_client import ArxivResult


class ChunkMetaData(BaseModel):
    chunk_id: str
    chunk_index: int
    paper_id: str
    title: str
    authors: list[str] | None
    primary_category: str
    categories: list[str] | None
    published: datetime
    source_text: str


class BasicChunker:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def chunk_result(self, arxiv_result: ArxivResult) -> list[dict]:

        full_content = f"Title: {arxiv_result.title}\n\nSummary: {arxiv_result.summary}"

        if arxiv_result.comment:
            full_content += f"\n\nComment: {arxiv_result.comment}"

        result_chunks = self.splitter.split_text(full_content)

        chunks = []
        for i, text in enumerate(result_chunks):
            chunks.append(
                ChunkMetaData(
                    chunk_id=f"{arxiv_result.entry_id}_{arxiv_result.title}_{i}",
                    chunk_index=i,
                    paper_id=arxiv_result.entry_id,
                    title=arxiv_result.title,
                    authors=arxiv_result.authors if arxiv_result.authors else [],
                    primary_category=arxiv_result.primary_category,
                    categories=arxiv_result.categories
                    if arxiv_result.categories
                    else [],
                    published=arxiv_result.published,
                    source_text=text,
                )
            )

        return chunks

    def chunk_all_results(self, arxiv_results: list[ArxivResult]):
        all_chunks = []
        for arxiv_result in arxiv_results:
            all_chunks.extend(self.chunk_result(arxiv_result))
        return all_chunks
