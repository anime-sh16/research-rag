import json
import logging
import os
from datetime import datetime

from src.config.config import settings
from src.ingestion.arxiv_client import ArxivClient
from src.ingestion.chunker import BasicChunker
from src.ingestion.vector_store import VectorStore

logger = logging.getLogger(__name__)


class SimpleIngestionPipeline:
    def __init__(
        self,
        query: str,
        max_results: int = settings.ingestion.max_results,
        chunk_size: int = settings.ingestion.chunk_size,
        chunk_overlap: int = settings.ingestion.chunk_overlap,
    ):
        self.arxiv_client = ArxivClient()
        self.chunker = BasicChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.query = query
        self.max_results = max_results

    def process(self):
        logger.info("Starting ingestion pipeline for query: '%s'", self.query)

        arxiv_results = self.arxiv_client.get_arxiv_results(
            self.query, max_results=self.max_results
        )
        all_chunks = self.chunker.chunk_all_results(arxiv_results)

        temp_output_file = settings.data.temp_dir / f"{self.query}.jsonl"
        self.save_chunks_to_json(all_chunks, temp_output_file)
        logger.info("Saved %d chunks to %s", len(all_chunks), temp_output_file)

        vector_store = VectorStore()
        vector_store.ensure_collection(collection_name=settings.db.collection_name)
        vector_store.upsert_chunks(all_chunks)
        logger.info("Ingestion complete. %d chunks upserted.", len(all_chunks))

        return all_chunks

    def save_chunks_to_json(self, chunks: list, output_file: str):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as f:
            for chunk in chunks:
                f.write(
                    json.dumps(
                        chunk.model_dump(),
                        indent=4,
                        default=lambda obj: (
                            obj.isoformat() if isinstance(obj, datetime) else str(obj)
                        ),
                    )
                )
                f.write("\n")
