import json
import os
from datetime import datetime

from src.config.config import data_settings, db_settings
from src.ingestion.arxiv_client import ArxivClient
from src.ingestion.chunker import BasicChunker
from src.ingestion.vector_store import VectorStore


class SimpleIngestionPipeline:
    def __init__(
        self,
        query: str,
        max_results: int = 10,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ):
        self.arxiv_client = ArxivClient()
        self.chunker = BasicChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.query = query
        self.max_results = max_results

    def process(self):
        arxiv_results = self.arxiv_client.get_arxiv_results(
            self.query, max_results=self.max_results
        )
        all_chunks = self.chunker.chunk_all_results(arxiv_results)
        
        # save chunks to json tmp file
        temp_output_file = data_settings.temp_dir / f"{self.query}.jsonl"
        self.save_chunks_to_json(all_chunks, temp_output_file)
        
        # save chunks to vector store
        vector_store = VectorStore()
        vector_store.ensure_collection(collection_name=db_settings.collection_name)
        vector_store.upsert_chunks(all_chunks)
        
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
