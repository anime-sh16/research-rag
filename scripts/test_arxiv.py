import json
from datetime import datetime

from src.ingestion.arxiv_client import ArxivClient
from src.ingestion.chunker import BasicChunker

client = ArxivClient()
chunker = BasicChunker()
results = client.get_arxiv_results("machine learning", max_results=2)

all_chunks = chunker.chunk_all_results(results)

for chunk in all_chunks:
    # pretty print chunks
    chunk_json = chunk.model_dump()
    print(
        json.dumps(
            chunk_json,
            indent=4,
            default=lambda obj: (
                obj.isoformat() if isinstance(obj, datetime) else str(obj)
            ),
        )
    )
    print("=============")

print(f"Total Chunks: {len(all_chunks)}")
