import json
import logging
import os
from datetime import datetime

from pydantic import BaseModel

from src.config.config import settings
from src.ingestion.arxiv_client import ArxivClient, ArxivResult
from src.ingestion.chunker import BasicChunker, ChunkMetaData
from src.ingestion.vector_store import VectorStore

logger = logging.getLogger(__name__)


class TopicIngestionStats(BaseModel):
    topic: str
    fetched: int  # how many arxiv returned
    selected: int  # how many passed dedup + cap
    chunks: int = 0  # how many chunks produced


class IngestionRunSummary(BaseModel):
    run_id: str  # timestamp string
    topics: list[TopicIngestionStats]
    total_papers: int
    total_chunks: int


class SimpleIngestionPipeline:
    def __init__(
        self,
        topics: list[str] | str = settings.ingestion.topics,
        target_papers_no: int = settings.ingestion.target_papers_no,
        chunk_size: int = settings.ingestion.chunk_size,
        chunk_overlap: int = settings.ingestion.chunk_overlap,
    ):
        self.arxiv_client = ArxivClient()
        self.chunker = BasicChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.topics = topics if isinstance(topics, list) else [topics]
        self.target_papers_no_per_topic = target_papers_no
        self.seen_ids = set()
        self.all_unique_papers: list[list[ArxivResult]] = []

    def fetch_paper_single_topic(self, topic: str) -> TopicIngestionStats:
        # Phase 1: fetch metadata only (no PDF downloads) for the full candidate pool
        topic_papers = self.arxiv_client.get_arxiv_results(
            query=topic,
            max_results=settings.ingestion.fetch_per_topic,
            download_pdf=False,
        )

        topic_papers_unique: list[ArxivResult] = []

        for paper in topic_papers:
            if len(topic_papers_unique) >= self.target_papers_no_per_topic:
                break

            if paper.entry_id not in self.seen_ids:
                self.seen_ids.add(paper.entry_id)
                topic_papers_unique.append(paper)

        # Phase 2: download PDFs only for the selected papers
        for paper in topic_papers_unique:
            self.arxiv_client.populate_full_text(paper)

        self.all_unique_papers.append(topic_papers_unique)

        logger.info(
            "Fetched %d unique papers for '%s'.", len(topic_papers_unique), topic
        )

        return TopicIngestionStats(
            topic=topic,
            fetched=len(topic_papers),
            selected=len(topic_papers_unique),
        )

    def chunk_single_topic(
        self, topic_papers: list[ArxivResult]
    ) -> list[ChunkMetaData]:
        return self.chunker.chunk_all_results(topic_papers)

    def process_single_topic(
        self, topic: str
    ) -> tuple[TopicIngestionStats, list[ChunkMetaData]]:
        topic_stats = self.fetch_paper_single_topic(topic)
        topic_chunks = self.chunk_single_topic(self.all_unique_papers[-1])
        topic_stats.chunks = len(topic_chunks)
        return topic_stats, topic_chunks

    def process(self) -> IngestionRunSummary:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info(
            "Starting ingestion pipeline for %d topics (run_id=%s).",
            len(self.topics),
            run_id,
        )

        all_stats: list[TopicIngestionStats] = []
        all_chunks: list[ChunkMetaData] = []

        for topic in self.topics:
            try:
                topic_stats, topic_chunks = self.process_single_topic(topic)
                all_stats.append(topic_stats)
                all_chunks.extend(topic_chunks)
            except Exception as e:
                logger.error("Failed to process topic '%s': %s", topic, e)

        run_summary = IngestionRunSummary(
            run_id=run_id,
            topics=all_stats,
            total_papers=sum(s.selected for s in all_stats),
            total_chunks=len(all_chunks),
        )

        chunks_file = settings.data.temp_dir / f"chunks_{run_id}.jsonl"
        summary_file = settings.data.temp_dir / f"summary_{run_id}.json"

        self.save_chunks_to_jsonl(all_chunks, chunks_file)
        self.save_summary_to_json(run_summary, summary_file)
        logger.info(
            "Saved %d chunks → %s | summary → %s",
            len(all_chunks),
            chunks_file,
            summary_file,
        )

        vector_store = VectorStore()
        vector_store.ensure_collection(collection_name=settings.db.collection_name)
        vector_store.upsert_chunks(all_chunks)
        logger.info("Ingestion complete. %d chunks upserted.", len(all_chunks))

        return run_summary

    def save_chunks_to_jsonl(self, chunks: list, output_file):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as f:
            for chunk in chunks:
                f.write(
                    json.dumps(
                        chunk.model_dump(),
                        default=lambda obj: (
                            obj.isoformat() if isinstance(obj, datetime) else str(obj)
                        ),
                    )
                )
                f.write("\n")

    def save_summary_to_json(self, summary: IngestionRunSummary, output_file):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as f:
            f.write(summary.model_dump_json(indent=2))
