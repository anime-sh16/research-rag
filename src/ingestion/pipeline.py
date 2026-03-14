import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

from src.config.config import settings
from src.config.logging_config import setup_ingestion_logging
from src.ingestion.arxiv_client import ArxivClient, ArxivResult
from src.ingestion.chunker import BasicChunker, ChunkMetaData
from src.ingestion.vector_store import VectorStore

logger = logging.getLogger(__name__)


class TopicIngestionStats(BaseModel):
    topic: str
    fetched: int  # how many arxiv returned
    filtered_by_category: int = 0  # how many dropped by category allowlist
    selected: int  # how many passed category filter + dedup + cap
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
        allowed_categories: set[str] | None = None,
    ):
        self.arxiv_client = ArxivClient()
        self.chunker = BasicChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.topics = topics if isinstance(topics, list) else [topics]
        self.target_papers_no_per_topic = target_papers_no
        self.allowed_categories = (
            allowed_categories
            if allowed_categories is not None
            else settings.ingestion.allowed_categories
        )
        self.seen_ids = set()

    def fetch_paper_single_topic(
        self, topic: str
    ) -> tuple[TopicIngestionStats, list[ArxivResult]]:
        # Phase 1: fetch metadata only (no PDF downloads) for the full candidate pool
        topic_papers = self.arxiv_client.get_arxiv_results(
            query=topic,
            max_results=settings.ingestion.fetch_per_topic,
            download_pdf=False,
        )

        topic_papers_unique: list[ArxivResult] = []
        filtered_by_category = 0

        for paper in topic_papers:
            if len(topic_papers_unique) >= self.target_papers_no_per_topic:
                break

            if paper.primary_category not in self.allowed_categories:
                filtered_by_category += 1
                logger.debug(
                    "Skipping '%s' (category=%s, not in allowlist).",
                    paper.title,
                    paper.primary_category,
                )
                continue

            if paper.entry_id not in self.seen_ids:
                self.seen_ids.add(paper.entry_id)
                topic_papers_unique.append(paper)

        # Phase 2: download PDFs only for the selected papers
        for paper in topic_papers_unique:
            self.arxiv_client.populate_full_text(paper)

        logger.info(
            "Fetched %d unique papers for '%s'.", len(topic_papers_unique), topic
        )

        stats = TopicIngestionStats(
            topic=topic,
            fetched=len(topic_papers),
            filtered_by_category=filtered_by_category,
            selected=len(topic_papers_unique),
        )
        return stats, topic_papers_unique

    def chunk_single_topic(
        self, topic_papers: list[ArxivResult]
    ) -> list[ChunkMetaData]:
        return self.chunker.chunk_all_results(topic_papers)

    def process_single_topic(
        self, topic: str
    ) -> tuple[TopicIngestionStats, list[ChunkMetaData]]:
        topic_stats, topic_papers = self.fetch_paper_single_topic(topic)
        topic_chunks = self.chunk_single_topic(topic_papers)
        topic_stats.chunks = len(topic_chunks)
        return topic_stats, topic_chunks

    def process(self) -> IngestionRunSummary:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        _log_handler = setup_ingestion_logging(run_id)

        try:
            logger.info(
                "Starting ingestion pipeline for %d topics (run_id=%s).",
                len(self.topics),
                run_id,
            )

            vector_store = VectorStore()
            vector_store.ensure_collection(collection_name=settings.db.collection_name)

            all_stats: list[TopicIngestionStats] = []
            all_chunks: list[ChunkMetaData] = []
            total_upserted = 0

            for i, topic in enumerate(self.topics):
                if i > 0:
                    # ArXiv rate-limits rapid successive queries — brief cooldown
                    logger.info(
                        "Waiting 30s before next topic to respect ArXiv rate limits."
                    )
                    time.sleep(30)
                try:
                    topic_stats, topic_chunks = self.process_single_topic(topic)
                    all_stats.append(topic_stats)
                    all_chunks.extend(topic_chunks)
                    vector_store.upsert_chunks(topic_chunks)
                    total_upserted += len(topic_chunks)
                    logger.info(
                        "Upserted %d chunks for topic '%s'.", len(topic_chunks), topic
                    )
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

            self._save_chunks_to_jsonl(all_chunks, chunks_file)
            self._save_summary_to_json(run_summary, summary_file)
            logger.info(
                "Saved %d chunks → %s | summary → %s",
                len(all_chunks),
                chunks_file,
                summary_file,
            )

            logger.info(
                "Ingestion complete. %d / %d chunks upserted to Qdrant.",
                total_upserted,
                len(all_chunks),
            )

        finally:
            logging.getLogger().removeHandler(_log_handler)
            _log_handler.close()

        return run_summary

    @classmethod
    def process_from_jsonl(cls, chunks_path: Path) -> "IngestionRunSummary":
        """Load pre-made chunks from a JSONL file and upsert to Qdrant."""
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        _log_handler = setup_ingestion_logging(run_id)

        try:
            chunks = cls._load_chunks_from_jsonl(chunks_path)
            logger.info(
                "Loaded %d chunks from %s (run_id=%s).",
                len(chunks),
                chunks_path,
                run_id,
            )

            vector_store = VectorStore()
            vector_store.ensure_collection(collection_name=settings.db.collection_name)
            vector_store.upsert_chunks(chunks)

            # Build per-topic stats from chunk metadata
            papers_by_topic: dict[str, set[str]] = {}
            chunks_by_topic: dict[str, int] = {}
            for chunk in chunks:
                topic = chunk.topic or "Unknown"
                papers_by_topic.setdefault(topic, set()).add(chunk.paper_id)
                chunks_by_topic[topic] = chunks_by_topic.get(topic, 0) + 1

            all_stats = [
                TopicIngestionStats(
                    topic=topic,
                    fetched=len(papers_by_topic[topic]),
                    selected=len(papers_by_topic[topic]),
                    chunks=chunks_by_topic[topic],
                )
                for topic in papers_by_topic
            ]

            run_summary = IngestionRunSummary(
                run_id=run_id,
                topics=all_stats,
                total_papers=sum(s.selected for s in all_stats),
                total_chunks=len(chunks),
            )

            logger.info("JSONL ingestion complete. %d chunks upserted.", len(chunks))
            return run_summary

        finally:
            logging.getLogger().removeHandler(_log_handler)
            _log_handler.close()

    @classmethod
    def process_from_pdfs(cls, pdf_dir: Path) -> "IngestionRunSummary":
        """Extract text from local PDFs, chunk, embed, and upsert to Qdrant."""
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        _log_handler = setup_ingestion_logging(run_id)

        try:
            if not pdf_dir.is_dir():
                raise NotADirectoryError(f"PDF directory not found: {pdf_dir}")

            pdf_files = sorted(pdf_dir.rglob("*.pdf"))
            if not pdf_files:
                raise FileNotFoundError(f"No PDF files found in {pdf_dir}")

            logger.info(
                "Found %d PDFs in %s (run_id=%s).", len(pdf_files), pdf_dir, run_id
            )

            chunker = BasicChunker()
            all_chunks: list[ChunkMetaData] = []

            for pdf_path in pdf_files:
                # Use top-level subdirectory name as topic (e.g., data/pdfs/LLM/sub/paper.pdf → "LLM")
                relative_parts = pdf_path.relative_to(pdf_dir).parts
                topic = relative_parts[0] if len(relative_parts) > 1 else "local"
                full_text = ArxivClient.extract_text_from_pdf(pdf_path)
                if not full_text:
                    logger.warning("Skipping %s — no text extracted.", pdf_path.name)
                    continue

                paper_id = pdf_path.stem
                result = ArxivResult(
                    entry_id=paper_id,
                    title=paper_id.replace("_", " ").replace("-", " "),
                    topic=topic,
                    published=datetime.fromtimestamp(pdf_path.stat().st_mtime),
                    summary="",
                    authors=None,
                    comment=None,
                    primary_category="unknown",
                    categories=None,
                    pdf_url=None,
                    full_text=full_text,
                )
                chunks = chunker.chunk_result(result)
                all_chunks.extend(chunks)
                logger.info(
                    "'%s' → %d chunks (topic=%s).", pdf_path.name, len(chunks), topic
                )

            logger.info("Total chunks from PDFs: %d.", len(all_chunks))

            vector_store = VectorStore()
            vector_store.ensure_collection(collection_name=settings.db.collection_name)
            vector_store.upsert_chunks(all_chunks)

            # Build per-topic stats
            papers_by_topic: dict[str, set[str]] = {}
            chunks_by_topic: dict[str, int] = {}
            for chunk in all_chunks:
                t = chunk.topic or "local"
                papers_by_topic.setdefault(t, set()).add(chunk.paper_id)
                chunks_by_topic[t] = chunks_by_topic.get(t, 0) + 1

            all_stats = [
                TopicIngestionStats(
                    topic=t,
                    fetched=len(papers_by_topic[t]),
                    selected=len(papers_by_topic[t]),
                    chunks=chunks_by_topic[t],
                )
                for t in papers_by_topic
            ]

            # Save chunks for reproducibility
            chunks_file = settings.data.temp_dir / f"chunks_{run_id}.jsonl"
            summary_file = settings.data.temp_dir / f"summary_{run_id}.json"

            run_summary = IngestionRunSummary(
                run_id=run_id,
                topics=all_stats,
                total_papers=sum(s.selected for s in all_stats),
                total_chunks=len(all_chunks),
            )

            cls._save_chunks_to_jsonl(all_chunks, chunks_file)
            cls._save_summary_to_json(run_summary, summary_file)
            logger.info(
                "Saved %d chunks → %s | summary → %s",
                len(all_chunks),
                chunks_file,
                summary_file,
            )

            logger.info("PDF ingestion complete. %d chunks upserted.", len(all_chunks))
            return run_summary

        finally:
            logging.getLogger().removeHandler(_log_handler)
            _log_handler.close()

    @staticmethod
    def _load_chunks_from_jsonl(path: Path) -> list[ChunkMetaData]:
        if not path.exists():
            raise FileNotFoundError(f"Chunks file not found: {path}")

        chunks: list[ChunkMetaData] = []
        skipped = 0
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    chunks.append(ChunkMetaData.model_validate(json.loads(line)))
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(
                        "Skipping malformed line %d: %s (%s)", line_no, line[:80], e
                    )
                    skipped += 1
        if skipped:
            logger.warning("Skipped %d malformed lines in %s.", skipped, path)
        return chunks

    @staticmethod
    def _save_chunks_to_jsonl(chunks: list, output_file):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(
                    json.dumps(
                        chunk.model_dump(),
                        ensure_ascii=False,
                        default=lambda obj: (
                            obj.isoformat() if isinstance(obj, datetime) else str(obj)
                        ),
                    )
                )
                f.write("\n")

    @staticmethod
    def _save_summary_to_json(summary: "IngestionRunSummary", output_file):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as f:
            f.write(summary.model_dump_json(indent=2))
