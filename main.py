import argparse
import json
import logging
import sys
from pathlib import Path

from src.config.config import settings
from src.ingestion.chunker import ChunkMetaData
from src.ingestion.pipeline import SimpleIngestionPipeline
from src.ingestion.vector_store import VectorStore


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the ArXiv ingestion pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--topics",
        nargs="+",
        default=None,
        metavar="TOPIC",
        help="Override topics from config (space-separated, quote multi-word topics).",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=settings.ingestion.target_papers_no,
        metavar="N",
        help="Target papers to select per topic after dedup.",
    )
    parser.add_argument(
        "--from-chunks",
        type=Path,
        default=None,
        metavar="PATH",
        help="Skip fetch/chunk — embed and upsert directly from a saved chunks JSONL file.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def _upsert_from_jsonl(path: Path) -> None:
    if not path.exists():
        logging.error("Chunks file not found: %s", path)
        sys.exit(1)

    chunks: list[ChunkMetaData] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(ChunkMetaData.model_validate(json.loads(line)))

    logging.info("Loaded %d chunks from %s", len(chunks), path)

    store = VectorStore()
    store.ensure_collection(collection_name=settings.db.collection_name)
    store.upsert_chunks(chunks)
    logging.info("Done.")


def main() -> None:
    args = _parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    if args.from_chunks:
        _upsert_from_jsonl(args.from_chunks)
        return

    topics = args.topics or settings.ingestion.topics
    pipeline = SimpleIngestionPipeline(
        topics=topics,
        target_papers_no=args.target,
    )
    summary = pipeline.process()
    print(summary.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
