import argparse
import logging
import sys
from pathlib import Path

from src.config.config import settings
from src.ingestion.pipeline import SimpleIngestionPipeline


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
        "--from-pdfs",
        type=Path,
        default=None,
        metavar="DIR",
        help="Skip ArXiv fetch — extract text from local PDFs, chunk, embed, and upsert.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    if args.from_chunks:
        summary = SimpleIngestionPipeline.process_from_jsonl(args.from_chunks)
    elif args.from_pdfs:
        summary = SimpleIngestionPipeline.process_from_pdfs(args.from_pdfs)
    else:
        topics = args.topics or settings.ingestion.topics
        pipeline = SimpleIngestionPipeline(
            topics=topics,
            target_papers_no=args.target,
        )
        summary = pipeline.process()

    print(summary.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
