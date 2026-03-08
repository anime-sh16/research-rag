import argparse
import logging
import sys

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

    topics = args.topics or settings.ingestion.topics
    pipeline = SimpleIngestionPipeline(
        topics=topics,
        target_papers_no=args.target,
    )
    summary = pipeline.process()
    print(summary.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
