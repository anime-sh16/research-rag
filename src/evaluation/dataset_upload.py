import json
import logging
from pathlib import Path

from langsmith import Client

from src.config.config import settings

logger = logging.getLogger(__name__)


def upload_dataset(
    evalset_path: Path = settings.evaluation.evalset_path,
    dataset_name: str = settings.evaluation.dataset_name,
) -> str:
    """Upload evalset.json to LangSmith and return the dataset ID.
    Each question becomes a LangSmith Example with:
        - inputs:   {"question": ...}
        - outputs:  {"ground_truth": ...}
        - metadata: {id, question_type, question_subtype, source_papers}"""
    client = Client()

    with open(evalset_path) as f:
        evalset = json.load(f)

    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description=(
            f"RAGAS evaluation set for ArXiv RAG pipeline ({len(evalset)} questions). "
        ),
    )

    for q in evalset:
        client.create_example(
            inputs={"question": q["question"]},
            outputs={"ground_truth": q["ground_truth"]},
            metadata={
                "id": q["id"],
                "question_type": q["question_type"],
                "question_subtype": q["question_subtype"],
                "source_papers": q["source_papers"],
            },
            dataset_id=dataset.id,
        )

    logger.info(
        "Uploaded %d examples to dataset '%s' (id=%s)",
        len(evalset),
        dataset_name,
        dataset.id,
    )
    return str(dataset.id)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    dataset_id = upload_dataset()
    print(f"Dataset created: {dataset_id}")
