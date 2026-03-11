"""RAGAS evaluation runner integrated with LangSmith Experiments.

Usage:
    uv run python src/evaluation/ragas_runner.py --experiment <name>

Flow:
    1. LangSmith evaluate() iterates the dataset, calling run_pipeline() per row
    2. Each pipeline call produces a full operational trace (retrieval + generation)
    3. RAGAS evaluator functions score each (run, example) pair
    4. Scores are attached as feedback on each run in the LangSmith Experiment
    5. A local snapshot is saved to evaluation/results/<name>.json
"""

import argparse
import asyncio
import json
import logging
import os
from datetime import datetime

from google import genai
from langsmith import evaluate
from langsmith.schemas import Example, Run
from ragas.embeddings.base import embedding_factory
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

from src.api.main import run_pipeline
from src.config.config import settings

logger = logging.getLogger(__name__)

# Evaluator LLM & Embeddings (Gemini via RAGAS factories)
# embedding_factory reads GOOGLE_API_KEY from os.environ
os.environ["GOOGLE_API_KEY"] = settings.google_api_key.get_secret_value()

_gemini_client = genai.Client(api_key=settings.google_api_key.get_secret_value())
_evaluator_llm = llm_factory(
    model=settings.evaluation.evaluator_model,
    provider="google",
    client=_gemini_client,
)
_evaluator_embeddings = embedding_factory(
    provider="google",
    model=settings.db.embedding_model,
)

# RAGAS metric instances (created once, reused across evaluations)
_faithfulness = Faithfulness(llm=_evaluator_llm)
_answer_relevancy = AnswerRelevancy(
    llm=_evaluator_llm,
    embeddings=_evaluator_embeddings,
)
_context_precision = ContextPrecision(llm=_evaluator_llm)
_context_recall = ContextRecall(llm=_evaluator_llm)


# Target function: called once per dataset row by LangSmith evaluate()
def target(inputs: dict) -> dict:
    """Run the RAG pipeline and format output for RAGAS evaluation."""
    result = run_pipeline(inputs["question"])
    contexts = [chunk["text"] for chunk in result["sources"] if chunk.get("text")]
    return {
        "answer": result["answer"],
        "contexts": contexts,
    }


# RAGAS evaluator functions (called per-question by LangSmith)
def _safe_ragas_score(coro) -> float | None:
    """Run an async RAGAS scorer, returning None on failure."""
    try:
        result = asyncio.run(coro)
        return result.value
    except Exception:
        logger.exception("RAGAS scoring failed")
        return None


def eval_faithfulness(run: Run, example: Example) -> dict:
    score = _safe_ragas_score(
        _faithfulness.ascore(
            user_input=example.inputs["question"],
            response=run.outputs["answer"],
            retrieved_contexts=run.outputs["contexts"],
        )
    )
    return {"key": "faithfulness", "score": score}


def eval_answer_relevancy(run: Run, example: Example) -> dict:
    score = _safe_ragas_score(
        _answer_relevancy.ascore(
            user_input=example.inputs["question"],
            response=run.outputs["answer"],
        )
    )
    return {"key": "answer_relevancy", "score": score}


def eval_context_precision(run: Run, example: Example) -> dict:
    outputs = run.outputs or {}
    score = _safe_ragas_score(
        _context_precision.ascore(
            user_input=example.inputs["question"],
            retrieved_contexts=outputs.get("contexts", []),
            reference=example.outputs["ground_truth"],
        )
    )
    return {"key": "context_precision", "score": score}


def eval_context_recall(run: Run, example: Example) -> dict:
    outputs = run.outputs or {}
    score = _safe_ragas_score(
        _context_recall.ascore(
            user_input=example.inputs["question"],
            retrieved_contexts=outputs.get("contexts", []),
            reference=example.outputs["ground_truth"],
        )
    )
    return {"key": "context_recall", "score": score}


# Snapshot: save results locally
def _save_snapshot(experiment_name: str, results) -> str:
    """Extract per-question scores from ExperimentResults and save as JSON."""
    results_dir = settings.evaluation.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    per_question = []
    metric_scores: dict[str, list[float]] = {
        "faithfulness": [],
        "answer_relevancy": [],
        "context_precision": [],
        "context_recall": [],
    }

    for row in results:
        q_scores = {}
        for eval_result in row["evaluation_results"]["results"]:
            if eval_result.score is not None:
                q_scores[eval_result.key] = eval_result.score
                metric_scores[eval_result.key].append(eval_result.score)

        per_question.append(
            {
                "id": row["example"].metadata.get("id", "unknown"),
                "question": row["example"].inputs["question"],
                "question_type": row["example"].metadata.get("question_type"),
                "question_subtype": row["example"].metadata.get("question_subtype"),
                "scores": q_scores,
            }
        )

    aggregate = {}
    for key, values in metric_scores.items():
        aggregate[key] = round(sum(values) / len(values), 4) if values else None

    snapshot = {
        "experiment": experiment_name,
        "timestamp": datetime.now().isoformat(),
        "pipeline_version": settings.pipeline_version,
        "aggregate": aggregate,
        "per_question": per_question,
    }

    snapshot_path = results_dir / f"{experiment_name}.json"
    with open(snapshot_path, "w") as f:
        json.dump(snapshot, f, indent=2)

    return str(snapshot_path)


def run_evaluation(experiment_name: str) -> None:
    """Run RAGAS evaluation on the fixed eval set and save results."""
    dataset_name = settings.evaluation.dataset_name

    logger.info(
        "Starting evaluation experiment '%s' against dataset '%s'",
        experiment_name,
        dataset_name,
    )

    results = evaluate(
        target,
        data=dataset_name,
        evaluators=[
            eval_faithfulness,
            eval_answer_relevancy,
            eval_context_precision,
            eval_context_recall,
        ],
        experiment_prefix=experiment_name,
        metadata={
            "eval_run": True,
            "pipeline_version": settings.pipeline_version,
        },
    )

    snapshot_path = _save_snapshot(experiment_name, results)

    logger.info("Evaluation complete. Snapshot saved to %s", snapshot_path)
    print(f"\nResults saved to {snapshot_path}")

    with open(snapshot_path) as f:
        snapshot = json.load(f)
    print("\nAggregate scores:")
    for metric, score in snapshot["aggregate"].items():
        print(f"  {metric}: {score}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Run RAGAS evaluation on the fixed eval set."
    )
    parser.add_argument(
        "--experiment",
        default=settings.pipeline_version,
        help="Name for this experiment (default: current pipeline_version)",
    )
    args = parser.parse_args()
    run_evaluation(args.experiment)
