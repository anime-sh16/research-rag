import argparse
import asyncio
import json
import logging
import os
from datetime import datetime

import instructor
from google import genai
from google.genai import types
from langsmith import evaluate
from langsmith.schemas import Example, Run
from ragas.embeddings.base import embedding_factory
from ragas.llms.base import InstructorLLM
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

from src.api.main import run_pipeline
from src.config.config import settings

# Suppress noisy third-party loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("google_genai").setLevel(logging.INFO)
logging.getLogger("src.retrieval").setLevel(logging.WARNING)
logging.getLogger("src.generation").setLevel(logging.WARNING)
logging.getLogger("tenacity").setLevel(logging.INFO)
logging.getLogger("langsmith").setLevel(logging.INFO)


logger = logging.getLogger(__name__)

os.environ["GOOGLE_API_KEY"] = settings.google_api_key.get_secret_value()

_gemini_client = genai.Client(
    api_key=settings.google_api_key.get_secret_value(),
    http_options=types.HttpOptions(timeout=120000),
)
_async_instructor = instructor.from_genai(
    _gemini_client,
    mode=instructor.Mode.GENAI_STRUCTURED_OUTPUTS,
    use_async=True,
)
_evaluator_llm = InstructorLLM(
    client=_async_instructor,
    model=settings.evaluation.evaluator_model,
    provider="google",
    thinking_config=types.ThinkingConfig(thinking_level="low", include_thoughts=True),
)
_evaluator_embeddings = embedding_factory(
    provider="google",
    model=settings.db.embedding_model,
)

_faithfulness = Faithfulness(llm=_evaluator_llm)
_answer_relevancy = AnswerRelevancy(
    llm=_evaluator_llm, embeddings=_evaluator_embeddings
)
_context_precision = ContextPrecision(llm=_evaluator_llm)
_context_recall = ContextRecall(llm=_evaluator_llm)


def make_target(prompt_version: str | None = None):
    """Return a target function that uses the specified prompt variant."""

    def target(inputs: dict) -> dict:
        result = run_pipeline(inputs["question"], prompt_version=prompt_version)
        contexts = [chunk["text"] for chunk in result["sources"] if chunk.get("text")]
        return {"answer": result["answer"], "contexts": contexts}

    return target


def _safe_ragas_score(metric_name: str, coro) -> float | None:
    """Run an async RAGAS scorer, returning None on failure."""
    try:
        result = asyncio.run(coro)
        score = result.value
        logger.debug("  [PASS] %-22s score=%.4f", metric_name, score)
        return score
    except Exception as e:
        logger.warning("  [FAIL] %-22s error=%s: %s", metric_name, type(e).__name__, e)
        return None


def eval_faithfulness(run: Run, example: Example) -> dict:
    outputs = run.outputs or {}
    answer = outputs.get("answer")
    contexts = outputs.get("contexts", [])
    if not answer:
        return {"key": "faithfulness", "score": None}
    score = _safe_ragas_score(
        "faithfulness",
        _faithfulness.ascore(
            user_input=example.inputs["question"],
            response=answer,
            retrieved_contexts=contexts,
        ),
    )
    return {"key": "faithfulness", "score": score}


def eval_answer_relevancy(run: Run, example: Example) -> dict:
    outputs = run.outputs or {}
    answer = outputs.get("answer")
    if not answer:
        return {"key": "answer_relevancy", "score": None}
    score = _safe_ragas_score(
        "answer_relevancy",
        _answer_relevancy.ascore(
            user_input=example.inputs["question"],
            response=answer,
        ),
    )
    return {"key": "answer_relevancy", "score": score}


def eval_context_precision(run: Run, example: Example) -> dict:
    outputs = run.outputs or {}
    score = _safe_ragas_score(
        "context_precision",
        _context_precision.ascore(
            user_input=example.inputs["question"],
            retrieved_contexts=outputs.get("contexts", []),
            reference=example.outputs["ground_truth"],
        ),
    )
    return {"key": "context_precision", "score": score}


def eval_context_recall(run: Run, example: Example) -> dict:
    outputs = run.outputs or {}
    score = _safe_ragas_score(
        "context_recall",
        _context_recall.ascore(
            user_input=example.inputs["question"],
            retrieved_contexts=outputs.get("contexts", []),
            reference=example.outputs["ground_truth"],
        ),
    )
    return {"key": "context_recall", "score": score}


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

        run = row.get("run")
        run_outputs = (run.outputs or {}) if run else {}
        example_outputs = row["example"].outputs or {}

        per_question.append(
            {
                "id": row["example"].metadata.get("id", "unknown"),
                "question": row["example"].inputs["question"],
                "question_type": row["example"].metadata.get("question_type"),
                "question_subtype": row["example"].metadata.get("question_subtype"),
                "answer": run_outputs.get("answer"),
                "reference": example_outputs.get("ground_truth"),
                "scores": q_scores,
            }
        )

    aggregate = {
        key: round(sum(vals) / len(vals), 4) if vals else None
        for key, vals in metric_scores.items()
    }

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


def run_evaluation(experiment_name: str, prompt_version: str | None = None) -> None:
    """Run RAGAS evaluation on the fixed eval set and save results."""
    dataset_name = settings.evaluation.dataset_name

    logger.info("Experiment : %s", experiment_name)
    logger.info("Dataset    : %s", dataset_name)
    logger.info("Model      : %s", settings.evaluation.evaluator_model)
    logger.info("Prompt     : %s", prompt_version or "default")

    results = evaluate(
        make_target(prompt_version),
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
    logger.info("Snapshot   : %s", snapshot_path)

    with open(snapshot_path) as f:
        snapshot = json.load(f)

    print("\nAggregate scores:")
    for metric, score in snapshot["aggregate"].items():
        status = f"{score:.4f}" if score is not None else "N/A (all failed)"
        print(f"  {metric:<22} {status}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)-8s %(name)s — %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Run RAGAS evaluation on the fixed eval set."
    )
    parser.add_argument(
        "--experiment",
        default=settings.pipeline_version,
        help="Name for this experiment (default: current pipeline_version)",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Prompt variant to use (default: RAGChain default). Must be a key in PROMPT_VARIANTS.",
    )
    args = parser.parse_args()
    run_evaluation(args.experiment, prompt_version=args.prompt)
