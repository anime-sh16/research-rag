import json
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Module-level guard: patch ragas_runner's heavy init before first import
#
# ragas_runner.py instantiates Gemini clients, instructor wrappers, and RAGAS
# metric objects at module level (lines 47–68). These constructors are safe
# in most environments, but embedding_factory may validate the model name
# against the live API. Patching the constructors here makes the module
# importable in any CI environment without live credentials.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def _patch_ragas_runner_init():
    """Prevent module-level API client construction in ragas_runner."""
    with (
        patch("google.genai.Client", return_value=MagicMock()),
        patch("instructor.from_genai", return_value=MagicMock()),
        patch("ragas.llms.base.InstructorLLM", return_value=MagicMock()),
        patch("ragas.embeddings.base.embedding_factory", return_value=MagicMock()),
        patch("ragas.metrics.collections.Faithfulness", return_value=MagicMock()),
        patch("ragas.metrics.collections.AnswerRelevancy", return_value=MagicMock()),
        patch("ragas.metrics.collections.ContextPrecision", return_value=MagicMock()),
        patch("ragas.metrics.collections.ContextRecall", return_value=MagicMock()),
    ):
        yield


def _make_evalset(n: int = 2) -> list[dict]:
    return [
        {
            "id": f"q_{i:03d}",
            "question": f"Question {i}?",
            "ground_truth": f"Answer {i}.",
            "question_type": "factual",
            "question_subtype": "definition",
            "source_papers": [f"paper_{i}"],
        }
        for i in range(n)
    ]


def _make_fake_run(answer: str = "Generated answer.", contexts: list | None = None):
    run = MagicMock()
    run.outputs = {
        "answer": answer,
        "contexts": contexts or ["Some retrieved chunk."],
    }
    return run


def _make_fake_example(question: str = "What is attention?", ground_truth: str = "GT"):
    example = MagicMock()
    example.inputs = {"question": question}
    example.outputs = {"ground_truth": ground_truth}
    example.metadata = {
        "id": "q_001",
        "question_type": "factual",
        "question_subtype": "definition",
    }
    return example


class TestUploadDataset:
    def test_creates_dataset_with_correct_name(self, tmp_path) -> None:
        evalset = _make_evalset(2)
        evalset_file = tmp_path / "evalset.json"
        evalset_file.write_text(json.dumps(evalset))

        mock_client = MagicMock()
        mock_dataset = MagicMock()
        mock_dataset.id = "fake-dataset-id"
        mock_client.create_dataset.return_value = mock_dataset

        with patch("src.evaluation.dataset_upload.Client", return_value=mock_client):
            from src.evaluation.dataset_upload import upload_dataset

            upload_dataset(evalset_path=evalset_file, dataset_name="my-eval-set")

        mock_client.create_dataset.assert_called_once()
        call_kwargs = mock_client.create_dataset.call_args.kwargs
        assert call_kwargs["dataset_name"] == "my-eval-set"

    def test_creates_one_example_per_question(self, tmp_path) -> None:
        evalset = _make_evalset(3)
        evalset_file = tmp_path / "evalset.json"
        evalset_file.write_text(json.dumps(evalset))

        mock_client = MagicMock()
        mock_dataset = MagicMock()
        mock_dataset.id = "fake-dataset-id"
        mock_client.create_dataset.return_value = mock_dataset

        with patch("src.evaluation.dataset_upload.Client", return_value=mock_client):
            from src.evaluation.dataset_upload import upload_dataset

            upload_dataset(evalset_path=evalset_file, dataset_name="my-eval-set")

        assert mock_client.create_example.call_count == 3

    def test_example_inputs_contain_question(self, tmp_path) -> None:
        evalset = _make_evalset(1)
        evalset_file = tmp_path / "evalset.json"
        evalset_file.write_text(json.dumps(evalset))

        mock_client = MagicMock()
        mock_dataset = MagicMock()
        mock_dataset.id = "fake-dataset-id"
        mock_client.create_dataset.return_value = mock_dataset

        with patch("src.evaluation.dataset_upload.Client", return_value=mock_client):
            from src.evaluation.dataset_upload import upload_dataset

            upload_dataset(evalset_path=evalset_file, dataset_name="my-eval-set")

        call_kwargs = mock_client.create_example.call_args.kwargs
        assert "question" in call_kwargs["inputs"]
        assert call_kwargs["inputs"]["question"] == evalset[0]["question"]

    def test_example_outputs_contain_ground_truth(self, tmp_path) -> None:
        evalset = _make_evalset(1)
        evalset_file = tmp_path / "evalset.json"
        evalset_file.write_text(json.dumps(evalset))

        mock_client = MagicMock()
        mock_dataset = MagicMock()
        mock_dataset.id = "fake-dataset-id"
        mock_client.create_dataset.return_value = mock_dataset

        with patch("src.evaluation.dataset_upload.Client", return_value=mock_client):
            from src.evaluation.dataset_upload import upload_dataset

            upload_dataset(evalset_path=evalset_file, dataset_name="my-eval-set")

        call_kwargs = mock_client.create_example.call_args.kwargs
        assert "ground_truth" in call_kwargs["outputs"]

    def test_returns_dataset_id_as_string(self, tmp_path) -> None:
        evalset = _make_evalset(1)
        evalset_file = tmp_path / "evalset.json"
        evalset_file.write_text(json.dumps(evalset))

        mock_client = MagicMock()
        mock_dataset = MagicMock()
        mock_dataset.id = "abc-123"
        mock_client.create_dataset.return_value = mock_dataset

        with patch("src.evaluation.dataset_upload.Client", return_value=mock_client):
            from src.evaluation.dataset_upload import upload_dataset

            result = upload_dataset(evalset_path=evalset_file, dataset_name="test")

        assert result == "abc-123"


# ---------------------------------------------------------------------------
# TestSaveSnapshot
# ---------------------------------------------------------------------------


class TestSaveSnapshot:
    """_save_snapshot is a pure I/O function — no LLM required."""

    def _make_results_rows(
        self,
        scores: dict,
        example=None,
        answer: str = "Generated answer.",
        ground_truth: str = "GT",
    ) -> list[dict]:
        """Build a minimal fake ExperimentResults iterable."""
        example = example or _make_fake_example(ground_truth=ground_truth)
        eval_results = []
        for key, score in scores.items():
            r = MagicMock()
            r.key = key
            r.score = score
            eval_results.append(r)

        run = _make_fake_run(answer=answer)

        return [
            {
                "example": example,
                "run": run,
                "evaluation_results": {"results": eval_results},
            }
        ]

    def test_creates_json_file_in_results_dir(self, tmp_path) -> None:
        rows = self._make_results_rows({"faithfulness": 0.9})

        with patch("src.evaluation.ragas_runner.settings") as mock_settings:
            mock_settings.evaluation.results_dir = tmp_path
            mock_settings.pipeline_version = "v1-baseline"

            from src.evaluation.ragas_runner import _save_snapshot

            path = _save_snapshot("my-experiment", rows)

        assert (tmp_path / "my-experiment.json").exists()
        assert path.endswith("my-experiment.json")

    def test_snapshot_contains_experiment_name(self, tmp_path) -> None:
        rows = self._make_results_rows({"faithfulness": 0.8})

        with patch("src.evaluation.ragas_runner.settings") as mock_settings:
            mock_settings.evaluation.results_dir = tmp_path
            mock_settings.pipeline_version = "v1-baseline"

            from src.evaluation.ragas_runner import _save_snapshot

            _save_snapshot("exp-name", rows)

        with open(tmp_path / "exp-name.json") as f:
            data = json.load(f)
        assert data["experiment"] == "exp-name"

    def test_aggregate_scores_are_correct(self, tmp_path) -> None:
        row1 = self._make_results_rows({"faithfulness": 0.8})
        row2 = self._make_results_rows({"faithfulness": 0.6})
        rows = row1 + row2

        with patch("src.evaluation.ragas_runner.settings") as mock_settings:
            mock_settings.evaluation.results_dir = tmp_path
            mock_settings.pipeline_version = "v1-baseline"

            from src.evaluation.ragas_runner import _save_snapshot

            _save_snapshot("agg-test", rows)

        with open(tmp_path / "agg-test.json") as f:
            data = json.load(f)
        # avg of 0.8 and 0.6 = 0.7
        assert data["aggregate"]["faithfulness"] == pytest.approx(0.7, abs=1e-4)

    def test_none_score_excluded_from_aggregate(self, tmp_path) -> None:
        """Scores that are None (evaluator failed) must not be averaged in."""
        r = MagicMock()
        r.key = "faithfulness"
        r.score = None
        row = [
            {
                "example": _make_fake_example(),
                "run": _make_fake_run(),
                "evaluation_results": {"results": [r]},
            }
        ]

        with patch("src.evaluation.ragas_runner.settings") as mock_settings:
            mock_settings.evaluation.results_dir = tmp_path
            mock_settings.pipeline_version = "v1-baseline"

            from src.evaluation.ragas_runner import _save_snapshot

            _save_snapshot("none-test", row)

        with open(tmp_path / "none-test.json") as f:
            data = json.load(f)
        assert data["aggregate"]["faithfulness"] is None

    def test_per_question_list_length_matches_row_count(self, tmp_path) -> None:
        rows = self._make_results_rows({"faithfulness": 0.9}) + self._make_results_rows(
            {"faithfulness": 0.7}
        )

        with patch("src.evaluation.ragas_runner.settings") as mock_settings:
            mock_settings.evaluation.results_dir = tmp_path
            mock_settings.pipeline_version = "v1-baseline"

            from src.evaluation.ragas_runner import _save_snapshot

            _save_snapshot("count-test", rows)

        with open(tmp_path / "count-test.json") as f:
            data = json.load(f)
        assert len(data["per_question"]) == 2

    def test_per_question_contains_answer_from_run(self, tmp_path) -> None:
        rows = self._make_results_rows(
            {"faithfulness": 0.9}, answer="The transformer uses self-attention."
        )

        with patch("src.evaluation.ragas_runner.settings") as mock_settings:
            mock_settings.evaluation.results_dir = tmp_path
            mock_settings.pipeline_version = "v1-baseline"

            from src.evaluation.ragas_runner import _save_snapshot

            _save_snapshot("answer-test", rows)

        with open(tmp_path / "answer-test.json") as f:
            data = json.load(f)
        assert (
            data["per_question"][0]["answer"] == "The transformer uses self-attention."
        )

    def test_per_question_contains_reference_from_ground_truth(self, tmp_path) -> None:
        rows = self._make_results_rows(
            {"faithfulness": 0.9},
            ground_truth="Attention is a mechanism in transformers.",
        )

        with patch("src.evaluation.ragas_runner.settings") as mock_settings:
            mock_settings.evaluation.results_dir = tmp_path
            mock_settings.pipeline_version = "v1-baseline"

            from src.evaluation.ragas_runner import _save_snapshot

            _save_snapshot("ref-test", rows)

        with open(tmp_path / "ref-test.json") as f:
            data = json.load(f)
        assert (
            data["per_question"][0]["reference"]
            == "Attention is a mechanism in transformers."
        )

    def test_per_question_answer_is_none_when_run_missing(self, tmp_path) -> None:
        """Row without a 'run' key (e.g. pipeline failure) must not raise."""
        r = MagicMock()
        r.key = "faithfulness"
        r.score = 0.5
        row = [
            {
                "example": _make_fake_example(),
                "evaluation_results": {"results": [r]},
                # no "run" key — simulates a failed pipeline execution
            }
        ]

        with patch("src.evaluation.ragas_runner.settings") as mock_settings:
            mock_settings.evaluation.results_dir = tmp_path
            mock_settings.pipeline_version = "v1-baseline"

            from src.evaluation.ragas_runner import _save_snapshot

            _save_snapshot("no-run-test", row)

        with open(tmp_path / "no-run-test.json") as f:
            data = json.load(f)
        assert data["per_question"][0]["answer"] is None


class TestTarget:
    def test_returns_answer_and_contexts_keys(self) -> None:
        fake_result = {
            "answer": "Some answer.",
            "sources": [{"text": "chunk 1"}, {"text": "chunk 2"}],
        }
        with patch(
            "src.evaluation.ragas_runner.run_pipeline", return_value=fake_result
        ):
            from src.evaluation.ragas_runner import target

            result = target({"question": "What is attention?"})

        assert "answer" in result
        assert "contexts" in result

    def test_contexts_are_extracted_from_sources(self) -> None:
        fake_result = {
            "answer": "Answer.",
            "sources": [
                {"text": "chunk A"},
                {"text": "chunk B"},
                {"other_field": "no text key"},
            ],
        }
        with patch(
            "src.evaluation.ragas_runner.run_pipeline", return_value=fake_result
        ):
            from src.evaluation.ragas_runner import target

            result = target({"question": "query"})

        # Only sources with a 'text' key should be included
        assert result["contexts"] == ["chunk A", "chunk B"]

    def test_answer_matches_pipeline_output(self) -> None:
        fake_result = {"answer": "Specific answer.", "sources": []}
        with patch(
            "src.evaluation.ragas_runner.run_pipeline", return_value=fake_result
        ):
            from src.evaluation.ragas_runner import target

            result = target({"question": "query"})

        assert result["answer"] == "Specific answer."


class TestSafeRagasScore:
    def test_returns_score_on_success(self) -> None:
        mock_result = MagicMock()
        mock_result.value = 0.85

        async def _coro():
            return mock_result

        from src.evaluation.ragas_runner import _safe_ragas_score

        score = _safe_ragas_score("faithfulness", _coro())
        assert score == pytest.approx(0.85)

    def test_returns_none_on_exception(self) -> None:
        async def _failing_coro():
            raise RuntimeError("API error")

        from src.evaluation.ragas_runner import _safe_ragas_score

        score = _safe_ragas_score("faithfulness", _failing_coro())
        assert score is None


class TestEvaluatorFunctionShapes:
    """Evaluator functions must return {key, score} — shape is the contract."""

    def _patch_safe_score(self, score_value):
        return patch(
            "src.evaluation.ragas_runner._safe_ragas_score", return_value=score_value
        )

    def test_eval_faithfulness_returns_correct_key(self) -> None:
        with self._patch_safe_score(0.9):
            from src.evaluation.ragas_runner import eval_faithfulness

            result = eval_faithfulness(_make_fake_run(), _make_fake_example())
        assert result["key"] == "faithfulness"

    def test_eval_faithfulness_score_propagated(self) -> None:
        with self._patch_safe_score(0.75):
            from src.evaluation.ragas_runner import eval_faithfulness

            result = eval_faithfulness(_make_fake_run(), _make_fake_example())
        assert result["score"] == pytest.approx(0.75)

    def test_eval_answer_relevancy_returns_correct_key(self) -> None:
        with self._patch_safe_score(0.8):
            from src.evaluation.ragas_runner import eval_answer_relevancy

            result = eval_answer_relevancy(_make_fake_run(), _make_fake_example())
        assert result["key"] == "answer_relevancy"

    def test_eval_context_precision_returns_correct_key(self) -> None:
        with self._patch_safe_score(0.7):
            from src.evaluation.ragas_runner import eval_context_precision

            result = eval_context_precision(_make_fake_run(), _make_fake_example())
        assert result["key"] == "context_precision"

    def test_eval_context_recall_returns_correct_key(self) -> None:
        with self._patch_safe_score(0.65):
            from src.evaluation.ragas_runner import eval_context_recall

            result = eval_context_recall(_make_fake_run(), _make_fake_example())
        assert result["key"] == "context_recall"

    def test_none_score_is_propagated_not_coerced(self) -> None:
        """A None score (evaluator failure) must pass through, not become 0."""
        with self._patch_safe_score(None):
            from src.evaluation.ragas_runner import eval_faithfulness

            result = eval_faithfulness(_make_fake_run(), _make_fake_example())
        assert result["score"] is None

    def test_eval_context_precision_uses_ground_truth(self) -> None:
        """reference kwarg passed to ascore must be example.outputs['ground_truth']."""
        example = _make_fake_example(ground_truth="Expected answer.")
        with (
            patch("src.evaluation.ragas_runner._context_precision") as mock_metric,
            patch("src.evaluation.ragas_runner._safe_ragas_score", return_value=0.5),
        ):
            from src.evaluation.ragas_runner import eval_context_precision

            eval_context_precision(_make_fake_run(), example)

        call_kwargs = mock_metric.ascore.call_args.kwargs
        assert call_kwargs["reference"] == "Expected answer."

    def test_eval_context_recall_uses_ground_truth(self) -> None:
        """reference kwarg passed to ascore must be example.outputs['ground_truth']."""
        example = _make_fake_example(ground_truth="Expected answer.")
        with (
            patch("src.evaluation.ragas_runner._context_recall") as mock_metric,
            patch("src.evaluation.ragas_runner._safe_ragas_score", return_value=0.5),
        ):
            from src.evaluation.ragas_runner import eval_context_recall

            eval_context_recall(_make_fake_run(), example)

        call_kwargs = mock_metric.ascore.call_args.kwargs
        assert call_kwargs["reference"] == "Expected answer."

    def test_eval_context_precision_empty_contexts_when_run_outputs_none(self) -> None:
        """When run.outputs is None, retrieved_contexts must default to []."""
        run = MagicMock()
        run.outputs = None
        with (
            patch("src.evaluation.ragas_runner._context_precision") as mock_metric,
            patch("src.evaluation.ragas_runner._safe_ragas_score", return_value=0.5),
        ):
            from src.evaluation.ragas_runner import eval_context_precision

            eval_context_precision(run, _make_fake_example())

        call_kwargs = mock_metric.ascore.call_args.kwargs
        assert call_kwargs["retrieved_contexts"] == []

    def test_eval_context_recall_empty_contexts_when_run_outputs_none(self) -> None:
        """When run.outputs is None, retrieved_contexts must default to []."""
        run = MagicMock()
        run.outputs = None
        with (
            patch("src.evaluation.ragas_runner._context_recall") as mock_metric,
            patch("src.evaluation.ragas_runner._safe_ragas_score", return_value=0.5),
        ):
            from src.evaluation.ragas_runner import eval_context_recall

            eval_context_recall(run, _make_fake_example())

        call_kwargs = mock_metric.ascore.call_args.kwargs
        assert call_kwargs["retrieved_contexts"] == []
