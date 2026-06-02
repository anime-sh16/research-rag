import json
from pathlib import Path

import pytest

from scripts.ragas_gate import GateResult, composite_score, evaluate_gate, load_result

BASELINE = {
    "experiment": "v3.1.1-prefetch-scale-30",
    "pipeline_version": "v3.1.1-prefetch-scale-30",
    "aggregate": {
        "faithfulness": 0.9844,
        "answer_relevancy": 0.897,
        "context_precision": 0.8328,
        "context_recall": 0.9106,
    },
}


def _write(tmp_path: Path, name: str, payload: dict) -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(payload))
    return p


def test_composite_score_is_mean_of_four_metrics() -> None:
    composite = composite_score(BASELINE["aggregate"])
    assert composite == pytest.approx(0.9062, abs=0.0005)


def test_load_result_parses_aggregate(tmp_path: Path) -> None:
    path = _write(tmp_path, "baseline.json", BASELINE)
    loaded = load_result(path)
    assert loaded["aggregate"]["faithfulness"] == 0.9844


def test_gate_passes_when_new_composite_equals_baseline(tmp_path: Path) -> None:
    baseline_path = _write(tmp_path, "baseline.json", BASELINE)
    new_path = _write(tmp_path, "new.json", BASELINE)
    result = evaluate_gate(baseline_path, new_path, tolerance=0.02)
    assert result.passed is True
    assert result.delta == pytest.approx(0.0, abs=1e-6)


def test_gate_passes_when_new_composite_within_tolerance(tmp_path: Path) -> None:
    new = json.loads(json.dumps(BASELINE))
    new["aggregate"]["faithfulness"] -= 0.05  # composite drops by ~0.0125
    new_path = _write(tmp_path, "new.json", new)
    baseline_path = _write(tmp_path, "baseline.json", BASELINE)
    result = evaluate_gate(baseline_path, new_path, tolerance=0.02)
    assert result.passed is True


def test_gate_fails_when_new_composite_drops_beyond_tolerance(tmp_path: Path) -> None:
    new = json.loads(json.dumps(BASELINE))
    new["aggregate"]["faithfulness"] -= 0.5  # composite drops by ~0.125
    new_path = _write(tmp_path, "new.json", new)
    baseline_path = _write(tmp_path, "baseline.json", BASELINE)
    result = evaluate_gate(baseline_path, new_path, tolerance=0.02)
    assert result.passed is False
    assert result.delta < -0.02


def test_gate_result_renders_markdown_summary() -> None:
    result = GateResult(
        baseline_composite=0.9062,
        new_composite=0.8800,
        delta=-0.0262,
        tolerance=0.02,
        passed=False,
        baseline_metrics=BASELINE["aggregate"],
        new_metrics={
            "faithfulness": 0.95,
            "answer_relevancy": 0.85,
            "context_precision": 0.80,
            "context_recall": 0.90,
        },
    )
    md = result.to_markdown()
    assert "FAIL" in md
    assert "0.9062" in md
    assert "0.8800" in md
    assert "-0.0262" in md
