"""Compare a RAGAS run to a baseline; non-zero exit if composite drops beyond tolerance."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

METRIC_KEYS = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)


def composite_score(aggregate: dict[str, float]) -> float:
    return sum(aggregate[k] for k in METRIC_KEYS) / len(METRIC_KEYS)


def load_result(path: Path) -> dict:
    return json.loads(Path(path).read_text())


@dataclass
class GateResult:
    baseline_composite: float
    new_composite: float
    delta: float
    tolerance: float
    passed: bool
    baseline_metrics: dict[str, float]
    new_metrics: dict[str, float]

    def to_markdown(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        header = (
            f"### RAGAS Gate: {status}\n"
            f"- Baseline composite: **{self.baseline_composite:.4f}**\n"
            f"- New composite: **{self.new_composite:.4f}**\n"
            f"- Delta: **{self.delta:+.4f}** (tolerance −{self.tolerance:.2f})\n\n"
        )
        rows = ["| Metric | Baseline | New | Delta |", "|---|---:|---:|---:|"]
        for key in METRIC_KEYS:
            b = self.baseline_metrics[key]
            n = self.new_metrics[key]
            rows.append(f"| {key} | {b:.4f} | {n:.4f} | {n - b:+.4f} |")
        return header + "\n".join(rows)


def evaluate_gate(baseline_path: Path, new_path: Path, tolerance: float) -> GateResult:
    baseline = load_result(baseline_path)["aggregate"]
    new = load_result(new_path)["aggregate"]
    b_comp = composite_score(baseline)
    n_comp = composite_score(new)
    delta = n_comp - b_comp
    return GateResult(
        baseline_composite=b_comp,
        new_composite=n_comp,
        delta=delta,
        tolerance=tolerance,
        passed=delta >= -tolerance,
        baseline_metrics=baseline,
        new_metrics=new,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="RAGAS regression gate.")
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--new", required=True, type=Path)
    parser.add_argument("--tolerance", type=float, default=0.02)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="If provided, write the markdown summary to this path.",
    )
    args = parser.parse_args()

    result = evaluate_gate(args.baseline, args.new, args.tolerance)
    md = result.to_markdown()
    print(md)
    if args.output:
        args.output.write_text(md)
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
