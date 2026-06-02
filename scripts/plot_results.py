"""Generate the README RAGAS graphs from evaluation snapshots.

Charts three runs — BASELINE → MIDPOINT → CURRENT — and produces three PNGs
in evaluation/results/:
  1. COMPARISON_OUT     — grouped bar chart of aggregate metrics, all three runs
  2. EARLY_HEATMAP_OUT  — per-question-type metric delta, baseline → midpoint
  3. RECENT_HEATMAP_OUT — per-question-type metric delta, midpoint → current

To chart different runs, edit the snapshot paths / labels / colors / output
names in the "EDIT HERE" block below — nothing else needs to change; titles are
built automatically from the labels. The script is wired for exactly three runs.

The midpoint (v2-hybrid-rerank-v2) uses the promptv2 run — the adopted default
and production lineage into v3 — so all three graphs chain on the same snapshot.

Run:  uv run python scripts/plot_results.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

RESULTS = Path(__file__).resolve().parent.parent / "evaluation" / "results"

METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
METRIC_LABELS = [
    "Faithfulness",
    "Answer Relevancy",
    "Context Precision",
    "Context Recall",
]
METRIC_LABELS_SHORT = [
    "Faithfulness",
    "Ans. Relevancy",
    "Ctx. Precision",
    "Ctx. Recall",
]

# question-type row order (matches the original v1->v2.2 heatmap)
TYPE_ORDER = ["factual", "conceptual", "multi-hop", "cross-paper", "numerical"]

# ─────────────────────────────────────────────────────────────────────────────
# EDIT HERE — point these at the three snapshots you want to chart.
# Three runs are charted: BASELINE → MIDPOINT → CURRENT.
# Titles are built automatically from the labels, so just change paths + labels.
# v2.2 uses the promptv2 run (true production lineage into v3).
# ─────────────────────────────────────────────────────────────────────────────

# Snapshot JSON paths
BASELINE_SNAPSHOT = RESULTS / "v1-baseline" / "v1-baseline.json"
MIDPOINT_SNAPSHOT = (
    RESULTS / "v2-hybrid-rerank-v2" / "v2-hybrid-rerank-v2|promptv2.json"
)
CURRENT_SNAPSHOT = (
    RESULTS / "v3.1.1-prefetch-scale-30" / "v3.1.1-prefetch-scale-30.json"
)

# Labels shown in the legend and titles
BASELINE_LABEL = "v1-baseline"
MIDPOINT_LABEL = "v2-hybrid-rerank-v2"
CURRENT_LABEL = "v3.1.1-prefetch-scale-30"

# Bar colors (baseline, midpoint, current)
BASELINE_COLOR = "#9b59b6"
MIDPOINT_COLOR = "#3498db"
CURRENT_COLOR = "#2ecc71"

# Output PNG filenames (written under evaluation/results/)
COMPARISON_OUT = "ragas_comparison.png"
EARLY_HEATMAP_OUT = "category_delta_heatmap.png"  # baseline → midpoint
RECENT_HEATMAP_OUT = "v2.2-to-v3.1.1_delta_heatmap.png"  # midpoint → current


def load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def per_type_means(snapshot: dict) -> dict[str, dict[str, float]]:
    """Mean of each metric within each question type."""
    buckets: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for q in snapshot["per_question"]:
        qtype = q.get("question_type", "unknown")
        for m in METRICS:
            v = q.get("scores", {}).get(m)
            if v is not None:
                buckets[qtype][m].append(v)
    return {
        t: {
            m: (sum(vals[m]) / len(vals[m]) if vals[m] else float("nan"))
            for m in METRICS
        }
        for t, vals in buckets.items()
    }


def plot_comparison() -> Path:
    """Grouped bar chart: baseline -> midpoint -> current aggregate RAGAS metrics."""
    series = [
        (BASELINE_LABEL, load(BASELINE_SNAPSHOT)["aggregate"], BASELINE_COLOR),
        (MIDPOINT_LABEL, load(MIDPOINT_SNAPSHOT)["aggregate"], MIDPOINT_COLOR),
        (CURRENT_LABEL, load(CURRENT_SNAPSHOT)["aggregate"], CURRENT_COLOR),
    ]

    x = np.arange(len(METRICS))
    width = 0.26
    fig, ax = plt.subplots(figsize=(12, 7))

    offsets = [-width, 0.0, width]
    for (label, agg, color), off in zip(series, offsets):
        vals = [agg[m] for m in METRICS]
        bars = ax.bar(x + off, vals, width, label=label, color=color)
        for b in bars:
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height(),
                f"{b.get_height():.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_title(
        f"RAGAS Metrics: {BASELINE_LABEL} → {MIDPOINT_LABEL} → {CURRENT_LABEL}",
        fontsize=14,
    )
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(METRIC_LABELS)
    ax.set_ylim(0.6, 1.05)
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    out = RESULTS / COMPARISON_OUT
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_delta_heatmap(
    base_path: Path, exp_path: Path, title: str, out_name: str
) -> Path:
    """Per-question-type metric delta heatmap between two snapshots."""
    base = per_type_means(load(base_path))
    exp = per_type_means(load(exp_path))

    types = [t for t in TYPE_ORDER if t in base and t in exp]
    delta = np.array([[exp[t][m] - base[t][m] for m in METRICS] for t in types])

    lim = max(0.05, float(np.nanmax(np.abs(delta))))
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    im = ax.imshow(delta, cmap="Spectral", norm=norm, aspect="auto")

    ax.set_xticks(np.arange(len(METRICS)))
    ax.set_xticklabels(METRIC_LABELS_SHORT)
    ax.set_yticks(np.arange(len(types)))
    ax.set_yticklabels(types)

    for i in range(len(types)):
        for j in range(len(METRICS)):
            ax.text(
                j,
                i,
                f"{delta[i, j]:+.3f}",
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                color="black",
            )

    ax.set_title(title, fontsize=13, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Δ Score")

    out = RESULTS / out_name
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    outputs = [
        plot_comparison(),
        plot_delta_heatmap(
            BASELINE_SNAPSHOT,
            MIDPOINT_SNAPSHOT,
            f"Score Delta: {BASELINE_LABEL} → {MIDPOINT_LABEL}",
            EARLY_HEATMAP_OUT,
        ),
        plot_delta_heatmap(
            MIDPOINT_SNAPSHOT,
            CURRENT_SNAPSHOT,
            f"Score Delta: {MIDPOINT_LABEL} → {CURRENT_LABEL}",
            RECENT_HEATMAP_OUT,
        ),
    ]
    for o in outputs:
        print(f"wrote {o}")
