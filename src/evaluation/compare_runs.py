import argparse
import json
import sys
from pathlib import Path

METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
METRIC_SHORT = {
    "faithfulness": "Faith",
    "answer_relevancy": "AR",
    "context_precision": "CP",
    "context_recall": "CR",
}


def load_snapshot(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def fmt_score(score: float | None) -> str:
    if score is None:
        return "  ERR "
    return f"{score:.4f}"


def fmt_delta(delta: float | None) -> str:
    if delta is None:
        return "   n/a"
    return f"{delta:+.4f}"


def composite(scores: dict) -> float | None:
    vals = [scores.get(m) for m in METRICS]
    valid = [v for v in vals if v is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)


def print_header(title: str) -> None:
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")


def print_aggregate(base: dict, exp: dict) -> None:
    print_header("AGGREGATE SCORES")
    base_agg = base.get("aggregate", {})
    exp_agg = exp.get("aggregate", {})

    print(
        f"\n  {'Metric':<22} {'Baseline':>8} {'Experiment':>10} {'Delta':>8}  Direction"
    )
    print(f"  {'-' * 62}")
    for m in METRICS:
        b = base_agg.get(m)
        e = exp_agg.get(m)
        d = (e - b) if (b is not None and e is not None) else None
        direction = ""
        if d is not None:
            if d > 0.005:
                direction = "Improved"
            elif d < -0.005:
                direction = "REGRESSED"
            else:
                direction = "~Stable"
        print(
            f"  {m:<22} {fmt_score(b):>8} {fmt_score(e):>10} {fmt_delta(d):>8}  {direction}"
        )


def print_per_question_table(
    base_by_id: dict, exp_by_id: dict, threshold: float
) -> tuple[list, list]:
    print_header(f"PER-QUESTION CHANGES (|delta| > {threshold})")

    all_ids = sorted(set(base_by_id.keys()) | set(exp_by_id.keys()))
    improved, regressed = [], []

    for qid in all_ids:
        bq = base_by_id.get(qid, {})
        eq = exp_by_id.get(qid, {})
        bs = bq.get("scores", {})
        es = eq.get("scores", {})

        bc = composite(bs)
        ec = composite(es)
        dc = (ec - bc) if (bc is not None and ec is not None) else None

        if dc is not None and abs(dc) > threshold:
            qtype = eq.get("question_type", bq.get("question_type", "?"))
            qsub = eq.get("question_subtype", bq.get("question_subtype", "?"))

            metric_deltas = []
            for m in METRICS:
                bv = bs.get(m)
                ev = es.get(m)
                if bv is not None and ev is not None:
                    d = ev - bv
                    if abs(d) > 0.01:
                        metric_deltas.append(f"{METRIC_SHORT[m]} {d:+.2f}")

            entry = {
                "id": qid,
                "type": f"{qtype}/{qsub}",
                "base_composite": bc,
                "exp_composite": ec,
                "delta": dc,
                "metric_deltas": ", ".join(metric_deltas) if metric_deltas else "~",
            }
            if dc > 0:
                improved.append(entry)
            else:
                regressed.append(entry)

    if regressed:
        print(f"\n  REGRESSIONS ({len(regressed)}):")
        print(
            f"  {'ID':<8} {'Type':<35} {'Base':>6} {'Exp':>6} {'Delta':>7}  Key changes"
        )
        print(f"  {'-' * 80}")
        for r in sorted(regressed, key=lambda x: x["delta"]):
            print(
                f"  {r['id']:<8} {r['type']:<35} {r['base_composite']:>6.3f} "
                f"{r['exp_composite']:>6.3f} {r['delta']:>+7.3f}  {r['metric_deltas']}"
            )

    if improved:
        print(f"\n  IMPROVEMENTS ({len(improved)}):")
        print(
            f"  {'ID':<8} {'Type':<35} {'Base':>6} {'Exp':>6} {'Delta':>7}  Key changes"
        )
        print(f"  {'-' * 80}")
        for r in sorted(improved, key=lambda x: -x["delta"]):
            print(
                f"  {r['id']:<8} {r['type']:<35} {r['base_composite']:>6.3f} "
                f"{r['exp_composite']:>6.3f} {r['delta']:>+7.3f}  {r['metric_deltas']}"
            )

    if not regressed and not improved:
        print("\n  No questions exceeded the threshold.")

    return improved, regressed


def print_type_summary(base_by_id: dict, exp_by_id: dict) -> None:
    print_header("BY QUESTION TYPE")

    from collections import defaultdict

    by_type: dict[str, list[tuple[float, float]]] = defaultdict(list)
    all_ids = sorted(set(base_by_id.keys()) | set(exp_by_id.keys()))

    for qid in all_ids:
        bq = base_by_id.get(qid, {})
        eq = exp_by_id.get(qid, {})
        bc = composite(bq.get("scores", {}))
        ec = composite(eq.get("scores", {}))
        if bc is None or ec is None:
            continue
        qtype = eq.get("question_type", bq.get("question_type", "unknown"))
        by_type[qtype].append((bc, ec))

    print(
        f"\n  {'Type':<20} {'Count':>5} {'Base Avg':>9} {'Exp Avg':>9} {'Delta':>8}  Verdict"
    )
    print(f"  {'-' * 65}")
    for qtype in sorted(by_type.keys()):
        pairs = by_type[qtype]
        n = len(pairs)
        b_avg = sum(b for b, _ in pairs) / n
        e_avg = sum(e for _, e in pairs) / n
        d = e_avg - b_avg
        verdict = "~Stable" if abs(d) < 0.01 else ("Improved" if d > 0 else "REGRESSED")
        print(f"  {qtype:<20} {n:>5} {b_avg:>9.4f} {e_avg:>9.4f} {d:>+8.4f}  {verdict}")


def print_question_detail(qid: str, base_by_id: dict, exp_by_id: dict) -> None:
    print_header(f"DETAIL: {qid}")

    bq = base_by_id.get(qid)
    eq = exp_by_id.get(qid)

    if not bq and not eq:
        print(f"\n  Question {qid} not found in either snapshot.")
        return

    q = eq or bq
    print(f"\n  Question: {q.get('question', 'N/A')}")
    print(f"  Type: {q.get('question_type', '?')}/{q.get('question_subtype', '?')}")

    source_papers = q.get("source_papers")
    if source_papers:
        print(f"  Expected papers: {', '.join(source_papers)}")

    print(f"\n  Reference: {q.get('reference', 'N/A')[:300]}")

    # Scores comparison
    print(f"\n  {'Metric':<22} {'Baseline':>8} {'Experiment':>10} {'Delta':>8}")
    print(f"  {'-' * 52}")
    bs = (bq or {}).get("scores", {})
    es = (eq or {}).get("scores", {})
    for m in METRICS:
        bv = bs.get(m)
        ev = es.get(m)
        d = (ev - bv) if (bv is not None and ev is not None) else None
        flag = " <<<" if (d is not None and abs(d) > 0.1) else ""
        print(
            f"  {m:<22} {fmt_score(bv):>8} {fmt_score(ev):>10} {fmt_delta(d):>8}{flag}"
        )

    # Retrieved sources (experiment only, if available)
    exp_sources = (eq or {}).get("retrieved_sources")
    base_sources = (bq or {}).get("retrieved_sources")

    if exp_sources or base_sources:
        print("\n  Retrieved chunks:")
        if base_sources:
            print("    Baseline:")
            for i, s in enumerate(base_sources):
                title = s.get("title", "?")[:60]
                ci = s.get("chunk_index", "?")
                score = s.get("score")
                score_str = f"{score:.4f}" if score is not None else "n/a"
                print(f"      [{i + 1}] {title:<60} chunk={ci} score={score_str}")

        if exp_sources:
            print("    Experiment:")
            for i, s in enumerate(exp_sources):
                title = s.get("title", "?")[:60]
                ci = s.get("chunk_index", "?")
                score = s.get("score")
                score_str = f"{score:.4f}" if score is not None else "n/a"
                print(f"      [{i + 1}] {title:<60} chunk={ci} score={score_str}")
    else:
        print(
            "\n  (No retrieved_sources data in snapshots — run with enriched snapshot for chunk details)"
        )

    # Answers
    for label, snapshot in [("Baseline", bq), ("Experiment", eq)]:
        if snapshot and snapshot.get("answer"):
            print(f"\n  {label} answer (truncated):")
            print(f"    {snapshot['answer'][:400]}")


def print_movement_counts(base_by_id: dict, exp_by_id: dict) -> None:
    print_header("PER-METRIC MOVEMENT COUNTS")

    all_ids = sorted(set(base_by_id.keys()) & set(exp_by_id.keys()))
    print(
        f"\n  {'Metric':<22} {'Improved':>8} {'Regressed':>9} {'Stable':>7} {'Missing':>8}"
    )
    print(f"  {'-' * 58}")

    for m in METRICS:
        up, down, same, missing = 0, 0, 0, 0
        for qid in all_ids:
            bv = base_by_id[qid].get("scores", {}).get(m)
            ev = exp_by_id[qid].get("scores", {}).get(m)
            if bv is None or ev is None:
                missing += 1
                continue
            d = ev - bv
            if abs(d) < 0.01:
                same += 1
            elif d > 0:
                up += 1
            else:
                down += 1
        print(f"  {m:<22} {up:>8} {down:>9} {same:>7} {missing:>8}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two RAGAS evaluation snapshots.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("baseline", help="Path to baseline snapshot JSON")
    parser.add_argument("experiment", help="Path to experiment snapshot JSON")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Composite delta threshold for flagging changes (default: 0.05)",
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="Show detailed comparison for a specific question ID (e.g., q_029)",
    )
    args = parser.parse_args()

    for p in [args.baseline, args.experiment]:
        if not Path(p).exists():
            print(f"Error: {p} not found", file=sys.stderr)
            sys.exit(1)

    base = load_snapshot(args.baseline)
    exp = load_snapshot(args.experiment)

    print(
        f"  Baseline:   {base.get('experiment', '?')} ({base.get('timestamp', '?')[:10]})"
    )
    print(
        f"  Experiment: {exp.get('experiment', '?')} ({exp.get('timestamp', '?')[:10]})"
    )

    base_by_id = {q["id"]: q for q in base.get("per_question", [])}
    exp_by_id = {q["id"]: q for q in exp.get("per_question", [])}

    if args.question:
        print_question_detail(args.question, base_by_id, exp_by_id)
        return

    print_aggregate(base, exp)
    print_movement_counts(base_by_id, exp_by_id)
    print_type_summary(base_by_id, exp_by_id)
    print_per_question_table(base_by_id, exp_by_id, args.threshold)


if __name__ == "__main__":
    main()
