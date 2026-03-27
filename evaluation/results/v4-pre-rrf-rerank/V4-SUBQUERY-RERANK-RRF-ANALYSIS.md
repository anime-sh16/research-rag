# V4 Sub-Query Level Reranking + RRF Merge — Evaluation Report

**Experiment:** `v4-pre-rrf-rerank`
**Baseline:** `v3.1.1-prefetch-scale-30`
**Eval set:** 41 questions (fixed)
**Date:** 2026-03-27


## What v3.1.1 Identified

The v3.1.1 analysis found 11 questions below 0.90 composite. After trace verification, the genuine pipeline issues were:

1. **Query decomposition coreference loss (q_034)** — sub-query lost "they" context, highest priority.
2. **Weak semantic signal eliminated by reranker (q_031)** — Factuality Challenges paper surfaced 1/15 candidates but got eliminated during global reranking.
3. **Semantic gap / embedding mismatch (q_032)** — QServe never entered the candidate pool due to vocabulary mismatch.

Cross-paper questions had the lowest composite (0.7630) with CP averaging 0.39, but 3 of those 6 cross-paper failures (q_028, q_029, q_030) were verified as **RAGAS CP scoring artifacts** — the retrieved chunks were relevant, answers were good (F>=0.94, AR>=0.91, CR=1.0), but RAGAS penalized chunks covering only one half of a comparative question.

---

## Changes Implemented (v3.1.1 → v4)

All changes scoped to `src/retrieval/retriever.py` across 3 commits (`79d354c`, `7d45072`, `1a1118d`).

### 1. Query Decomposition Prompt Improvement
- Each sub-query must now **explicitly name its entity/method/model** — no pronouns ("they", "it", "this approach") or implicit references that depend on other sub-queries for context.
- Targets the coreference loss identified in q_034.

### 2. Per-Sub-Query Reranking + RRF Merge
**Before (v3/v3.1.1):**
```
sub-query 1 → hybrid search → candidates₁ ─┐
                                             ├→ merge + dedup → rerank(full_query) → top-k
sub-query 2 → hybrid search → candidates₂ ─┘
```

**After (v4):**
```
sub-query 1 → hybrid search → rerank(sub-query₁) → ranked₁ ─┐
                                                               ├→ RRF merge → top-k
sub-query 2 → hybrid search → rerank(sub-query₂) → ranked₂ ─┘
```

- New `_rrf_merge()` method: standard RRF with `k=60`. For each candidate, `score = Σ 1/(k + rank)` across all lists, deduplicated by `(paper_id, chunk_index)`.
- Single sub-query path unchanged (no RRF needed — direct rerank output).
- Previous `_merge_candidates()` method now unused.

### 3. Parallel Reranker Calls
- Multi-sub-query reranking runs via `ThreadPoolExecutor` to avoid serializing API calls.
- Each sub-query's search + rerank executes concurrently.

---

## Aggregate Results

| Metric | v3.1.1 | v4 | Delta | Direction |
|---|---|---|---|---|
| **Context Precision** | 0.8328 | 0.7627 | −0.0701 | **Regressed** |
| **Context Recall** | 0.9106 | 0.9146 | +0.0040 | ~Flat |
| Faithfulness | 0.9844 | 0.9842 | −0.0002 | Flat |
| Answer Relevancy | 0.8970 | 0.8949 | −0.0021 | Flat |
| **Composite** | **0.9062** | **0.8891** | **−0.0155** | **Regressed** |

**v4 is a net regression.** Context Precision dropped 7 points; no other metric moved meaningfully.

---

## Score Progression (v1 → v4)

| Version | Change | Faith | AR | CP | CR | Composite |
|---|---|---|---|---|---|---|
| v1 (baseline) | Dense retrieval only | 0.8742 | 0.7509 | 0.6818 | 0.8415 | **0.7871** |
| v2 | + Hybrid search + Cohere rerank | 0.8918 | 0.7818 | 0.7680 | 0.9071 | **0.8372** |
| v3 | + Query decomposition + expansion | 0.9507 | 0.8742 | 0.8336 | 0.9208 | **0.8948** |
| v3.1 | + Prefetch scale 20 | 0.9787 | 0.8944 | 0.7868 | 0.9146 | **0.8936** |
| v3.1.1 | + Prefetch scale 30 | 0.9844 | 0.8970 | 0.8328 | 0.9106 | **0.9062** |
| **v4** | **+ Sub-query rerank + RRF** | **0.9842** | **0.8949** | **0.7627** | **0.9146** | **0.8891** |

v4 drops composite below v3 levels. The improvement trajectory that held from v1 through v3.1.1 is broken.

---

## Per-Question Movement

### Direction Summary (threshold ±0.01)

| Direction | Count |
|---|---|
| Improved | 6 |
| Regressed | 12 |
| Unchanged | 23 |

2:1 regression-to-improvement ratio. 56% of questions unaffected; of those that moved, twice as many got worse.

### Per-Metric Movement

| Metric | ↑ Improved | ↓ Regressed | = Stable |
|---|---|---|---|
| Context Precision | 3 | **11** | 27 |
| Context Recall | 2 | 1 | 38 |
| Answer Relevancy | 7 | 8 | 26 |
| Faithfulness | 5 | 6 | 29 |

**Context Precision is the sole driver of regression.** 11 questions lost CP, only 3 gained it. All other metrics are within noise.

---

## Detailed Analysis

### By Question Type

The eval set splits roughly 50/50 into comparative (21) and single-topic (20) questions:

| Metric | Comparative v3.1.1 | Comparative v4 | Δ | Single v3.1.1 | Single v4 | Δ |
|---|---|---|---|---|---|---|
| Context Precision | 0.7752 | 0.6463 | **−0.1289** | 0.8933 | 0.8850 | −0.0083 |
| Context Recall | 0.8730 | 0.8810 | +0.0079 | 0.9500 | 0.9500 | +0.0000 |
| Answer Relevancy | 0.9211 | 0.9146 | −0.0065 | 0.8717 | 0.8743 | +0.0026 |
| Faithfulness | 0.9833 | 0.9760 | −0.0073 | 0.9856 | 0.9929 | +0.0072 |

Single-topic questions are essentially unaffected (CP −0.008). The regression is concentrated in comparative questions (CP −0.129).

### CP Drop vs CR Change

Of the 11 questions where Context Precision regressed:

| Pattern | Count |
|---|---|
| CP dropped, CR improved | 2 |
| CP dropped, CR unchanged | **7** |
| CP dropped, CR also dropped | 2 |

In 7 of 11 cases, the same chunks were retrieved but reordered into a worse ranking. v4 is not finding new relevant chunks — it is rearranging the existing ones.

### Context Precision Distribution Shift

| CP Bucket | v3.1.1 | v4 | Change |
|---|---|---|---|
| 0.9 – 1.0 | 26 | 23 | −3 |
| 0.7 – 0.9 | 9 | 9 | = |
| 0.5 – 0.7 | 1 | 1 | = |
| 0.3 – 0.5 | 1 | 2 | +1 |
| 0.0 – 0.3 | 4 | 6 | +2 |

Movement at the extremes: 3 questions fell out of the near-perfect tier, 2 additional entered the near-zero tier. Middle tiers unchanged.

### Perfect CP Erosion

25 questions had CP=1.0 in v3.1.1. Of those, **6 dropped below 0.95** in v4 (5 comparative, 1 single-topic):

| Question (abbreviated) | CP v3.1.1 → v4 | Type |
|---|---|---|
| Hallucination bottleneck in RAG pipelines... | 1.000 → 0.250 | Comparative |
| Canonical vs agentic RAG architectures... | 1.000 → 0.756 | Comparative |
| LLM-profiled planner base vs search workflows... | 1.000 → 0.700 | Comparative |
| Decoupling reward modeling from policy training... | 1.000 → 0.833 | Comparative |
| Heavy-tailed data distributions, quantization... | 1.000 → 0.917 | Comparative |
| Hit@1 neural retriever-reranker pipeline... | 1.000 → 0.833 | Single-topic |

### Top Regressions

| Question (abbreviated) | Type | CP Δ | CR Δ | Composite Δ |
|---|---|---|---|---|
| Hallucination bottleneck in RAG pipelines... | Comparative | −0.750 | +0.000 | −0.213 |
| AWQ activations unquantized vs Quasar-ViT... | Comparative | −0.325 | −0.500 | −0.203 |
| Diffusion Probabilistic Models Markov chain... | Comparative | −0.417 | +0.000 | −0.141 |
| Synthetic instruction datasets distillation... | Comparative | −0.500 | +0.000 | −0.103 |
| LLM-profiled planner base vs search workflows... | Comparative | −0.300 | +0.000 | −0.077 |
| Canonical vs agentic RAG architectures... | Comparative | −0.244 | +0.000 | −0.056 |

All top regressions are comparative questions. CP collapses while CR stays flat.

### Top Improvements

| Question (abbreviated) | Type | CP Δ | CR Δ | Composite Δ |
|---|---|---|---|---|
| Appraisal-based reward formulation in RL... | Comparative | −0.083 | **+0.500** | +0.095 |
| WizardLM Evol-Instruct five prompt operations... | Single-topic | +0.000 | +0.000 | +0.068 |
| RePaint resampling LPIPS reduction... | Comparative | +0.167 | +0.000 | +0.042 |

The appraisal-based reward question (q_033, identified in v3.1.1 as CR=0.50) improved — per-sub-query reranking surfaced the second source paper (CR: 0.50 → 1.00). The WizardLM improvement was a faithfulness correction (0.73 → 1.00), unrelated to the architectural change.

---

## Impact on v3.1.1 Identified Issues

### Did v4 fix the 3 genuine pipeline issues from v3.1.1?

| Issue | v3.1.1 | v4 | Fixed? |
|---|---|---|---|
| **q_034** — coreference loss in sub-query | CR=0.50, Comp=0.796 | CR=0.50, Comp=0.654 | **No — regressed.** CP dropped 0.750→0.333. The prompt improvement alone was not enough, or the coreference fix didn't apply to this question's decomposition. |
| **q_031** — weak semantic signal (Factuality Challenges paper) | CR=0.50, Comp=0.846 | CR=0.50, Comp=0.633 | **No — regressed.** CP dropped 1.000→0.250. The paper still doesn't surface competitively, and RRF worsened the ordering of what was retrieved. |
| **q_032** — semantic gap (QServe) | CR=0.50, Comp=0.852 | CR=0.50, Comp=0.813 | **No.** CR unchanged (QServe still not retrieved). CP dropped 1.000→0.917. |

None of the 3 genuine failures improved. q_031 and q_034 actually worsened significantly.

### What about the RAGAS artifact questions (q_028, q_029, q_030)?

| Question | CP v3.1.1 | CP v4 | Change |
|---|---|---|---|
| q_028 (AWQ vs Quasar-ViT, hardware) | 0.000 | 0.000 | No change |
| q_029 (AWQ vs Quasar-ViT, activations) | 0.325 | 0.000 | Worse |
| q_030 (Self-RAG vs CATP-LLM) | 0.000 | 0.000 | No change |

These questions, already identified as RAGAS artifacts, were not improved by v4. q_029 worsened.

---

## What the Data Shows

1. **The architectural change (per-sub-query rerank + RRF) produced worse chunk ordering than global reranking.** 11 questions lost CP, and in 7 of those the same chunks were present — just ranked worse.

2. **Single-topic questions were unaffected.** These go through the single sub-query path (no RRF), which is architecturally identical to v3.1.1. This confirms the regression is caused by the RRF merge path, not the prompt change.

3. **The 3 genuine pipeline failures from v3.1.1 were not fixed.** The coreference prompt improvement was insufficient for q_034. The weak signal (q_031) and semantic gap (q_032) problems are retrieval-level issues that reranker placement cannot address.

4. **One previously-failing question improved.** q_033 (appraisal-based reward) gained CR from 0.50 → 1.00, suggesting per-sub-query reranking can help in specific cases where the global reranker was suppressing a relevant chunk from a secondary sub-query.

---

## Conclusion

**v4 is a failed experiment.** Composite dropped from 0.9062 to 0.8891, driven entirely by Context Precision regression on comparative questions. The genuine pipeline issues identified in v3.1.1 remain unresolved.

v3.1.1 remains the best configuration.

| Version | Composite |
|---|---|
| v3.1.1 (prefetch=30, global rerank) | **0.9062** |
| v4 (per-subquery rerank + RRF) | 0.8891 |
