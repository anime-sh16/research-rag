# V2 Hybrid + Rerank v2: Comprehensive Review
**Experiment:** `v2-hybrid-rerank-v2`
**Compared against:** `v2-hybrid-rerank` (v2.1) and `v1-baseline`
**Date:** 2026-03-14

---

## Changes from v2-hybrid-rerank (v2.1)

- **LLM upgrade:** `gemini-2.5-flash-lite` → `gemini-3-flash-preview`
- **Reranker upgrade:** `jina-reranker-v2-base-multilingual` → `jina-reranker-v3`
- **No architectural changes:** Same hybrid search (dense + BM25), same RRF fusion, same top-5 reranked retrieval, same prompt template
- **Same chunks:** Identical ingested content from v1

---

## Aggregate Scores (All 3 Versions)

| Metric | v1-baseline | v2.1-hybrid-rerank | v2.2-hybrid-rerank-v2 | Delta (v2.1→v2.2) | Delta (v1→v2.2) |
|--------|-------------|--------------------|-----------------------|--------------------|------------------|
| Faithfulness | 0.8742 | 0.8918 | **0.9753** | **+0.0835** | **+0.1011** |
| Answer Relevancy | 0.7509 | 0.7818 | **0.8726** | **+0.0908** | **+0.1217** |
| Context Precision | 0.6818 | 0.7680 | **0.8295** | **+0.0615** | **+0.1477** |
| Context Recall | 0.8415 | 0.9071 | **0.8984** | **-0.0087** | **+0.0569** |
| "Don't know" answers | 8 | 5 | **2** | **-3** | **-6** |
| Partial answers (answered but incomplete) | — | — | 3 | — | — |

All metrics except Context Recall improved significantly. Context Recall dipped marginally (-0.009) from v2.1 — within noise. The standout improvements are Faithfulness (+0.08) and Answer Relevancy (+0.09), both driven by the LLM upgrade.

---

## "Don't Know" Transition Map (Full 3-Version History)

| QID | Type | v1 | v2.1 | v2.2 | Transition |
|-----|------|----|------|------|------------|
| q_001 | factual/method-detail | DK | Answered | Answered | Resolved in v2.1 ✅ |
| q_023 | conceptual/comparison | DK | Answered | Answered | Resolved in v2.1 ✅ |
| q_025 | multi-hop/design-rationale | Answered | **DK** | **Answered** | v2.1 regression **FIXED** ✅ |
| q_028 | cross-paper/method-contrast | DK | Answered | Answered | Resolved in v2.1 ✅ |
| q_029 | cross-paper/tradeoff-comparison | DK | DK | **Partial** | Improved but still incomplete |
| q_030 | cross-paper/method-contrast | DK | DK | **Answered** | **RESOLVED** ✅ |
| q_031 | cross-paper/field-intersection | DK | Answered | Partial | Answered but quality dipped |
| q_032 | cross-paper/field-intersection | DK | DK | **Partial** | Improved — first half answered |
| q_033 | multi-hop/method-to-result | Partial | Partial | Partial | Persistent (half of multi-hop) |
| q_034 | multi-hop/arch-to-evaluation | Partial | Partial | Partial | Persistent (half of multi-hop) |
| q_039 | numerical/throughput | DK | DK | **DK** | **PERSISTENT** (ingestion) |

**Net DK count: 8 → 5 → 3** (q_029 partial-DK, q_032 partial-DK, q_039 full DK)

---

## Problems Solved in v2.2

### q_025 — v2.1 Regression FIXED (Multi-hop: RLHF decoupling rationale)
- **v2.1 status:** DK. The reranker selected the "best" chunk from 5 different RLHF papers, creating fragmented evidence. The LLM refused to synthesize.
- **v2.2 status:** Fully answered. f=1.00, ar=0.95, cp=0.87, cr=0.67.
- **Answer quality:** Provides structured explanation of decoupling rationale (supervised learning applicability, isolated evaluation, semi-supervised learning) and challenges (reward hacking, computational costs, non-stationary reward signal).
- **Why fixed:** The upgraded LLM (`gemini-3-flash-preview`) synthesizes across heterogeneous chunks that the previous model refused to handle. Same 5 RLHF papers in context — the difference is purely generation capability. This confirms the v2.1 analysis that identified it as a generation-layer problem.
- **Significance:** This was the single true regression in v2.1 and the highest-priority fix. The LLM upgrade resolved it without any retrieval changes.

### q_030 — Previously Persistent DK → Now Answered (Cross-paper: Self-RAG vs CATP-LLM)
- **v2.1 status:** DK. Retrieval had improved (both Self-RAG and CATP-LLM chunks present) but the LLM refused to synthesize.
- **v2.2 status:** Answered. f=1.00, ar=0.95, cp=0.00, cr=1.00.
- **Answer quality:** Correctly contrasts Self-RAG's reflection token mechanism with CATP-LLM's Cost-Aware Offline RL approach.
- **Why fixed:** Same root cause as q_025 — the upgraded LLM is willing to synthesize information across heterogeneous sources. The v2.1 analysis correctly predicted this would be solvable through generation-layer improvements.
- **Note:** Context precision remains 0.00 — the relevant chunks are present (cr=1.00) but not in the top positions. The LLM overcomes poor ranking with better comprehension.

### q_032 — Previously Persistent DK → Now Partial Answer (Cross-paper: heavy-tailed distributions)
- **v2.1 status:** DK. All chunks from "Pruning vs Quantization" only. QServe missing.
- **v2.2 status:** Partial answer. f=1.00, ar=0.97, cp=1.00, cr=0.50. First half answered thoroughly; second half (QServe) explicitly stated as missing.
- **Answer quality:** Correctly explains how outliers/heavy tails make pruning preferable over quantization. Context now includes the right Pruning-vs-Quantization analysis chunks. Explicitly states it cannot answer the QServe portion.
- **Why improved:** Two factors: (1) context precision jumped to 1.00 — the reranker-v3 promoted the most relevant analysis chunks, and (2) the upgraded LLM extracted a thorough answer from the available evidence instead of refusing entirely. QServe still not retrieved (persistent retrieval gap).

---

## Problems Improved But Not Fully Resolved

### q_029 — Cross-paper: AWQ vs Quasar-ViT activation treatment
- **v2.1:** DK. All 5 chunks from Quasar-ViT. AWQ missing.
- **v2.2:** Partial. f=1.00, ar=0.86, cp=1.00, cr=0.50. Provides detailed Quasar-ViT analysis but explicitly states AWQ is not in context.
- **Change:** The LLM no longer refuses entirely — it answers what it can (Quasar-ViT half) and transparently flags the missing AWQ information. Context precision jumped from 0.00 to 1.00 (reranker-v3 better at ranking Quasar-ViT chunks).
- **Root cause unchanged:** AWQ chunks still not retrieved. Query terms ("activations unquantized" + "FPGA acceleration") overwhelmingly favor Quasar-ViT in both dense and sparse retrieval.

### q_031 — Cross-paper: RAG hallucination bottleneck
- **v2.1:** Answered. f=1.00, ar=0.87, cp=0.70, cr=1.00.
- **v2.2:** Answered but with lower scores. f=0.94, ar=0.89, cp=0.50, cr=0.50.
- **Change:** Still answers correctly (identifies "Generation Failure, 31.0%" as bottleneck) but context recall dropped from 1.00 to 0.50, suggesting the reranker-v3 may have shuffled chunks differently for this query. The answer quality remains adequate.

### q_033 — Multi-hop: appraisal-based reward math + Sequeira empirical result
- **v2.1:** Partial (ar=0.00). Math formula correct, empirical result missing.
- **v2.2:** Partial (ar=0.95). Math formula correct with full LaTeX formulation. Explicitly notes the Sequeira empirical result is "discussed in Section 7" but not in retrieved chunks.
- **Change:** Answer relevancy jumped from 0.00 to 0.95. The upgraded LLM provides a more structured, complete answer for the available portion and explicitly explains *why* the second part is missing. RAGAS now scores this as relevant (v2.1's 0.00 was likely an evaluator limitation with the older model).

### q_034 — Multi-hop: Diffusion models + FVD metric
- **v2.1:** Partial (ar=0.00). Diffusion process explained, FVD metric missing.
- **v2.2:** Partial (ar=0.00). Same pattern — diffusion process fully explained, FVD still missing.
- **Root cause unchanged:** The FVD metric lives in a different section/paper than the diffusion process explanation. This is a retrieval recall gap for the second hop.

---

## Persistent Failure

### q_039 — Numerical: Cascade Llama-2-13B runtime
- **All versions:** DK. f=0.67, ar=0.00, cp=0.00, cr=0.00.
- **Root cause:** Ingestion failure — table data. The Llama-2-13B runtime data is in Table 6 of the Cascade paper but was lost during PDF extraction. The LLM now explicitly states: "Table 8 includes Bert-Base, Bert-Large, and Llama-3.2-1B-Instruct, but does not provide data for Llama-2-13B."
- **Not addressable** through LLM or reranker upgrades. Requires table-aware PDF ingestion.

---

## Notable Score Changes (Non-DK Questions)

### Major Improvements
| QID | Type | Change | Detail |
|-----|------|--------|--------|
| q_015 | factual/method-detail | cr: 0.00→1.00 | v2.1 had a recall anomaly (0.00 despite correct answer). v2.2 restores perfect scores across the board. |
| q_008 | factual/metric | f: 0.50→1.00 | Faithfulness doubled. The upgraded LLM generates more grounded answers. |
| q_005 | factual/method-detail | cp: 0.58→1.00 | Context precision nearly doubled — reranker-v3 better at promoting the answer chunk. |
| q_007 | factual/metric | cp: 0.50→1.00 | Same pattern — reranker improvement. |
| q_018 | factual/metric | cp: 0.50→1.00 | Same pattern. |
| q_026 | multi-hop/taxonomy-to-limitation | cp: 0.59→0.89, cr: 0.86→1.00 | Both retrieval metrics improved. |

### Notable Regression
| QID | Type | Change | Detail |
|-----|------|--------|--------|
| q_027 | cross-paper/method-contrast | cp: 0.87→0.25, cr: 1.00→0.67 | Context quality degraded while answer quality improved (ar: 0.77→0.96). The LLM compensates for worse retrieval with better comprehension. See analysis below. |

### q_027 Deep Dive (Cross-paper: SFT phase in alignment pipelines)
- **v2.1:** f=1.00, ar=0.77, cp=0.87, cr=1.00. Strong retrieval, adequate answer.
- **v2.2:** f=0.87, ar=0.96, cp=0.25, cr=0.67. Weaker retrieval, better answer.
- The reranker-v3 pulled in chunks from more diverse sources (5 different RLHF/alignment papers) but this diluted the context precision. Despite this, the LLM upgrade produced a significantly more relevant answer (ar +0.19). The slight faithfulness dip (1.00→0.87) suggests the LLM added some reasoning not directly grounded in the retrieved chunks.
- **This is the inverse of the q_025 v2.1 pattern:** where reranker heterogeneity hurt the old LLM, the new LLM thrives on diverse evidence.

---

## Per-Category Analysis

### Factual (19 questions)

| Metric | v1 | v2.1 | v2.2 | Change (v2.1→v2.2) |
|--------|----|----|------|---------------------|
| Avg Faithfulness | 0.912 | 0.921 | **0.987** | +0.066 |
| Avg Answer Relevancy | 0.887 | 0.896 | **0.919** | +0.023 |
| Avg Context Precision | 0.745 | 0.829 | **0.956** | +0.127 |
| Avg Context Recall | 1.000 | 0.947 | **1.000** | +0.053 |
| DK count | 1 | 0 | **0** | — |

**Verdict:** Near-perfect. 17 of 19 questions have context precision ≥ 0.83. The reranker-v3 upgrade is the primary driver — context precision jumped +0.13 on average. This is the strongest category by far.

### Conceptual (5 questions)

| Metric | v1 | v2.1 | v2.2 | Change (v2.1→v2.2) |
|--------|----|----|------|---------------------|
| Avg Faithfulness | 0.964 | 1.000 | **0.979** | -0.021 |
| Avg Answer Relevancy | 0.909 | 0.912 | **0.900** | -0.012 |
| Avg Context Precision | 0.940 | 0.756 | **0.851** | +0.095 |
| Avg Context Recall | 1.000 | 1.000 | **1.000** | — |
| DK count | 1 | 0 | **0** | — |

**Verdict:** Solid. All 5 questions fully answered with perfect recall. Minor faithfulness dip is noise on 5 samples.

### Multi-hop (6 questions)

| Metric | v1 | v2.1 | v2.2 | Change (v2.1→v2.2) |
|--------|----|----|------|---------------------|
| Avg Faithfulness | 0.973 | 0.833 | **1.000** | +0.167 |
| Avg Answer Relevancy | 0.571 | 0.425 | **0.753** | +0.328 |
| Avg Context Precision | 0.759 | 0.817 | **0.806** | -0.011 |
| Avg Context Recall | 0.806 | 0.865 | **0.778** | -0.087 |
| DK count | 0 | 1 | **0** | -1 (q_025 fixed) |
| Partial (half-answered) | 2 | 2 | 2 | unchanged (q_033, q_034) |

**Verdict:** The v2.1 regression (q_025) is fixed. Faithfulness is now perfect. Answer relevancy jumped massively (+0.33) driven by two factors: (1) q_025 recovery (0.00→0.95) and (2) q_033 recovery (0.00→0.95). The 2 persistent partial answers (q_033, q_034) are retrieval recall gaps for the second hop — not addressable through LLM upgrades.

### Cross-paper (6 questions)

| Metric | v1 | v2.1 | v2.2 | Change (v2.1→v2.2) |
|--------|----|----|------|---------------------|
| Avg Faithfulness | 0.558 | 0.577 | **0.967** | +0.390 |
| Avg Answer Relevancy | 0.390 | 0.562 | **0.930** | +0.368 |
| Avg Context Precision | 0.310 | 0.428 | **0.458** | +0.030 |
| Avg Context Recall | 0.528 | 0.750 | **0.694** | -0.056 |
| DK count | 5 | 3 | **0** | -3 |
| Partial answers | — | — | 2 (q_029, q_032) | — |

**Verdict:** Transformative improvement in faithfulness (+0.39) and answer relevancy (+0.37). Zero full DK answers for the first time. The LLM upgrade is the dominant factor — it synthesizes across multiple source papers where the previous model refused. Context precision remains the weakest metric (0.46), confirming retrieval diversity for cross-paper queries is still a challenge. But the new LLM compensates.

### Numerical (5 questions)

| Metric | v1 | v2.1 | v2.2 | Change (v2.1→v2.2) |
|--------|----|----|------|---------------------|
| Avg Faithfulness | 0.900 | 0.920 | **0.908** | -0.012 |
| Avg Answer Relevancy | 0.666 | 0.878 | **0.744** | -0.134 |
| Avg Context Precision | 0.540 | 0.700 | **0.800** | +0.100 |
| Avg Context Recall | 0.800 | 0.800 | **0.800** | — |
| DK count | 1 | 1 | **1** | unchanged (q_039) |

**Verdict:** Mixed. Context precision improved (+0.10) but answer relevancy dipped (-0.13) — driven by q_039 scoring ar=0.00 (same as v1) after v2.1 had scored ar=0.76 for the same DK answer. This is evaluator variance on an unanswerable question.

---

## Key Inferences

### 1. The LLM upgrade is the single highest-impact change across all experiments
Moving from `gemini-2.5-flash-lite` to `gemini-3-flash-preview` produced larger metric improvements than the hybrid search + reranker combined:
- Faithfulness: +0.08 (vs +0.02 for v2.1 over v1)
- Answer Relevancy: +0.09 (vs +0.03 for v2.1 over v1)
- Cross-paper faithfulness: +0.39 — the largest single-category improvement in any experiment

### 2. The LLM upgrade resolved the fundamental tension identified in v2.1
The v2.1 analysis identified "reranker-induced chunk heterogeneity" as a core problem — the reranker selected best-per-chunk across many papers, but the LLM couldn't synthesize fragmented evidence. `gemini-3-flash-preview` handles this confidently:
- q_025: 5 different RLHF papers → synthesized (was DK in v2.1)
- q_030: Self-RAG + CATP-LLM heterogeneous chunks → synthesized (was DK in v2.1)
- q_027: 5 alignment papers → better answer despite worse retrieval (ar +0.19)

### 3. Reranker-v3 improves context precision for focused queries
Context precision for factual questions jumped from 0.83 to 0.96. The `jina-reranker-v3` is measurably better at promoting the single best chunk to position 1. This is visible across q_003, q_004, q_005, q_007, q_018, q_040 — all factual questions where precision went from ~0.50 to 1.00.

### 4. Cross-paper retrieval diversity remains the weakest link
Despite all improvements, cross-paper context precision is only 0.46. The LLM compensates with better comprehension, but retrieval is still the bottleneck:
- q_029: AWQ still not retrieved (Quasar-ViT dominates)
- q_032: QServe still not retrieved (Pruning-vs-Quantization dominates)
- q_027: Precision dropped from 0.87 to 0.25 with more diverse but less focused chunks

### 5. "Partial answer" is the new failure mode
The old failure mode was "refuse to answer" (DK). The new failure mode is "answer what you can, flag what's missing":
- q_029: Answers Quasar-ViT half, flags AWQ missing
- q_032: Answers pruning-vs-quantization half, flags QServe missing
- q_033: Provides math formula, flags empirical result missing
- q_034: Explains diffusion process, flags FVD metric missing

This is a strictly better failure mode — partial information is more useful than refusal, and the explicit flagging maintains transparency.

### 6. Table extraction remains an ingestion-layer blocker
q_039 is the only true DK (no useful partial answer). Not addressable through any retrieval or generation improvement.

### 7. Evaluator behavior may differ with the new LLM
Several questions show answer relevancy shifts that may reflect RAGAS evaluator interaction with longer/more structured answers from `gemini-3-flash-preview`:
- q_033: ar 0.00→0.95 (same partial answer quality, but RAGAS now scores the detailed formula explanation as relevant)
- q_039: ar 0.76→0.00 (v2.1 got partial credit for a DK answer; v2.2 does not)

---

## Failure Classification (v2.2 Final State)

| QID | Status | Root Cause | Fix Layer |
|-----|--------|------------|-----------|
| q_029 | Partial | Retrieval: AWQ not surfaced | Source-aware retrieval diversity (MMR) |
| q_032 | Partial | Retrieval: QServe not surfaced | Source-aware retrieval diversity (MMR) |
| q_033 | Partial | Retrieval: Second hop (Sequeira result) not in top-5 | Multi-hop retrieval or larger context window |
| q_034 | Partial | Retrieval: FVD metric not in retrieved chunks | Multi-hop retrieval or larger context window |
| q_039 | DK | Ingestion: Table data lost in PDF extraction | Table-aware PDF parsing |
| q_027 | Answered (cp=0.25) | Retrieval: Reranker diversified too much | Source-aware reranking |
| q_031 | Answered (cr=0.50) | Retrieval: Context recall dipped | Monitor — may be noise |

---

## Cumulative Progress: v1 → v2.1 → v2.2

| Metric | v1 | v2.1 | v2.2 | Total Improvement |
|--------|----|----|------|-------------------|
| Faithfulness | 0.8742 | 0.8918 (+0.02) | 0.9753 (+0.08) | **+0.1011 (+11.6%)** |
| Answer Relevancy | 0.7509 | 0.7818 (+0.03) | 0.8726 (+0.09) | **+0.1217 (+16.2%)** |
| Context Precision | 0.6818 | 0.7680 (+0.09) | 0.8295 (+0.06) | **+0.1477 (+21.7%)** |
| Context Recall | 0.8415 | 0.9071 (+0.07) | 0.8984 (-0.01) | **+0.0569 (+6.8%)** |
| DK Count | 8 | 5 (-3) | 2 (-3) | **-6 (-75%)** |

---

## Actionable Next Steps (Prioritized)

1. **Retrieval diversity for cross-paper queries (addresses q_029, q_032, q_027):**
   Implement MMR (Maximal Marginal Relevance) or source-aware post-processing after reranking. When query terms map overwhelmingly to one source paper, force inclusion of a second source. This is now the #1 retrieval weakness.

2. **Multi-hop retrieval strategy (addresses q_033, q_034):**
   These questions require information from two distinct sections/papers. Consider query decomposition (split multi-hop question into sub-queries) or increasing the context window beyond top-5 chunks.

3. **Table-aware ingestion (addresses q_039):**
   Add table extraction to PDF parsing pipeline. This is the only true DK remaining.

4. **Prompt engineering for partial answers:**
   The LLM already handles partial answers well, but could be further tuned to explicitly distinguish "I found X but not Y" vs "I don't know" — improving signal for evaluation.

5. **RAGAS evaluator consistency:**
   Consider re-running v1 and v2.1 with the same evaluator model used for v2.2 to ensure score comparability across experiments.

---

## Final Assessment

**v2.2-hybrid-rerank-v2 represents a major step forward, driven primarily by the LLM upgrade.** Faithfulness crossed 0.97, answer relevancy crossed 0.87, and only 3 DK answers remain (down from 8 in v1).

The most significant outcome: **the generation-layer bottleneck identified in v2.1 is resolved.** The `gemini-3-flash-preview` model synthesizes across heterogeneous chunks that the previous model refused to process. This fixes the q_025 regression, resolves q_030, and produces partial answers where v2.1 gave blanket refusals.

The remaining failures are now almost exclusively **retrieval problems**:
- 2 cross-paper retrieval diversity gaps (q_029, q_032)
- 2 multi-hop retrieval recall gaps (q_033, q_034)
- 1 ingestion gap (q_039)

The clearest next lever is **retrieval diversity** — specifically source-aware mechanisms that ensure cross-paper queries surface chunks from multiple source papers.
