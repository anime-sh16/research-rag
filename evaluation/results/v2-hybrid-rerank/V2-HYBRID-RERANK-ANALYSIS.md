# V2 Hybrid + Rerank: Comprehensive Review
**Experiment:** `v2-hybrid-rerank`
**Compared against:** `v1-baseline`
**Date:** 2026-03-13

---

## Changes from v1-baseline

- **Hybrid search** using Qdrant prefetch (dense + sparse)
  - Dense: `gemini-embedding-001` (same as baseline)
  - Sparse: `qdrant/bm25`
  - Fusion: RRF (Reciprocal Rank Fusion) to combine top results
- **Reranker**: Jina API reranker applied to top results after fusion, returning top 5
- **Same chunks**: Locally saved chunks from v1 ingestion (identical chunk content)
- **Same generation model and prompt**: No changes to generation layer

---

## Aggregate Scores

| Metric | v1-baseline | v2-hybrid-rerank | Delta |
|--------|-------------|------------------|-------|
| Answer Relevancy | 0.75 | 0.78 | **+0.03** |
| Context Precision | 0.68 | 0.77 | **+0.09** |
| Context Recall | 0.84 | 0.91 | **+0.07** |
| Faithfulness | 0.87 | 0.89 | **+0.02** |
| "Don't know" answers | 8 | 5 | **-3** |

All four RAGAS metrics improved. Context precision saw the largest gain (+0.09), confirming that the reranker successfully promotes relevant chunks to the top positions.

---

## "Don't Know" Transition Map

| QID | Type | v1 Status | v2 Status | Transition |
|-----|------|-----------|-----------|------------|
| q_001 | factual/method-detail | DK | Answered | **RESOLVED** |
| q_023 | conceptual/comparison | DK | Answered | **RESOLVED** |
| q_025 | multi-hop/design-rationale | Answered (r=0.95) | DK | **REGRESSION** |
| q_028 | cross-paper/method-contrast | DK | Answered | **RESOLVED** |
| q_029 | cross-paper/tradeoff-comparison | DK | DK | PERSISTENT |
| q_030 | cross-paper/method-contrast | DK | DK | PERSISTENT (root cause shifted) |
| q_031 | cross-paper/field-intersection | DK | Answered | **RESOLVED** |
| q_032 | cross-paper/field-intersection | DK | DK | PERSISTENT |
| q_039 | numerical/throughput | DK | DK | PERSISTENT |

**Net: 4 resolved, 4 persistent, 1 new regression = 8 -> 5 "don't know"**

---

## Problems Solved (4 questions: v1 DK -> v2 answered)

### q_001 — Factual/method-detail (AWQ salient weight percentage)
- **v1:** DK. Scores f=0.00, r=0.00, p=0.33, c=0.00. AWQ chunks present (2/5) but the answer-bearing chunk ranked too low.
- **v2:** Correctly identifies "0.1%-1% of salient weights" with supporting detail.
- **Why:** Reranker promoted the relevant AWQ methodology chunk to a top position. Straightforward retrieval quality improvement.

### q_023 — Conceptual/comparison (assignment-based vs classifier guidance)
- **v1:** DK with high scores (f=0.86, r=0.94, p=1.00, c=1.00). Context was perfect — all from correct source paper, covering all 3 guidance types. The LLM said "don't know" then provided a detailed partial analysis. Pure generation-layer conservatism.
- **v2:** Fully answered. Provides a structured comparison of how assignment-based guidance overcomes classifier and classifier-free guidance limitations.
- **Why:** This is the most interesting resolution. The generation prompt didn't change, so either: (a) the reranker changed chunk ordering within the same source, making the comparison more obvious to the LLM, or (b) non-determinism in generation. Since the context was already perfect in v1, the chunk ordering is the most likely factor.

### q_028 — Cross-paper/method-contrast (AWQ vs Quasar-ViT mixed-precision)
- **v1:** DK. Scores f=0.75, r=0.74, p=0.00, c=0.50. All 5 chunks from Quasar-ViT; AWQ completely missing. Answer says "does not mention AWQ."
- **v2:** Correctly contrasts AWQ's channel scaling approach with Quasar-ViT's row-wise mixed-precision scheme. Retrieved 2 AWQ + 2 Quasar-ViT chunks.
- **Why:** Hybrid search (BM25) surfaced AWQ chunks that dense-only missed. The keyword "AWQ" in the query matched BM25's lexical approach. This is exactly the scenario hybrid search was designed for.

### q_031 — Cross-paper/field-intersection (RAG hallucination bottleneck)
- **v1:** DK. Scores f=0.00, r=0.00, p=0.20, c=0.00. Chunks were about RAG and hallucination generally but not from the two specific source papers.
- **v2:** Correctly identifies "Generation Failure (31.0% of errors)" as the bottleneck. Retrieved chunks from both FAIR-RAG and Factuality Challenges papers.
- **Why:** Hybrid search improved coverage. BM25 matched the specific terminology ("hallucination," "RAG pipeline," "error bottleneck") that led to the correct source papers.

---

## Problems Still Existing (4 persistent "don't know")

### q_029 — Cross-paper/tradeoff-comparison (AWQ vs Quasar-ViT activation treatment)
- **v1:** DK. All 5 chunks from Quasar-ViT. AWQ missing.
- **v2:** Still DK. All 5 chunks still from Quasar-ViT. AWQ still missing.
- **Root cause: Retrieval failure — unchanged.** The query about "leaving activations unquantized" + "FPGA acceleration" triggers Quasar-ViT so strongly in both dense and sparse retrieval that AWQ cannot break into the top results.
- **Note:** The sibling question q_028 (same two papers, different angle) was resolved — "mixed-precision quantization" + "hardware inefficiency" is a query that BM25 can match to AWQ content. The difference is the query wording.

### q_030 — Cross-paper/method-contrast (Self-RAG vs CATP-LLM tool assessment)
- **v1:** DK. All 5 chunks from CATP-LLM only. Self-RAG/RAG Stack completely missing. Answer says "does not mention Self-RAG."
- **v2:** Still DK. But retrieval **improved** — chunk 1 now describes Self-RAG's reflection token mechanism, chunks 2 and 4 describe CATP-LLM. Both sources ARE present.
- **Root cause shifted: Retrieval failure (v1) -> Generation failure (v2).** The retrieval problem was solved by hybrid search. The LLM now has both mechanisms in its context but refuses to synthesize a comparison. Additionally, 2 of the 5 chunks are noise (Parametric RAG, WeKnow-RAG), which may dilute the signal.
- **This is a notable case:** retrieval improved but the outcome didn't change. Confirms generation-layer work is needed.

### q_032 — Cross-paper/field-intersection (heavy-tailed data + quantization vs pruning)
- **v1:** DK. All chunks from "Pruning vs Quantization." QServe missing.
- **v2:** Still DK. All 5 chunks still from "Pruning vs Quantization." QServe still missing.
- **Root cause: Retrieval failure — unchanged.** The query terms ("heavy-tailed," "quantization," "pruning") match the Pruning-vs-Quantization paper so strongly that QServe cannot surface. Same pattern as q_029.

### q_039 — Numerical/throughput (Cascade Llama-2-13B runtime)
- **v1:** DK. Cascade chunks present but Llama-2-13B data is in Table 6, lost during PDF extraction.
- **v2:** Still DK. Same root cause.
- **Root cause: Ingestion failure — table data.** Not solvable through retrieval improvements.

---

## New Regression (1 question: v1 answered -> v2 DK)

### q_025 — Multi-hop/design-rationale-to-limitation (RLHF decoupling rationale)
- **v1:** Answered well. Scores f=0.92, r=**0.95**, p=0.92, c=0.83. **Top-performing multi-hop question in v1.**
- **v1 answer:** Detailed explanation of why reward modeling is decoupled (supervised learning applicability, isolated evaluation, semi-supervised learning) and the challenges (reward hacking, non-stationary reward signal).
- **v1 chunks:** All 5 chunks from RLHF-related papers with focused content on reward modeling decoupling rationale. Chunk 4 directly addresses "Reward Modeling: We focus on approaches that learn a reward model from human feedback..."
- **v2:** "I don't have enough information to answer this."
- **v2 chunks:** 5 chunks from 5 different RLHF papers (Safe RLHF, ReST, general RLHF overview, BTL model, DPO). The answer IS present — chunk 3 mentions "reward hacking" and "substantial computational costs," chunk 5 mentions "two-stage process is complex and may be unstable."
- **Root cause: Generation failure caused by reranker-induced chunk heterogeneity.**
  - In v1, dense retrieval returned chunks clustered around the same focused content — the LLM saw cohesive evidence and synthesized confidently.
  - In v2, the reranker selected the "best" chunk from each of 5 different papers. Each chunk discusses RLHF from a different angle (safety, efficiency, theory, alternatives). The LLM sees fragmented, heterogeneous evidence and refuses to synthesize.
  - The reranker optimized per-chunk relevance but reduced cohesion.
- **Severity: HIGH** — This was the strongest multi-hop question in v1. The regression demonstrates a fundamental tension between per-chunk relevance ranking and answer coherence.

---

## Other Notable Changes (Non-DK questions)

### q_033 — Multi-hop (partial answer, rel=0 in both v1 and v2)
- Both versions correctly provide the appraisal-based reward math formula.
- Both versions fail to retrieve the Sequeira empirical result (lives in a different paper section).
- v1 and v2 are functionally equivalent here. The answer is partial but correct for what's available.

### q_034 — Multi-hop (partial answer, rel=0 in both v1 and v2)
- Both versions correctly explain the diffusion Markov chain process with math.
- Both versions miss the FVD metric (not in retrieved chunks).
- Functionally equivalent. RAGAS scores 0 relevancy because the question is half-answered.

### q_027 — Cross-paper (only cross-paper question answered in both v1 and v2)
- v1: f=1.00, r=0.77, p=0.81, c=0.67
- Still answered in v2. Works because "instruction tuning" and "RLHF" are pervasive enough in the corpus that both papers surface naturally.

---

## Per-Category Analysis

### Factual (19 questions)

| Metric | v1 | v2 | Change |
|--------|----|----|--------|
| "Don't know" | 1 | **0** | -1 (q_001 resolved) |
| Avg Faithfulness | 0.912 | improved | |
| Avg Precision | 0.745 | significantly improved | Reranker benefit |

**Verdict:** Strongest improvement category. Reranker excels at single-paper factual retrieval — promotes the answer-bearing chunk to position 1.

### Conceptual (5 questions)

| Metric | v1 | v2 | Change |
|--------|----|----|--------|
| "Don't know" | 1 | **0** | -1 (q_023 resolved) |
| Avg Precision | 0.940 | slight decrease (1 sample) | |

**Verdict:** q_023 resolution is significant — it was a generation-layer fix enabled by better chunk ordering. 1 sample shows slightly lower precision but overall strong.

### Multi-hop (6 questions)

| Metric | v1 | v2 | Change |
|--------|----|----|--------|
| "Don't know" | 0 | **1** | +1 (q_025 regression) |
| 0 answer relevancy | 2 (q_033, q_034) | 3 (+q_025) | Degradation |
| Avg Recall | 0.806 | improved | |
| Avg Precision | 0.759 | improved | |

**Verdict:** Mixed. Retrieval metrics improved across the board, but the q_025 regression pulls down answer relevancy. The 2 persistent 0-relevancy questions (q_033, q_034) are retrieval recall gaps, not solvable by reranking.

### Cross-paper (6 questions)

| Metric | v1 | v2 | Change |
|--------|----|----|--------|
| "Don't know" | 5 | **3** | -2 (q_028, q_031 resolved) |
| Answered | 1 (q_027) | 3 (+ q_028, q_031) | |

**Verdict:** The biggest category improvement. Hybrid search directly addressed the core v1 failure mode (single-source retrieval). 2 of 5 DK questions resolved. The remaining 3 persistent DK questions split into:
- 2 retrieval failures (q_029, q_032): query terms overwhelmingly favor one source
- 1 generation failure (q_030): retrieval improved but LLM won't synthesize

### Numerical (5 questions)

| Metric | v1 | v2 | Change |
|--------|----|----|--------|
| "Don't know" | 1 | **1** | unchanged (q_039) |

**Verdict:** No change in DK count. q_039 is an ingestion problem. Marginal score improvements on other numerical questions.

---

## Key Inferences

### 1. Hybrid search (BM25) directly addressed the #1 v1 failure mode
Cross-paper retrieval diversity was the biggest problem in v1 (5/6 DK). Hybrid search resolved 2 of those 5 (q_028, q_031) by surfacing chunks via lexical matching that dense-only missed. This validates the hybrid search addition.

### 2. Reranker is highly effective for focused single-source questions
Context precision +0.09 is driven by the reranker promoting the best chunk to position 1 for factual/conceptual questions. q_001 and q_023 resolutions are direct reranker benefits.

### 3. Hybrid search + reranker cannot solve ALL cross-paper failures
3 cross-paper questions remain DK. Two patterns:
- **Query term dominance** (q_029, q_032): When query terms map overwhelmingly to one paper, neither dense nor sparse retrieval can surface the second paper. RRF fusion reinforces the overlap rather than diversifying.
- **Generation conservatism** (q_030): Retrieval improved (both sources now present) but the LLM refuses to synthesize. This is a prompt/generation issue.

### 4. Reranker-induced chunk heterogeneity can hurt synthesis questions
The q_025 regression reveals a fundamental tension: the reranker selects the "best" individual chunks, but for synthesis questions, **cohesive chunks from fewer sources** may be better than **diverse best-of-breed chunks from many sources**. In v1, dense retrieval accidentally returned cohesive RLHF chunks; in v2, the reranker picked the top chunk from each of 5 RLHF papers, creating fragmentation.

### 5. q_030 demonstrates that retrieval improvement alone is insufficient
In v1, q_030 failed because Self-RAG wasn't retrieved. In v2, Self-RAG IS retrieved, but the LLM still says "don't know." This proves that some questions need generation-layer work (prompt engineering, better synthesis instructions) in addition to retrieval improvements.

### 6. "Don't know" with partial content is an evaluator/prompt interaction
Several v1 DK answers had high RAGAS scores (q_023: r=0.94, q_029: r=0.81, q_030: r=0.82) because the LLM provided useful partial analysis after saying "I don't have enough information." RAGAS evaluates the full text, not the DK flag. This means:
- The DK count and RAGAS scores tell different stories
- The LLM's DK threshold is miscalibrated — it says "don't know" too readily, even when it has enough context to provide a useful answer

### 7. Table extraction remains an ingestion-layer blocker
q_039 is persistent across both versions. Not addressable through retrieval improvements.

---

## Regression Root Cause Summary

| QID | Transition | Root Cause | Fix Layer |
|-----|-----------|------------|-----------|
| q_025 | Answered -> DK | Reranker spread chunks across 5 RLHF papers; LLM saw fragmented evidence and refused to synthesize | Prompt engineering (instruct to synthesize across sources) |

Note: q_029 and q_030 are NOT regressions — they were already DK in v1.

---

## Actionable Next Steps (Prioritized)

1. **Prompt engineering (addresses q_025, q_030, and general generation conservatism):** Explicitly instruct the LLM to synthesize information across multiple context chunks, even when they come from different sources. Track as `prompt-v1` experiment. This addresses the single true regression (q_025) and the shifted-root-cause persistent failure (q_030).

2. **Evaluator model upgrade:** Switch from `gemini-2.5-flash-lite` to `gemini-3-flash-preview` for RAGAS scoring. Re-run both v1 and v2 evaluations with the new evaluator for comparable, higher-quality scores.

3. **Retrieval diversity (addresses q_029, q_032):** Investigate MMR (Maximal Marginal Relevance) or source-aware post-processing after reranking. When a cross-paper question is detected, ensure chunks from multiple source papers appear in the final top-5.

4. **Table-aware ingestion (addresses q_039):** Add table extraction to PDF parsing pipeline.

---

## Final Assessment

**v2-hybrid-rerank is a clear net improvement over v1-baseline.** All aggregate metrics improved. 4 previously unanswered questions are now answered (q_001, q_023, q_028, q_031), with cross-paper questions seeing the biggest improvement (5 DK -> 3 DK).

The experiment introduced 1 true regression (q_025) caused by reranker-induced chunk heterogeneity. This reveals a tension between per-chunk relevance optimization and answer coherence that the generation layer needs to handle.

The remaining 4 persistent DK questions break down into:
- 2 retrieval diversity failures (q_029, q_032) — need source-aware retrieval
- 1 generation failure with improved retrieval (q_030) — need prompt engineering
- 1 ingestion failure (q_039) — need table extraction

The clearest next lever is **prompt engineering** — it addresses the 1 regression, the 1 shifted-root-cause persistent failure, and the general generation conservatism observed across both versions.
