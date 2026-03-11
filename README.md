# ArXiv ML Research Assistant

[![CI](https://github.com/anime-sh16/research-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/anime-sh16/research-rag/actions/workflows/ci.yml)

An end-to-end production-grade RAG system for querying ArXiv ML research papers with natural language. Ask a question, get a grounded answer with source citations — retrieved from a corpus of curated ML papers spanning LLMs, diffusion models, RL alignment, vision transformers, and more.

Current Stage -- v1 Baseline
---

## Aim

The goal is to build a RAG system over ArXiv ML research, using measurement-driven iteration to improve systematically.

---

## How It Works

```
User Question
     │
     ▼
[Retriever] ──── Dense vector search (Gemini embeddings, Qdrant)
     │                  Top-5 chunks by cosine similarity
     ▼
[Generator] ──── Gemini 2.5 Flash Lite
     │                  Grounded by retrieved context only
     ▼
Answer + Sources (paper title, authors, arxiv ID, score)
```

The pipeline is orchestrated in `src/api/main.py` (`run_pipeline`), decoupled from the HTTP layer so it can be called directly by the evaluation runner.

---

## Data

**Source:** ArXiv API (free, structured, no scraping required)

**Topics (8 query areas):**
- Large language models & NLP transformers
- Retrieval-augmented generation & knowledge systems
- Diffusion & generative models
- Fine-tuning & instruction tuning
- Reinforcement learning & alignment (RLHF)
- Vision transformers & multimodal learning
- LLM agents, planning, tool use
- Inference optimization (quantization, pruning)

**Filtering:** Papers must belong to an allowed category allowlist (`cs.LG`, `cs.CL`, `cs.AI`, `cs.CV`, `cs.IR`, `cs.NE`, `cs.MA`, `stat.ML`) — off-domain papers are rejected at fetch time.

**Volume:** Up to 70 papers per topic, deduplicated by ArXiv ID. PDFs are downloaded selectively and text is extracted with PyMuPDF (ligature normalization, hyphenation fixing, reference section removal).

---

## Ingestion Pipeline

**Location:** `src/ingestion/`

### 1. ArXiv Client (`arxiv_client.py`)
Fetches paper metadata, filters by category, deduplicates, then downloads PDFs for accepted papers. Respects ArXiv rate limits with 5s delays and 3 retries with exponential backoff.

### 2. Chunking (`chunker.py`)
- **Method:** `RecursiveCharacterTextSplitter` (LangChain)
- **Tokenizer:** Tiktoken `o200k_base` for accurate token counting
- **Chunk size:** 512 tokens | **Overlap:** 64 tokens
- **Metadata per chunk:** `paper_id`, `title`, `authors`, `category`, `publication_date`, `source`

### 3. Embeddings & Vector Store (`vector_store.py`)
- **Model:** `gemini-embedding-001` (768-dim output, cosine distance)
- **Task types:** `RETRIEVAL_DOCUMENT` at ingest, `RETRIEVAL_QUERY` at search time
- **Database:** Qdrant Cloud (collection: `arxiv_paper_v0.5`)
- **Deduplication:** UUID5-based chunk IDs prevent re-embedding on re-runs
- **Rate limiting:** 100-chunk batches with 4s inter-batch sleep + exponential backoff on 429s

### 4. Ingestion Orchestrator (`pipeline.py`)
Runs multi-topic fetching with progress tracking. Saves output as `chunks_<timestamp>.jsonl` and `summary_<timestamp>.json`.

---

## Retrieval

**Location:** `src/retrieval/retriever.py`

Current approach: **dense vector search** — cosine similarity over Gemini embeddings, top-5 chunks per query.

- Query embeddings are cached (MD5-keyed JSONL) to avoid redundant API calls
- Exponential backoff on embedding rate limits
- LangSmith proxy metrics logged per query: `retrieval_avg_score`, `retrieval_score_spread`, `source_diversity`

---

## Generation Pipeline

**Location:** `src/generation/chain.py`

- **Model:** Gemini 2.5 Flash Lite (`temperature=0.1`)
- **Prompt (v1-baseline):** System instruction constrains answers to retrieved context only. Context chunks are numbered and titled. If context is insufficient, the model says so.
- **Tracing:** Full prompt, token usage, and cited paper IDs logged to LangSmith with tag `prompt_version:v1-baseline`
- **Retry logic:** Up to 5 attempts with max 480s wait on Gemini 429 errors

---

## FastAPI

**Location:** `src/api/main.py`

### `POST /query`

**Request:**
```json
{ "question": "What training objective does InstructGPT use?" }
```

**Response:**
```json
{
  "answer": "InstructGPT uses RLHF with a KL penalty...",
  "sources": [
    {
      "title": "Training language models to follow instructions...",
      "authors": ["Ouyang, Long", "..."],
      "paper_id": "2203.02155",
      "chunk_index": 3,
      "score": 0.84
    }
  ]
}
```

Auto-flags attached to LangSmith root span: `empty_retrieval`, `low_retrieval_score` (< 0.5), `single_source_warning`.

Run locally:
```bash
uv run uvicorn src.api.main:app --reload
```

Query the API:
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What training objective does InstructGPT use?"}'
```

---

## Evaluation

**Location:**
- `src/evaluation/`
- `./evaluation/`

### Evaluation Set
41 hand-curated questions with ground truth answers, covering:
- **Types:** Factual, conceptual, multi-hop, numerical, cross-paper
- **Subtypes:** Method-detail, metric, hyperparameter, formula, throughput, architecture, comparison, tradeoff, limitation

The eval set is **fixed and immutable** — all experiments run against the same 41 questions.

### RAGAS Metrics
| Metric | What it measures |
|---|---|
| **Faithfulness** | Does the answer hallucinate facts not in the retrieved context? |
| **Answer Relevancy** | Is the answer relevant to the question asked? |
| **Context Precision** | Are retrieved chunks actually relevant to the ground truth? |
| **Context Recall** | Does retrieval capture all context needed to answer? |

---

## Baseline Results

**Experiment:** `v1-baseline` | Dense retrieval + Gemini 2.5 Flash Lite | 2026-03-11

| Metric | Score |
|---|---|
| Faithfulness | **0.8742** |
| Answer Relevancy | **0.7509** |
| Context Precision | **0.6818** |
| Context Recall | **0.8415** |

**Key observations:**
- High faithfulness (0.87) — the model stays grounded in what it retrieves
- Context precision (0.68) is the primary bottleneck — dense search retrieves high-scoring chunks that aren't always relevant to the ground truth
- Weakest questions: multi-hop cross-paper queries and numerical formula lookups
- Strongest questions: direct factual lookups and method-detail questions

Full per-question breakdown: `evaluation/results/v1-baseline.json`

---

## Project Structure

```
research-rag/
├── src/
│   ├── api/
│   │   └── main.py               # FastAPI app + pipeline orchestration
│   ├── config/
│   │   └── config.py             # Pydantic settings (loaded from .env)
│   ├── ingestion/
│   │   ├── arxiv_client.py       # ArXiv API fetch + PDF extraction
│   │   ├── chunker.py            # Recursive chunking + metadata
│   │   ├── pipeline.py           # Multi-topic ingestion orchestrator
│   │   └── vector_store.py       # Gemini embeddings + Qdrant upsert
│   ├── retrieval/
│   │   └── retriever.py          # Dense search + embed cache + proxy metrics
│   ├── generation/
│   │   └── chain.py              # Gemini generation + LangSmith tracing
│   └── evaluation/
│       ├── ragas_runner.py       # RAGAS evaluation + LangSmith experiments
│       └── dataset_upload.py     # Upload evalset to LangSmith
├── evaluation/
│   ├── evalset.json              # 41 questions + ground truth (immutable)
│   └── results/
│       └── v1-baseline.json      # Baseline RAGAS snapshot
├── tests/                        # Unit + integration tests (mirrors src/)
├── scripts/                      # Dev utilities (verify connections, smoke tests)
├── .github/workflows/ci.yml      # Lint + format + test on every push
├── pyproject.toml                # Single config: uv, ruff, pytest, hatchling
└── .env.template                 # Copy to .env and fill in API keys
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Embeddings** | Google Gemini (`gemini-embedding-001`, 768d) |
| **LLM** | Google Gemini 2.5 Flash Lite |
| **Vector DB** | Qdrant Cloud |
| **PDF Extraction** | PyMuPDF |
| **Chunking** | LangChain `RecursiveCharacterTextSplitter` + Tiktoken |
| **API** | FastAPI + Uvicorn |
| **Evaluation** | RAGAS |
| **Tracing** | LangSmith |
| **Config** | Pydantic Settings |
| **Retry Logic** | Tenacity |
| **Linting/Formatting** | Ruff |
| **Testing** | Pytest |
| **Packaging** | uv + hatchling |

---

## Setup

**Prerequisites:** Python 3.12, [`uv`](https://docs.astral.sh/uv/)

```bash
git clone https://github.com/anime-sh16/research-rag.git
cd research-rag

# Install dependencies and project in editable mode
uv sync --all-groups

# Copy environment template and fill in your API keys
cp .env.template .env
```

### Required API Keys

| Variable | Where to get it |
|---|---|
| `GOOGLE_API_KEY` | [Google AI Studio](https://aistudio.google.com) |
| `QDRANT_URL` | Qdrant Cloud console |
| `QDRANT_API_KEY` | Qdrant Cloud console |
| `COHERE_API_KEY` | Cohere dashboard |
| `LANGSMITH_API_KEY` | LangSmith settings |
| `LANGSMITH_PROJECT` | LangSmith project name |
| `LANGSMITH_TRACING` | `true` to enable tracing |
| `LANGSMITH_ENDPOINT` | `https://api.smith.langchain.com` |

### Verify setup

```bash
uv run scripts/verify_connections.py
```

---

## Running

```bash
# Start the API server
uv run uvicorn src.api.main:app --reload

# Run evaluation against the full eval set
uv run python -m src.evaluation.ragas_runner

# Run tests
uv run pytest

# Lint / Format
uv run ruff check .
uv run ruff format .
```

---

## CI

Every push to `main` and every pull request runs:

1. `ruff check` — linting
2. `ruff format --check` — formatting
3. `pytest` — full test suite

---

## Future Work

- **Hybrid search** — add BM25 sparse retrieval fused with dense via Reciprocal Rank Fusion (RRF) to improve context precision on keyword-heavy queries
- **Reranking** — cross-encoder reranker (Cohere) as a post-retrieval filter to push most relevant chunks to top-k
- **Query expansion** — HyDE or sub-question decomposition to improve multi-hop and cross-paper recall
- **RAGAS regression gate** — block CI merges if evaluation scores drop below a stored baseline threshold
- **Metadata filtering** — filter by publication date, topic, or author at query time
- **Improved Chunking** - Try improving the chinking process
