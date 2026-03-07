# ArXiv ML Research Assistant

[![CI](https://github.com/anime-sh16/research-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/anime-sh16/research-rag/actions/workflows/ci.yml)

An end-to-end RAG system for querying ArXiv ML research papers with natural language.

> **Status:** Phase 0 — Project scaffolding and CI setup complete.

---

## Project Structure

```
research-rag/
├── src/
│   ├── config/        # pydantic-settings config, loads .env
│   ├── ingestion/     # ArXiv API fetch, chunking, embedding
│   ├── retrieval/     # hybrid search, reranking
│   ├── generation/    # LangGraph agent, prompts
│   ├── evaluation/    # RAGAS runner, test set
│   └── api/           # FastAPI app
├── scripts/           # one-off dev utilities (verify_connections, etc.)
├── tests/             # mirrors src/ structure
├── notebooks/         # exploratory analysis
├── .github/workflows/
├── pyproject.toml     # single config for uv, ruff, pytest, hatchling
└── .env.template      # copy to .env and fill in API keys
```

---

## Local Setup

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
| `GOOGLE_API_KEY` | Google AI Studio |
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

## Development

```bash
# Run tests
uv run pytest

# Lint
uv run ruff check .

# Format
uv run ruff format .
```

---

## CI

Every push to `main` and every pull request runs:

1. `ruff check` — linting
2. `ruff format --check` — formatting
3. `pytest` — full test suite

Future phases will add a RAGAS regression gate that blocks deploys if evaluation scores drop below baseline.
