# ArXiv ML Research Assistant

[![CI](https://github.com/anime-sh16/research-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/anime-sh16/research-rag/actions/workflows/ci.yml)

An end-to-end RAG system for querying ArXiv ML research papers with natural language.

> **Status:** Phase 0 — Project scaffolding and CI setup complete.

---

## Project Structure

```
research-rag/
├── src/
│   ├── ingestion/
│   ├── retrieval/
│   ├── generation/
│   ├── evaluation/
│   └── api/
├── tests/     
├── notebooks/ 
├── .github/
├── pyproject.toml
└── main.py

```

---

## Tech Stack

| Layer | Tool |
|---|---|
| Language | Python 3.12 |
| Package Manager | uv |
| Linting / Formatting | ruff |
| Testing | pytest |
| CI | GitHub Actions |

---

## Local Setup

**Prerequisites:** Python 3.12, [`uv`](https://docs.astral.sh/uv/)

```bash
git clone git remote add origin https://github.com/anime-sh16/research-rag.git
cd research-rag

# Install dependencies
uv sync --all-groups

# Copy environment template
cp .env.template .env
# Fill in your API keys in .env
```

---

## Development

```bash
# Run tests
uv run pytest tests/ -v

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
3. `pytest tests/` — full test suite
