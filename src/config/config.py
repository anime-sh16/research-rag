import os
from pathlib import Path

import arxiv
from pydantic import BaseModel, SecretStr
from pydantic_settings import BaseSettings


class DataConfig(BaseModel):
    temp_dir: Path = Path("data/tmp")
    pdf_dir: Path = Path("data/pdfs")
    ingested_chunks_file: str = "ingested_chunks.jsonl"
    query_cache_file: str = "query_cache.jsonl"


class DBConfig(BaseModel):
    collection_name: str = "arxiv_paper_1"
    embedding_dimension: int = 768
    full_embedding_dimension: int = 3072  # gemini-embedding-001 defaults to 3072. process teh normalization automatically
    embedding_model: str = "gemini-embedding-001"
    ingest_task_type: str = "SEMANTIC_SIMILARITY"
    retrieval_task_type: str = "RETRIEVAL_QUERY"


class IngestionConfig(BaseModel):
    topics: list[str] = [
        "language models NLP",
        "retrieval augmented generation knowledge",
        "generative models diffusion GAN",
        "fine-tuning transfer learning adaptation",
        "reinforcement learning human feedback alignment",
        "vision transformers multimodal learning",
        "agents planning tool use autonomous",
        "efficient inference compression optimization",
    ]
    fetch_sort_by: arxiv.SortCriterion = arxiv.SortCriterion.Relevance
    fetch_per_topic: int = 250
    target_papers_no: int = 70
    chunk_size: int = 512
    chunk_overlap: int = 64


class GenerationConfig(BaseModel):
    model: str = "gemini-2.5-flash-lite"
    temperature: float = 0.1
    top_k: int = 5


class APIConfig(BaseModel):
    title: str = "ArXiv RAG API"


class Settings(BaseSettings):
    # Secrets — required, loaded from .env; SecretStr prevents accidental logging
    google_api_key: SecretStr
    qdrant_url: str
    qdrant_api_key: SecretStr
    cohere_api_key: SecretStr
    langsmith_tracing: str = "false"
    langsmith_endpoint: str = "https://api.smith.langchain.com"
    langsmith_api_key: SecretStr
    langsmith_project: str

    # App config groups — have defaults, overridable via env (e.g. DB__COLLECTION_NAME=foo)
    data: DataConfig = DataConfig()
    db: DBConfig = DBConfig()
    ingestion: IngestionConfig = IngestionConfig()
    generation: GenerationConfig = GenerationConfig()
    api: APIConfig = APIConfig()

    model_config = {
        "env_file": ".env",
        "env_nested_delimiter": "__",
    }


settings = Settings()

# LangSmith SDK reads directly from os.environ, not from pydantic model
os.environ["LANGSMITH_TRACING"] = settings.langsmith_tracing
os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key.get_secret_value()
os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint
os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
