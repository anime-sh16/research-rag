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
    collection_name: str = "arxiv_paper_v1_hybrid"
    dense_name: str = "dense"
    sparse_name: str = "sparse"
    sparse_model: str = "Qdrant/bm25"
    embedding_dimension: int = 768
    full_embedding_dimension: int = 3072  # gemini-embedding-001 defaults to 3072. process teh normalization automatically
    embedding_model: str = "gemini-embedding-001"
    ingest_task_type: str = "RETRIEVAL_DOCUMENT"
    retrieval_task_type: str = "RETRIEVAL_QUERY"


class IngestionConfig(BaseModel):
    topics: list[str] = [
        "large language models LLM NLP transformers",
        "retrieval augmented generation knowledge",
        "diffusion probabilistic models image generation deep learning",
        "fine-tuning pretrained language models instruction tuning",
        "reinforcement learning human feedback alignment",
        "vision transformers ViT multimodal learning",
        "LLM agents planning tool use reasoning",
        "LLM inference optimization quantization pruning",
    ]
    # ArXiv primary_category allowlist — rejects off-domain papers (math, physics, chemistry)
    allowed_categories: set[str] = {
        "cs.LG",  # Machine Learning
        "cs.CL",  # Computation and Language (NLP)
        "cs.AI",  # Artificial Intelligence
        "cs.CV",  # Computer Vision and Pattern Recognition
        "cs.IR",  # Information Retrieval
        "cs.NE",  # Neural and Evolutionary Computing
        "cs.MA",  # Multiagent Systems (LLM agents papers)
        "stat.ML",  # Statistics - Machine Learning
    }
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


class RetrievalConfig(BaseModel):
    hybrid_prefetch_k: int = 20
    rerank_top_n: int = 5
    jina_rerank_model: str = "jina-reranker-v2-base-multilingual"


class EvaluationConfig(BaseModel):
    dataset_name: str = "arxiv-rag-eval-set"
    evalset_path: Path = Path("evaluation/evalset.json")
    results_dir: Path = Path("evaluation/results")
    evaluator_model: str = "gemini-2.5-flash-lite"


class Settings(BaseSettings):
    # Secrets — required, loaded from .env; SecretStr prevents accidental logging
    google_api_key: SecretStr
    qdrant_url: str
    qdrant_api_key: SecretStr
    jina_api_key: SecretStr
    jina_rerank_url: str = "https://api.jina.ai/v1/rerank"

    langsmith_tracing: str = "false"
    langsmith_endpoint: str = "https://api.smith.langchain.com"
    langsmith_api_key: SecretStr
    langsmith_project: str

    # Cross-cutting — used in tracing tags, evaluation snapshots, experiment metadata
    pipeline_version: str = "v1-baseline"

    # App config groups — have defaults, overridable via env (e.g. DB__COLLECTION_NAME=foo)
    data: DataConfig = DataConfig()
    db: DBConfig = DBConfig()
    ingestion: IngestionConfig = IngestionConfig()
    generation: GenerationConfig = GenerationConfig()
    api: APIConfig = APIConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    retrieval: RetrievalConfig = RetrievalConfig()

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
