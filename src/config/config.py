import os

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    google_api_key: str
    qdrant_url: str
    qdrant_api_key: str
    cohere_api_key: str
    langsmith_tracing: str
    langsmith_endpoint: str
    langsmith_api_key: str
    langsmith_project: str

    model_config = {"env_file": ".env"}


settings = Settings()

os.environ["LANGSMITH_TRACING"] = settings.langsmith_tracing
os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint
os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
