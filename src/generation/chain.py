import logging

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from langsmith import traceable, wrappers
from langsmith.run_helpers import get_current_run_tree
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from src.config.config import settings

logger = logging.getLogger(__name__)


def _is_rate_limit_error(exc: BaseException) -> bool:
    return isinstance(exc, genai_errors.ClientError) and (
        getattr(exc, "status_code", None) == 429 or "429" in str(exc)
    )


PROMPT_VERSION = "v1-baseline"

SYSTEM_INSTRUCTION = "You are a research assistant. Answer questions based only on the provided context from ArXiv papers. If the context does not contain enough information, say 'I don't have enough information to answer this.'"

PROMPT_TEMPLATE = """Context:
{context}

Question: {question}"""


class RAGChain:
    def __init__(self, model: str = settings.generation.model):
        self.client = wrappers.wrap_gemini(
            genai.Client(api_key=settings.google_api_key.get_secret_value()),
            tracing_extra={
                "tags": ["gemini", "python"],
                "metadata": {
                    "integration": "google-genai",
                },
            },
        )
        self.model = model

    def _format_context(self, chunks: list[dict]) -> str:
        parts = []
        for i, chunk in enumerate(chunks, start=1):
            title = chunk.get("title", "Unknown Paper")
            text = chunk.get("text", "")
            parts.append(f"[{i}] {title}\n{text}")
        return "\n\n".join(parts)

    @traceable(run_type="llm", name="generation/gemini")
    @retry(
        retry=retry_if_exception(_is_rate_limit_error),
        wait=wait_exponential(multiplier=60, min=60, max=480),
        stop=stop_after_attempt(5),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def generate(self, query: str, chunks: list[dict]) -> str:
        logger.info(
            "Generating answer for query: '%s' using %d chunks.", query, len(chunks)
        )
        context = self._format_context(chunks)
        prompt = PROMPT_TEMPLATE.format(context=context, question=query)

        # Log prompt details to LangSmith
        run = get_current_run_tree()
        if run:
            run.add_metadata(
                {
                    "prompt_version": PROMPT_VERSION,
                    "full_prompt_sent": prompt,
                    "system_instruction": SYSTEM_INSTRUCTION,
                    "model": self.model,
                    "context_chunks_used": [c.get("paper_id") for c in chunks],
                }
            )
            run.add_tags([f"prompt_version:{PROMPT_VERSION}"])

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=settings.generation.temperature,
            ),
        )

        if run and hasattr(response, "usage_metadata") and response.usage_metadata:
            run.add_metadata(
                {
                    "input_tokens": response.usage_metadata.prompt_token_count,
                    "output_tokens": response.usage_metadata.candidates_token_count,
                }
            )

        logger.debug("Generated answer: '%s'", response.text[:100])
        return response.text
