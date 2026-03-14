import logging

import httpx
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


def _is_service_unavailable_error(exc: BaseException) -> bool:
    return isinstance(exc, genai_errors.ServerError) and (
        getattr(exc, "status_code", None) == 503 or "503" in str(exc)
    )


def _is_timeout_error(exc: BaseException) -> bool:
    if isinstance(exc, (TimeoutError, httpx.TimeoutException)):
        return True
    if isinstance(exc, genai_errors.ServerError) and (
        getattr(exc, "status_code", None) == 504
        or "504" in str(exc)
        or "DEADLINE_EXCEEDED" in str(exc)
    ):
        return True
    return False


DEFAULT_PROMPT_VERSION = "v2"

PROMPT_VARIANTS: dict[str, dict[str, str]] = {
    "v1": {
        "system": (
            "You are a research assistant. Answer questions based only on the provided context"
            " from ArXiv papers. If the context does not contain enough information, say"
            " 'I don't have enough information to answer this.'"
        ),
        "template": "<CONTEXT>\n{context}\n</CONTEXT>\nQuestion: {question}",
    },
    "v2": {
        "system": (
            "You are an expert ML research assistant specializing in ArXiv papers.\n\n"
            "Your job is to answer the user's question directly and precisely using only"
            " the numbered sources provided in <CONTEXT>.\n\n"
            "Rules:\n"
            "- Answer directly. Never say 'based on the context', 'according to the context',"
            " or similar phrases. Just state the facts.\n"
            "- Cite sources inline using [1], [2], etc. wherever a claim comes from a specific source.\n"
            "- If multiple sources support a claim, cite all of them. For example: [1][3].\n"
            "- Answer every part of the question you can. If only partial information is available,"
            " answer those parts fully and explicitly state what is missing for the rest.\n"
            "- If the context contains no relevant information at all, say exactly:"
            " 'I don't have enough information to answer this.' Do not guess or extrapolate."
            "Output:\n"
            "A choesive response to the user's query based on the context provided."
        ),
        "template": ("<CONTEXT>\n{context}\n</CONTEXT>\n\nQuestion: {question}"),
    },
}


class RAGChain:
    def __init__(
        self,
        model: str = settings.generation.model,
        prompt_version: str = DEFAULT_PROMPT_VERSION,
    ):
        if prompt_version not in PROMPT_VARIANTS:
            raise ValueError(
                f"Unknown prompt_version '{prompt_version}'. "
                f"Valid options: {list(PROMPT_VARIANTS)}"
            )
        self.prompt_version = prompt_version
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

    def _resolve_prompt(self, prompt_version: str | None) -> tuple[str, str, str]:
        """Return (system_instruction, template) for the given version."""
        version = prompt_version or self.prompt_version
        if version not in PROMPT_VARIANTS:
            raise ValueError(
                f"Unknown prompt_version '{version}'. "
                f"Valid options: {list(PROMPT_VARIANTS)}"
            )
        variant = PROMPT_VARIANTS[version]
        return variant["system"], variant["template"], version

    def _format_context(self, chunks: list[dict]) -> str:
        parts = []
        for i, chunk in enumerate(chunks, start=1):
            title = chunk.get("title", "Unknown Paper")
            text = chunk.get("text", "")
            parts.append(f"[{i}] {title}\n{text}")
        return "\n\n".join(parts)

    @traceable(run_type="llm", name="generation/gemini")
    @retry(
        retry=retry_if_exception(_is_service_unavailable_error),
        wait=wait_exponential(multiplier=180, min=180, max=900, exp_base=1),
        stop=stop_after_attempt(5),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    @retry(
        retry=retry_if_exception(_is_rate_limit_error),
        wait=wait_exponential(multiplier=60, min=60, max=480),
        stop=stop_after_attempt(5),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    @retry(
        retry=retry_if_exception(_is_timeout_error),
        wait=wait_exponential(multiplier=5, min=5, max=60),
        stop=stop_after_attempt(3),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def generate(
        self, query: str, chunks: list[dict], prompt_version: str | None = None
    ) -> str:
        system_instruction, template, resolved_version = self._resolve_prompt(
            prompt_version
        )
        logger.info(
            "Generating answer for query: '%s' using %d chunks (prompt=%s).",
            query,
            len(chunks),
            resolved_version,
        )
        context = self._format_context(chunks)
        prompt = template.format(context=context, question=query)

        # Log prompt details to LangSmith
        run = get_current_run_tree()
        if run:
            run.add_metadata(
                {
                    "prompt_version": resolved_version,
                    "full_prompt_sent": prompt,
                    "system_instruction": system_instruction,
                    "model": self.model,
                    "context_chunks_used": [c.get("paper_id") for c in chunks],
                }
            )
            run.add_tags([f"prompt_version:{resolved_version}"])

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=settings.generation.temperature,
                thinking_config=types.ThinkingConfig(
                    thinking_level="low",
                ),
                http_options=types.HttpOptions(timeout=60000),
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
