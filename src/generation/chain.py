import logging

from google import genai
from google.genai import types

from src.config.config import settings

logger = logging.getLogger(__name__)

SYSTEM_INSTRUCTION = "You are a research assistant. Answer questions based only on the provided context from ArXiv papers. If the context does not contain enough information, say 'I don't have enough information to answer this.'"

PROMPT_TEMPLATE = """Context:
{context}

Question: {question}"""


class RAGChain:
    def __init__(self, model: str = settings.generation.model):
        self.client = genai.Client(api_key=settings.google_api_key.get_secret_value())
        self.model = model

    def _format_context(self, chunks: list[dict]) -> str:
        parts = []
        for i, chunk in enumerate(chunks, start=1):
            title = chunk.get("title", "Unknown Paper")
            text = chunk.get("text", "")
            parts.append(f"[{i}] {title}\n{text}")
        return "\n\n".join(parts)

    def generate(self, query: str, chunks: list[dict]) -> str:
        logger.info(
            "Generating answer for query: '%s' using %d chunks.", query, len(chunks)
        )
        context = self._format_context(chunks)
        prompt = PROMPT_TEMPLATE.format(context=context, question=query)

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=settings.generation.temperature,
            ),
        )

        logger.debug("Generated answer: '%s'", response.text[:100])
        return response.text
