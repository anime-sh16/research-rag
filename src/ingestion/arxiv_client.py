import logging
import os
import re
from datetime import datetime

import arxiv
import fitz
import requests
from pydantic import BaseModel

from src.config.config import settings

logger = logging.getLogger(__name__)
SORT_BY = settings.ingestion.fetch_sort_by

_LIGATURES = {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl"}


class ArxivResult(BaseModel):
    entry_id: str
    title: str
    topic: str | None
    published: datetime
    summary: str
    authors: list[str] | None
    comment: str | None
    primary_category: str
    categories: list[str] | None
    pdf_url: str | None
    full_text: str | None


class ArxivClient:
    def __init__(self):
        self.client = arxiv.Client(
            page_size=settings.ingestion.fetch_per_topic,
            delay_seconds=5,
            num_retries=3,
        )

    def get_arxiv_results(
        self,
        query: str,
        max_results: int = 10,
        sort_by: arxiv.SortCriterion = SORT_BY,
        download_pdf: bool = True,
    ) -> list[ArxivResult]:
        logger.info("Fetching up to %d papers for query: '%s'", max_results, query)
        search = arxiv.Search(query=query, max_results=max_results, sort_by=sort_by)
        results = []
        for r in self.client.results(search):
            try:
                results.append(
                    self._parse_arxiv_result(r, query, download_pdf=download_pdf)
                )
            except Exception:
                logger.exception("Skipping paper %s due to parse error.", r.entry_id)

        logger.info("Fetched %d papers.", len(results))
        return results

    def populate_full_text(self, result: ArxivResult) -> None:
        """Download and extract PDF text for a result that was fetched metadata-only."""
        if result.full_text is not None:
            return
        if not result.pdf_url or not result.topic:
            logger.warning(
                "Cannot populate full_text for %s: missing pdf_url or topic.",
                result.entry_id,
            )
            return
        result.full_text = self._extract_pdf_text(
            result.pdf_url, result.entry_id, result.topic
        )

    def _parse_arxiv_result(
        self, arxiv_result: arxiv.Result, topic: str, download_pdf: bool = True
    ) -> ArxivResult:
        entry_id = arxiv_result.entry_id.split("/")[-1].split("v")[0]
        authors = (
            [author.name for author in arxiv_result.authors]
            if arxiv_result.authors
            else []
        )
        pdf_url = getattr(arxiv_result, "pdf_url", None)
        full_text = (
            self._extract_pdf_text(pdf_url, entry_id, topic)
            if download_pdf and pdf_url
            else None
        )

        return ArxivResult(
            entry_id=entry_id,
            title=arxiv_result.title,
            topic=topic,
            published=arxiv_result.published,
            summary=arxiv_result.summary,
            authors=authors,
            comment=arxiv_result.comment or None,
            primary_category=arxiv_result.primary_category,
            categories=arxiv_result.categories,
            pdf_url=pdf_url,
            full_text=full_text,
        )

    def _download_pdf_locally(
        self, pdf_url: str, entry_id: str, output_dir: str
    ) -> str:
        # Check if already exists
        file_path = os.path.join(output_dir, f"{entry_id}.pdf")
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            return file_path

        os.makedirs(output_dir, exist_ok=True)

        logger.info("Downloading PDF from %s to %s", pdf_url, file_path)
        response = requests.get(pdf_url, stream=True, timeout=60)
        response.raise_for_status()
        with open(file_path, "wb") as f:
            for chunk in response.iter_content(1024 * 8):
                if chunk:
                    f.write(chunk)

        return file_path

    @staticmethod
    def extract_text_from_pdf(pdf_path: str | os.PathLike) -> str | None:
        """Extract and clean text from a local PDF file."""
        try:
            doc = fitz.open(pdf_path)
            text_parts: list[str] = []
            for page in doc:
                text_parts.append(page.get_text())

            full_text = "\n".join(text_parts)

            # fix ligature artifacts (ﬁ → fi, etc.)
            for lig, rep in _LIGATURES.items():
                full_text = full_text.replace(lig, rep)

            # rejoin words hyphenated across line breaks: "pro-\nposed" → "proposed"
            full_text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", full_text)

            # drop everything from References section onwards — noise for retrieval
            full_text = re.sub(
                r"\n(References|REFERENCES|Bibliography)\s*\n.*$",
                "",
                full_text,
                flags=re.DOTALL,
            )

            # collapse multiple spaces/tabs to one (two-column layout artifact)
            full_text = re.sub(r"[ \t]+", " ", full_text)

            # collapse 3+ newlines to 2
            return re.sub(r"(\n\s*){3,}", "\n\n", full_text)
        except (fitz.FileDataError, OSError, ValueError) as exc:
            logger.exception(
                "Failed to extract text from PDF %s: %s",
                pdf_path,
                exc,
            )
            return None

    def _extract_pdf_text(self, pdf_url: str, entry_id: str, topic: str) -> str | None:
        try:
            out_dir = str(settings.data.pdf_dir / topic.strip())
            pdf_path = self._download_pdf_locally(pdf_url, entry_id, out_dir)
            return self.extract_text_from_pdf(pdf_path)
        except (requests.RequestException, OSError) as exc:
            logger.exception(
                "Failed to download/extract PDF for %s: %s",
                entry_id,
                exc,
            )
            return None
