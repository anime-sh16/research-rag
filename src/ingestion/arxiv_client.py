import logging
from datetime import datetime

import arxiv
from pydantic import BaseModel

from src.config.config import settings

logger = logging.getLogger(__name__)
SORT_BY = settings.ingestion.fetch_sort_by


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


class ArxivClient:
    def __init__(self):
        self.client = arxiv.Client()

    def get_arxiv_results(
        self,
        query: str,
        max_results: int = 10,
        sort_by: arxiv.SortCriterion = SORT_BY,
    ) -> list[ArxivResult]:
        logger.info("Fetching up to %d papers for query: '%s'", max_results, query)
        search = arxiv.Search(query=query, max_results=max_results, sort_by=sort_by)
        results = [
            self._parse_arxiv_result(r, query) for r in self.client.results(search)
        ]
        logger.info("Fetched %d papers.", len(results))
        return results

    def _parse_arxiv_result(
        self, arxiv_result: arxiv.Result, topic: str
    ) -> ArxivResult:
        entry_id = arxiv_result.entry_id.split("/")[-1].split("v")[0]
        authors = (
            [author.name for author in arxiv_result.authors]
            if arxiv_result.authors
            else []
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
        )
