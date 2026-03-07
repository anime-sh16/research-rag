from datetime import datetime

import arxiv
from pydantic import BaseModel


class ArxivResult(BaseModel):
    entry_id: str
    title: str
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
        self, query, max_results=10, sort_by=arxiv.SortCriterion.SubmittedDate
    ):
        """
        Get a list of ArxivResult objects for a given query.
        Args:
            query (str): The query to search for.
            max_results (int, optional): The maximum number of results to return. Defaults to 10.
            sort_by (arxiv.SortCriterion, optional): The criterion to sort the results by. Defaults to arxiv.SortCriterion.SubmittedDate.

        Returns:
            list: A list of ArxivResult objects.
        """
        search = arxiv.Search(query=query, max_results=max_results, sort_by=sort_by)
        results = []

        for r in self.client.results(search):
            r_parsed = self._parse_arxiv_result(r)
            results.append(r_parsed)

        return results

    def _parse_arxiv_result(self, arxiv_result):

        entry_id = arxiv_result.entry_id
        entry_id = entry_id.split("/")[-1].split("v")[0]

        if arxiv_result.authors:
            authors = [author.name for author in arxiv_result.authors]
        else:
            authors = []

        return ArxivResult(
            entry_id=entry_id,
            title=arxiv_result.title,
            published=arxiv_result.published,
            summary=arxiv_result.summary,
            authors=authors,
            comment=arxiv_result.comment,
            primary_category=arxiv_result.primary_category,
            categories=arxiv_result.categories,
        )

    def print_results(self, results: list[ArxivResult]):
        for result in results:
            print(result)
