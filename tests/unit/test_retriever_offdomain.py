from unittest.mock import patch

import pytest

from src.retrieval.retriever import OffDomainQuery, Retriever


@pytest.fixture
def retriever():
    with (
        patch("src.retrieval.retriever.AsyncQdrantClient"),
        patch("src.retrieval.retriever.wrappers.wrap_gemini"),
        patch("src.retrieval.retriever.Client"),
    ):
        yield Retriever(top_k=5)


async def test_retrieve_raises_offdomain_when_not_research_query(retriever):
    with patch.object(
        retriever,
        "_extract_subquery",
        return_value={"is_research_query": False, "subquery": []},
    ):
        with pytest.raises(OffDomainQuery):
            await retriever.retrieve("what's the weather in Tokyo?")


async def test_retrieve_proceeds_when_is_research_query_true(retriever):
    with (
        patch.object(
            retriever,
            "_extract_subquery",
            return_value={
                "is_research_query": True,
                "subquery": [{"query": "attention", "expansion_terms": []}],
            },
        ),
        patch.object(retriever, "_search_subquery", return_value=[]),
        patch.object(retriever, "_rerank", return_value=[]),
    ):
        result = await retriever.retrieve("what is attention?")
        assert result == []


async def test_retrieve_fails_open_when_field_missing(retriever):
    # Extraction fallback (no is_research_query key) must NOT be treated off-domain.
    with (
        patch.object(
            retriever,
            "_extract_subquery",
            return_value={"subquery": [{"query": "x", "expansion_terms": []}]},
        ),
        patch.object(retriever, "_search_subquery", return_value=[]),
        patch.object(retriever, "_rerank", return_value=[]),
    ):
        result = await retriever.retrieve("x")
        assert result == []
