import arxiv
from google import genai
from langsmith import wrappers
from qdrant_client import QdrantClient

from src.config.config import settings


def test_gemini(client):
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents="Explain quantum computing in one line.",
    )
    print("=====Gemini Response:=====")
    print(response.text)
    print("=============")


def test_langsmith():
    pass


def test_arxiv():
    client = arxiv.Client()
    search = arxiv.Search(
        query="quantum", max_results=10, sort_by=arxiv.SortCriterion.SubmittedDate
    )

    print("=====Arxiv Results:=====")
    for r in client.results(search):
        print(r.title)

    print("=============")


def test_qdrant():
    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    collections = client.get_collections()
    print("=====Qdrant Collections:=====")
    print(collections)
    print("=============")


def main():
    # genai.Client() reads GOOGLE_API_KEY / GEMINI_API_KEY from the environment
    gemini_client = genai.Client(api_key=settings.google_api_key)

    # Wrap teh Gemini client to enable LangSmith tracing
    client = wrappers.wrap_gemini(
        gemini_client,
        tracing_extra={
            "tags": ["gemini", "python"],
            "metadata": {
                "integration": "google-genai",
            },
        },
    )

    try:
        test_gemini(client)
    except Exception as e:
        print(f"Error testing Gemini: {e}")

    try:
        test_arxiv()
    except Exception as e:
        print(f"Error testing Arxiv: {e}")

    try:
        test_qdrant()
    except Exception as e:
        print(f"Error testing Qdrant: {e}")


if __name__ == "__main__":
    main()
