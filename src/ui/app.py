import os

import streamlit as st

from src.ui.api_client import APIClient

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
REPO_URL = os.environ.get("REPO_URL", "https://github.com/")


@st.cache_resource
def get_client() -> APIClient:
    return APIClient(base_url=API_BASE_URL)


def render_sources(sources: list[dict]) -> None:
    if not sources:
        st.info("No sources returned.")
        return
    for i, source in enumerate(sources, start=1):
        with st.expander(f"[{i}] {source['title']} — score {source['score']:.3f}"):
            authors = source.get("authors") or []
            st.write(f"**Authors:** {', '.join(authors) if authors else 'Unknown'}")
            st.write(f"**ArXiv ID:** `{source['paper_id']}`")
            st.write(f"**Chunk index:** {source['chunk_index']}")


def render_sidebar(client: APIClient) -> None:
    st.sidebar.title("research-rag")
    try:
        health = client.health()
        st.sidebar.success(
            f"API healthy — pipeline `{health.get('pipeline_version', 'unknown')}`"
        )
        st.sidebar.caption(f"commit `{health.get('git_sha', 'unknown')}`")
    except Exception as e:
        st.sidebar.error(f"API unreachable: {e}")
    st.sidebar.markdown(f"[View source on GitHub]({REPO_URL})")


def main() -> None:
    st.set_page_config(page_title="research-rag", layout="wide")
    client = get_client()
    render_sidebar(client)

    st.title("Ask a question about ArXiv ML papers")
    question = st.text_area(
        "Question", height=100, placeholder="What does LoRA optimize?"
    )
    submit = st.button("Ask", type="primary", disabled=not question.strip())

    if submit:
        with st.spinner("Retrieving and generating..."):
            try:
                result = client.query(question.strip())
            except Exception as e:
                st.error(f"Query failed: {e}")
                return
        st.subheader("Answer")
        st.markdown(result.answer)
        st.subheader("Sources")
        render_sources(result.sources)


if __name__ == "__main__":
    main()
