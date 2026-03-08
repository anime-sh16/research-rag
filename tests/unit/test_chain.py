from unittest.mock import MagicMock, patch

import pytest

from src.generation.chain import SYSTEM_INSTRUCTION, RAGChain


@pytest.fixture
def mock_gemini_response() -> MagicMock:
    response = MagicMock()
    response.text = "This is a generated answer."
    return response


@pytest.fixture
def chain(mock_gemini_response: MagicMock):
    """RAGChain with the Gemini client fully patched out."""
    with patch("src.generation.chain.genai.Client") as mock_client_cls:
        mock_client = mock_client_cls.return_value
        mock_client.models.generate_content.return_value = mock_gemini_response
        instance = RAGChain()
        # Attach mock so individual tests can inspect calls
        instance._mock_client = mock_client
        yield instance


@pytest.fixture
def sample_chunks() -> list[dict]:
    return [
        {
            "title": "Attention Is All You Need",
            "text": "The transformer architecture uses self-attention.",
        },
        {
            "title": "BERT: Pre-training of Deep Bidirectional Transformers",
            "text": "BERT is pre-trained using masked language modelling.",
        },
    ]


class TestFormatContext:
    def test_numbers_chunks_sequentially(
        self, chain: RAGChain, sample_chunks: list[dict]
    ) -> None:
        result = chain._format_context(sample_chunks)
        assert result.startswith("[1]")
        assert "[2]" in result

    def test_includes_title_and_text(
        self, chain: RAGChain, sample_chunks: list[dict]
    ) -> None:
        result = chain._format_context(sample_chunks)
        assert "Attention Is All You Need" in result
        assert "self-attention" in result

    def test_chunks_separated_by_double_newline(
        self, chain: RAGChain, sample_chunks: list[dict]
    ) -> None:
        result = chain._format_context(sample_chunks)
        assert "\n\n" in result

    def test_missing_title_falls_back_to_unknown(self, chain: RAGChain) -> None:
        chunks = [{"text": "Some text with no title."}]
        result = chain._format_context(chunks)
        assert "Unknown Paper" in result

    def test_missing_text_produces_empty_text(self, chain: RAGChain) -> None:
        chunks = [{"title": "Paper Without Text"}]
        result = chain._format_context(chunks)
        assert "Paper Without Text" in result

    def test_empty_chunks_returns_empty_string(self, chain: RAGChain) -> None:
        assert chain._format_context([]) == ""

    def test_single_chunk_has_no_separator(self, chain: RAGChain) -> None:
        chunks = [{"title": "Solo", "text": "Only chunk."}]
        result = chain._format_context(chunks)
        assert result.count("\n\n") == 0


class TestGenerate:
    def test_returns_model_text(
        self, chain: RAGChain, sample_chunks: list[dict]
    ) -> None:
        result = chain.generate("What is attention?", sample_chunks)
        assert result == "This is a generated answer."

    def test_calls_generate_content_once(
        self, chain: RAGChain, sample_chunks: list[dict]
    ) -> None:
        chain.generate("What is attention?", sample_chunks)
        chain._mock_client.models.generate_content.assert_called_once()

    def test_prompt_contains_question(
        self, chain: RAGChain, sample_chunks: list[dict]
    ) -> None:
        query = "What is attention?"
        chain.generate(query, sample_chunks)
        call_kwargs = chain._mock_client.models.generate_content.call_args
        prompt = call_kwargs.kwargs.get("contents") or call_kwargs.args[1]
        assert query in prompt

    def test_prompt_contains_context(
        self, chain: RAGChain, sample_chunks: list[dict]
    ) -> None:
        chain.generate("What is BERT?", sample_chunks)
        call_kwargs = chain._mock_client.models.generate_content.call_args
        prompt = call_kwargs.kwargs.get("contents") or call_kwargs.args[1]
        assert "self-attention" in prompt
        assert "masked language modelling" in prompt

    def test_system_instruction_is_set(
        self, chain: RAGChain, sample_chunks: list[dict]
    ) -> None:
        chain.generate("Any question?", sample_chunks)
        call_kwargs = chain._mock_client.models.generate_content.call_args
        config = call_kwargs.kwargs.get("config") or call_kwargs.args[2]
        assert config.system_instruction == SYSTEM_INSTRUCTION

    def test_temperature_is_low(
        self, chain: RAGChain, sample_chunks: list[dict]
    ) -> None:
        chain.generate("Any question?", sample_chunks)
        call_kwargs = chain._mock_client.models.generate_content.call_args
        config = call_kwargs.kwargs.get("config") or call_kwargs.args[2]
        assert config.temperature == pytest.approx(0.1)

    def test_uses_correct_model(
        self, chain: RAGChain, sample_chunks: list[dict]
    ) -> None:
        chain.generate("Any question?", sample_chunks)
        call_kwargs = chain._mock_client.models.generate_content.call_args
        model = call_kwargs.kwargs.get("model") or call_kwargs.args[0]
        assert model == "gemini-2.5-flash-lite"

    def test_generate_with_empty_chunks(self, chain: RAGChain) -> None:
        """Should not raise — empty context is valid, model decides what to do."""
        result = chain.generate("What is LoRA?", [])
        assert isinstance(result, str)

    def test_custom_model_is_used(self, mock_gemini_response: MagicMock) -> None:
        with patch("src.generation.chain.genai.Client") as mock_client_cls:
            mock_client = mock_client_cls.return_value
            mock_client.models.generate_content.return_value = mock_gemini_response
            custom_chain = RAGChain(model="gemini-2.5-pro")
            custom_chain.generate("test", [])
            call_kwargs = mock_client.models.generate_content.call_args
            model = call_kwargs.kwargs.get("model") or call_kwargs.args[0]
            assert model == "gemini-2.5-pro"
