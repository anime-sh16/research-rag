from unittest.mock import MagicMock, patch

import pytest

from src.config.config import settings
from src.generation.chain import DEFAULT_PROMPT_VERSION, PROMPT_VARIANTS, RAGChain


@pytest.fixture
def mock_gemini_response() -> MagicMock:
    response = MagicMock()
    response.text = "This is a generated answer."
    return response


@pytest.fixture
def chain(mock_gemini_response: MagicMock):
    """RAGChain with the Gemini client and LangSmith wrapper fully patched out."""
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_gemini_response
    with (
        patch("src.generation.chain.genai.Client", return_value=mock_client),
        patch("src.generation.chain.wrappers.wrap_gemini", return_value=mock_client),
        patch("src.generation.chain.get_current_run_tree", return_value=None),
    ):
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


class TestPromptVariants:
    """PROMPT_VARIANTS is the single source of truth for prompt config."""

    def test_v1_and_v2_variants_exist(self) -> None:
        assert "v1" in PROMPT_VARIANTS
        assert "v2" in PROMPT_VARIANTS

    def test_each_variant_has_system_and_template_keys(self) -> None:
        for version, variant in PROMPT_VARIANTS.items():
            assert "system" in variant, f"variant '{version}' missing 'system'"
            assert "template" in variant, f"variant '{version}' missing 'template'"

    def test_default_prompt_version_is_v2(self) -> None:
        assert DEFAULT_PROMPT_VERSION == "v2"

    def test_invalid_prompt_version_raises_on_init(self, mock_gemini_response) -> None:
        mock_client = MagicMock()
        with (
            patch("src.generation.chain.genai.Client", return_value=mock_client),
            patch(
                "src.generation.chain.wrappers.wrap_gemini", return_value=mock_client
            ),
        ):
            with pytest.raises(ValueError, match="Unknown prompt_version"):
                RAGChain(prompt_version="nonexistent")

    def test_v2_system_instruction_contains_citation_rules(self) -> None:
        """v2 prompt must include citation format — this is a key behavioral requirement."""
        system = PROMPT_VARIANTS["v2"]["system"]
        assert "[1]" in system or "cite" in system.lower()

    def test_v1_and_v2_system_instructions_are_different(self) -> None:
        assert PROMPT_VARIANTS["v1"]["system"] != PROMPT_VARIANTS["v2"]["system"]


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

    def test_system_instruction_matches_default_prompt_variant(
        self, chain: RAGChain, sample_chunks: list[dict]
    ) -> None:
        """System instruction sent to the model must match the configured prompt variant."""
        chain.generate("Any question?", sample_chunks)
        call_kwargs = chain._mock_client.models.generate_content.call_args
        config = call_kwargs.kwargs.get("config") or call_kwargs.args[2]
        assert (
            config.system_instruction == PROMPT_VARIANTS[chain.prompt_version]["system"]
        )

    def test_prompt_version_override_uses_v1_system_instruction(
        self, chain: RAGChain, sample_chunks: list[dict]
    ) -> None:
        """Per-call prompt_version override must change which system instruction is sent."""
        chain.generate("Any question?", sample_chunks, prompt_version="v1")
        call_kwargs = chain._mock_client.models.generate_content.call_args
        config = call_kwargs.kwargs.get("config") or call_kwargs.args[2]
        assert config.system_instruction == PROMPT_VARIANTS["v1"]["system"]

    def test_generate_raises_on_invalid_prompt_version_override(
        self, chain: RAGChain, sample_chunks: list[dict]
    ) -> None:
        with pytest.raises(ValueError, match="Unknown prompt_version"):
            chain.generate("query", sample_chunks, prompt_version="bad_version")

    def test_thinking_config_is_passed(
        self, chain: RAGChain, sample_chunks: list[dict]
    ) -> None:
        """ThinkingConfig must be forwarded to the model — it was added in v2."""
        chain.generate("Any question?", sample_chunks)
        call_kwargs = chain._mock_client.models.generate_content.call_args
        config = call_kwargs.kwargs.get("config") or call_kwargs.args[2]
        assert config.thinking_config is not None

    def test_temperature_matches_config_setting(
        self, chain: RAGChain, sample_chunks: list[dict]
    ) -> None:
        chain.generate("Any question?", sample_chunks)
        call_kwargs = chain._mock_client.models.generate_content.call_args
        config = call_kwargs.kwargs.get("config") or call_kwargs.args[2]
        assert config.temperature == pytest.approx(settings.generation.temperature)

    def test_uses_correct_model(
        self, chain: RAGChain, sample_chunks: list[dict]
    ) -> None:
        chain.generate("Any question?", sample_chunks)
        call_kwargs = chain._mock_client.models.generate_content.call_args
        model = call_kwargs.kwargs.get("model") or call_kwargs.args[0]
        assert model == settings.generation.model

    def test_generate_with_empty_chunks(self, chain: RAGChain) -> None:
        """Should not raise — empty context is valid, model decides what to do."""
        result = chain.generate("What is LoRA?", [])
        assert isinstance(result, str)

    def test_custom_model_is_used(self, mock_gemini_response: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_gemini_response
        with (
            patch("src.generation.chain.genai.Client", return_value=mock_client),
            patch(
                "src.generation.chain.wrappers.wrap_gemini", return_value=mock_client
            ),
            patch("src.generation.chain.get_current_run_tree", return_value=None),
        ):
            custom_chain = RAGChain(model="gemini-2.5-pro")
            custom_chain.generate("test", [])
            call_kwargs = mock_client.models.generate_content.call_args
            model = call_kwargs.kwargs.get("model") or call_kwargs.args[0]
            assert model == "gemini-2.5-pro"
