"""Version resolution: PIPELINE_VERSION env → repo VERSION file → "dev"."""

from pathlib import Path

from src.config.config import Settings, _read_version_file

VERSION_FILE = Path(__file__).resolve().parents[2] / "VERSION"


def test_version_file_exists_and_is_nonempty() -> None:
    assert VERSION_FILE.is_file()
    assert VERSION_FILE.read_text(encoding="utf-8").strip() != ""


def test_read_version_file_returns_stripped_contents() -> None:
    expected = VERSION_FILE.read_text(encoding="utf-8").strip()
    assert _read_version_file() == expected


def test_explicit_pipeline_version_wins() -> None:
    settings = Settings(pipeline_version="v9-custom-experiment")
    assert settings.pipeline_version == "v9-custom-experiment"


def test_blank_pipeline_version_falls_back_to_version_file() -> None:
    settings = Settings(pipeline_version="")
    assert settings.pipeline_version == _read_version_file()


def test_git_sha_defaults_to_unknown_when_unset() -> None:
    # GIT_SHA is not set in the test environment (only build-stamped images set it).
    settings = Settings()
    assert settings.git_sha == "unknown"
