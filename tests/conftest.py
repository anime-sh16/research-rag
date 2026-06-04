import os

import pytest

# Provide dummy values so Settings() can be instantiated during unit tests.
# All external API calls are mocked in individual tests; these values never reach real services.
os.environ.setdefault("GOOGLE_API_KEY", "dummy-for-tests")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("QDRANT_API_KEY", "dummy-for-tests")
os.environ.setdefault("JINA_API_KEY", "dummy-for-tests")
os.environ.setdefault("LANGSMITH_API_KEY", "dummy-for-tests")
os.environ.setdefault("LANGSMITH_PROJECT", "test-project")
os.environ.setdefault("PIPELINE_VERSION", "test")

# Disable LangSmith tracing during tests to avoid polluting production traces
os.environ["LANGSMITH_TRACING"] = "false"


@pytest.fixture(autouse=True)
def _disable_rate_limiter():
    """With TestClient every request shares one client IP, so an enabled per-IP
    limiter would trip across unrelated tests. Disable by default; the dedicated
    rate-limit test re-enables it explicitly."""
    from src.api.main import limiter

    limiter.enabled = False
    yield
