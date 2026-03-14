import os

# Provide dummy values so Settings() can be instantiated during unit tests.
# All external API calls are mocked in individual tests; these values never reach real services.
os.environ.setdefault("GOOGLE_API_KEY", "dummy-for-tests")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("QDRANT_API_KEY", "dummy-for-tests")
os.environ.setdefault("JINA_API_KEY", "dummy-for-tests")
os.environ.setdefault("LANGSMITH_API_KEY", "dummy-for-tests")
os.environ.setdefault("LANGSMITH_PROJECT", "test-project")

# Disable LangSmith tracing during tests to avoid polluting production traces
os.environ["LANGSMITH_TRACING"] = "false"
