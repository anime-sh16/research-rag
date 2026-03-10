import os

# Disable LangSmith tracing during tests to avoid polluting production traces
os.environ["LANGSMITH_TRACING"] = "false"
