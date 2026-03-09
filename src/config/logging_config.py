"""Project-wide file logging setup.

- Ingestion runs  →  logs/ingestion/{run_id}/ingestion.log  (plain FileHandler)
- API server      →  logs/api/api.log  (daily-rotating, 30-day retention)

Both handlers are additive on top of whatever the caller already configured
(e.g. basicConfig stdout for the CLI, uvicorn's own handler for the API).
"""

import logging
import os
from logging.handlers import TimedRotatingFileHandler

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATE_FORMAT = "%H:%M:%S"


def setup_ingestion_logging(run_id: str) -> logging.FileHandler:
    """Attach a per-run FileHandler to the root logger.

    Returns the handler so callers can remove it when the run finishes.
    Log file: logs/ingestion/{run_id}/ingestion.log
    """
    log_dir = os.path.join("logs", "ingestion", run_id)
    os.makedirs(log_dir, exist_ok=True)
    handler = logging.FileHandler(
        os.path.join(log_dir, "ingestion.log"), encoding="utf-8"
    )
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    logging.getLogger().addHandler(handler)
    return handler


def setup_api_logging() -> logging.Handler:
    """Attach a daily-rotating FileHandler to the root logger.

    Rotates at midnight, retains 30 days of logs.
    Log file: logs/api/api.log
    """
    log_dir = os.path.join("logs", "api")
    os.makedirs(log_dir, exist_ok=True)
    handler = TimedRotatingFileHandler(
        os.path.join(log_dir, "api.log"),
        when="midnight",
        backupCount=30,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    logging.getLogger().addHandler(handler)
    return handler
