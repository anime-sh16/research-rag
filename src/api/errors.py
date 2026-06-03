"""Structured, leak-free error responses for the public API.

Maps upstream/internal exceptions to client-safe HTTP statuses and a structured
body. Full exception detail is logged server-side by the caller — never returned
to the client.
"""

import httpx
from google.genai import errors as genai_errors
from pydantic import BaseModel

OFF_DOMAIN_MESSAGE = (
    "I can only answer questions about the ML research papers in my knowledge "
    "base. Try asking about a method, model, or finding from a paper."
)


class ErrorResponse(BaseModel):
    error: str  # stable category code, e.g. "service_unavailable"
    detail: str  # generic, client-safe message (never the raw exception text)


_RETRYABLE_STATUS = {429, 503, 504}
_RETRYABLE_MARKERS = ("429", "503", "504", "DEADLINE_EXCEEDED")


def _is_retryable_upstream(exc: BaseException) -> bool:
    """True for transient upstream failures (rate limit / unavailable / timeout)."""
    if isinstance(exc, (TimeoutError, httpx.TimeoutException)):
        return True
    if isinstance(exc, genai_errors.APIError):
        if getattr(exc, "status_code", None) in _RETRYABLE_STATUS:
            return True
        if any(marker in str(exc) for marker in _RETRYABLE_MARKERS):
            return True
    return False


def map_exception(exc: BaseException) -> tuple[int, ErrorResponse]:
    """Map an exception to (http_status, client-safe ErrorResponse)."""
    if _is_retryable_upstream(exc):
        return 503, ErrorResponse(
            error="service_unavailable",
            detail="The service is temporarily unavailable. Please retry shortly.",
        )
    return 500, ErrorResponse(
        error="internal_error",
        detail="An unexpected error occurred.",
    )
