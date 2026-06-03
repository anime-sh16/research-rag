import httpx
import pytest
from google.genai import errors as genai_errors

from src.api.errors import ErrorResponse, map_exception


def test_timeout_error_maps_to_503():
    status, body = map_exception(TimeoutError("upstream slow"))
    assert status == 503
    assert body.error == "service_unavailable"


def test_httpx_timeout_maps_to_503():
    status, body = map_exception(httpx.ReadTimeout("slow"))
    assert status == 503
    assert body.error == "service_unavailable"


def test_generic_exception_maps_to_500():
    status, body = map_exception(ValueError("internal detail"))
    assert status == 500
    assert body.error == "internal_error"


def test_mapped_body_never_leaks_original_message():
    status, body = map_exception(ValueError("SECRET-STACK-TRACE"))
    assert "SECRET-STACK-TRACE" not in body.detail
    assert "SECRET-STACK-TRACE" not in body.error


def test_error_response_is_pydantic_model():
    body = ErrorResponse(error="x", detail="y")
    assert body.model_dump() == {"error": "x", "detail": "y"}


@pytest.mark.parametrize("status_code", [429, 503, 504])
def test_genai_api_errors_map_to_503(status_code):
    exc = genai_errors.APIError.__new__(genai_errors.APIError)
    exc.status_code = status_code
    exc.message = f"upstream {status_code}"
    status, body = map_exception(exc)
    assert status == 503
    assert body.error == "service_unavailable"
