"""Tests for MCPSessionMiddleware's call_next handling."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from starlette.datastructures import Headers

from auth.mcp_session_middleware import MCPSessionMiddleware


def _make_request(path="/mcp"):
    request = Mock()
    request.url = SimpleNamespace(path=path)
    request.headers = Headers({})
    request.method = "POST"
    request.state = SimpleNamespace()
    return request


@pytest.mark.asyncio
async def test_call_next_failure_propagates_without_a_retry():
    """A downstream failure (e.g. a client disconnect mid-stream) must not be
    retried by calling call_next a second time - BaseHTTPMiddleware has already
    consumed the request stream, so a retry re-enters the whole downstream ASGI
    stack (including auth) and can double-execute a tool call before failing
    identically again.
    """
    middleware = MCPSessionMiddleware(app=Mock())
    request = _make_request()

    call_count = 0

    async def call_next(_request):
        nonlocal call_count
        call_count += 1
        raise RuntimeError("No response returned.")

    with pytest.raises(RuntimeError, match="No response returned."):
        await middleware.dispatch(request, call_next)

    assert call_count == 1


@pytest.mark.asyncio
async def test_session_extraction_failure_still_calls_call_next_once():
    """A broken session-extraction step (bad headers/claims) must fall back to
    processing the request without a session context, not fail the request."""
    middleware = MCPSessionMiddleware(app=Mock())
    request = _make_request()
    # Force the extraction logic to raise by making header access blow up.
    request.headers = Mock()
    request.headers.__iter__ = Mock(side_effect=RuntimeError("bad headers"))

    call_count = 0

    async def call_next(_request):
        nonlocal call_count
        call_count += 1
        return "ok"

    result = await middleware.dispatch(request, call_next)

    assert result == "ok"
    assert call_count == 1


@pytest.mark.asyncio
async def test_non_mcp_path_calls_call_next_once():
    middleware = MCPSessionMiddleware(app=Mock())
    request = _make_request(path="/health")

    call_count = 0

    async def call_next(_request):
        nonlocal call_count
        call_count += 1
        return "ok"

    result = await middleware.dispatch(request, call_next)

    assert result == "ok"
    assert call_count == 1
