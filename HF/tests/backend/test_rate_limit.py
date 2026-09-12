from __future__ import annotations

import httpx
import pytest
from fastapi import HTTPException

from api.hf_api.rate_limit import UpstashLimiter


def test_upstash_limiter_uses_opaque_keys_and_returns_window() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer test-token"
        body = request.read().decode()
        assert "client-address" not in body
        assert "hf:rate:read:" in body
        return httpx.Response(200, json={"result": [1, 42]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    limiter = UpstashLimiter("https://redis.example", "test-token", client=client)

    remaining, reset = limiter.check("client-address", "read")
    assert remaining == 179
    assert reset > 0


def test_upstash_limiter_fails_closed_on_remote_error() -> None:
    client = httpx.Client(
        transport=httpx.MockTransport(lambda _request: httpx.Response(503))
    )
    limiter = UpstashLimiter("https://redis.example", "test-token", client=client)

    with pytest.raises(RuntimeError, match="Distributed rate limiter unavailable"):
        limiter.check("client-address", "ingest")


def test_upstash_limiter_returns_retry_after_when_exceeded() -> None:
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(200, json={"result": [3, 15]})
        )
    )
    limiter = UpstashLimiter("https://redis.example", "test-token", client=client)

    with pytest.raises(HTTPException) as error:
        limiter.check("client-address", "ingest")
    assert error.value.status_code == 429
    assert error.value.headers == {"Retry-After": "15"}
