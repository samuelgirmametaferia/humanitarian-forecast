from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from threading import Lock
from typing import Protocol

import httpx
from fastapi import HTTPException, Request, status


@dataclass(frozen=True)
class RateClass:
    limit: int
    window_seconds: int


RATE_CLASSES = {
    "read": RateClass(180, 60),
    "history": RateClass(40, 60),
    "ingest": RateClass(2, 60),
    "learn": RateClass(2, 300),
}


class Limiter(Protocol):
    def check(self, key: str, rate_name: str) -> tuple[int, int]: ...


class InMemoryLimiter:
    def __init__(self) -> None:
        self._buckets: dict[tuple[str, str, int], int] = {}
        self._lock = Lock()

    def check(self, key: str, rate_name: str) -> tuple[int, int]:
        rate = RATE_CLASSES[rate_name]
        now = int(time.time())
        window = now // rate.window_seconds
        bucket = (key, rate_name, window)
        with self._lock:
            used = self._buckets.get(bucket, 0) + 1
            self._buckets[bucket] = used
        reset = (window + 1) * rate.window_seconds
        _raise_if_exceeded(used, rate, reset, now)
        return max(0, rate.limit - used), reset


_SCRIPT = """
local count = redis.call('INCR', KEYS[1])
if count == 1 then redis.call('EXPIRE', KEYS[1], ARGV[1]) end
local ttl = redis.call('TTL', KEYS[1])
return {count, ttl}
""".strip()


class UpstashLimiter:
    def __init__(
        self,
        url: str,
        token: str,
        *,
        client: httpx.Client | None = None,
    ) -> None:
        if not url.startswith("https://"):
            raise RuntimeError("Upstash Redis URL must use HTTPS")
        self._url = url.rstrip("/")
        self._token = token
        self._client = client or httpx.Client(timeout=httpx.Timeout(3.0))

    def check(self, key: str, rate_name: str) -> tuple[int, int]:
        rate = RATE_CLASSES[rate_name]
        now = int(time.time())
        bucket_key = f"hf:rate:{rate_name}:{_opaque_key(key)}"
        try:
            response = self._client.post(
                self._url,
                json=["EVAL", _SCRIPT, "1", bucket_key, str(rate.window_seconds)],
                headers={"Authorization": f"Bearer {self._token}"},
            )
            response.raise_for_status()
            result = response.json().get("result")
            if (
                not isinstance(result, list)
                or len(result) != 2
                or not all(isinstance(value, int) for value in result)
            ):
                raise ValueError("invalid limiter response")
            used, ttl = result
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            raise RuntimeError("Distributed rate limiter unavailable") from exc
        reset = now + max(1, ttl)
        _raise_if_exceeded(used, rate, reset, now)
        return max(0, rate.limit - used), reset


def _opaque_key(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:32]


def _raise_if_exceeded(used: int, rate: RateClass, reset: int, now: int) -> None:
    if used > rate.limit:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            "Rate limit exceeded",
            headers={"Retry-After": str(max(1, reset - now))},
        )


limiter: Limiter = InMemoryLimiter()


def configure_limiter(
    *,
    production: bool,
    redis_url: str | None,
    redis_token: str | None,
) -> None:
    global limiter
    if production:
        if not redis_url or not redis_token:
            raise RuntimeError("Production rate limiting requires Upstash Redis configuration")
        limiter = UpstashLimiter(redis_url, redis_token)
    else:
        limiter = InMemoryLimiter()


def enforce_rate(request: Request, rate_name: str) -> dict[str, str]:
    host = request.client.host if request.client else "unknown"
    remaining, reset = limiter.check(host, rate_name)
    rate = RATE_CLASSES[rate_name]
    return {
        "RateLimit-Limit": str(rate.limit),
        "RateLimit-Remaining": str(remaining),
        "RateLimit-Reset": str(reset),
    }
