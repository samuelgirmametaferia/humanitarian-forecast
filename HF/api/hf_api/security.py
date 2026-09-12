from __future__ import annotations

import hashlib
import hmac
import time

from fastapi import HTTPException, Request, status

MAX_INGEST_BYTES = 1_500_000
MAX_CLOCK_SKEW_SECONDS = 300


def body_sha256(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def signature_payload(timestamp: str, run_id: str, digest: str) -> bytes:
    return f"{timestamp}.{run_id}.{digest}".encode()


def sign_body(secret: str, timestamp: str, run_id: str, body: bytes) -> str:
    return hmac.new(
        secret.encode(),
        signature_payload(timestamp, run_id, body_sha256(body)),
        hashlib.sha256,
    ).hexdigest()


def verify_ingest_signature(
    *, secret: str, timestamp: str, run_id: str, signature: str, body: bytes,
    now: int | None = None,
) -> None:
    if len(body) > MAX_INGEST_BYTES:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "Request body too large")
    try:
        sent_at = int(timestamp)
    except ValueError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid signature timestamp") from exc
    current = int(time.time()) if now is None else now
    if abs(current - sent_at) > MAX_CLOCK_SKEW_SECONDS:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Expired signature timestamp")
    expected = sign_body(secret, timestamp, run_id, body)
    if not hmac.compare_digest(expected, signature):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid request signature")


async def require_json(request: Request) -> bytes:
    content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if content_type != "application/json":
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "application/json required")
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_INGEST_BYTES:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "Request body too large")
    body = await request.body()
    if len(body) > MAX_INGEST_BYTES:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "Request body too large")
    return body
