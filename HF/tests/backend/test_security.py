from __future__ import annotations

import pytest
from fastapi import HTTPException

from api.hf_api.security import sign_body, verify_ingest_signature

TEST_KEY = "unit-test-signing-material"


def test_signature_round_trip() -> None:
    body = b'{"schemaVersion":"ingest-envelope.v1"}'
    timestamp = "2000"
    signature = sign_body(TEST_KEY, timestamp, "run-0001", body)
    verify_ingest_signature(
        secret=TEST_KEY,
        timestamp=timestamp,
        run_id="run-0001",
        signature=signature,
        body=body,
        now=2000,
    )


def test_signature_rejects_tampering() -> None:
    with pytest.raises(HTTPException) as error:
        verify_ingest_signature(
            secret=TEST_KEY,
            timestamp="2000",
            run_id="run-0001",
            signature="0" * 64,
            body=b"{}",
            now=2000,
        )
    assert error.value.status_code == 401


def test_signature_rejects_stale_request() -> None:
    body = b"{}"
    signature = sign_body(TEST_KEY, "1000", "run-0001", body)
    with pytest.raises(HTTPException) as error:
        verify_ingest_signature(
            secret=TEST_KEY,
            timestamp="1000",
            run_id="run-0001",
            signature=signature,
            body=body,
            now=2000,
        )
    assert error.value.status_code == 401
