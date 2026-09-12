from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.hf_api.schemas import ForecastSnapshot
from api.index import app

FIXTURES = Path(__file__).parents[1] / "fixtures"
client = TestClient(app)


def test_health_and_latest_demo() -> None:
    health = client.get("/api/v1/health")
    assert health.status_code == 200
    assert health.json()["mode"] == "demo"
    assert health.json()["writesEnabled"] is False

    latest = client.get("/api/v1/forecast/latest")
    assert latest.status_code == 200
    assert latest.json()["mode"] == "demo"
    assert latest.headers["ratelimit-limit"] == "180"

    history = client.get("/api/v1/history")
    assert history.status_code == 200
    payload = history.json()
    assert payload["updateCycleDays"] == 3
    assert len(payload["entries"]) == 1
    assert payload["entries"][0]["kind"] == "prediction_batch"
    assert payload["entries"][0]["mode"] == "demo"
    assert payload["entries"][0]["metrics"] == {}


def test_forecast_fixture_contracts() -> None:
    valid = json.loads((FIXTURES / "forecast.valid.json").read_text())
    ForecastSnapshot.model_validate(valid)
    invalid = json.loads((FIXTURES / "forecast.invalid.json").read_text())
    with pytest.raises(ValueError):
        ForecastSnapshot.model_validate(invalid)


def test_missing_forecast_returns_404() -> None:
    response = client.get("/api/v1/forecast/not-present")
    assert response.status_code == 404


def test_ingest_fails_closed_in_demo() -> None:
    response = client.post(
        "/api/v1/ingest",
        content=b"{}",
        headers={
            "content-type": "application/json",
            "X-HF-Timestamp": "0",
            "X-HF-Run-ID": "demo-run-01",
            "X-HF-Signature": "invalid",
        },
    )
    assert response.status_code == 503
