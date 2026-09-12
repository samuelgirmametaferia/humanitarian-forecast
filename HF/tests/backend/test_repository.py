from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from api.hf_api.repository import ProductionRepository, ReplayConflict
from api.hf_api.schemas import ForecastSnapshot, ProjectHistoryEvent

FIXTURE = Path(__file__).parents[1] / "fixtures" / "forecast.valid.json"


class FakeCursor:
    def __init__(self, rowcount: int = 0, rows: list[Any] | None = None) -> None:
        self.rowcount = rowcount
        self.rows = rows or []

    def execute(self, query: str, params: tuple[Any, ...] = ()) -> FakeCursor:
        del query, params
        return self

    def fetchone(self) -> Any | None:
        return self.rows[0] if self.rows else None

    def fetchall(self) -> list[Any]:
        return self.rows


class FakeConnection:
    def __init__(self, responder: Callable[[str, tuple[Any, ...]], FakeCursor]) -> None:
        self.responder = responder
        self.queries: list[tuple[str, tuple[Any, ...]]] = []

    def __enter__(self) -> FakeConnection:
        return self

    def __exit__(self, *args: object) -> None:
        del args

    def cursor(self) -> FakeCursor:
        return FakeCursor()

    def execute(self, query: str, params: tuple[Any, ...] = ()) -> FakeCursor:
        self.queries.append((query, params))
        return self.responder(query, params)


def snapshot() -> ForecastSnapshot:
    return ForecastSnapshot.model_validate(json.loads(FIXTURE.read_text()))


def test_publish_once_inserts_forecast_before_run_claim() -> None:
    connection = FakeConnection(lambda _query, _params: FakeCursor(rowcount=1))
    repository = ProductionRepository("postgresql://test", connect=lambda: connection)

    assert repository.publish_once("run-0001", "a" * 64, snapshot()) is True
    statements = [query for query, _params in connection.queries]
    forecast_index = next(
        index for index, query in enumerate(statements) if "INSERT INTO hf_forecasts" in query
    )
    claim_index = next(
        index for index, query in enumerate(statements) if "INSERT INTO hf_ingest_runs" in query
    )
    assert any("CREATE TABLE" in query for query in statements)
    assert forecast_index < claim_index


def test_publish_once_replay_raises_after_forecast_insert_for_transactional_rollback() -> None:
    def responder(query: str, _params: tuple[Any, ...]) -> FakeCursor:
        if "INSERT INTO hf_ingest_runs" in query:
            return FakeCursor(rowcount=0)
        return FakeCursor(rowcount=1)

    connection = FakeConnection(responder)
    repository = ProductionRepository("postgresql://test", connect=lambda: connection)

    try:
        repository.publish_once("run-0001", "a" * 64, snapshot())
    except ReplayConflict:
        pass
    else:
        raise AssertionError("expected a replay conflict")
    statements = [query for query, _params in connection.queries]
    assert any("INSERT INTO hf_forecasts" in query for query in statements)
    assert any("INSERT INTO hf_ingest_runs" in query for query in statements)


def test_repository_deserializes_latest_snapshot() -> None:
    payload = snapshot().model_dump(mode="json")

    def responder(query: str, _params: tuple[Any, ...]) -> FakeCursor:
        if "SELECT payload" in query:
            return FakeCursor(rows=[(payload,)])
        return FakeCursor(rowcount=1)

    connection = FakeConnection(responder)
    repository = ProductionRepository("postgresql://test", connect=lambda: connection)

    assert repository.latest().id == "demo-2026-09-11"


def history_event() -> ProjectHistoryEvent:
    return ProjectHistoryEvent.model_validate({
        "id": "evaluation:test-1",
        "kind": "evaluation",
        "occurredAt": "2026-09-12T00:00:00Z",
        "mode": "production",
        "modelName": "theswarm_fine_v2",
        "modelVersion": "fine-v2",
        "sourceSnapshotId": None,
        "metrics": {"top1Within20Km": 0.11},
        "changes": ["Compared against the current model"],
        "reportUrl": None,
    })


def test_repository_persists_and_orders_history_events() -> None:
    payload = history_event().model_dump(mode="json")

    def responder(query: str, _params: tuple[Any, ...]) -> FakeCursor:
        if "SELECT payload FROM hf_history_events" in query:
            return FakeCursor(rows=[(payload,)])
        return FakeCursor(rowcount=1)

    connection = FakeConnection(responder)
    repository = ProductionRepository("postgresql://test", connect=lambda: connection)
    repository.add_history_events([history_event()])
    entries = repository.project_history(10)

    assert entries == [history_event()]
    statements = [query for query, _params in connection.queries]
    assert any("INSERT INTO hf_history_events" in query for query in statements)
    assert any("ORDER BY occurred_at DESC" in query for query in statements)
