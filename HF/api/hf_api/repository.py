from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Callable
from importlib import import_module
from threading import Lock
from typing import Any, Protocol, cast

from .schemas import ForecastSnapshot, ProjectHistoryEvent

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS hf_forecasts (
    id text PRIMARY KEY,
    generated_at timestamptz NOT NULL,
    payload jsonb NOT NULL
);
CREATE INDEX IF NOT EXISTS hf_forecasts_generated_at_idx
    ON hf_forecasts (generated_at DESC);
CREATE TABLE IF NOT EXISTS hf_ingest_runs (
    run_id text PRIMARY KEY,
    body_sha256 char(64) NOT NULL,
    forecast_id text NOT NULL UNIQUE REFERENCES hf_forecasts(id),
    published_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS hf_history_events (
    id text PRIMARY KEY,
    occurred_at timestamptz NOT NULL,
    payload jsonb NOT NULL
);
CREATE INDEX IF NOT EXISTS hf_history_events_occurred_at_idx
    ON hf_history_events (occurred_at DESC);
CREATE TABLE IF NOT EXISTS hf_feature_payloads (
    run_id text PRIMARY KEY,
    generated_at timestamptz NOT NULL,
    payload jsonb NOT NULL
);
CREATE INDEX IF NOT EXISTS hf_feature_payloads_generated_at_idx
    ON hf_feature_payloads (generated_at ASC);
"""


class Cursor(Protocol):
    rowcount: int

    def execute(self, query: str, params: tuple[Any, ...] = ()) -> Cursor: ...
    def fetchone(self) -> Any | None: ...
    def fetchall(self) -> list[Any]: ...


class Connection(Protocol):
    def __enter__(self) -> Connection: ...
    def __exit__(self, *args: object) -> None: ...
    def cursor(self) -> Cursor: ...
    def execute(self, query: str, params: tuple[Any, ...] = ()) -> Cursor: ...


Connect = Callable[[], Connection]


class Repository(ABC):
    @abstractmethod
    def publish_once(
        self,
        run_id: str,
        body_digest: str,
        snapshot: ForecastSnapshot,
    ) -> bool:
        raise NotImplementedError

    @abstractmethod
    def latest(self) -> ForecastSnapshot:
        raise NotImplementedError

    @abstractmethod
    def history(self, limit: int) -> list[ForecastSnapshot]:
        raise NotImplementedError

    @abstractmethod
    def by_id(self, forecast_id: str) -> ForecastSnapshot | None:
        raise NotImplementedError

    @abstractmethod
    def add_history_events(self, entries: list[ProjectHistoryEvent]) -> None:
        raise NotImplementedError

    @abstractmethod
    def project_history(self, limit: int) -> list[ProjectHistoryEvent]:
        raise NotImplementedError

    @abstractmethod
    def store_feature_payload(self, run_id: str, generated_at: Any, payload: dict[str, Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    def feature_payloads(self, since: Any, limit: int) -> list[dict[str, Any]]:
        raise NotImplementedError


class DemoRepository(Repository):
    def publish_once(
        self,
        run_id: str,
        body_digest: str,
        snapshot: ForecastSnapshot,
    ) -> bool:
        del run_id, body_digest, snapshot
        raise RuntimeError("Demo mode never publishes production forecasts")

    def latest(self) -> ForecastSnapshot:
        raise RuntimeError("Demo data is provided from a static fixture")

    def history(self, limit: int) -> list[ForecastSnapshot]:
        del limit
        raise RuntimeError("Demo data is provided from a static fixture")

    def by_id(self, forecast_id: str) -> ForecastSnapshot | None:
        del forecast_id
        raise RuntimeError("Demo data is provided from a static fixture")

    def add_history_events(self, entries: list[ProjectHistoryEvent]) -> None:
        del entries
        raise RuntimeError("Demo mode never persists history")

    def project_history(self, limit: int) -> list[ProjectHistoryEvent]:
        del limit
        return []

    def store_feature_payload(self, run_id: str, generated_at: Any, payload: dict[str, Any]) -> None:
        del run_id, generated_at, payload
        raise RuntimeError("Demo mode never persists feature payloads")

    def feature_payloads(self, since: Any, limit: int) -> list[dict[str, Any]]:
        del since, limit
        return []


class ProductionRepository(Repository):
    def __init__(self, database_url: str | None, connect: Connect | None = None) -> None:
        if not database_url:
            raise RuntimeError("Production repository requires HF_DATABASE_URL")
        self._database_url = database_url
        self._connect_override = connect
        self._schema_ready = False
        self._schema_lock = Lock()

    def _connect(self) -> Connection:
        if self._connect_override is not None:
            return self._connect_override()
        try:
            psycopg = import_module("psycopg")
        except ImportError as exc:
            raise RuntimeError("psycopg is required for production persistence") from exc
        return cast(Connection, psycopg.connect(self._database_url))

    def _ensure_schema(self) -> None:
        if self._schema_ready:
            return
        with self._schema_lock:
            if self._schema_ready:
                return
            with self._connect() as connection:
                connection.execute(_SCHEMA_SQL)
            self._schema_ready = True

    def publish_once(
        self,
        run_id: str,
        body_digest: str,
        snapshot: ForecastSnapshot,
    ) -> bool:
        self._ensure_schema()
        payload = json.dumps(snapshot.model_dump(mode="json"), separators=(",", ":"))
        with self._connect() as connection:
            forecast = connection.execute(
                "INSERT INTO hf_forecasts (id, generated_at, payload) VALUES (%s, %s, %s::jsonb) "
                "ON CONFLICT (id) DO NOTHING",
                (snapshot.id, snapshot.generatedAt, payload),
            )
            if forecast.rowcount != 1:
                raise ReplayConflict
            claim = connection.execute(
                "INSERT INTO hf_ingest_runs (run_id, body_sha256, forecast_id) "
                "VALUES (%s, %s, %s) ON CONFLICT (run_id) DO NOTHING",
                (run_id, body_digest, snapshot.id),
            )
            if claim.rowcount != 1:
                raise ReplayConflict
        return True

    def latest(self) -> ForecastSnapshot:
        self._ensure_schema()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload FROM hf_forecasts ORDER BY generated_at DESC LIMIT 1"
            ).fetchone()
        if row is None:
            raise RuntimeError("No production forecast has been published")
        return _snapshot_from_row(row)

    def history(self, limit: int) -> list[ForecastSnapshot]:
        self._ensure_schema()
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload FROM hf_forecasts ORDER BY generated_at DESC LIMIT %s",
                (limit,),
            ).fetchall()
        return [_snapshot_from_row(row) for row in rows]

    def by_id(self, forecast_id: str) -> ForecastSnapshot | None:
        self._ensure_schema()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload FROM hf_forecasts WHERE id = %s",
                (forecast_id,),
            ).fetchone()
        return None if row is None else _snapshot_from_row(row)

    def add_history_events(self, entries: list[ProjectHistoryEvent]) -> None:
        if not entries:
            return
        self._ensure_schema()
        with self._connect() as connection:
            for entry in entries:
                payload = json.dumps(entry.model_dump(mode="json"), separators=(",", ":"))
                connection.execute(
                    "INSERT INTO hf_history_events (id, occurred_at, payload) "
                    "VALUES (%s, %s, %s::jsonb) ON CONFLICT (id) DO NOTHING",
                    (entry.id, entry.occurredAt, payload),
                )

    def project_history(self, limit: int) -> list[ProjectHistoryEvent]:
        self._ensure_schema()
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload FROM hf_history_events ORDER BY occurred_at DESC LIMIT %s",
                (limit,),
            ).fetchall()
        return [_history_event_from_row(row) for row in rows]

    def store_feature_payload(self, run_id: str, generated_at: Any, payload: dict[str, Any]) -> None:
        """Persist the raw feature contract of a published run so matured
        forecasts can be labeled with realized UCDP outcomes and retrained on."""
        self._ensure_schema()
        body = json.dumps(payload, separators=(",", ":"))
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO hf_feature_payloads (run_id, generated_at, payload) "
                "VALUES (%s, %s, %s::jsonb) ON CONFLICT (run_id) DO NOTHING",
                (run_id, generated_at, body),
            )

    def feature_payloads(self, since: Any, limit: int) -> list[dict[str, Any]]:
        self._ensure_schema()
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT run_id, generated_at, payload FROM hf_feature_payloads "
                "WHERE generated_at > %s ORDER BY generated_at ASC LIMIT %s",
                (since, limit),
            ).fetchall()
        return [
            {"runId": row[0], "generatedAt": row[1].isoformat() if hasattr(row[1], "isoformat") else row[1], "featurePayload": _json_value(row[2])}
            for row in rows
        ]


class ReplayConflict(RuntimeError):
    pass


def _snapshot_from_row(row: Any) -> ForecastSnapshot:
    value = row[0]
    if isinstance(value, str):
        value = json.loads(value)
    return ForecastSnapshot.model_validate(value)


def _history_event_from_row(row: Any) -> ProjectHistoryEvent:
    value = row[0]
    if isinstance(value, str):
        value = json.loads(value)
    return ProjectHistoryEvent.model_validate(value)


def _json_value(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str) else value
