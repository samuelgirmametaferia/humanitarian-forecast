from __future__ import annotations

import secrets
from datetime import UTC, datetime

from fastapi import FastAPI, Header, HTTPException, Query, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from .hf_api.config import Settings
from .hf_api.inference import InferenceService
from .hf_api.learning import reconciliation_status
from .hf_api.providers import DemoForecastProvider, ForecastProvider, ProductionForecastProvider
from .hf_api.rate_limit import configure_limiter, enforce_rate
from .hf_api.repository import ProductionRepository, ReplayConflict
from .hf_api.schemas import (
    ForecastSnapshot,
    HealthResponse,
    IngestEnvelope,
    ModelMetadata,
    ProjectHistoryEvent,
    ProjectHistoryResponse,
)
from .hf_api.security import body_sha256, require_json, verify_ingest_signature

settings = Settings.from_env()
if settings.provider_mode == "production":
    settings.validate_production()
configure_limiter(
    production=settings.provider_mode == "production",
    redis_url=settings.redis_url,
    redis_token=settings.redis_token,
)
production_repository = (
    ProductionRepository(settings.database_url) if settings.provider_mode == "production" else None
)
provider: ForecastProvider = (
    DemoForecastProvider()
    if production_repository is None
    else ProductionForecastProvider(repository=production_repository)
)
inference = InferenceService(settings.model_dir) if production_repository is not None else None
app = FastAPI(title="Humanitarian Forecaster API", version="1.0.0", docs_url=None, redoc_url=None)
if settings.allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.allowed_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Accept", "Content-Type", "X-HF-Run-ID", "X-HF-Signature", "X-HF-Timestamp"],
    )


def add_rate_headers(response: Response, headers: dict[str, str]) -> None:
    for name, value in headers.items():
        response.headers[name] = value


@app.exception_handler(RuntimeError)
async def unavailable_handler(_request: Request, _exc: RuntimeError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": "Service capability unavailable"})


@app.get("/api/health", response_model=HealthResponse)
@app.get("/api/v1/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if settings.provider_mode == "demo" else "degraded",
        mode=settings.provider_mode,
        writesEnabled=settings.provider_mode == "production",
    )


@app.get("/api/v1/forecast/latest", response_model=ForecastSnapshot)
def latest(request: Request, response: Response) -> ForecastSnapshot:
    add_rate_headers(response, enforce_rate(request, "read"))
    return provider.latest()


@app.get("/api/v1/forecast/history", response_model=list[ForecastSnapshot])
def history(
    request: Request,
    response: Response,
    limit: int = Query(20, ge=1, le=100),
) -> list[ForecastSnapshot]:
    add_rate_headers(response, enforce_rate(request, "history"))
    return provider.history(limit)


@app.get("/api/v1/forecast/{forecast_id}", response_model=ForecastSnapshot)
def forecast_by_id(forecast_id: str, request: Request, response: Response) -> ForecastSnapshot:
    add_rate_headers(response, enforce_rate(request, "read"))
    result = provider.by_id(forecast_id)
    if result is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Forecast not found")
    return result


@app.get("/api/v1/model", response_model=ModelMetadata)
def model_metadata(request: Request, response: Response) -> ModelMetadata:
    add_rate_headers(response, enforce_rate(request, "read"))
    return provider.latest().model


@app.get("/api/v1/history", response_model=ProjectHistoryResponse)
def project_history(
    request: Request,
    response: Response,
    limit: int = Query(100, ge=1, le=500),
) -> ProjectHistoryResponse:
    add_rate_headers(response, enforce_rate(request, "history"))
    entries: list[ProjectHistoryEvent] = []
    if production_repository is not None:
        entries = production_repository.project_history(limit)
    if not entries:
        snapshots = provider.history(limit)
        entries = [
            ProjectHistoryEvent(
                id=f"prediction:{snapshot.id}",
                kind="prediction_batch",
                occurredAt=snapshot.generatedAt,
                mode=snapshot.mode,
                modelName=snapshot.model.name,
                modelVersion=snapshot.model.version,
                sourceSnapshotId=snapshot.id,
                metrics={},
                changes=["Base model"] if index == len(snapshots) - 1 else [],
            )
            for index, snapshot in enumerate(snapshots)
        ]
    return ProjectHistoryResponse(entries=entries[:limit])


@app.post("/api/v1/ingest", status_code=status.HTTP_201_CREATED)
async def ingest(
    request: Request,
    x_hf_timestamp: str = Header(alias="X-HF-Timestamp"),
    x_hf_run_id: str = Header(alias="X-HF-Run-ID"),
    x_hf_signature: str = Header(alias="X-HF-Signature"),
) -> dict[str, str]:
    enforce_rate(request, "ingest")
    if (
        settings.provider_mode != "production"
        or not settings.ingest_hmac_secret
        or production_repository is None
        or inference is None
    ):
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Production ingestion is disabled")
    body = await require_json(request)
    verify_ingest_signature(
        secret=settings.ingest_hmac_secret,
        timestamp=x_hf_timestamp,
        run_id=x_hf_run_id,
        signature=x_hf_signature,
        body=body,
    )
    try:
        envelope = IngestEnvelope.model_validate_json(body)
    except ValidationError as exc:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "Invalid ingest envelope",
        ) from exc
    if not secrets.compare_digest(envelope.runId, x_hf_run_id):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Run ID mismatch")
    snapshot = inference.predict(
        envelope.featurePayload,
        run_id=envelope.runId,
        generated_at=envelope.generatedAt,
        source_health=envelope.sourceSummary,
    )
    try:
        production_repository.publish_once(
            envelope.runId,
            body_sha256(body),
            snapshot,
        )
        production_repository.add_history_events(envelope.historyEvents)
    except ReplayConflict as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, "Run ID has already been published") from exc
    return {"status": "published", "forecastId": snapshot.id}


@app.post("/api/v1/context/refresh", status_code=status.HTTP_501_NOT_IMPLEMENTED)
def context_refresh(request: Request) -> dict[str, str]:
    enforce_rate(request, "ingest")
    raise HTTPException(
        status.HTTP_501_NOT_IMPLEMENTED,
        "Context refresh is performed by GitHub Actions",
    )


@app.get("/api/v1/learn/reconcile")
def reconcile(
    request: Request,
    authorization: str | None = Header(default=None),
) -> dict[str, str | bool]:
    enforce_rate(request, "learn")
    if not settings.cron_secret:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Reconciliation is disabled")
    expected = f"Bearer {settings.cron_secret}"
    if authorization is None or not secrets.compare_digest(authorization, expected):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid cron authorization")
    return reconciliation_status(datetime.now(UTC))
