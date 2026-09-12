from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

Strict = ConfigDict(extra="forbid")
Probability = Annotated[float, Field(ge=0, le=1)]


class Position(BaseModel):
    model_config = Strict
    latitude: Annotated[float, Field(ge=-90, le=90)]
    longitude: Annotated[float, Field(ge=-180, le=180)]


class ModelMetadata(BaseModel):
    model_config = Strict
    name: str = Field(max_length=120)
    version: str = Field(max_length=40)
    artifactSha256: str = Field(pattern=r"^(demo|[a-f0-9]{64})$")
    featureContract: Literal["feature-contract.v1"]
    candidateCount: int = Field(ge=1, le=512)
    validationTop1Within20Km: Probability
    validationDiverseTop5Within20Km: Probability
    developmentTop1Within20Km: Probability
    candidateOracleWithin20Km: Probability


class ForecastZone(BaseModel):
    model_config = Strict
    id: str = Field(min_length=3, max_length=80)
    rank: int = Field(ge=1, le=10)
    label: str = Field(min_length=1, max_length=80)
    position: Position
    probability: Probability
    uncertainty: Probability | None
    radiusKm: float = Field(ge=10, le=100)
    siteType: Literal["conflict-frequency site", "recent cross-conflict event"]
    daysSinceLastEvent: int = Field(ge=0, le=10000)
    elevationM: int = Field(ge=-500, le=5000)
    ruggednessM: float = Field(ge=0, le=3000)
    populationExposureBand: Literal["lower", "moderate", "higher"] | None
    populationEstimate: int | None = Field(default=None, ge=0)
    populationDensityPerKm2: float | None = Field(default=None, ge=0)
    populationAreaKm2: float | None = Field(default=None, gt=0)
    populationYear: int | None = Field(default=None, ge=2015, le=2030)
    populationSource: str | None = Field(default=None, max_length=120)
    adminArea: str | None = Field(default=None, max_length=120)
    drivers: list[str] = Field(min_length=1, max_length=5)

class ForecastObservation(BaseModel):
    model_config = Strict
    id: str = Field(max_length=80)
    kind: Literal["observed event", "source signal"]
    position: Position
    occurredAt: datetime
    precision: Literal["exact", "named place", "regional"]
    summary: str = Field(min_length=1, max_length=240)


class SourceHealth(BaseModel):
    model_config = Strict
    id: str = Field(max_length=40)
    label: str = Field(max_length=80)
    status: Literal["healthy", "delayed", "unavailable"]
    lastSuccessfulAt: datetime | None
    articlesProcessed: int = Field(ge=0, le=100000)


class ForecastSnapshot(BaseModel):
    model_config = Strict
    schemaVersion: Literal["forecast-snapshot.v1"]
    id: str = Field(min_length=8, max_length=80)
    mode: Literal["demo", "production"]
    country: Literal["Ethiopia"]
    conflict: str = Field(min_length=1, max_length=100)
    generatedAt: datetime
    observationCutoff: datetime
    horizonDays: int = Field(ge=1, le=30)
    model: ModelMetadata
    zones: list[ForecastZone] = Field(min_length=1, max_length=10)
    observations: list[ForecastObservation] = Field(max_length=500)
    sourceHealth: list[SourceHealth] = Field(min_length=1, max_length=12)
    warning: str = Field(min_length=20, max_length=500)


class Candidate(BaseModel):
    model_config = Strict
    id: str = Field(max_length=80)
    valid: bool
    eastKm: float = Field(ge=-2000, le=2000)
    northKm: float = Field(ge=-2000, le=2000)
    features: list[float] = Field(min_length=37, max_length=37)


class FeaturePayload(BaseModel):
    model_config = Strict
    schemaVersion: Literal["feature-contract.v1"]
    country: Literal["Ethiopia"]
    conflictId: str = Field(max_length=100)
    conflictLabel: str = Field(min_length=1, max_length=100)
    anchor: Position
    observationCutoff: datetime
    horizonDays: int = Field(ge=1, le=30)
    events: list[list[float]] = Field(min_length=16, max_length=16)
    candidates: list[Candidate] = Field(min_length=64, max_length=64)


HistoryEventKind = Literal[
    "dataset_update",
    "model_run",
    "training",
    "evaluation",
    "prediction_batch",
    "model_version",
]


class ProjectHistoryEvent(BaseModel):
    model_config = Strict
    id: str = Field(min_length=3, max_length=120)
    kind: HistoryEventKind
    occurredAt: datetime
    mode: Literal["demo", "production"]
    modelName: str | None = Field(default=None, max_length=120)
    modelVersion: str | None = Field(default=None, max_length=40)
    sourceSnapshotId: str | None = Field(default=None, max_length=80)
    metrics: dict[str, float] = Field(default_factory=dict)
    changes: list[str] = Field(default_factory=list, max_length=20)
    reportUrl: str | None = Field(default=None, max_length=500)


class ProjectHistoryResponse(BaseModel):
    model_config = Strict
    entries: list[ProjectHistoryEvent]
    updateCycleDays: Literal[3] = 3


class IngestEnvelope(BaseModel):
    model_config = Strict
    schemaVersion: Literal["ingest-envelope.v1"]
    runId: str = Field(min_length=8, max_length=100, pattern=r"^[a-zA-Z0-9._:-]+$")
    generatedAt: datetime
    featurePayload: FeaturePayload
    sourceSummary: list[SourceHealth] = Field(min_length=1, max_length=12)
    historyEvents: list[ProjectHistoryEvent] = Field(default_factory=list, max_length=50)


class TrainingPair(BaseModel):
    model_config = Strict
    runId: str = Field(min_length=8, max_length=100)
    generatedAt: datetime
    featurePayload: FeaturePayload


class TrainingPairsResponse(BaseModel):
    model_config = Strict
    pairs: list[TrainingPair] = Field(max_length=1000)


class RegistryHealth(BaseModel):
    model_config = Strict
    configured: bool
    activeVersion: str | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    model_config = Strict
    status: Literal["ok", "degraded"]
    mode: Literal["demo", "production"]
    writesEnabled: bool
    modelRegistry: RegistryHealth | None = None
