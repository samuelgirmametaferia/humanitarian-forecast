from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any

from .schemas import (
    FeaturePayload,
    ForecastSnapshot,
    ForecastZone,
    ModelMetadata,
    Position,
    SourceHealth,
)
from .serving import CandidatePrediction, OnnxEnsemble, load_feature_contract

_WARNING = (
    "Coarse humanitarian early-warning research signal. Not a tactical coordinate forecast, "
    "verified event feed, safe route, or evacuation order. Advisory radii are display buffers, "
    "not uncertainty bounds. Uncertainty and population exposure are unavailable unless supplied "
    "by separately validated data products."
)


def _package_sha256(manifest: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for member in manifest["members"]:
        digest.update(bytes.fromhex(member["sha256"]))
    return digest.hexdigest()


def _driver_labels(candidate: CandidatePrediction) -> list[str]:
    labels: list[str] = [candidate.site_type]
    if candidate.days_since_last_event_at_site <= 30:
        labels.append("site activity recorded within 30 days of the cutoff")
    labels.append("ensemble candidate ranking")
    return labels


class InferenceService:
    def __init__(self, model_dir: str | Path, runtime: Any | None = None) -> None:
        self.model_dir = Path(model_dir)
        self._runtime = runtime
        self._ensemble: OnnxEnsemble | None = None
        self._metadata: ModelMetadata | None = None
        self._lock = Lock()

    def _load(self) -> tuple[OnnxEnsemble, ModelMetadata]:
        if self._ensemble is not None and self._metadata is not None:
            return self._ensemble, self._metadata
        with self._lock:
            if self._ensemble is None:
                ensemble = OnnxEnsemble(self.model_dir, runtime=self._runtime)
                champion = json.loads((self.model_dir / "champion-info.json").read_text())
                validation = champion["ethiopia_metrics"]["validation"]
                development = champion["ethiopia_metrics"]["development"]
                manifest = ensemble.manifest
                self._ensemble = ensemble
                self._metadata = ModelMetadata(
                    name=manifest["model"],
                    version="fine-v2",
                    artifactSha256=_package_sha256(manifest),
                    featureContract=manifest["featureContract"],
                    candidateCount=manifest["input"]["candidates"][0],
                    validationTop1Within20Km=validation["within_20km_at_top1"],
                    validationDiverseTop5Within20Km=validation["diverse_within_20km_at_top5"],
                    developmentTop1Within20Km=development["within_20km_at_top1"],
                    candidateOracleWithin20Km=validation["oracle_within_20km"],
                )
            if self._metadata is None:
                raise RuntimeError("model metadata failed to initialize")
            return self._ensemble, self._metadata

    def metadata(self) -> ModelMetadata:
        _, metadata = self._load()
        return metadata.model_copy(deep=True)

    def predict(
        self,
        payload: FeaturePayload,
        *,
        run_id: str,
        generated_at: datetime,
        source_health: list[SourceHealth],
    ) -> ForecastSnapshot:
        ensemble, metadata = self._load()
        raw = payload.model_dump(mode="json")
        contract = load_feature_contract(raw)
        prediction = ensemble.predict(contract)
        zones = [self._zone(candidate) for candidate in prediction.candidates]
        return ForecastSnapshot(
            schemaVersion="forecast-snapshot.v1",
            id=f"forecast:{run_id}",
            mode="production",
            country="Ethiopia",
            conflict=payload.conflictLabel,
            generatedAt=generated_at.astimezone(UTC),
            observationCutoff=payload.observationCutoff,
            horizonDays=payload.horizonDays,
            model=metadata,
            zones=zones,
            observations=[],
            sourceHealth=source_health,
            warning=_WARNING,
        )

    @staticmethod
    def _zone(candidate: CandidatePrediction) -> ForecastZone:
        return ForecastZone(
            id=candidate.id,
            rank=candidate.rank,
            label=f"Candidate zone {candidate.rank}",
            position=Position(
                latitude=round(candidate.latitude, 2),
                longitude=round(candidate.longitude, 2),
            ),
            probability=candidate.probability,
            uncertainty=None,
            radiusKm=candidate.advisory_radius_km,
            siteType=candidate.site_type,
            daysSinceLastEvent=candidate.days_since_last_event_at_site,
            elevationM=candidate.elevation_m,
            ruggednessM=candidate.ruggedness_m,
            populationExposureBand=None,
            drivers=_driver_labels(candidate),
        )
