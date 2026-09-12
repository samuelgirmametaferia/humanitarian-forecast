from __future__ import annotations

import hashlib
import json
import sys
import time
import urllib.request
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

_REGISTRY_REFRESH_SECONDS = 600.0
# Registry members are multi-megabyte GitHub Release assets; on a slow path a
# single member can take a minute, so the timeout must tolerate that.
_REGISTRY_TIMEOUT_SECONDS = 120.0
_REGISTRY_FETCH_ATTEMPTS = 3
_REGISTRY_CACHE_ROOT = Path("/tmp/hf-model-registry")


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


def _fetch(url: str, destination: Path) -> None:
    last_error: Exception | None = None
    for _ in range(_REGISTRY_FETCH_ATTEMPTS):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "HumanitarianForecast/1.0"})
            with urllib.request.urlopen(request, timeout=_REGISTRY_TIMEOUT_SECONDS) as response:
                with destination.open("wb") as out:
                    while chunk := response.read(1 << 16):
                        out.write(chunk)
            return
        except Exception as exc:  # noqa: BLE001 - retried below, raised after the last attempt
            last_error = exc
    assert last_error is not None
    raise last_error


class InferenceService:
    def __init__(
        self,
        model_dir: str | Path,
        registry_url: str | None = None,
        runtime: Any | None = None,
    ) -> None:
        self.model_dir = Path(model_dir)
        # Stable URL of the registry manifest (GitHub Releases asset). The
        # serving layer polls it, promotes new versions automatically, and
        # falls back to the bundled package whenever the registry is
        # unreachable or serves anything that fails validation.
        self.registry_url = registry_url
        self._runtime = runtime
        self._ensemble: OnnxEnsemble | None = None
        self._metadata: ModelMetadata | None = None
        self._active_dir: Path = self.model_dir
        self._registry_digest: str | None = None
        self._registry_checked_at: float = 0.0
        self._registry_error: str | None = None
        self._registry_index: list[dict[str, Any]] | None = None
        self._registry_index_checked_at: float = 0.0
        self._lock = Lock()

    def _registry_manifest(self) -> dict[str, Any] | None:
        if not self.registry_url:
            return None
        request = urllib.request.Request(
            self.registry_url, headers={"User-Agent": "HumanitarianForecast/1.0"}
        )
        with urllib.request.urlopen(request, timeout=_REGISTRY_TIMEOUT_SECONDS) as response:
            raw = response.read()
        manifest = json.loads(raw)
        digest = hashlib.sha256(raw).hexdigest()
        base = self.registry_url.rsplit("/", 1)[0]
        cache_dir = _REGISTRY_CACHE_ROOT / digest[:16]
        # manifest.json is written LAST: its presence marks the directory as
        # fully downloaded. A previous attempt killed mid-download leaves the
        # directory without it and is re-fetched instead of half-loading.
        if not (cache_dir / "manifest.json").exists():
            cache_dir.mkdir(parents=True, exist_ok=True)
            for member in manifest["members"]:
                _fetch(f"{base}/{member['path']}", cache_dir / member["path"])
            _fetch(f"{base}/champion-info.json", cache_dir / "champion-info.json")
            (cache_dir / "manifest.json").write_bytes(raw)
        return {"digest": digest, "dir": cache_dir, "manifest": manifest}

    def _registry_dir(self) -> Path | None:
        """Refresh the registry model inline when the poll interval elapses.

        Vercel freezes the function between requests, so a background download
        thread would rarely get CPU time; the refresh runs on the request path
        instead. It is a no-op (one small manifest fetch) whenever the registry
        is unchanged, and only the first request after a promotion pays the
        multi-second member download.
        """
        if not self.registry_url:
            return None
        if time.monotonic() - self._registry_checked_at < _REGISTRY_REFRESH_SECONDS:
            return None
        self._registry_checked_at = time.monotonic()
        try:
            latest = self._registry_manifest()
            if latest is None or latest["digest"] == self._registry_digest:
                return None
            # Validation happens inside OnnxEnsemble: schema, parity status,
            # and per-member sha256 checks. A bad registry never loads.
            ensemble = OnnxEnsemble(latest["dir"], runtime=self._runtime)
        except Exception as exc:  # noqa: BLE001 - any registry failure keeps the current model
            print(f"model registry unavailable, keeping current model: {exc}", file=sys.stderr)
            self._registry_error = str(exc)
            return None
        self._registry_error = None
        self._registry_digest = latest["digest"]
        self._ensemble = ensemble
        self._active_dir = Path(ensemble.model_dir)
        self._metadata = self._metadata_for(ensemble)
        print(
            f"model registry promoted to {latest['manifest'].get('version')}",
            file=sys.stderr,
        )
        return latest["dir"]

    @staticmethod
    def _metadata_for(ensemble: OnnxEnsemble) -> ModelMetadata:
        manifest = ensemble.manifest
        champion = json.loads((Path(ensemble.model_dir) / "champion-info.json").read_text())
        validation = champion["ethiopia_metrics"]["validation"]
        development = champion["ethiopia_metrics"]["development"]
        return ModelMetadata(
            name=manifest["model"],
            version=str(manifest.get("version", "fine-v2")),
            artifactSha256=_package_sha256(manifest),
            featureContract=manifest["featureContract"],
            candidateCount=manifest["input"]["candidates"][0],
            validationTop1Within20Km=validation["within_20km_at_top1"],
            validationDiverseTop5Within20Km=validation["diverse_within_20km_at_top5"],
            developmentTop1Within20Km=development["within_20km_at_top1"],
            candidateOracleWithin20Km=validation["oracle_within_20km"],
        )

    def _load(self) -> tuple[OnnxEnsemble, ModelMetadata]:
        with self._lock:
            self._registry_dir()
            if self._ensemble is None:
                ensemble = OnnxEnsemble(self.model_dir, runtime=self._runtime)
                self._ensemble = ensemble
                self._active_dir = Path(ensemble.model_dir)
                self._metadata = self._metadata_for(ensemble)
            if self._metadata is None:
                raise RuntimeError("model metadata failed to initialize")
            return self._ensemble, self._metadata

    def registry_models(self) -> dict[str, Any]:
        """Published registry model history (newest last), for the registry endpoint.

        Forces a refresh attempt first so the endpoint reflects the live
        registry and surfaces why a promotion has not happened.
        """
        if not self.registry_url:
            return {"active": None, "history": [], "error": None}
        self._registry_dir()
        if (
            self._registry_index is None
            or time.monotonic() - self._registry_index_checked_at > _REGISTRY_REFRESH_SECONDS
        ):
            self._registry_index_checked_at = time.monotonic()
            try:
                base = self.registry_url.rsplit("/", 1)[0]
                destination = _REGISTRY_CACHE_ROOT / "registry-index.json"
                destination.parent.mkdir(parents=True, exist_ok=True)
                _fetch(f"{base}/registry-index.json", destination)
                index = json.loads(destination.read_text())
                models = index.get("models")
                if isinstance(models, list) and models and all(isinstance(m, dict) for m in models):
                    self._registry_index = models
            except Exception as exc:  # noqa: BLE001 - a missing index is not fatal
                print(f"model registry index unavailable: {exc}", file=sys.stderr)
        models = self._registry_index or []
        return {
            "active": models[-1] if models else None,
            "history": models[:-1],
            "error": self._registry_error,
        }

    def registry_status(self) -> dict[str, Any] | None:
        """Registry wiring and last refresh outcome, for the health endpoint."""
        if not self.registry_url:
            return None
        return {
            "configured": True,
            "activeVersion": self._metadata.version if self._metadata else None,
            "error": self._registry_error,
        }

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
