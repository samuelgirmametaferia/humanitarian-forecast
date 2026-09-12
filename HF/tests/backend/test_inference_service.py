from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from api.hf_api.inference import InferenceService
from api.hf_api.schemas import FeaturePayload, ForecastSnapshot, SourceHealth

MODEL_DIR = Path(__file__).parents[2] / "api" / "models" / "ethiopia_serving"


def feature_payload() -> FeaturePayload:
    return FeaturePayload.model_validate(
        {
            "schemaVersion": "feature-contract.v1",
            "country": "Ethiopia",
            "conflictId": "unseen-conflict",
            "conflictLabel": "Contract test conflict",
            "anchor": {"latitude": 9.0, "longitude": 39.0},
            "observationCutoff": "2026-09-11T00:00:00Z",
            "horizonDays": 7,
            "events": [[1.0] + [0.0] * 18 for _ in range(16)],
            "candidates": [
                {
                    "id": f"candidate-{index}",
                    "valid": index < 8,
                    "eastKm": float(index * 20),
                    "northKm": 0.0,
                    "features": [0.0] * 37,
                }
                for index in range(64)
            ],
        }
    )


class FakeSession:
    def __init__(self, path: str, providers: list[str]) -> None:
        del providers
        self.offset = 0.05 if path.endswith("member-1.onnx") else 0.0

    def run(
        self,
        outputs: list[str],
        feeds: dict[str, np.ndarray[Any, Any]],
    ) -> list[np.ndarray[Any, Any]]:
        del outputs, feeds
        logits = np.linspace(2.0, -2.0, 64, dtype=np.float32) + self.offset
        return [logits[np.newaxis]]


def test_inference_service_builds_honest_production_snapshot() -> None:
    runtime = SimpleNamespace(InferenceSession=FakeSession)
    service = InferenceService(MODEL_DIR, runtime=runtime)
    source = SourceHealth(
        id="reliefweb",
        label="ReliefWeb",
        status="healthy",
        lastSuccessfulAt=datetime(2026, 9, 11, tzinfo=UTC),
        articlesProcessed=4,
    )
    snapshot = service.predict(
        feature_payload(),
        run_id="run-20260911",
        generated_at=datetime(2026, 9, 11, 1, tzinfo=UTC),
        source_health=[source],
    )

    ForecastSnapshot.model_validate(snapshot.model_dump())
    assert snapshot.id == "forecast:run-20260911"
    assert snapshot.mode == "production"
    assert len(snapshot.zones) == 5
    assert snapshot.zones[0].label == "Candidate zone 1"
    assert snapshot.zones[0].uncertainty is None
    assert snapshot.zones[0].populationExposureBand is None
    assert snapshot.observations == []
    assert snapshot.model.name == "theswarm_fine_v2"
    assert snapshot.model.artifactSha256 != "demo"
    assert len(snapshot.model.artifactSha256) == 64


def test_packaged_metadata_comes_from_validated_artifacts() -> None:
    service = InferenceService(MODEL_DIR, runtime=SimpleNamespace(InferenceSession=FakeSession))
    metadata = service.metadata()
    manifest = json.loads((MODEL_DIR / "manifest.json").read_text())

    assert manifest["status"] == "parity-validated"
    assert metadata.featureContract == "feature-contract.v1"
    assert metadata.candidateCount == 64
    assert metadata.validationTop1Within20Km == 0.134
    assert metadata.validationDiverseTop5Within20Km == 0.33
    assert metadata.developmentTop1Within20Km == 0.042
    assert metadata.candidateOracleWithin20Km == 0.673


def test_registry_promotes_new_version_and_falls_back_on_failure(tmp_path: Any) -> None:
    runtime = SimpleNamespace(InferenceSession=FakeSession)

    # A registry "release": the serving package with a new version stamp.
    registry = tmp_path / "registry"
    registry.mkdir()
    for name in ("member-0.onnx", "member-1.onnx", "champion-info.json"):
        (registry / name).write_bytes((MODEL_DIR / name).read_bytes())
    manifest = json.loads((MODEL_DIR / "manifest.json").read_text())
    manifest["version"] = "retrain-test-1"
    (registry / "manifest.json").write_text(json.dumps(manifest))

    service = InferenceService(
        MODEL_DIR,
        registry_url=registry.as_uri() + "/manifest.json",
        runtime=runtime,
    )
    promoted = service.metadata()
    assert promoted.version == "retrain-test-1"
    assert promoted.artifactSha256 != "demo"

    # An unusable registry (missing members) keeps the current model.
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "manifest.json").write_text(json.dumps({"schema": "ethiopia-serving-package.v1"}))
    fallback = InferenceService(
        MODEL_DIR,
        registry_url=broken.as_uri() + "/manifest.json",
        runtime=runtime,
    )
    assert fallback.metadata().version == "fine-v2"


def test_predict_forces_registry_refresh_inside_poll_window(tmp_path: Any) -> None:
    runtime = SimpleNamespace(InferenceSession=FakeSession)

    registry = tmp_path / "registry"
    registry.mkdir()
    for member in ("member-0.onnx", "member-1.onnx", "champion-info.json"):
        (registry / member).write_bytes((MODEL_DIR / member).read_bytes())
    manifest = json.loads((MODEL_DIR / "manifest.json").read_text())
    manifest_path = registry / "manifest.json"

    # v1 is live; the service promotes and starts its poll-window clock.
    manifest["version"] = "retrain-window-1"
    manifest_path.write_text(json.dumps(manifest))
    service = InferenceService(
        MODEL_DIR,
        registry_url=manifest_path.as_uri(),
        runtime=runtime,
    )
    assert service.metadata().version == "retrain-window-1"

    # A promotion lands while the instance is still inside its poll window.
    # A published forecast must still carry the new champion.
    manifest["version"] = "retrain-window-2"
    manifest_path.write_text(json.dumps(manifest))
    snapshot = service.predict(
        feature_payload(),
        run_id="run-20260911-window",
        generated_at=datetime(2026, 9, 11, 1, tzinfo=UTC),
        source_health=[
            SourceHealth(
                id="reliefweb",
                label="ReliefWeb",
                status="healthy",
                lastSuccessfulAt=None,
                articlesProcessed=4,
            )
        ],
    )
    assert snapshot.model.version == "retrain-window-2"
