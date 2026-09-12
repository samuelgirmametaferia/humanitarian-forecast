from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from humanitarian_forecast.core.model_store import discover_models
from humanitarian_forecast.core.paths import PATHS

Predictor = Callable[[Mapping[str, Any]], Mapping[str, Any]]
_REJECTED_MARKERS = ("reject", "archive", "failed", "superseded")
_MODEL_SUFFIXES = {".pt", ".onnx", ".npz", ".json"}


@dataclass(frozen=True)
class RegisteredModel:
    id: str
    directory: Path
    subsystem: str
    version: str
    status: str
    description: str
    artifacts: tuple[str, ...]
    geospatial: bool
    eligible_for_fresh_input: bool
    adapter_registered: bool

    def public_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "subsystem": self.subsystem,
            "version": self.version,
            "status": self.status,
            "description": self.description,
            "artifacts": list(self.artifacts),
            "geospatial": self.geospatial,
            "eligibleForFreshInput": self.eligible_for_fresh_input,
            "adapterRegistered": self.adapter_registered,
        }


class ModelRegistry:
    """Catalog model artifacts and execute validated adapters on one fresh payload.

    Discovery is deliberately separate from execution: an old checkpoint can remain
    inspectable without being allowed into a live ensemble. Time-split evaluation is
    model evidence, not an inference cutoff; accepted models may score newer inputs.
    """

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or PATHS.models
        self._adapters: dict[str, Predictor] = {}

    def register_adapter(self, model_id: str, predictor: Predictor) -> None:
        if model_id in self._adapters:
            raise ValueError(f"adapter already registered: {model_id}")
        self._adapters[model_id] = predictor

    def models(self, *, geospatial_only: bool = False) -> list[RegisteredModel]:
        records = [self._record(directory, info) for directory, info in discover_models(self.root)]
        return [record for record in records if record.geospatial or not geospatial_only]

    def ready_models(self) -> list[RegisteredModel]:
        return [model for model in self.models(geospatial_only=True) if model.eligible_for_fresh_input and model.adapter_registered]

    def predict_all(self, payload: Mapping[str, Any], model_ids: Iterable[str] | None = None) -> dict[str, Mapping[str, Any]]:
        ready = {model.id: model for model in self.ready_models()}
        selected = list(model_ids) if model_ids is not None else sorted(ready)
        unknown = [model_id for model_id in selected if model_id not in ready]
        if unknown:
            raise ValueError(f"models are not ready for fresh inference: {', '.join(unknown)}")
        return {
            model_id: {
                "model": ready[model_id].public_dict(),
                "prediction": dict(self._adapters[model_id](payload)),
            }
            for model_id in selected
        }

    def manifest(self) -> dict[str, Any]:
        models = self.models()
        return {
            "schema": "humanitarian-forecast-model-registry/v1",
            "modelCount": len(models),
            "geospatialModelCount": sum(model.geospatial for model in models),
            "freshInputReadyCount": len(self.ready_models()),
            "models": [model.public_dict() for model in models],
        }

    def _record(self, directory: Path, info: Mapping[str, Any]) -> RegisteredModel:
        subsystem = str(info.get("subsystem") or directory.parent.relative_to(self.root))
        version = str(info.get("version") or directory.name)
        status = str(info.get("status", "unknown"))
        artifacts = tuple(str(item) for item in info.get("artifacts", ()))
        model_id = directory.relative_to(self.root).as_posix()
        geospatial = subsystem.startswith(("location/", "risk/"))
        has_model_artifact = any(Path(item).suffix.lower() in _MODEL_SUFFIXES for item in artifacts)
        accepted = not any(marker in status.lower() for marker in _REJECTED_MARKERS)
        return RegisteredModel(
            id=model_id,
            directory=directory,
            subsystem=subsystem,
            version=version,
            status=status,
            description=str(info.get("description", "")),
            artifacts=artifacts,
            geospatial=geospatial,
            eligible_for_fresh_input=geospatial and accepted and has_model_artifact,
            adapter_registered=model_id in self._adapters,
        )
