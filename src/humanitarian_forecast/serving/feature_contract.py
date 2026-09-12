from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

EVENT_COUNT = 16
EVENT_DIM = 19
CANDIDATE_COUNT = 64
CANDIDATE_DIM = 37


@dataclass(frozen=True)
class Position:
    latitude: float
    longitude: float


@dataclass(frozen=True)
class CandidateInput:
    id: str
    valid: bool
    east_km: float
    north_km: float
    features: tuple[float, ...]


@dataclass(frozen=True)
class FeatureContract:
    country: str
    conflict_id: str
    conflict_label: str
    anchor: Position
    observation_cutoff: datetime
    horizon_days: int
    events: np.ndarray
    candidates: tuple[CandidateInput, ...]

    @property
    def candidate_features(self) -> np.ndarray:
        return np.asarray([candidate.features for candidate in self.candidates], dtype=np.float32)

    @property
    def candidate_coordinates(self) -> np.ndarray:
        return np.asarray(
            [
                (candidate.east_km / 1000.0, candidate.north_km / 1000.0)
                for candidate in self.candidates
            ],
            dtype=np.float32,
        )

    @property
    def candidate_valid(self) -> np.ndarray:
        return np.asarray([candidate.valid for candidate in self.candidates], dtype=np.bool_)


class ContractError(ValueError):
    pass


def _object(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{name} must be an object")
    return value


def _keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    missing = expected - value.keys()
    extra = value.keys() - expected
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing {sorted(missing)}")
        if extra:
            details.append(f"unexpected {sorted(extra)}")
        raise ContractError(f"{name}: {', '.join(details)}")


def _finite(value: Any, name: str, low: float | None = None, high: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{name} must be a number")
    result = float(value)
    outside_range = low is not None and result < low or high is not None and result > high
    if not math.isfinite(result) or outside_range:
        raise ContractError(f"{name} is outside its allowed range")
    return result


def _string(value: Any, name: str, minimum: int = 0, maximum: int = 100) -> str:
    if not isinstance(value, str) or not minimum <= len(value) <= maximum:
        raise ContractError(f"{name} must be a string between {minimum} and {maximum} characters")
    return value


def _matrix(value: Any, rows: int, columns: int, name: str) -> np.ndarray:
    if not isinstance(value, list) or len(value) != rows:
        raise ContractError(f"{name} must contain exactly {rows} rows")
    parsed = []
    for row_index, row in enumerate(value):
        if not isinstance(row, list) or len(row) != columns:
            raise ContractError(f"{name}[{row_index}] must contain exactly {columns} values")
        parsed.append([_finite(item, f"{name}[{row_index}]") for item in row])
    return np.asarray(parsed, dtype=np.float32)


def load_feature_contract(source: str | bytes | Path | Mapping[str, Any]) -> FeatureContract:
    if isinstance(source, Path):
        raw: Any = json.loads(source.read_text())
    elif isinstance(source, (bytes, str)):
        raw = json.loads(source)
    else:
        raw = source
    payload = _object(raw, "feature contract")
    required = {
        "schemaVersion",
        "country",
        "conflictId",
        "conflictLabel",
        "anchor",
        "observationCutoff",
        "horizonDays",
        "events",
        "candidates",
    }
    _keys(payload, required, "feature contract")
    if payload["schemaVersion"] != "feature-contract.v1" or payload["country"] != "Ethiopia":
        raise ContractError("feature contract must be feature-contract.v1 for Ethiopia")
    anchor = _object(payload["anchor"], "anchor")
    _keys(anchor, {"latitude", "longitude"}, "anchor")
    try:
        cutoff_value = _string(payload["observationCutoff"], "observationCutoff")
        cutoff = datetime.fromisoformat(cutoff_value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ContractError("observationCutoff must be an ISO 8601 timestamp") from exc
    horizon = payload["horizonDays"]
    if isinstance(horizon, bool) or not isinstance(horizon, int) or not 1 <= horizon <= 30:
        raise ContractError("horizonDays must be an integer between 1 and 30")
    events = _matrix(payload["events"], EVENT_COUNT, EVENT_DIM, "events")
    items = payload["candidates"]
    if not isinstance(items, list) or len(items) != CANDIDATE_COUNT:
        raise ContractError(f"candidates must contain exactly {CANDIDATE_COUNT} entries")
    candidates = []
    candidate_keys = {"id", "valid", "eastKm", "northKm", "features"}
    for index, item in enumerate(items):
        candidate = _object(item, f"candidates[{index}]")
        _keys(candidate, candidate_keys, f"candidates[{index}]")
        if not isinstance(candidate["valid"], bool):
            raise ContractError(f"candidates[{index}].valid must be boolean")
        features = _matrix(
            [candidate["features"]],
            1,
            CANDIDATE_DIM,
            f"candidates[{index}].features",
        )[0]
        candidates.append(
            CandidateInput(
                id=_string(candidate["id"], f"candidates[{index}].id", maximum=80),
                valid=candidate["valid"],
                east_km=_finite(candidate["eastKm"], f"candidates[{index}].eastKm", -2000, 2000),
                north_km=_finite(candidate["northKm"], f"candidates[{index}].northKm", -2000, 2000),
                features=tuple(float(value) for value in features),
            )
        )
    if not any(candidate.valid for candidate in candidates):
        raise ContractError("at least one candidate must be valid")
    return FeatureContract(
        country="Ethiopia",
        conflict_id=_string(payload["conflictId"], "conflictId"),
        conflict_label=_string(payload["conflictLabel"], "conflictLabel", minimum=1),
        anchor=Position(
            latitude=_finite(anchor["latitude"], "anchor.latitude", -90, 90),
            longitude=_finite(anchor["longitude"], "anchor.longitude", -180, 180),
        ),
        observation_cutoff=cutoff,
        horizon_days=horizon,
        events=events,
        candidates=tuple(candidates),
    )
