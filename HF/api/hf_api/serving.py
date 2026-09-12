from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np

EVENT_COUNT = 16
EVENT_DIM = 19
CANDIDATE_COUNT = 64
CANDIDATE_DIM = 37
EARTH_KM = 111.32
SiteType = Literal["conflict-frequency site", "recent cross-conflict event"]


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


@dataclass(frozen=True)
class CandidatePrediction:
    id: str
    index: int
    rank: int
    probability: float
    latitude: float
    longitude: float
    distance_from_anchor_km: float
    site_type: SiteType
    days_since_last_event_at_site: int
    elevation_m: int
    ruggedness_m: float
    advisory_radius_km: float


@dataclass(frozen=True)
class EnsemblePrediction:
    probabilities: np.ndarray
    member_logits: tuple[np.ndarray, ...]
    candidates: tuple[CandidatePrediction, ...]
    unseen_identity: bool


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


def load_feature_contract(source: Mapping[str, Any]) -> FeatureContract:
    payload = _object(source, "feature contract")
    required = {
        "schemaVersion", "country", "conflictId", "conflictLabel", "anchor",
        "observationCutoff", "horizonDays", "events", "candidates",
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
            [candidate["features"]], 1, CANDIDATE_DIM, f"candidates[{index}].features",
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


def greedy_order(
    probabilities: np.ndarray,
    coordinates_km: np.ndarray,
    k: int,
    spread_km: float,
) -> list[int]:
    picked: list[int] = []
    for _ in range(k):
        best_index, best_score = -1, -1.0
        for index, probability in enumerate(probabilities):
            if probability <= 0 or index in picked:
                continue
            score = float(probability)
            if picked:
                minimum = min(
                    math.hypot(
                        coordinates_km[index, 0] - coordinates_km[selected, 0],
                        coordinates_km[index, 1] - coordinates_km[selected, 1],
                    )
                    for selected in picked
                )
                score *= min(1.0, minimum / spread_km)
            if score > best_score:
                best_index, best_score = index, score
        if best_index < 0:
            break
        picked.append(best_index)
    remaining = [
        int(index)
        for index in np.argsort(-probabilities)
        if index not in picked and probabilities[index] > 0
    ]
    return (picked + remaining)[:k]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _softmax(logits: np.ndarray, valid: np.ndarray) -> np.ndarray:
    scores = np.where(valid, logits, -np.inf)
    maximum = np.max(scores)
    exponents = np.where(valid, np.exp(scores - maximum), 0.0)
    total = float(exponents.sum())
    if not math.isfinite(total) or total <= 0:
        raise RuntimeError("ranker produced no finite probability mass")
    return exponents / total


class OnnxEnsemble:
    def __init__(self, model_dir: str | Path, runtime: Any | None = None) -> None:
        self.model_dir = Path(model_dir)
        self.manifest: dict[str, Any] = json.loads((self.model_dir / "manifest.json").read_text())
        if self.manifest.get("schema") != "ethiopia-serving-package.v1":
            raise RuntimeError("unsupported serving package")
        if self.manifest.get("status") != "parity-validated":
            raise RuntimeError("serving package has not passed parity validation")
        if runtime is None:
            try:
                import onnxruntime as runtime_module  # type: ignore[import-untyped]
            except ImportError as exc:
                raise RuntimeError("onnxruntime is required for production inference") from exc
            runtime = runtime_module
        self.sessions = []
        for member in self.manifest["members"]:
            path = self.model_dir / member["path"]
            if _sha256(path) != member["sha256"]:
                raise RuntimeError(f"model checksum mismatch: {path.name}")
            self.sessions.append(
                runtime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            )
        self.country_map = self.manifest["countryMap"]
        self.conflict_map = self.manifest["conflictMap"]
        self.selection = self.manifest["selection"]

    def predict(
        self,
        contract: FeatureContract,
        top_k: int | None = None,
        advisory_radius_km: float = 20.0,
    ) -> EnsemblePrediction:
        country = int(self.country_map.get(contract.country, 0))
        conflict = int(self.conflict_map.get(contract.conflict_id, 0))
        valid = contract.candidate_valid
        feeds = {
            "events": contract.events[np.newaxis].astype(np.float32),
            "candidates": contract.candidate_features[np.newaxis].astype(np.float32),
            "valid": valid[np.newaxis],
            "country": np.asarray([country], dtype=np.int64),
            "conflict": np.asarray([conflict], dtype=np.int64),
        }
        member_logits = tuple(
            np.asarray(session.run(["logits"], feeds)[0][0]) for session in self.sessions
        )
        probabilities = np.mean([_softmax(logits, valid) for logits in member_logits], axis=0)
        masked = np.where(valid, probabilities, 0.0)
        coordinates_km = contract.candidate_coordinates * 1000.0
        limit = top_k or int(self.selection.get("top_k", 5))
        if self.selection.get("method") == "greedy_spread":
            order = greedy_order(
                masked, coordinates_km, limit, float(self.selection.get("spread_km", 30.0)),
            )
        else:
            order = [int(index) for index in np.argsort(-masked) if valid[index]][:limit]
        predictions = tuple(
            self._candidate(contract, index, rank, float(probabilities[index]), advisory_radius_km)
            for rank, index in enumerate(order, 1)
        )
        return EnsemblePrediction(
            probabilities=probabilities,
            member_logits=member_logits,
            candidates=predictions,
            unseen_identity=country == 0 or conflict == 0,
        )


    @staticmethod
    def _candidate(
        contract: FeatureContract,
        index: int,
        rank: int,
        probability: float,
        advisory_radius_km: float,
    ) -> CandidatePrediction:
        candidate = contract.candidates[index]
        latitude = contract.anchor.latitude + candidate.north_km / EARTH_KM
        longitude = contract.anchor.longitude + candidate.east_km / (
            EARTH_KM * max(0.1, math.cos(math.radians(contract.anchor.latitude)))
        )
        features: Sequence[float] = candidate.features
        return CandidatePrediction(
            id=candidate.id,
            index=index,
            rank=rank,
            probability=probability,
            latitude=latitude,
            longitude=longitude,
            distance_from_anchor_km=math.hypot(candidate.east_km, candidate.north_km),
            site_type=(
                "recent cross-conflict event" if features[27] > 0.5 else "conflict-frequency site"
            ),
            days_since_last_event_at_site=int(round(math.expm1(features[31] * 8))),
            elevation_m=int(round(features[35] * 1000)),
            ruggedness_m=features[36] * 1000,
            advisory_radius_km=advisory_radius_km,
        )
