from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .feature_contract import FeatureContract

EARTH_KM = 111.32


@dataclass(frozen=True)
class CandidatePrediction:
    id: str
    index: int
    rank: int
    probability: float
    latitude: float
    longitude: float
    distance_from_anchor_km: float
    site_type: str
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
        self.manifest = json.loads((self.model_dir / "manifest.json").read_text())
        if self.manifest.get("schema") != "ethiopia-serving-package.v1":
            raise RuntimeError("unsupported serving package")
        if runtime is None:
            try:
                import onnxruntime as runtime_module
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
                masked,
                coordinates_km,
                limit,
                float(self.selection.get("spread_km", 30.0)),
            )
        else:
            order = [int(index) for index in np.argsort(-masked) if valid[index]][:limit]
        predictions = tuple(
            self._candidate(contract, index, rank, float(probabilities[index]), advisory_radius_km)
            for rank, index in enumerate(order, 1)
        )
        return EnsemblePrediction(
            probabilities,
            member_logits,
            predictions,
            country == 0 or conflict == 0,
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
