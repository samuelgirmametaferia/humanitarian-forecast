from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from lightgbm import Booster

from humanitarian_forecast.location.training.train_geo_lambdarank_v2 import (
    _flatten_inference_features,
    _scores_to_matrix,
    _softmax,
)

SCALE_KM = 1000.0


def _cap_residual(residual: np.ndarray, cap_km: float) -> np.ndarray:
    cap = float(cap_km) / SCALE_KM
    norm = np.linalg.norm(residual, axis=-1, keepdims=True)
    return residual * np.minimum(1.0, cap / np.maximum(norm, 1e-9))


def _matrix(flat: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.zeros(valid.shape, dtype=np.float32)
    out[valid] = flat.astype(np.float32)
    return out


@dataclass(frozen=True)
class ChildSupportResult:
    probability: np.ndarray
    coordinates: np.ndarray
    valid: np.ndarray
    parent_index: np.ndarray
    is_child: np.ndarray
    parent_probability: np.ndarray
    gate_probability: np.ndarray | None


class CandidateChildSupport:
    """Inference-only parent-preserving local support expansion.

    No target/outcome coordinate is accepted by this API.  The coarse regional
    ranker remains frozen; residual models can spawn a nearby child hypothesis
    and split parent probability mass according to a validation-frozen policy.
    """

    def __init__(
        self,
        coarse_model: Path,
        coarse_config: Path,
        refiner_dir: Path,
        policy_path: Path,
        gate_model: Path | None = None,
    ) -> None:
        self.coarse = Booster(model_file=str(coarse_model))
        self.east = Booster(model_file=str(refiner_dir / "east_refiner.txt"))
        self.north = Booster(model_file=str(refiner_dir / "north_refiner.txt"))
        self.gate = Booster(model_file=str(gate_model)) if gate_model else None
        config = json.loads(coarse_config.read_text())
        self.iteration = int(config.get("selected_iteration", self.coarse.num_trees()))
        self.temperature = float(config.get("probability_temperature", 0.2))
        self.policy = json.loads(policy_path.read_text())

    @staticmethod
    def _gate_features(
        base: np.ndarray,
        residual: np.ndarray,
        raw: np.ndarray,
        probability: np.ndarray,
        valid: np.ndarray,
    ) -> np.ndarray:
        entropy = -(probability * np.log(np.maximum(probability, 1e-12))).sum(axis=1)
        order = np.argsort(-raw, axis=1)
        rank = np.empty_like(order)
        rank[np.arange(len(order))[:, None], order] = np.arange(order.shape[1])[None, :]
        residual_norm = np.linalg.norm(residual, axis=-1) * SCALE_KM
        extra = np.stack(
            [
                residual[..., 0],
                residual[..., 1],
                residual_norm / 200.0,
                raw,
                probability,
                np.repeat(entropy[:, None], probability.shape[1], axis=1),
                rank / np.maximum(valid.sum(axis=1, keepdims=True) - 1, 1),
            ],
            axis=-1,
        ).astype(np.float32)
        return np.concatenate([base, extra[valid]], axis=1).astype(np.float32, copy=False)

    def predict(
        self,
        events: np.ndarray,
        candidate_features: np.ndarray,
        candidate_coordinates: np.ndarray,
        candidate_valid: np.ndarray,
    ) -> ChildSupportResult:
        events = np.asarray(events, dtype=np.float32)
        candidate_features = np.asarray(candidate_features, dtype=np.float32)
        coordinates = np.asarray(candidate_coordinates, dtype=np.float32)
        valid = np.asarray(candidate_valid, dtype=bool)
        if events.ndim != 3 or candidate_features.ndim != 3 or coordinates.ndim != 3:
            raise ValueError("batched events/candidate tensors are required")
        if valid.shape != coordinates.shape[:2]:
            raise ValueError("candidate_valid shape does not match coordinates")

        flat, _ = _flatten_inference_features(events, candidate_features, valid)
        raw = _scores_to_matrix(self.coarse.predict(flat, num_iteration=self.iteration), valid)
        probability = _softmax(raw, valid, self.temperature).astype(np.float32)
        residual = np.zeros((*valid.shape, 2), dtype=np.float32)
        residual[..., 0][valid] = self.east.predict(flat).astype(np.float32)
        residual[..., 1][valid] = self.north.predict(flat).astype(np.float32)

        gate_probability = None
        if self.gate is not None:
            gate_flat = self.gate.predict(self._gate_features(flat, residual, raw, probability, valid))
            gate_probability = _matrix(gate_flat, valid)

        cap_km = float(self.policy["cap_km"])
        alpha = float(self.policy["alpha"])
        child_fraction = float(self.policy["child_mass_fraction"])
        parent_max = float(
            self.policy.get(
                "parent_probability_max",
                self.policy.get("spawn_if_parent_probability_lte", 1.0),
            )
        )
        eligible = valid & (probability <= parent_max)
        if gate_probability is not None:
            threshold = float(self.policy.get("gate_threshold", 0.5))
            eligible &= gate_probability >= threshold

        refined = coordinates + alpha * _cap_residual(residual, cap_km)
        parent_probability = np.where(eligible, probability * (1.0 - child_fraction), probability)
        child_probability = np.where(eligible, probability * child_fraction, 0.0)
        expanded_probability = np.concatenate([parent_probability, child_probability], axis=1)
        expanded_coordinates = np.concatenate([coordinates, refined], axis=1)
        expanded_valid = np.concatenate([valid, eligible], axis=1)
        k = valid.shape[1]
        parent_index = np.tile(np.arange(k, dtype=np.int32), (len(valid), 2))
        is_child = np.concatenate(
            [np.zeros(valid.shape, dtype=bool), np.ones(valid.shape, dtype=bool)],
            axis=1,
        )
        return ChildSupportResult(
            probability=expanded_probability,
            coordinates=expanded_coordinates,
            valid=expanded_valid,
            parent_index=parent_index,
            is_child=is_child,
            parent_probability=probability,
            gate_probability=gate_probability,
        )
