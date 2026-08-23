#!/usr/bin/env python3
"""Inference for versioned coarse humanitarian-risk challenger artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch

from humanitarian_forecast.risk.features import engineered_temporal_features
from humanitarian_forecast.risk.model import MultiscaleTemporalRiskModel, TemporalRiskTransformer


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(requested)


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    logits = np.clip(logits, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-logits))


def _calibrate(probs: np.ndarray, calibration: dict[str, float] | None) -> np.ndarray:
    if not calibration:
        return probs
    clipped = np.clip(probs, 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped))
    logits = float(calibration["scale"]) * logits + float(calibration["bias"])
    return _sigmoid(logits)


def _resolve_reference(model_dir: Path, configured: str) -> Path:
    candidate = Path(configured)
    if candidate.exists():
        return candidate
    # model_dir is normally <repo>/models/risk/ethiopia/vN.
    if len(model_dir.parents) >= 4:
        candidate = model_dir.parents[3] / configured
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"reference checkpoint not found: {configured}")


class HumanitarianRiskPredictor:
    """Self-contained coarse risk predictor for a versioned model directory."""

    def __init__(self, model_dir: Path, device: str = "auto") -> None:
        self.model_dir = Path(model_dir)
        self.device = choose_device(device)
        checkpoint = torch.load(
            self.model_dir / "multiscale_best.pt", map_location="cpu", weights_only=False
        )
        self.model = MultiscaleTemporalRiskModel(**checkpoint["model_config"])
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.to(self.device).eval()

        self.ensemble = json.loads((self.model_dir / "ensemble.json").read_text(encoding="utf-8"))
        self.weights = {name: float(value) for name, value in self.ensemble["weights"].items()}
        self.threshold = float(self.ensemble["threshold"])
        self.calibration = self.ensemble.get("affine_logit_calibration")

        self.reference: TemporalRiskTransformer | None = None
        if self.weights.get("reference", 0.0) > 0:
            configured = self.ensemble.get("reference_checkpoint")
            if not configured:
                raise ValueError("reference ensemble weight is non-zero but no checkpoint is configured")
            reference_state = torch.load(
                _resolve_reference(self.model_dir, configured), map_location="cpu", weights_only=False
            )
            self.reference = TemporalRiskTransformer(**reference_state["model_config"])
            self.reference.load_state_dict(reference_state["model_state"])
            self.reference.to(self.device).eval()

        self.tree = None
        if self.weights.get("tree", 0.0) > 0:
            self.tree = joblib.load(self.model_dir / "tree_classifier.joblib")

    def predict_array(self, features: np.ndarray, batch_size: int = 1024) -> dict[str, np.ndarray]:
        features = np.asarray(features, dtype=np.float32)
        if features.ndim == 2:
            features = features[None, ...]
        if features.ndim != 3:
            raise ValueError(f"expected [N,T,F] or [T,F], got shape={features.shape}")

        tensor = torch.from_numpy(features)
        multiscale_outputs: list[np.ndarray] = []
        reference_outputs: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(tensor), batch_size):
                xb = tensor[start:start + batch_size].to(self.device)
                multiscale_outputs.append(self.model(xb).cpu().numpy())
                if self.reference is not None:
                    reference_outputs.append(self.reference(xb).cpu().numpy())
        multiscale = np.concatenate(multiscale_outputs)
        expert_probs: dict[str, np.ndarray] = {
            "multiscale": _sigmoid(multiscale[:, 1]),
        }
        if reference_outputs:
            reference = np.concatenate(reference_outputs)
            expert_probs["reference"] = _sigmoid(reference[:, 1])
        if self.tree is not None:
            engineered = engineered_temporal_features(features)
            expert_probs["tree"] = self.tree.predict_proba(engineered)[:, 1]

        probability = np.zeros(len(features), dtype=np.float64)
        for name, weight in self.weights.items():
            if weight == 0:
                continue
            if name not in expert_probs:
                raise RuntimeError(f"missing configured expert: {name}")
            probability += weight * expert_probs[name]
        probability = _calibrate(probability, self.calibration)
        return {
            "intensity_log1p": multiscale[:, 0].astype(np.float64),
            "escalation_probability": probability,
            "risk_flag": probability >= self.threshold,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="NPY or NPZ containing [N,T,F] feature histories")
    parser.add_argument("--key", default="x", help="NPZ array key (default: x)")
    parser.add_argument("--index", type=int, default=0, help="Example index to print")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.input.suffix == ".npz":
        data = np.load(args.input)[args.key]
    else:
        data = np.load(args.input)
    if not 0 <= args.index < len(data):
        raise IndexError(f"index {args.index} outside 0..{len(data)-1}")

    predictor = HumanitarianRiskPredictor(args.model_dir, args.device)
    result = predictor.predict_array(data[args.index])
    print(json.dumps({
        "intensity_log1p": float(result["intensity_log1p"][0]),
        "escalation_probability": float(result["escalation_probability"][0]),
        "risk_flag": bool(result["risk_flag"][0]),
        "warning": (
            "Coarse humanitarian early-warning signal only; not a verified front, "
            "tactical forecast, route recommendation, or evacuation order."
        ),
    }, indent=2))


if __name__ == "__main__":
    main()
