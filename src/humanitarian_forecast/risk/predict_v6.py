#!/usr/bin/env python3
"""Inference for the v6 coarse humanitarian early-warning bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from humanitarian_forecast.risk.model import MultiscaleTemporalRiskModel
from humanitarian_forecast.risk.predict import HumanitarianRiskPredictor, choose_device


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    logits = np.clip(logits, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-logits))


def _calibrate(probability: np.ndarray, calibration: dict[str, float] | None) -> np.ndarray:
    if not calibration:
        return probability
    clipped = np.clip(probability, 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped))
    logits = float(calibration["scale"]) * logits + float(calibration["bias"])
    return _sigmoid(logits)


def _resolve_model_dir(model_dir: Path, configured: str) -> Path:
    candidate = Path(configured)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    relative = (model_dir / candidate).resolve()
    if relative.exists():
        return relative
    raise FileNotFoundError(f"configured base model directory not found: {configured}")


class HumanitarianRiskV6Predictor:
    """Predict calibrated area-level risk and intensity from 26-channel spatial histories."""

    def __init__(self, model_dir: Path, device: str = "auto") -> None:
        self.model_dir = Path(model_dir)
        self.device = choose_device(device)
        self.ensemble = json.loads((self.model_dir / "ensemble.json").read_text(encoding="utf-8"))
        self.threshold = float(self.ensemble["threshold"])
        self.calibration = self.ensemble.get("affine_logit_calibration")
        self.weights = {
            key: float(value) for key, value in self.ensemble["probability_weights"].items()
        }

        base_dir = _resolve_model_dir(self.model_dir, self.ensemble["base_model_dir"])
        self.base = HumanitarianRiskPredictor(base_dir, str(self.device))

        cls_state = torch.load(
            self.model_dir / self.ensemble["spatial_classifier"],
            map_location="cpu",
            weights_only=False,
        )
        self.spatial_classifier = MultiscaleTemporalRiskModel(**cls_state["model_config"])
        self.spatial_classifier.load_state_dict(cls_state["model_state"])
        self.spatial_classifier.to(self.device).eval()

        intensity_state = torch.load(
            self.model_dir / self.ensemble["intensity_model"],
            map_location="cpu",
            weights_only=False,
        )
        self.intensity_model = MultiscaleTemporalRiskModel(**intensity_state["model_config"])
        self.intensity_model.load_state_dict(intensity_state["model_state"])
        self.intensity_model.to(self.device).eval()
        self.spatial_feature_dim = int(cls_state["model_config"]["feature_dim"])
        self.local_feature_dim = int(self.base.model.config["feature_dim"])
        self.conformal = self.ensemble.get("intensity_conformal_absolute_radii_log1p", {})

    def predict_array(self, spatial_features: np.ndarray, batch_size: int = 1024) -> dict[str, np.ndarray]:
        spatial_features = np.asarray(spatial_features, dtype=np.float32)
        if spatial_features.ndim == 2:
            spatial_features = spatial_features[None, ...]
        if spatial_features.ndim != 3:
            raise ValueError(
                f"expected [N,T,F] or [T,F] spatial histories, got {spatial_features.shape}"
            )
        if spatial_features.shape[-1] != self.spatial_feature_dim:
            raise ValueError(
                f"v6 expects {self.spatial_feature_dim} spatial channels, "
                f"got {spatial_features.shape[-1]}"
            )

        local_features = spatial_features[..., : self.local_feature_dim]
        base = self.base.predict_array(local_features, batch_size=batch_size)
        tensor = torch.from_numpy(spatial_features)
        cls_outputs: list[np.ndarray] = []
        intensity_outputs: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(tensor), batch_size):
                xb = tensor[start:start + batch_size].to(self.device)
                cls_outputs.append(self.spatial_classifier(xb).cpu().numpy())
                intensity_outputs.append(self.intensity_model(xb).cpu().numpy())
        cls = np.concatenate(cls_outputs)
        intensity = np.concatenate(intensity_outputs)[:, 0].astype(np.float64)
        spatial_probability = _sigmoid(cls[:, 1])
        raw_probability = (
            self.weights["base_v5"] * base["escalation_probability"]
            + self.weights["spatial"] * spatial_probability
        )
        probability = _calibrate(raw_probability, self.calibration)

        result: dict[str, np.ndarray] = {
            "intensity_log1p": intensity,
            "escalation_probability": probability,
            "risk_flag": probability >= self.threshold,
        }
        for name, radius in self.conformal.items():
            r = float(radius)
            result[f"intensity_{name}_lower_log1p"] = np.maximum(0.0, intensity - r)
            result[f"intensity_{name}_upper_log1p"] = intensity + r
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("models/risk/ethiopia/v6"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--key", default="x")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    data = np.load(args.input)[args.key] if args.input.suffix == ".npz" else np.load(args.input)
    if not 0 <= args.index < len(data):
        raise IndexError(f"index {args.index} outside 0..{len(data)-1}")
    predictor = HumanitarianRiskV6Predictor(args.model_dir, args.device)
    result = predictor.predict_array(data[args.index])
    payload = {
        "intensity_log1p": float(result["intensity_log1p"][0]),
        "escalation_probability": float(result["escalation_probability"][0]),
        "risk_flag": bool(result["risk_flag"][0]),
        "uncertainty": {
            name: float(values[0])
            for name, values in result.items()
            if name.startswith("intensity_p")
        },
        "warning": (
            "Coarse humanitarian early-warning signal only; not a verified front, tactical "
            "forecast, route recommendation, unit-position estimate, or evacuation order."
        ),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
