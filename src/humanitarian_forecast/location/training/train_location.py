#!/usr/bin/env python3
"""Train and calibrate a coarse probabilistic next-location model on MPS."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from humanitarian_forecast.location.models.direct import ProbabilisticLocationTransformer, gaussian_location_loss

SCALE_KM = 1000.0


def device_for(name: str) -> torch.device:
    if name == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(name)


def collect(model, loader, device):
    centers, sigmas, targets = [], [], []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            center, sigma = model(x.to(device))
            centers.append(center.cpu()); sigmas.append(sigma.cpu()); targets.append(y)
    return torch.cat(centers), torch.cat(sigmas), torch.cat(targets)


def summarize(center, sigma, target, multiplier: float, radius_floor_km: float) -> dict[str, float]:
    error_km = torch.linalg.vector_norm(center - target, dim=1) * SCALE_KM
    radius_km = torch.maximum(sigma * SCALE_KM * multiplier, torch.tensor(radius_floor_km))
    return {
        "samples": len(error_km),
        "mean_error_km": float(error_km.mean()),
        "median_error_km": float(error_km.median()),
        "p90_error_km": float(error_km.quantile(0.90)),
        "mean_radius_km": float(radius_km.mean()),
        "median_radius_km": float(radius_km.median()),
        "circle_coverage": float((error_km <= radius_km).float().mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("models/location/direct/v2"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--target-coverage", type=float, default=0.80)
    parser.add_argument("--radius-floor-km", type=float, default=50.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260810)
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = device_for(args.device)
    print(f"device={device}")

    data = np.load(args.data)
    x = torch.from_numpy(data["x"]).float(); y = torch.from_numpy(data["y"]).float()
    train_end = int(len(x) * 0.70); validation_end = int(len(x) * 0.85)
    train = TensorDataset(x[:train_end], y[:train_end])
    validation = TensorDataset(x[train_end:validation_end], y[train_end:validation_end])
    test = TensorDataset(x[validation_end:], y[validation_end:])
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation, batch_size=args.batch_size)
    test_loader = DataLoader(test, batch_size=args.batch_size)

    model = ProbabilisticLocationTransformer(
        feature_dim=x.shape[-1], sequence_length=x.shape[1]
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train(); total = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            center, sigma = model(xb)
            loss = gaussian_location_loss(center, sigma, yb)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); total += loss.item() * len(xb)
        vc, vs, vy = collect(model, validation_loader, device)
        val_loss = float(gaussian_location_loss(vc, vs, vy))
        print(f"epoch={epoch:02d} train_nll={total/len(train):.4f} val_nll={val_loss:.4f}")
        if val_loss < best:
            best = val_loss
            torch.save({
                "model_config": model.config,
                "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "validation_nll": val_loss, "seed": args.seed,
                "data_policy": "UCDP+ReliefWeb only; Telegram excluded",
            }, args.output_dir / "location_best.pt")

    state = torch.load(args.output_dir / "location_best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state"]); model.to(device)
    vc, vs, vy = collect(model, validation_loader, device)
    ratio = torch.linalg.vector_norm(vc - vy, dim=1) / vs.clamp_min(1e-6)
    multiplier = float(ratio.quantile(args.target_coverage))
    tc, ts, ty = collect(model, test_loader, device)
    report = {
        "validation_nll": best, "target_coverage": args.target_coverage,
        "calibration_multiplier": multiplier, "radius_floor_km": args.radius_floor_km,
        "validation": summarize(vc, vs, vy, multiplier, args.radius_floor_km),
        "untouched_test": summarize(tc, ts, ty, multiplier, args.radius_floor_km),
        "split": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "data_policy": "UCDP georeferenced events + ReliefWeb context; no Telegram",
    }
    (args.output_dir / "location_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.output_dir / "location_calibration.json").write_text(json.dumps({
        "multiplier": multiplier, "target_coverage": args.target_coverage,
        "minimum_publishable_radius_km": args.radius_floor_km,
    }, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
