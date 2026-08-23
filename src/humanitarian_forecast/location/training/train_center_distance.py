#!/usr/bin/env python3
"""Train a deterministic center directly for mean geodesic-proxy error."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from humanitarian_forecast.location.models.direct import ProbabilisticLocationTransformer

SCALE_KM = 1000.0


def distance_loss(center: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # Coordinates are local east/north offsets in 1,000 km units, so this is
    # the appropriate differentiable proxy for the reported center distance.
    return torch.sqrt(((center - target) ** 2).sum(-1) + 1e-8).mean()


def collect(model, loader, device):
    centers, targets = [], []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            center, _ = model(x.to(device))
            centers.append(center.cpu()); targets.append(y)
    return torch.cat(centers), torch.cat(targets)


def metrics(center, target):
    error = torch.linalg.vector_norm(center - target, dim=-1) * SCALE_KM
    return {
        "samples": len(error), "mean_error_km": float(error.mean()),
        "median_error_km": float(error.median()),
        "p90_error_km": float(error.quantile(0.9)),
        "within_25km": float((error <= 25).float().mean()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=20260816)
    a = p.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    data = np.load(a.data)
    x = torch.from_numpy(data["x"]).float(); y = torch.from_numpy(data["y"]).float()
    train_end, validation_end = int(.70 * len(x)), int(.85 * len(x))
    loaders = [DataLoader(TensorDataset(x[i:j], y[i:j]), batch_size=a.batch_size,
                          shuffle=(i == 0))
               for i, j in ((0, train_end), (train_end, validation_end),
                            (validation_end, len(x)))]
    model = ProbabilisticLocationTransformer(
        feature_dim=x.shape[-1], sequence_length=x.shape[1],
        d_model=128, heads=8, layers=3, ff_dim=384, dropout=.1,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.learning_rate, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=a.epochs)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    for epoch in range(1, a.epochs + 1):
        model.train(); total = 0.0
        for xb, yb in loaders[0]:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            center, _ = model(xb)
            loss = distance_loss(center, yb)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); total += float(loss) * len(xb)
        scheduler.step()
        vc, vy = collect(model, loaders[1], device)
        value = float(distance_loss(vc, vy))
        print(f"epoch={epoch:02d} train_mean_km={total/train_end*SCALE_KM:.2f} "
              f"validation_mean_km={value*SCALE_KM:.2f}", flush=True)
        if value < best:
            best = value
            torch.save({"model_config": model.config,
                        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "selection_metric": "validation_mean_center_error",
                        "data_policy": "UCDP+ReliefWeb only; Telegram excluded"},
                       a.output_dir / "center_best.pt")
    state = torch.load(a.output_dir / "center_best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state"]); model.to(device)
    vc, vy = collect(model, loaders[1], device)
    tc, ty = collect(model, loaders[2], device)
    report = {"loss": "mean local center distance", "validation": metrics(vc, vy),
              "untouched_test": metrics(tc, ty),
              "data_policy": "UCDP georeferenced events + ReliefWeb context; no Telegram"}
    (a.output_dir / "center_metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
