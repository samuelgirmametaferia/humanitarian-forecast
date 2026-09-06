#!/usr/bin/env python3
"""Train a motion-aware GeoFusion challenger for coarse geographic early warning.

This is additive research infrastructure: it never overwrites the promoted v9
ranker.  Checkpoint selection is performed only on the chronological validation
partition.  The final historical partition is reported as a development block
because repeated model research has already inspected that period.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset

from humanitarian_forecast.core.model_store import ModelInfo, write_info
from humanitarian_forecast.location.models.geo_fusion import (
    GeoFusionCandidateRanker,
    spatial_distribution_loss,
)

SCALE_KM = 1000.0


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def choose_device(value: str) -> torch.device:
    if value == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(value)


def identity_maps(meta: list[dict[str, object]], end: int) -> tuple[dict[object, int], dict[object, int]]:
    countries = {value: i + 1 for i, value in enumerate(sorted({m["country"] for m in meta[:end]}))}
    conflicts = {value: i + 1 for i, value in enumerate(sorted({m["conflict_id"] for m in meta[:end]}))}
    return countries, conflicts


def normalization_stats(
    x: torch.Tensor,
    candidates: torch.Tensor,
    valid: torch.Tensor,
    end: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    event_mean = x[:end].mean(dim=(0, 1))
    event_std = x[:end].std(dim=(0, 1)).clamp_min(1e-4)
    valid_candidates = candidates[:end][valid[:end]]
    candidate_mean = valid_candidates.mean(dim=0)
    candidate_std = valid_candidates.std(dim=0).clamp_min(1e-4)
    return event_mean, event_std, candidate_mean, candidate_std


def build_dataset(
    x: torch.Tensor,
    features: torch.Tensor,
    coordinates: torch.Tensor,
    valid: torch.Tensor,
    labels: torch.Tensor,
    y: torch.Tensor,
    meta: list[dict[str, object]],
    country_map: dict[object, int],
    conflict_map: dict[object, int],
) -> TensorDataset:
    country = torch.tensor([country_map.get(m["country"], 0) for m in meta], dtype=torch.long)
    conflict = torch.tensor([conflict_map.get(m["conflict_id"], 0) for m in meta], dtype=torch.long)
    return TensorDataset(x, features, coordinates, valid, labels, y, country, conflict)


def make_model(
    *,
    x: torch.Tensor,
    features: torch.Tensor,
    valid: torch.Tensor,
    stats_end: int,
    countries: int,
    conflicts: int,
    d_model: int,
    heads: int,
    event_layers: int,
    candidate_layers: int,
    ff_dim: int,
    dropout: float,
) -> GeoFusionCandidateRanker:
    event_mean, event_std, candidate_mean, candidate_std = normalization_stats(
        x, features, valid, stats_end
    )
    return GeoFusionCandidateRanker(
        event_dim=x.shape[-1],
        candidate_dim=features.shape[-1],
        sequence_length=x.shape[1],
        countries=countries,
        conflicts=conflicts,
        event_mean=event_mean,
        event_std=event_std,
        candidate_mean=candidate_mean,
        candidate_std=candidate_std,
        d_model=d_model,
        heads=heads,
        event_layers=event_layers,
        candidate_layers=candidate_layers,
        ff_dim=ff_dim,
        dropout=dropout,
    )


def collect(
    model: GeoFusionCandidateRanker,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    logits_out: list[torch.Tensor] = []
    displacement_out: list[torch.Tensor] = []
    coordinates_out: list[torch.Tensor] = []
    valid_out: list[torch.Tensor] = []
    targets_out: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for xb, fb, cb, vb, _, yb, countryb, conflictb in loader:
            logits, displacement = model(
                xb.to(device), fb.to(device), vb.to(device), countryb.to(device), conflictb.to(device)
            )
            logits_out.append(logits.cpu())
            displacement_out.append(displacement.cpu())
            coordinates_out.append(cb)
            valid_out.append(vb)
            targets_out.append(yb)
    return (
        torch.cat(logits_out),
        torch.cat(displacement_out),
        torch.cat(coordinates_out),
        torch.cat(valid_out),
        torch.cat(targets_out),
    )


def spatial_metrics(
    logits: torch.Tensor,
    displacement: torch.Tensor,
    coordinates: torch.Tensor,
    valid: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    distance = torch.linalg.vector_norm(coordinates - target[:, None], dim=-1) * SCALE_KM
    distance = distance.masked_fill(~valid, float("inf"))
    probability = torch.softmax(logits, dim=-1)
    top = logits.argmax(dim=-1)
    row = torch.arange(len(logits))
    error = distance[row, top]
    oracle = distance.amin(dim=-1)
    expected_center = (probability[..., None] * coordinates).sum(dim=1)
    center_error = torch.linalg.vector_norm(expected_center - target, dim=-1) * SCALE_KM

    result: dict[str, float] = {
        "samples": float(len(error)),
        "mean_error_km": float(error.mean()),
        "median_error_km": float(error.median()),
        "p90_error_km": float(error.quantile(0.90)),
        "candidate_oracle_mean_km": float(oracle.mean()),
        "probability_center_mean_error_km": float(center_error.mean()),
        "mean_entropy": float((-(probability * probability.clamp_min(1e-9).log()).sum(-1)).mean()),
    }
    for radius in (25, 50, 100, 200):
        inside = distance <= radius
        result[f"within_{radius}km"] = float((error <= radius).float().mean())
        result[f"probability_mass_within_{radius}km"] = float(
            (probability * inside.float()).sum(-1).mean()
        )
    for k in (3, 5):
        topk = logits.topk(k=min(k, logits.shape[1]), dim=-1).indices
        topk_distance = distance.gather(1, topk)
        result[f"top{k}_within_100km"] = float((topk_distance <= 100).any(-1).float().mean())
        result[f"top{k}_within_200km"] = float((topk_distance <= 200).any(-1).float().mean())

    true_displacement = torch.linalg.vector_norm(target, dim=-1) * SCALE_KM
    result["displacement_mae_km"] = float(
        torch.mean(torch.abs(displacement * SCALE_KM - true_displacement))
    )
    # Broad-area checkpoint score: reward calibrated probability mass near the
    # outcome while retaining top-1 usefulness.  It is deliberately insensitive
    # to sub-25km point precision.
    result["broad_area_score"] = (
        0.55 * result["probability_mass_within_100km"]
        + 0.20 * result["probability_mass_within_200km"]
        + 0.15 * result["within_100km"]
        + 0.10 * result["top3_within_100km"]
    )
    return result


def train_phase(
    *,
    model: GeoFusionCandidateRanker,
    loader: DataLoader,
    validation_loader: DataLoader | None,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    hard_weight: float,
    soft50_weight: float,
    soft100_weight: float,
    distance_weight: float,
    displacement_weight: float,
    select_best: bool,
) -> tuple[dict[str, torch.Tensor], int, list[dict[str, float]]]:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.02)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs), eta_min=max(learning_rate * 0.08, 1e-6)
    )
    displacement_loss = nn.SmoothL1Loss(beta=0.05)
    best_score = -math.inf
    best_epoch = epochs
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        count = 0
        for xb, fb, cb, vb, lb, yb, countryb, conflictb in loader:
            xb = xb.to(device)
            fb = fb.to(device)
            cb = cb.to(device)
            vb = vb.to(device)
            lb = lb.to(device)
            yb = yb.to(device)
            countryb = countryb.to(device)
            conflictb = conflictb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, displacement = model(xb, fb, vb, countryb, conflictb)
            spatial, _ = spatial_distribution_loss(
                logits,
                cb,
                yb,
                lb,
                vb,
                hard_weight=hard_weight,
                soft50_weight=soft50_weight,
                soft100_weight=soft100_weight,
                distance_weight=distance_weight,
            )
            displacement_target = torch.linalg.vector_norm(yb, dim=-1)
            loss = spatial + displacement_weight * displacement_loss(displacement, displacement_target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(xb)
            count += len(xb)
        scheduler.step()

        record: dict[str, float] = {"epoch": float(epoch), "training_loss": total_loss / count}
        if validation_loader is not None:
            outputs = collect(model, validation_loader, device)
            metrics = spatial_metrics(*outputs)
            record.update({f"validation_{key}": value for key, value in metrics.items()})
            score = metrics["broad_area_score"]
            if score > best_score:
                best_score = score
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(json.dumps(record, sort_keys=True), flush=True)
        history.append(record)

    if select_best:
        if best_state is None:
            raise RuntimeError("no validation checkpoint selected")
        return best_state, best_epoch, history
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, epochs, history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--version", default="v1")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--event-layers", type=int, default=4)
    parser.add_argument("--candidate-layers", type=int, default=2)
    parser.add_argument("--ff-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.12)
    parser.add_argument("--hard-weight", type=float, default=0.35)
    parser.add_argument("--soft50-weight", type=float, default=0.75)
    parser.add_argument("--soft100-weight", type=float, default=0.45)
    parser.add_argument("--distance-weight", type=float, default=0.5)
    parser.add_argument("--displacement-weight", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty model directory: {args.output_dir}")

    seed_all(args.seed)
    bundle = np.load(args.data)
    x = torch.from_numpy(bundle["x"]).float()
    features = torch.from_numpy(bundle["candidate_features"]).float()
    coordinates = torch.from_numpy(bundle["candidate_coordinates"]).float()
    valid = torch.from_numpy(bundle["candidate_valid"])
    labels = torch.from_numpy(bundle["label"]).long()
    y = torch.from_numpy(bundle["y"]).float()
    meta = [json.loads(str(value)) for value in bundle["meta"]]
    n = len(x)
    train_end = int(0.70 * n)
    valid_end = int(0.85 * n)
    device = choose_device(args.device)
    print(
        f"device={device} samples={n:,} train={train_end:,} "
        f"validation={valid_end-train_end:,} development={n-valid_end:,} "
        f"event_dim={x.shape[-1]} candidate_dim={features.shape[-1]}",
        flush=True,
    )

    # Phase 1: validation-only architecture/checkpoint selection.
    country1, conflict1 = identity_maps(meta, train_end)
    dataset1 = build_dataset(
        x, features, coordinates, valid, labels, y, meta, country1, conflict1
    )
    train_loader = DataLoader(
        Subset(dataset1, range(0, train_end)),
        batch_size=args.batch_size,
        shuffle=True,
    )
    validation_loader = DataLoader(
        Subset(dataset1, range(train_end, valid_end)), batch_size=args.batch_size
    )
    model1 = make_model(
        x=x,
        features=features,
        valid=valid,
        stats_end=train_end,
        countries=len(country1) + 1,
        conflicts=len(conflict1) + 1,
        d_model=args.d_model,
        heads=args.heads,
        event_layers=args.event_layers,
        candidate_layers=args.candidate_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
    )
    best_state, best_epoch, phase1_history = train_phase(
        model=model1,
        loader=train_loader,
        validation_loader=validation_loader,
        device=device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        hard_weight=args.hard_weight,
        soft50_weight=args.soft50_weight,
        soft100_weight=args.soft100_weight,
        distance_weight=args.distance_weight,
        displacement_weight=args.displacement_weight,
        select_best=True,
    )
    model1.load_state_dict(best_state)
    validation_metrics = spatial_metrics(*collect(model1, validation_loader, device))

    # Phase 2: fresh retrain on first 85% for exactly the validation-selected
    # epoch count.  The last block is only a development benchmark now.
    seed_all(args.seed)
    country2, conflict2 = identity_maps(meta, valid_end)
    dataset2 = build_dataset(
        x, features, coordinates, valid, labels, y, meta, country2, conflict2
    )
    phase2_loader = DataLoader(
        Subset(dataset2, range(0, valid_end)),
        batch_size=args.batch_size,
        shuffle=True,
    )
    development_loader = DataLoader(
        Subset(dataset2, range(valid_end, n)), batch_size=args.batch_size
    )
    model2 = make_model(
        x=x,
        features=features,
        valid=valid,
        stats_end=valid_end,
        countries=len(country2) + 1,
        conflicts=len(conflict2) + 1,
        d_model=args.d_model,
        heads=args.heads,
        event_layers=args.event_layers,
        candidate_layers=args.candidate_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
    )
    phase2_state, _, phase2_history = train_phase(
        model=model2,
        loader=phase2_loader,
        validation_loader=None,
        device=device,
        epochs=best_epoch,
        learning_rate=args.learning_rate,
        hard_weight=args.hard_weight,
        soft50_weight=args.soft50_weight,
        soft100_weight=args.soft100_weight,
        distance_weight=args.distance_weight,
        displacement_weight=args.displacement_weight,
        select_best=False,
    )
    model2.load_state_dict(phase2_state)
    development_metrics = spatial_metrics(*collect(model2, development_loader, device))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_type": "GeoFusionCandidateRanker",
        "model_config": model2.config,
        "model_state": phase2_state,
        "country_map": country2,
        "conflict_map": conflict2,
        "selected_epochs": best_epoch,
        "feature_contract": {
            "event_dim": int(x.shape[-1]),
            "candidate_dim": int(features.shape[-1]),
            "candidate_count": int(features.shape[1]),
            "broad_area_primary": True,
        },
    }
    torch.save(checkpoint, args.output_dir / "geo_fusion_best.pt")

    report = {
        "protocol": "phase1 train70/validate15; fresh phase2 train85/development15",
        "evaluation_caveat": (
            "The final historical period is a development block because model research has inspected it. "
            "Promotion requires rolling-origin and prospective confirmation."
        ),
        "selection_metric": "validation broad_area_score",
        "selected_epochs": best_epoch,
        "validation": validation_metrics,
        "development": development_metrics,
        "phase1_history": phase1_history,
        "phase2_history": phase2_history,
        "loss": {
            "hard_weight": args.hard_weight,
            "soft50_weight": args.soft50_weight,
            "soft100_weight": args.soft100_weight,
            "distance_weight": args.distance_weight,
            "displacement_weight": args.displacement_weight,
        },
    }
    (args.output_dir / "geo_fusion_metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    write_info(
        args.output_dir,
        ModelInfo(
            subsystem="location/geo_fusion",
            version=args.version,
            status="research-challenger",
            description=(
                "Motion-aware broad-area GeoFusion candidate ranker with temporal convolution, "
                "candidate-to-history cross-attention, candidate competition, and distance-soft supervision."
            ),
            metrics={"validation": validation_metrics, "development": development_metrics},
            lineage={
                "dataset": str(args.data),
                "baseline": "models/location/candidate_ranker/v9",
                "restore_tag": "geo-supercharge-preflight-2026-08-23",
            },
            training={
                "selected_epochs": best_epoch,
                "seed": args.seed,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "architecture": model2.config,
                "split": {"train": 0.70, "validation": 0.15, "development": 0.15},
            },
            notes=[
                "Existing promoted v9 is not overwritten.",
                "Primary checkpoint criterion rewards broad 100/200 km probability coverage, not exact tactical precision.",
                "All candidate/history inputs are cutoff-safe under the source dataset contract.",
            ],
        ),
    )
    print(json.dumps({
        "selected_epochs": best_epoch,
        "validation": validation_metrics,
        "development": development_metrics,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
