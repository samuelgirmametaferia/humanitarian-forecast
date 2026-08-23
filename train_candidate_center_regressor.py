#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from candidate_center_model import CandidateCenterRegressor, center_loss


SCALE = 1000.0


def collect(model, loader, device):
    centers, targets = [], []
    model.eval()
    with torch.no_grad():
        for events, features, coordinates, valid, _, target, country, conflict in loader:
            _, center, _ = model(
                events.to(device),
                features.to(device),
                coordinates.to(device),
                valid.to(device),
                country.to(device),
                conflict.to(device),
            )
            centers.append(center.cpu())
            targets.append(target)
    return torch.cat(centers), torch.cat(targets)


def metrics(center, target):
    error = torch.linalg.vector_norm(center - target, dim=-1) * SCALE
    return {
        "samples": len(error),
        "mean_error_km": float(error.mean()),
        "median_error_km": float(error.median()),
        "p90_error_km": float(error.quantile(0.9)),
        "within_25km": float((error <= 25).float().mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--cross-entropy-weight", type=float, default=0.05)
    parser.add_argument("--residual-weight", type=float, default=0.1)
    parser.add_argument("--maximum-residual", type=float, default=0.5)
    parser.add_argument("--validation-gate-km", type=float, default=205.6698455810547)
    parser.add_argument("--seed", type=int, default=20260816)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    data = np.load(args.data)
    events = torch.from_numpy(data["x"]).float()
    features = torch.from_numpy(data["candidate_features"]).float()
    coordinates = torch.from_numpy(data["candidate_coordinates"]).float()
    valid = torch.from_numpy(data["candidate_valid"])
    label = torch.from_numpy(data["label"])
    target = torch.from_numpy(data["y"]).float()
    meta = [json.loads(str(row)) for row in data["meta"]]
    n = len(events)
    train_end = int(0.7 * n)
    validation_end = int(0.85 * n)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("device", device, flush=True)

    def identity_maps(end):
        countries = {
            value: i + 1 for i, value in enumerate(sorted({m["country"] for m in meta[:end]}))
        }
        conflicts = {
            value: i + 1 for i, value in enumerate(sorted({m["conflict_id"] for m in meta[:end]}))
        }
        return countries, conflicts

    def make_dataset(country_map, conflict_map):
        country_ids = torch.tensor([country_map.get(m["country"], 0) for m in meta])
        conflict_ids = torch.tensor([conflict_map.get(m["conflict_id"], 0) for m in meta])
        return TensorDataset(
            events, features, coordinates, valid, label, target, country_ids, conflict_ids
        )

    def fresh(country_map, conflict_map):
        return CandidateCenterRegressor(
            events.shape[-1],
            features.shape[-1],
            events.shape[1],
            len(country_map) + 1,
            len(conflict_map) + 1,
            maximum_residual=args.maximum_residual,
        ).to(device)

    def train_epoch(model, loader, optimizer):
        model.train()
        total = 0.0
        for batch in loader:
            eb, fb, cb, vb, lb, yb, countryb, conflictb = [value.to(device) for value in batch]
            optimizer.zero_grad(set_to_none=True)
            logits, center, correction = model(eb, fb, cb, vb, countryb, conflictb)
            loss, _, _, _ = center_loss(
                logits,
                center,
                correction,
                yb,
                lb,
                args.cross_entropy_weight,
                args.residual_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            total += float(loss.detach()) * len(eb)
        return total / len(loader.dataset)

    countries1, conflicts1 = identity_maps(train_end)
    dataset1 = make_dataset(countries1, conflicts1)
    train_loader = DataLoader(
        torch.utils.data.Subset(dataset1, range(train_end)),
        batch_size=args.batch_size,
        shuffle=True,
    )
    validation_loader = DataLoader(
        torch.utils.data.Subset(dataset1, range(train_end, validation_end)),
        batch_size=args.batch_size,
    )
    model = fresh(countries1, conflicts1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    best = float("inf")
    best_epoch = 1
    best_state = None
    stale = 0
    for epoch in range(1, args.epochs + 1):
        train_value = train_epoch(model, train_loader, optimizer)
        schedule.step()
        validation_center, validation_target = collect(model, validation_loader, device)
        validation = metrics(validation_center, validation_target)
        print(
            f"phase1 epoch={epoch:02d} train_loss={train_value:.4f} "
            f"validation_mean_km={validation['mean_error_km']:.2f}",
            flush=True,
        )
        if validation["mean_error_km"] < best:
            best = validation["mean_error_km"]
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= 4:
                break

    if best >= args.validation_gate_km:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "decision": "rejected_before_test",
            "best_validation_mean_km": best,
            "validation_gate_km": args.validation_gate_km,
            "selected_epochs": best_epoch,
            "protocol": "phase1 train70/validate15 only; test remained unread because validation gate failed",
            "data_policy": "cutoff-safe candidates and features; Telegram excluded",
        }
        (args.output_dir / "candidate_center_metrics.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        print(json.dumps(report, indent=2), flush=True)
        return

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    countries2, conflicts2 = identity_maps(validation_end)
    dataset2 = make_dataset(countries2, conflicts2)
    phase2_loader = DataLoader(
        torch.utils.data.Subset(dataset2, range(validation_end)),
        batch_size=args.batch_size,
        shuffle=True,
    )
    model2 = fresh(countries2, conflicts2)
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=args.learning_rate, weight_decay=0.01)
    for epoch in range(1, best_epoch + 1):
        train_epoch(model2, phase2_loader, optimizer2)
        print(f"phase2 epoch={epoch:02d}/{best_epoch}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_config": model2.config,
            "model_state": {key: value.detach().cpu() for key, value in model2.state_dict().items()},
            "selected_epochs": best_epoch,
        },
        args.output_dir / "candidate_center_best.pt",
    )
    phase1 = fresh(countries1, conflicts1)
    phase1.load_state_dict(best_state)
    validation_center, validation_target = collect(phase1, validation_loader, device)
    test_loader = DataLoader(
        torch.utils.data.Subset(dataset2, range(validation_end, n)),
        batch_size=args.batch_size,
    )
    test_center, test_target = collect(model2, test_loader, device)
    report = {
        "selected_epochs": best_epoch,
        "cross_entropy_weight": args.cross_entropy_weight,
        "residual_weight": args.residual_weight,
        "maximum_residual_km": args.maximum_residual * SCALE,
        "validation": metrics(validation_center, validation_target),
        "untouched_test": metrics(test_center, test_target),
        "protocol": "phase1 train70/validate15; phase2 fresh train85/test15",
        "data_policy": "cutoff-safe candidates and features; Telegram excluded",
    }
    (args.output_dir / "candidate_center_metrics.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
