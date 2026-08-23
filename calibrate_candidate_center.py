#!/usr/bin/env python3
"""Select center aggregation on validation, then evaluate a frozen phase-2 model once."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from candidate_rank_model import ConflictCandidateRanker, loss_fn


SCALE = 1000.0


def collect(model, loader, device):
    logits, coordinates, targets = [], [], []
    model.eval()
    with torch.no_grad():
        for x, features, candidate_coordinates, valid, _, target, country, conflict in loader:
            logits.append(
                model(
                    x.to(device),
                    features.to(device),
                    valid.to(device),
                    country.to(device),
                    conflict.to(device),
                ).cpu()
            )
            coordinates.append(candidate_coordinates)
            targets.append(target)
    return torch.cat(logits), torch.cat(coordinates), torch.cat(targets)


def weighted_mean(probability, coordinates):
    return (probability[:, :, None] * coordinates).sum(1)


def weighted_geometric_median(probability, coordinates, iterations=20):
    center = weighted_mean(probability, coordinates)
    for _ in range(iterations):
        distance = torch.linalg.vector_norm(coordinates - center[:, None], dim=-1).clamp_min(1e-5)
        weight = probability / distance
        updated = (weight[:, :, None] * coordinates).sum(1) / weight.sum(1, keepdim=True)
        center = updated
    return center


def predict(logits, coordinates, temperature, aggregation):
    probability = (logits / temperature).softmax(-1)
    if aggregation == "weighted_geometric_median":
        return weighted_geometric_median(probability, coordinates)
    return weighted_mean(probability, coordinates)


def history_references(events):
    valid = events[:, :, 0] > 0.5
    positions = events[:, :, 1:3]
    denominator = valid.sum(1, keepdim=True).clamp_min(1)
    references = {
        "anchor": torch.zeros_like(positions[:, 0]),
        "history_mean": (positions * valid[:, :, None]).sum(1) / denominator,
    }
    for count in (4, 8):
        subset_valid = valid[:, -count:]
        subset = positions[:, -count:]
        references[f"recent_{count}_mean"] = (
            (subset * subset_valid[:, :, None]).sum(1)
            / subset_valid.sum(1, keepdim=True).clamp_min(1)
        )
    days_ago = torch.expm1(events[:, :, 3] * 6).clamp_min(0)
    for scale in (30.0, 90.0, 365.0):
        weight = torch.exp(-days_ago / scale) * valid
        references[f"decay_{int(scale)}d"] = (
            (positions * weight[:, :, None]).sum(1) / weight.sum(1, keepdim=True).clamp_min(1e-6)
        )
    return references


def metrics(prediction, target):
    error = torch.linalg.vector_norm(prediction - target, dim=-1) * SCALE
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
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--distance-weight", type=float, default=0.0)
    parser.add_argument("--center-weight", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=20260816)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    data = np.load(args.data)
    x = torch.from_numpy(data["x"]).float()
    features = torch.from_numpy(data["candidate_features"]).float()
    coordinates = torch.from_numpy(data["candidate_coordinates"]).float()
    valid = torch.from_numpy(data["candidate_valid"])
    label = torch.from_numpy(data["label"])
    target = torch.from_numpy(data["y"]).float()
    meta = [json.loads(str(row)) for row in data["meta"]]
    n = len(x)
    train_end = int(0.7 * n)
    validation_end = int(0.85 * n)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("device", device, flush=True)

    countries = {value: i + 1 for i, value in enumerate(sorted({m["country"] for m in meta[:train_end]}))}
    conflicts = {value: i + 1 for i, value in enumerate(sorted({m["conflict_id"] for m in meta[:train_end]}))}
    country_ids = torch.tensor([countries.get(m["country"], 0) for m in meta])
    conflict_ids = torch.tensor([conflicts.get(m["conflict_id"], 0) for m in meta])
    dataset = TensorDataset(x, features, coordinates, valid, label, target, country_ids, conflict_ids)
    train_loader = DataLoader(
        torch.utils.data.Subset(dataset, range(train_end)),
        batch_size=args.batch_size,
        shuffle=True,
    )
    validation_loader = DataLoader(
        torch.utils.data.Subset(dataset, range(train_end, validation_end)),
        batch_size=args.batch_size,
    )

    model = ConflictCandidateRanker(
        x.shape[-1], features.shape[-1], x.shape[1], len(countries) + 1, len(conflicts) + 1
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    best_value = float("inf")
    best_state = None
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in train_loader:
            xb, fb, cb, vb, lb, yb, countryb, conflictb = [value.to(device) for value in batch]
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb, fb, vb, countryb, conflictb)
            loss, _, _, _ = loss_fn(
                logits, cb, yb, lb, args.distance_weight, args.center_weight
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
        schedule.step()
        validation_logits, validation_coordinates, validation_target = collect(
            model, validation_loader, device
        )
        value = metrics(
            weighted_mean(validation_logits.softmax(-1), validation_coordinates),
            validation_target,
        )["mean_error_km"]
        print(f"epoch={epoch:02d} validation_mean_km={value:.2f}", flush=True)
        if value < best_value:
            best_value = value
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    model.load_state_dict(best_state)
    validation_logits, validation_coordinates, validation_target = collect(model, validation_loader, device)
    validation_references = history_references(x[train_end:validation_end])
    trials = []
    for aggregation in ("weighted_mean", "weighted_geometric_median"):
        for temperature in (0.25, 0.35, 0.5, 0.65, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0):
            candidate_prediction = predict(
                validation_logits, validation_coordinates, temperature, aggregation
            )
            baseline = metrics(candidate_prediction, validation_target)
            trials.append({
                "aggregation": aggregation,
                "temperature": temperature,
                "history_reference": "none",
                "candidate_weight": 1.0,
                **baseline,
            })
            for reference_name, reference in validation_references.items():
                for candidate_weight in (0.6, 0.7, 0.8, 0.85, 0.9, 0.95):
                    blended = candidate_weight * candidate_prediction + (1 - candidate_weight) * reference
                    result = metrics(blended, validation_target)
                    trials.append({
                        "aggregation": aggregation,
                        "temperature": temperature,
                        "history_reference": reference_name,
                        "candidate_weight": candidate_weight,
                        **result,
                    })
    selected = min(trials, key=lambda row: row["mean_error_km"])
    print("selected", json.dumps(selected), flush=True)

    phase2_state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    phase2_state["center_calibration"] = {
        "aggregation": selected["aggregation"],
        "temperature": selected["temperature"],
        "history_reference": selected["history_reference"],
        "candidate_weight": selected["candidate_weight"],
        "selected_on": "chronological validation only",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(phase2_state, args.output)

    phase2_countries = {
        value: i + 1 for i, value in enumerate(sorted({m["country"] for m in meta[:validation_end]}))
    }
    phase2_conflicts = {
        value: i + 1 for i, value in enumerate(sorted({m["conflict_id"] for m in meta[:validation_end]}))
    }
    phase2_country_ids = torch.tensor([phase2_countries.get(m["country"], 0) for m in meta])
    phase2_conflict_ids = torch.tensor([phase2_conflicts.get(m["conflict_id"], 0) for m in meta])
    phase2_dataset = TensorDataset(
        x, features, coordinates, valid, label, target, phase2_country_ids, phase2_conflict_ids
    )
    test_loader = DataLoader(
        torch.utils.data.Subset(phase2_dataset, range(validation_end, n)),
        batch_size=args.batch_size,
    )
    phase2 = ConflictCandidateRanker(**phase2_state["model_config"]).to(device)
    phase2.load_state_dict(phase2_state["model_state"])
    test_logits, test_coordinates, test_target = collect(phase2, test_loader, device)
    test_prediction = predict(
        test_logits,
        test_coordinates,
        selected["temperature"],
        selected["aggregation"],
    )
    if selected["history_reference"] != "none":
        reference = history_references(x[validation_end:])[selected["history_reference"]]
        test_prediction = (
            selected["candidate_weight"] * test_prediction
            + (1 - selected["candidate_weight"]) * reference
        )
    test_result = metrics(test_prediction, test_target)
    report = {
        "selection_protocol": "train70/validate15; select epoch and calibration on validation only; frozen train85 checkpoint evaluated once on test15",
        "selected_epoch": best_epoch,
        "selected_calibration": {
            "aggregation": selected["aggregation"],
            "temperature": selected["temperature"],
            "history_reference": selected["history_reference"],
            "candidate_weight": selected["candidate_weight"],
        },
        "validation": {
            key: selected[key]
            for key in selected
            if key not in ("aggregation", "temperature", "history_reference", "candidate_weight")
        },
        "untouched_test": test_result,
        "validation_trials": trials,
        "data_policy": "cutoff-safe candidates and features; Telegram excluded",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
