#!/usr/bin/env python3
"""Build a fixed chronological manifest and evaluate cutoff-safe baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

KM_SCALE = 1000.0


def metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
    error = np.linalg.norm(prediction - target, axis=1) * KM_SCALE
    return {
        "samples": int(len(error)),
        "mean_error_km": float(error.mean()),
        "median_error_km": float(np.median(error)),
        "p90_error_km": float(np.quantile(error, 0.9)),
        "within_25km": float((error <= 25.0).mean()),
    }


def history_mode(x: np.ndarray, cell_km: float = 100.0) -> np.ndarray:
    result = []
    for row in x:
        valid = row[:, 0] > 0
        points = row[valid, 1:3]
        cells = np.rint(points * KM_SCALE / cell_km).astype(np.int64)
        unique, counts = np.unique(cells, axis=0, return_counts=True)
        selected = unique[np.argmax(counts)]
        # Prefer the most recent observation when a cell contains several points.
        result.append(points[(cells == selected).all(axis=1)][-1])
    return np.asarray(result, dtype=np.float32)


def bootstrap_mean_ci(
    prediction: np.ndarray, target: np.ndarray, seed: int, draws: int = 1000
) -> list[float]:
    error = np.linalg.norm(prediction - target, axis=1) * KM_SCALE
    rng = np.random.default_rng(seed)
    means = np.empty(draws)
    for i in range(draws):
        means[i] = rng.choice(error, len(error), replace=True).mean()
    return [float(v) for v in np.quantile(means, [0.025, 0.975])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260816)
    args = parser.parse_args()

    data_bytes = args.data.read_bytes()
    data_hash = hashlib.sha256(data_bytes).hexdigest()
    data = np.load(args.data)
    x, y, raw_meta = data["x"], data["y"], data["meta"]
    n = len(y)
    train_end, validation_end = int(n * 0.70), int(n * 0.85)
    bounds = {"train": (0, train_end), "validation": (train_end, validation_end),
              "test": (validation_end, n)}

    examples = []
    for index, raw in enumerate(raw_meta):
        meta = json.loads(str(raw))
        split = next(name for name, (start, end) in bounds.items() if start <= index < end)
        examples.append({
            "example_id": hashlib.sha256(
                f"{data_hash}:{index}:{meta['target_date']}:{meta['conflict_id']}".encode()
            ).hexdigest()[:20],
            "index": index,
            "split": split,
            "target_date": meta["target_date"],
            "country": meta["country"],
            "conflict_id": meta["conflict_id"],
            "horizon_days": meta.get("gap_days"),
        })
    manifest = {
        "schema_version": 1,
        "source": str(args.data),
        "source_sha256": data_hash,
        "split_policy": "global chronological 70/15/15",
        "counts": {name: end - start for name, (start, end) in bounds.items()},
        "examples": examples,
    }
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest["manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")

    report: dict[str, object] = {
        "data_sha256": data_hash,
        "manifest_sha256": manifest["manifest_sha256"],
        "selection_rule": "Baselines only; no checkpoint selection on test",
        "splits": {},
    }
    vx, vy = x[train_end:validation_end], y[train_end:validation_end]
    validation_history_mean = np.asarray([
        row[row[:, 0] > 0, 1:3].mean(axis=0) for row in vx
    ])
    shrinkage_trials = []
    for persistence_weight in np.linspace(0.0, 0.5, 11):
        prediction = (1.0 - persistence_weight) * validation_history_mean
        mean_error = float(np.linalg.norm(prediction - vy, axis=1).mean() * KM_SCALE)
        shrinkage_trials.append((mean_error, float(persistence_weight)))
    _, selected_persistence_weight = min(shrinkage_trials)
    report["history_mean_shrinkage"] = {
        "selection_split": "validation",
        "selected_persistence_weight": selected_persistence_weight,
        "trials": [
            {"persistence_weight": weight, "validation_mean_error_km": error}
            for error, weight in shrinkage_trials
        ],
    }
    for name, (start, end) in bounds.items():
        sx, sy = x[start:end], y[start:end]
        predictions = {
            "last_location": np.zeros_like(sy),
            "history_mean": np.asarray([
                row[row[:, 0] > 0, 1:3].mean(axis=0) for row in sx
            ]),
            "historical_100km_cell_mode": history_mode(sx),
        }
        predictions["validation_selected_shrunk_history_mean"] = (
            (1.0 - selected_persistence_weight) * predictions["history_mean"]
        )
        split_result = {}
        for baseline, prediction in predictions.items():
            value = metrics(prediction, sy)
            value["mean_error_95ci_km"] = bootstrap_mean_ci(
                prediction, sy, args.seed + start
            )
            split_result[baseline] = value
        report["splits"][name] = split_result
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
