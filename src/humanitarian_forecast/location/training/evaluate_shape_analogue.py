#!/usr/bin/env python3
"""Cutoff-safe 3D conflict-shape analogue expert.

The upstream ``shapefinder`` package currently pins pandas 1.1.5, which is not
compatible with this project's Python 3.14 environment. This implementation
captures the core research idea without silently changing runtimes: represent
longitude/east, latitude/north, and recency as a 3D point cloud, approximate
Earth-Mover geometry using fixed sliced-Wasserstein projections, retrieve only
historical analogues whose outcomes were observable before the query cutoff,
and convert their subsequent displacements into a spatial probability field.

This is a research expert, not a tactical tracker. Outputs are evaluated as
broad humanitarian areas over the same candidate support as existing experts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SCALE_KM = 1000.0


def parse_rows(meta: np.ndarray) -> list[dict[str, object]]:
    return [json.loads(str(value)) for value in meta]


def shape_signature(events: np.ndarray, *, projections: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(projections, 3)).astype(np.float32)
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-8)
    quantiles = np.linspace(0.0, 1.0, 16, dtype=np.float32)
    out = np.zeros((len(events), projections * len(quantiles) + 18), dtype=np.float32)
    for i, event in enumerate(events):
        valid = event[:, 0] > 0.5
        e = event[valid]
        if not len(e):
            continue
        days = np.maximum(0.0, np.expm1(e[:, 3] * 6.0))
        points = np.stack([
            e[:, 1] * 2.0,                 # east, roughly 500-km units
            e[:, 2] * 2.0,                 # north
            np.clip(days / 90.0, 0.0, 2.0),
        ], axis=-1)
        cursor = 0
        for direction in directions:
            projected = points @ direction
            values = np.quantile(projected, quantiles).astype(np.float32)
            out[i, cursor:cursor + len(quantiles)] = values
            cursor += len(quantiles)
        fatal = np.maximum(0.0, np.expm1(e[:, 4] * 6.0))
        civ = np.maximum(0.0, np.expm1(e[:, 5] * 6.0))
        motion = e[:, 19:21] if e.shape[1] >= 21 else np.zeros((len(e), 2), np.float32)
        recent = e[-min(4, len(e)):]
        summary = np.asarray([
            len(e) / 16.0,
            np.log1p(fatal.sum()) / 8.0,
            np.log1p(civ.sum()) / 8.0,
            np.log1p(fatal[-min(4, len(fatal)):].sum()) / 8.0,
            float(np.mean(e[:, 6])) if e.shape[1] > 8 else 0.0,
            float(np.mean(e[:, 7])) if e.shape[1] > 8 else 0.0,
            float(np.mean(e[:, 8])) if e.shape[1] > 8 else 0.0,
            float(np.std(e[:, 1])), float(np.std(e[:, 2])),
            float(np.mean(recent[:, 1])), float(np.mean(recent[:, 2])),
            float(np.mean(motion[:, 0])), float(np.mean(motion[:, 1])),
            float(np.mean(np.linalg.norm(motion, axis=-1))),
            float(e[-1, 1]), float(e[-1, 2]),
            float(np.sin(np.arctan2(e[-1, 1], e[-1, 2]))),
            float(np.cos(np.arctan2(e[-1, 1], e[-1, 2]))),
        ], dtype=np.float32)
        out[i, cursor:cursor + len(summary)] = summary
    return out


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp = np.exp(shifted)
    return exp / max(float(exp.sum()), 1e-12)


def metrics(probability: np.ndarray, coordinates: np.ndarray, valid: np.ndarray, target: np.ndarray) -> dict[str, float]:
    distance = np.linalg.norm(coordinates - target[:, None], axis=-1) * SCALE_KM
    distance = np.where(valid, distance, np.inf)
    top = probability.argmax(axis=1)
    error = distance[np.arange(len(distance)), top]
    report: dict[str, float] = {
        "samples": float(len(error)),
        "mean_error_km": float(error.mean()),
        "median_error_km": float(np.median(error)),
        "p90_error_km": float(np.quantile(error, 0.90)),
        "candidate_oracle_mean_km": float(np.mean(np.min(distance, axis=1))),
        "mean_entropy": float(np.mean(-np.sum(probability * np.log(np.maximum(probability, 1e-12)), axis=1))),
    }
    for radius in (25, 50, 100, 200):
        inside = distance <= radius
        report[f"within_{radius}km"] = float(np.mean(error <= radius))
        report[f"probability_mass_within_{radius}km"] = float(np.mean(np.sum(probability * inside, axis=1)))
    order = np.argsort(-probability, axis=1)
    chosen3 = np.take_along_axis(distance, order[:, :3], axis=1)
    report["top3_within_100km"] = float(np.mean(np.any(chosen3 <= 100, axis=1)))
    report["top3_within_200km"] = float(np.mean(np.any(chosen3 <= 200, axis=1)))
    report["broad_area_score"] = (
        0.55 * report["probability_mass_within_100km"]
        + 0.20 * report["probability_mass_within_200km"]
        + 0.15 * report["within_100km"]
        + 0.10 * report["top3_within_100km"]
    )
    return report


def predict_slice(
    signatures: np.ndarray,
    rows: list[dict[str, object]],
    coordinates: np.ndarray,
    valid: np.ndarray,
    outcomes: np.ndarray,
    query_indices: np.ndarray,
    bank_indices: np.ndarray,
    k: int,
    analogue_temperature: float,
    spatial_sigma_km: float,
) -> tuple[np.ndarray, dict[str, float]]:
    result = np.zeros((len(query_indices), coordinates.shape[1]), dtype=np.float64)
    bank_days = np.asarray([np.datetime64(rows[i]["target_date"], "D").astype(int) for i in bank_indices], dtype=np.int64)
    fallback = 0
    used = []
    for qi, index in enumerate(query_indices):
        target_day = int(np.datetime64(rows[index]["target_date"], "D").astype(int))
        cutoff = target_day - int(rows[index]["gap_days"])
        country = rows[index]["country"]
        conflict = rows[index]["conflict_id"]
        legal = bank_days <= cutoff
        # Prefer same conflict; back off to the same country when the conflict is new.
        same_conflict = np.asarray([rows[j]["conflict_id"] == conflict for j in bank_indices], dtype=bool)
        same_country = np.asarray([rows[j]["country"] == country for j in bank_indices], dtype=bool)
        mask = legal & same_conflict
        if int(mask.sum()) < max(3, min(k, 8)):
            mask = legal & same_country
            fallback += 1
        eligible = bank_indices[mask]
        if not len(eligible):
            p = valid[index].astype(np.float64)
            result[qi] = p / max(float(p.sum()), 1.0)
            continue
        delta = signatures[eligible] - signatures[index]
        distance = np.sqrt(np.mean(delta * delta, axis=1))
        take = np.argsort(distance)[: min(k, len(distance))]
        selected = eligible[take]
        d = distance[take]
        weight = softmax(-d / max(analogue_temperature, 1e-5))
        used.append(len(selected))
        candidate_xy = coordinates[index]
        spatial = np.linalg.norm(candidate_xy[:, None, :] - outcomes[selected][None, :, :], axis=-1) * SCALE_KM
        kernel = np.exp(-0.5 * (spatial / spatial_sigma_km) ** 2) @ weight
        kernel = np.where(valid[index], kernel, 0.0)
        if kernel.sum() <= 0:
            kernel = valid[index].astype(np.float64)
        result[qi] = kernel / max(float(kernel.sum()), 1e-12)
    return result, {
        "country_fallback_fraction": fallback / max(1, len(query_indices)),
        "mean_analogues_used": float(np.mean(used)) if used else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--country", default="Ethiopia")
    parser.add_argument("--projections", type=int, default=24)
    parser.add_argument("--neighbors", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260824)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty directory: {args.output_dir}")

    z = np.load(args.data, allow_pickle=True)
    x = z["x"].astype(np.float32, copy=False)
    coords = z["candidate_coordinates"].astype(np.float32, copy=False)
    valid = z["candidate_valid"]
    target = z["y"].astype(np.float32, copy=False)
    rows = parse_rows(z["meta"])
    n = len(x)
    train_end = int(0.70 * n)
    valid_end = int(0.85 * n)
    country_mask = np.asarray([row["country"] == args.country for row in rows])
    validation_idx = np.flatnonzero(country_mask & (np.arange(n) >= train_end) & (np.arange(n) < valid_end))
    development_idx = np.flatnonzero(country_mask & (np.arange(n) >= valid_end))
    if not len(validation_idx) or not len(development_idx):
        raise ValueError("country has no validation/development examples under the global chronology")

    relevant = np.flatnonzero(country_mask)
    signatures = np.zeros((n, args.projections * 16 + 18), np.float32)
    signatures[relevant] = shape_signature(x[relevant], projections=args.projections, seed=args.seed)
    # Standardize using only country rows whose outcomes are in the training block.
    fit_idx = np.flatnonzero(country_mask & (np.arange(n) < train_end))
    mean = signatures[fit_idx].mean(axis=0)
    std = signatures[fit_idx].std(axis=0)
    std = np.maximum(std, 1e-4)
    signatures[relevant] = (signatures[relevant] - mean) / std

    bank_for_validation = fit_idx
    # Hyperparameter selection is validation-only.
    trials = []
    best = None
    for temp in (0.15, 0.30, 0.60, 1.0):
        for sigma in (50.0, 100.0, 175.0, 250.0):
            p, diagnostics = predict_slice(
                signatures, rows, coords, valid, target,
                validation_idx, bank_for_validation, args.neighbors, temp, sigma,
            )
            report = metrics(p, coords[validation_idx], valid[validation_idx], target[validation_idx])
            trial = {"analogue_temperature": temp, "spatial_sigma_km": sigma, "validation": report, "diagnostics": diagnostics}
            trials.append(trial)
            if best is None or report["broad_area_score"] > best[0]:
                best = (report["broad_area_score"], temp, sigma, p, report, diagnostics)
    assert best is not None
    _, temperature, sigma, validation_probability, validation_report, validation_diag = best

    # Development analogue bank includes every same-country outcome available before
    # each individual forecast cutoff; predict_slice performs the per-row cutoff filter.
    dev_bank = np.flatnonzero(country_mask & (np.arange(n) < valid_end))
    development_probability, development_diag = predict_slice(
        signatures, rows, coords, valid, target,
        development_idx, dev_bank, args.neighbors, temperature, sigma,
    )
    development_report = metrics(
        development_probability, coords[development_idx], valid[development_idx], target[development_idx]
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "shape_analogue_probabilities.npz",
        validation_indices=validation_idx,
        validation_probability=validation_probability.astype(np.float32),
        development_indices=development_idx,
        development_probability=development_probability.astype(np.float32),
    )
    report = {
        "schema": "sliced-wasserstein-conflict-shape-analogue-v1",
        "country": args.country,
        "signature": {"projections": args.projections, "quantiles_per_projection": 16},
        "selected": {"neighbors": args.neighbors, "analogue_temperature": temperature, "spatial_sigma_km": sigma},
        "validation": validation_report,
        "development": development_report,
        "diagnostics": {"validation": validation_diag, "development": development_diag},
        "trials": trials,
        "causality": "Each query can use only analogue outcomes whose target date is on or before that query's observation cutoff.",
        "package_note": "Official shapefinder 0.1.0 pins pandas 1.1.5 and is not installable on this Python 3.14 runtime; this expert implements a compatible 3D historical-shape hypothesis using sliced-Wasserstein signatures.",
        "evaluation_caveat": "Final historical block is development-only after prior model research.",
    }
    (args.output_dir / "shape_analogue_metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
