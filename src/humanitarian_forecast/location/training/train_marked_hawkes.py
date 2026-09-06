#!/usr/bin/env python3
"""Train a marked spatiotemporal Hawkes-style broad-area expert.

The expert learns a non-negative mixture over causal space/time excitation bases.
It is deliberately much smaller and more interpretable than the neural models,
providing a different inductive bias for ensemble diversity.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

SCALE_KM = 1000.0
RADII = (25.0, 50.0, 100.0, 200.0, 400.0)
HALF_LIVES = (3.0, 14.0, 60.0, 180.0)


def _basis(events: np.ndarray, candidate_coordinates: np.ndarray) -> np.ndarray:
    valid = events[:, :, 0] > 0.5
    hist_xy = events[:, :, 1:3]
    days = np.maximum(0.0, np.expm1(events[:, :, 3] * 6.0))
    fatal = np.log1p(np.maximum(0.0, np.expm1(events[:, :, 4] * 6.0)))
    civ = np.log1p(np.maximum(0.0, np.expm1(events[:, :, 5] * 6.0)))
    violence = events[:, :, 6:9]
    diff = candidate_coordinates[:, :, None, :] - hist_xy[:, None, :, :]
    distance = np.linalg.norm(diff, axis=-1) * SCALE_KM
    mask = valid[:, None, :].astype(np.float32)
    parts: list[np.ndarray] = []
    names: list[str] = []
    marks = [
        ("event", np.ones_like(days)),
        ("fatal", fatal),
        ("civilian", civ),
        ("state", violence[:, :, 0]),
        ("nonstate", violence[:, :, 1]),
        ("onesided", violence[:, :, 2]),
    ]
    for radius in RADII:
        spatial = np.exp(-0.5 * (distance / radius) ** 2) * mask
        for half_life in HALF_LIVES:
            temporal = np.exp(-np.log(2.0) * days[:, None, :] / half_life)
            base = spatial * temporal
            for name, mark in marks:
                value = (base * mark[:, None, :]).sum(axis=-1)
                parts.append(np.log1p(value)[..., None].astype(np.float32))
                names.append(f"{name}_r{int(radius)}_h{int(half_life)}")
    # Smooth distance-to-last and directional propagation bases.
    last = hist_xy[:, -1]
    last_dist = np.linalg.norm(candidate_coordinates - last[:, None, :], axis=-1) * SCALE_KM
    parts.append((-last_dist / 200.0)[..., None].astype(np.float32)); names.append("distance_last")
    if events.shape[-1] >= 21:
        step = events[:, -1, 19:21]
        sn = np.linalg.norm(step, axis=-1, keepdims=True)
        dot = np.sum(candidate_coordinates * step[:, None, :], axis=-1)
        cn = np.linalg.norm(candidate_coordinates, axis=-1)
        cosine = dot / np.maximum(cn * sn, 1e-5)
        parts.append(cosine[..., None].astype(np.float32)); names.append("last_motion_alignment")
    return np.concatenate(parts, axis=-1), np.asarray(names)


def _soft_target(coordinates: np.ndarray, valid: np.ndarray, target: np.ndarray, radius_km: float) -> np.ndarray:
    distance = np.linalg.norm(coordinates - target[:, None], axis=-1) * SCALE_KM
    q = np.exp(-0.5 * (distance / radius_km) ** 2) * valid
    total = q.sum(axis=1, keepdims=True)
    bad = total[:, 0] <= 1e-12
    if np.any(bad):
        nearest = np.argmin(np.where(valid[bad], distance[bad], np.inf), axis=1)
        q[bad] = 0.0
        q[np.flatnonzero(bad), nearest] = 1.0
        total = q.sum(axis=1, keepdims=True)
    return q / total


def _softmax(score: np.ndarray, valid: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    s = score / max(temperature, 1e-5)
    s = np.where(valid, s, -np.inf)
    m = np.max(s, axis=1, keepdims=True)
    e = np.exp(np.where(valid, s - m, -np.inf))
    e = np.where(valid, e, 0.0)
    return e / np.maximum(e.sum(axis=1, keepdims=True), 1e-12)


def _ce(q: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(-np.sum(q * np.log(np.maximum(p, 1e-12)), axis=1)))


def metrics(probability: np.ndarray, coordinates: np.ndarray, valid: np.ndarray, target: np.ndarray, q: np.ndarray) -> dict[str, float]:
    distance = np.linalg.norm(coordinates - target[:, None], axis=-1) * SCALE_KM
    distance = np.where(valid, distance, np.inf)
    top = probability.argmax(axis=1)
    error = distance[np.arange(len(distance)), top]
    report: dict[str, float] = {
        "samples": float(len(error)),
        "distance_soft_cross_entropy": _ce(q, probability),
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
    chosen = np.take_along_axis(distance, order[:, :3], axis=1)
    report["top3_within_100km"] = float(np.mean(np.any(chosen <= 100, axis=1)))
    report["top3_within_200km"] = float(np.mean(np.any(chosen <= 200, axis=1)))
    report["broad_area_score"] = (
        0.55 * report["probability_mass_within_100km"]
        + 0.20 * report["probability_mass_within_200km"]
        + 0.15 * report["within_100km"]
        + 0.10 * report["top3_within_100km"]
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--country", default="Ethiopia")
    parser.add_argument("--target-radius-km", type=float, default=100.0)
    parser.add_argument("--l2", type=float, default=0.02)
    parser.add_argument("--max-iter", type=int, default=150)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty directory: {args.output_dir}")

    z = np.load(args.data, allow_pickle=True)
    x = z["x"].astype(np.float32, copy=False)
    coordinates = z["candidate_coordinates"].astype(np.float32, copy=False)
    valid = z["candidate_valid"]
    target = z["y"].astype(np.float32, copy=False)
    rows = [json.loads(str(value)) for value in z["meta"]]
    n = len(x); train_end = int(0.70 * n); valid_end = int(0.85 * n)
    country = np.asarray([row["country"] == args.country for row in rows])
    train_idx = np.flatnonzero(country & (np.arange(n) < train_end))
    val_idx = np.flatnonzero(country & (np.arange(n) >= train_end) & (np.arange(n) < valid_end))
    dev_idx = np.flatnonzero(country & (np.arange(n) >= valid_end))
    relevant = np.concatenate([train_idx, val_idx, dev_idx])
    basis_all, names = _basis(x[relevant], coordinates[relevant])
    offsets = {int(index): pos for pos, index in enumerate(relevant)}
    def take(idx: np.ndarray) -> np.ndarray:
        return basis_all[np.asarray([offsets[int(i)] for i in idx], dtype=np.int64)]

    train_basis = take(train_idx); val_basis = take(val_idx); dev_basis = take(dev_idx)
    q_train = _soft_target(coordinates[train_idx], valid[train_idx], target[train_idx], args.target_radius_km)
    q_val = _soft_target(coordinates[val_idx], valid[val_idx], target[val_idx], args.target_radius_km)
    q_dev = _soft_target(coordinates[dev_idx], valid[dev_idx], target[dev_idx], args.target_radius_km)

    feature_scale = np.sqrt(np.mean(train_basis * train_basis, axis=(0, 1)))
    feature_scale = np.maximum(feature_scale, 1e-4)
    train_basis = train_basis / feature_scale
    val_basis = val_basis / feature_scale
    dev_basis = dev_basis / feature_scale

    def objective(w: np.ndarray) -> tuple[float, np.ndarray]:
        score = np.tensordot(train_basis, w, axes=([-1], [0]))
        p = _softmax(score, valid[train_idx])
        value = _ce(q_train, p) + args.l2 * float(np.mean(w * w))
        # Exact gradient of soft-label cross entropy + L2.
        diff = p - q_train
        grad = np.mean(np.sum(diff[..., None] * train_basis, axis=1), axis=0)
        grad += 2.0 * args.l2 * w / len(w)
        return value, grad.astype(np.float64)

    initial = np.full(train_basis.shape[-1], 0.02, dtype=np.float64)
    result = minimize(
        fun=lambda w: objective(w), x0=initial, jac=True, method="L-BFGS-B",
        bounds=[(0.0, 8.0)] * len(initial), options={"maxiter": args.max_iter, "ftol": 1e-9},
    )
    weights = result.x
    val_score = np.tensordot(val_basis, weights, axes=([-1], [0]))
    best = None
    for temperature in np.geomspace(0.2, 5.0, 41):
        p = _softmax(val_score, valid[val_idx], float(temperature))
        report = metrics(p, coordinates[val_idx], valid[val_idx], target[val_idx], q_val)
        key = report["distance_soft_cross_entropy"]
        if best is None or key < best[0]:
            best = (key, float(temperature), p, report)
    assert best is not None
    _, temperature, val_probability, val_report = best
    dev_score = np.tensordot(dev_basis, weights, axes=([-1], [0]))
    dev_probability = _softmax(dev_score, valid[dev_idx], temperature)
    dev_report = metrics(dev_probability, coordinates[dev_idx], valid[dev_idx], target[dev_idx], q_dev)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "marked_hawkes_probabilities.npz",
        validation_indices=val_idx,
        validation_probability=val_probability.astype(np.float32),
        development_indices=dev_idx,
        development_probability=dev_probability.astype(np.float32),
    )
    ranked = np.argsort(-weights)
    top_weights = [{"feature": str(names[i]), "weight": float(weights[i])} for i in ranked[:20] if weights[i] > 1e-8]
    report = {
        "schema": "marked-hawkes-mixture-v1",
        "country": args.country,
        "fit": {"success": bool(result.success), "message": str(result.message), "iterations": int(result.nit), "objective": float(result.fun)},
        "target_radius_km": args.target_radius_km,
        "temperature": temperature,
        "nonzero_features": int(np.sum(weights > 1e-8)),
        "top_weights": top_weights,
        "validation": val_report,
        "development": dev_report,
        "evaluation_caveat": "Final block is a development benchmark, not a pristine prospective test.",
    }
    (args.output_dir / "marked_hawkes_metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output_dir / "marked_hawkes_model.json").write_text(json.dumps({"features": names.tolist(), "weights": weights.tolist(), "scale": feature_scale.tolist(), "temperature": temperature}, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
