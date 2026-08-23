#!/usr/bin/env python3
"""Train a gradient-boosted broad-area candidate ranking expert.

The model is deliberately heterogeneous with the neural candidate rankers.  It
uses only cutoff-safe candidate/history features and a LambdaRank objective with
geographic relevance grades (25/50/100/200 km).  No Torch import is used here so
LightGBM's native OpenMP runtime remains isolated on macOS.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from lightgbm import LGBMRanker, log_evaluation

from humanitarian_forecast.core.model_store import ModelInfo, write_info

SCALE_KM = 1000.0


def _motion_geometry(events: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    candidate_xy = candidates[..., 1:3]
    last = events[:, -1, 19:21]
    last = np.repeat(last[:, None, :], candidates.shape[1], axis=1)
    recent = events[:, -4:, 19:21].mean(axis=1)
    recent = np.repeat(recent[:, None, :], candidates.shape[1], axis=1)

    def relations(step: np.ndarray) -> list[np.ndarray]:
        cn = np.linalg.norm(candidate_xy, axis=-1, keepdims=True)
        sn = np.linalg.norm(step, axis=-1, keepdims=True)
        dot = np.sum(candidate_xy * step, axis=-1, keepdims=True)
        cosine = dot / np.maximum(cn * sn, 1e-5)
        cross = candidate_xy[..., 0:1] * step[..., 1:2] - candidate_xy[..., 1:2] * step[..., 0:1]
        projection = dot / np.maximum(sn, 1e-5)
        return [step, cosine, cross, projection]

    return np.concatenate([candidate_xy, *relations(last), *relations(recent)], axis=-1).astype(np.float32)


def _relevance(distance_km: np.ndarray) -> np.ndarray:
    # LightGBM relevance must be non-negative integers.  The gain schedule used
    # below then sharply rewards getting the correct broad area near the top.
    result = np.zeros(distance_km.shape, dtype=np.int32)
    result[distance_km <= 200] = 1
    result[distance_km <= 100] = 2
    result[distance_km <= 50] = 3
    result[distance_km <= 25] = 4
    return result


def _flatten_features(
    events: np.ndarray,
    candidates: np.ndarray,
    valid: np.ndarray,
    coordinates: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    geometry = _motion_geometry(events, candidates)
    full = np.concatenate([candidates, geometry], axis=-1).astype(np.float32, copy=False)
    distance = np.linalg.norm(coordinates - target[:, None], axis=-1) * SCALE_KM
    group = valid.sum(axis=1).astype(np.int32)
    flat_x = full[valid]
    flat_distance = distance[valid].astype(np.float32)
    flat_y = _relevance(flat_distance)
    return flat_x, flat_y, group, flat_distance


def _scores_to_matrix(scores: np.ndarray, valid: np.ndarray) -> np.ndarray:
    result = np.full(valid.shape, -np.inf, dtype=np.float64)
    result[valid] = scores
    return result


def _softmax(scores: np.ndarray, valid: np.ndarray, temperature: float) -> np.ndarray:
    scaled = scores / max(1e-4, float(temperature))
    scaled = np.where(valid, scaled, -np.inf)
    max_score = np.max(scaled, axis=1, keepdims=True)
    exp = np.exp(np.where(valid, scaled - max_score, -np.inf))
    exp = np.where(valid, exp, 0.0)
    return exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)


def metrics(
    scores: np.ndarray,
    probability: np.ndarray,
    coordinates: np.ndarray,
    valid: np.ndarray,
    target: np.ndarray,
) -> dict[str, float]:
    distance = np.linalg.norm(coordinates - target[:, None], axis=-1) * SCALE_KM
    distance = np.where(valid, distance, np.inf)
    top = np.argmax(scores, axis=1)
    error = distance[np.arange(len(distance)), top]
    out: dict[str, float] = {
        "samples": float(len(error)),
        "mean_error_km": float(np.mean(error)),
        "median_error_km": float(np.median(error)),
        "p90_error_km": float(np.quantile(error, 0.90)),
        "candidate_oracle_mean_km": float(np.mean(np.min(distance, axis=1))),
        "mean_entropy": float(np.mean(-np.sum(probability * np.log(np.maximum(probability, 1e-12)), axis=1))),
    }
    for radius in (25, 50, 100, 200):
        inside = distance <= radius
        out[f"within_{radius}km"] = float(np.mean(error <= radius))
        out[f"probability_mass_within_{radius}km"] = float(np.mean(np.sum(probability * inside, axis=1)))
    order = np.argsort(-scores, axis=1)
    for k in (3, 5):
        chosen = np.take_along_axis(distance, order[:, :k], axis=1)
        out[f"top{k}_within_100km"] = float(np.mean(np.any(chosen <= 100, axis=1)))
        out[f"top{k}_within_200km"] = float(np.mean(np.any(chosen <= 200, axis=1)))
    out["broad_area_score"] = (
        0.55 * out["probability_mass_within_100km"]
        + 0.20 * out["probability_mass_within_200km"]
        + 0.15 * out["within_100km"]
        + 0.10 * out["top3_within_100km"]
    )
    return out


def _select_temperature(
    scores: np.ndarray,
    coordinates: np.ndarray,
    valid: np.ndarray,
    target: np.ndarray,
) -> tuple[float, dict[str, float]]:
    best: tuple[float, float, dict[str, float]] | None = None
    # Log-spaced search because raw tree-score scale is arbitrary.
    for temperature in np.geomspace(0.20, 6.0, 31):
        probability = _softmax(scores, valid, float(temperature))
        report = metrics(scores, probability, coordinates, valid, target)
        key = report["broad_area_score"]
        if best is None or key > best[0]:
            best = (key, float(temperature), report)
    assert best is not None
    return best[1], best[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--version", default="v1")
    parser.add_argument("--estimators", type=int, default=350)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--min-child-samples", type=int, default=80)
    parser.add_argument("--feature-fraction", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=20260823)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty model directory: {args.output_dir}")

    bundle = np.load(args.data)
    x = bundle["x"].astype(np.float32, copy=False)
    candidates = bundle["candidate_features"].astype(np.float32, copy=False)
    coordinates = bundle["candidate_coordinates"].astype(np.float32, copy=False)
    valid = bundle["candidate_valid"]
    target = bundle["y"].astype(np.float32, copy=False)
    n = len(x)
    train_end = int(0.70 * n)
    valid_end = int(0.85 * n)
    if x.shape[-1] < 25:
        raise ValueError("LambdaRank geo expert requires motion-v3 history")

    print("building train candidate rows", flush=True)
    train_x, train_y, train_group, _ = _flatten_features(
        x[:train_end], candidates[:train_end], valid[:train_end], coordinates[:train_end], target[:train_end]
    )
    print("building validation candidate rows", flush=True)
    val_x, val_y, val_group, _ = _flatten_features(
        x[train_end:valid_end], candidates[train_end:valid_end], valid[train_end:valid_end],
        coordinates[train_end:valid_end], target[train_end:valid_end]
    )
    print(
        f"train_rows={len(train_x):,} validation_rows={len(val_x):,} features={train_x.shape[1]}",
        flush=True,
    )

    ranker = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        label_gain=[0, 1, 3, 7, 15],
        n_estimators=args.estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        min_child_samples=args.min_child_samples,
        colsample_bytree=args.feature_fraction,
        reg_lambda=4.0,
        reg_alpha=0.15,
        verbosity=-1,
        n_jobs=-1,
        random_state=args.seed,
    )
    ranker.fit(
        train_x,
        train_y,
        group=train_group,
        eval_set=[(val_x, val_y)],
        eval_group=[val_group],
        eval_at=[1, 3, 5],
        callbacks=[log_evaluation(25)],
    )
    del train_x, train_y, val_x, val_y

    # Select tree count and probability temperature solely on validation broad
    # area score.  This avoids assuming LightGBM's NDCG optimum matches the
    # humanitarian forecast metric exactly.
    validation_valid = valid[train_end:valid_end]
    best: tuple[float, int, float, dict[str, float]] | None = None
    flat_features, _, _, _ = _flatten_features(
        x[train_end:valid_end], candidates[train_end:valid_end], validation_valid,
        coordinates[train_end:valid_end], target[train_end:valid_end]
    )
    steps = sorted(set(list(range(25, args.estimators + 1, 25)) + [args.estimators]))
    for iteration in steps:
        raw_flat = ranker.predict(flat_features, num_iteration=iteration)
        raw = _scores_to_matrix(raw_flat, validation_valid)
        temperature, report = _select_temperature(
            raw, coordinates[train_end:valid_end], validation_valid, target[train_end:valid_end]
        )
        score = report["broad_area_score"]
        print(json.dumps({"iteration": iteration, "temperature": temperature, "validation": report}), flush=True)
        if best is None or score > best[0]:
            best = (score, iteration, temperature, report)
    assert best is not None
    _, best_iteration, temperature, validation_report = best

    print("building development candidate rows", flush=True)
    dev_features, _, _, _ = _flatten_features(
        x[valid_end:], candidates[valid_end:], valid[valid_end:], coordinates[valid_end:], target[valid_end:]
    )
    dev_flat = ranker.predict(dev_features, num_iteration=best_iteration)
    dev_raw = _scores_to_matrix(dev_flat, valid[valid_end:])
    dev_probability = _softmax(dev_raw, valid[valid_end:], temperature)
    development_report = metrics(
        dev_raw, dev_probability, coordinates[valid_end:], valid[valid_end:], target[valid_end:]
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ranker.booster_.save_model(
        str(args.output_dir / "geo_lambdarank.txt"), num_iteration=best_iteration
    )
    config = {
        "model_type": "LightGBM LambdaRank",
        "selected_iteration": best_iteration,
        "probability_temperature": temperature,
        "feature_dim": int(candidates.shape[-1] + 14),
        "event_feature_contract": "motion-v3",
        "candidate_feature_dim": int(candidates.shape[-1]),
        "relevance": {"within_25km": 4, "within_50km": 3, "within_100km": 2, "within_200km": 1, "else": 0},
    }
    (args.output_dir / "ensemble_config.json").write_text(json.dumps(config, indent=2) + "\n")
    report = {
        "validation": validation_report,
        "development": development_report,
        "configuration": config,
        "protocol": "70% train; 15% validation selects tree count and temperature; final 15% development benchmark",
        "evaluation_caveat": "Final historical block has been inspected by ongoing architecture research and is not a pristine prospective test.",
    }
    (args.output_dir / "geo_lambdarank_metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    write_info(
        args.output_dir,
        ModelInfo(
            subsystem="location/geo_lambdarank",
            version=args.version,
            status="research-challenger",
            description="Distance-graded LightGBM LambdaRank expert with explicit recent-motion candidate geometry.",
            metrics={"validation": validation_report, "development": development_report},
            lineage={
                "dataset": str(args.data),
                "baseline": "models/location/candidate_ranker/v9",
                "restore_tag": "geo-supercharge-preflight-2026-08-23",
            },
            training={
                "selected_iteration": best_iteration,
                "estimators_requested": args.estimators,
                "learning_rate": args.learning_rate,
                "num_leaves": args.num_leaves,
                "seed": args.seed,
            },
            calibration={"temperature": temperature, "selection": "validation broad_area_score"},
            notes=[
                "Promoted v9 remains untouched.",
                "This expert is intended for heterogeneous ensembling with neural geo models.",
            ],
        ),
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
