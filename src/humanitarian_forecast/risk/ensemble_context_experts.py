#!/usr/bin/env python3
"""Validation-only ensemble of coarse humanitarian-risk context experts.

Classification weights are chosen with binary log loss (a proper scoring rule),
not by looking at the development block. Intensity weights are chosen by
validation MAE. The script is deliberately limited to area-level humanitarian
risk/intensity outputs.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, log_loss, mean_absolute_error, brier_score_loss

from humanitarian_forecast.risk.predict_v6 import HumanitarianRiskV6Predictor
from humanitarian_forecast.risk.train_challenger import (
    apply_affine_logit_calibration,
    best_balanced_threshold,
    classification_metrics,
    chronological_bounds,
    fit_affine_logit_calibration,
)


def simplex_grid(names: list[str], step: float, min_base: float = 0.0):
    units = int(round(1.0 / step))
    for values in itertools.product(range(units + 1), repeat=len(names) - 1):
        used = sum(values)
        if used > units:
            continue
        full = list(values) + [units - used]
        weights = np.asarray(full, dtype=np.float64) / units
        if weights[0] + 1e-12 < min_base:
            continue
        yield weights


def choose_probability_weights(truth: np.ndarray, probs: np.ndarray, names: list[str], step: float, min_base: float):
    best = (math.inf, None, None)
    for weights in simplex_grid(names, step, min_base):
        blend = np.clip(weights @ probs, 1e-6, 1.0 - 1e-6)
        score = float(log_loss(truth, blend))
        if score < best[0] - 1e-12:
            best = (score, weights, blend)
    if best[1] is None:
        raise RuntimeError("no probability ensemble weights selected")
    return best


def choose_intensity_weights(truth: np.ndarray, preds: np.ndarray, names: list[str], step: float, min_base: float):
    best = (math.inf, None, None)
    for weights in simplex_grid(names, step, min_base):
        blend = weights @ preds
        score = float(mean_absolute_error(truth, blend))
        if score < best[0] - 1e-12:
            best = (score, weights, blend)
    if best[1] is None:
        raise RuntimeError("no intensity ensemble weights selected")
    return best


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-model", type=Path, required=True)
    p.add_argument("--base-data", type=Path, required=True)
    p.add_argument("--access-model", type=Path, required=True)
    p.add_argument("--access-data", type=Path, required=True)
    p.add_argument("--population-model", type=Path, required=True)
    p.add_argument("--population-data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--step", type=float, default=0.05)
    p.add_argument("--min-base-weight", type=float, default=0.50)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output}")

    paths = [args.base_data, args.access_data, args.population_data]
    bundles = [np.load(path, allow_pickle=True) for path in paths]
    for bundle in bundles[1:]:
        if not np.array_equal(bundle["y"], bundles[0]["y"]) or not np.array_equal(bundle["meta"], bundles[0]["meta"]):
            raise ValueError("expert datasets are not label/meta aligned")
    y = bundles[0]["y"].astype(np.float32)
    train_end, valid_end = chronological_bounds(len(y))

    specs = [
        ("v6", args.base_model, bundles[0]["x"]),
        ("accessibility", args.access_model, bundles[1]["x"]),
        ("population", args.population_model, bundles[2]["x"]),
    ]
    probability = []
    intensity = []
    for name, model_dir, x in specs:
        print(f"predicting {name} on {len(x):,} samples", flush=True)
        predictor = HumanitarianRiskV6Predictor(model_dir, args.device)
        pred = predictor.predict_array(x.astype(np.float32, copy=False), batch_size=1024)
        probability.append(pred["escalation_probability"].astype(np.float64))
        intensity.append(pred["intensity_log1p"].astype(np.float64))
    probability = np.stack(probability)
    intensity = np.stack(intensity)
    names = [row[0] for row in specs]

    truth_val = y[train_end:valid_end, 1]
    truth_dev = y[valid_end:, 1]
    val_prob_matrix = probability[:, train_end:valid_end]
    dev_prob_matrix = probability[:, valid_end:]
    val_int_matrix = intensity[:, train_end:valid_end]
    dev_int_matrix = intensity[:, valid_end:]

    val_logloss, prob_w, raw_val = choose_probability_weights(
        truth_val, val_prob_matrix, names, args.step, args.min_base_weight
    )
    raw_dev = prob_w @ dev_prob_matrix
    calibration = fit_affine_logit_calibration(truth_val, raw_val)
    val_prob = apply_affine_logit_calibration(raw_val, calibration)
    dev_prob = apply_affine_logit_calibration(raw_dev, calibration)
    threshold = best_balanced_threshold(truth_val, val_prob)

    val_mae, int_w, val_intensity = choose_intensity_weights(
        y[train_end:valid_end, 0], val_int_matrix, names, args.step, args.min_base_weight
    )
    dev_intensity = int_w @ dev_int_matrix

    validation = classification_metrics(truth_val, val_prob, threshold)
    development = classification_metrics(truth_dev, dev_prob, threshold)
    validation["intensity_mae"] = float(mean_absolute_error(y[train_end:valid_end, 0], val_intensity))
    development["intensity_mae"] = float(mean_absolute_error(y[valid_end:, 0], dev_intensity))

    components = {}
    for i, name in enumerate(names):
        components[name] = {
            "validation_ap": float(average_precision_score(truth_val, val_prob_matrix[i])),
            "development_ap": float(average_precision_score(truth_dev, dev_prob_matrix[i])),
            "validation_log_loss": float(log_loss(truth_val, np.clip(val_prob_matrix[i], 1e-6, 1-1e-6))),
            "development_log_loss": float(log_loss(truth_dev, np.clip(dev_prob_matrix[i], 1e-6, 1-1e-6))),
            "validation_intensity_mae": float(mean_absolute_error(y[train_end:valid_end, 0], val_int_matrix[i])),
            "development_intensity_mae": float(mean_absolute_error(y[valid_end:, 0], dev_int_matrix[i])),
        }

    report = {
        "schema": "coarse-humanitarian-context-ensemble-v1",
        "selection": {
            "probability_metric": "binary log loss on chronological validation (proper scoring rule)",
            "intensity_metric": "MAE on chronological validation",
            "weight_step": args.step,
            "minimum_v6_weight": args.min_base_weight,
            "probability_weights": dict(zip(names, map(float, prob_w))),
            "intensity_weights": dict(zip(names, map(float, int_w))),
            "raw_validation_log_loss": val_logloss,
            "validation_intensity_mae": val_mae,
            "affine_logit_calibration": calibration,
            "threshold": threshold,
        },
        "validation": validation,
        "development_holdout": development,
        "components": components,
        "evaluation_caveat": "Final historical block is a development benchmark; weights/calibration use validation only. Rolling-origin stability remains a separate promotion gate.",
        "safety_scope": "coarse humanitarian escalation/intensity only; no next-event coordinate or route output",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
