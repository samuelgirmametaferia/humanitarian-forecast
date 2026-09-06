#!/usr/bin/env python3
"""Train a strongly regularized country-local LambdaRank broad-area expert."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
from lightgbm import LGBMRanker, log_evaluation

from humanitarian_forecast.core.model_store import ModelInfo, write_info
from humanitarian_forecast.location.training.train_geo_lambdarank_v2 import (
    _flatten_features,
    _scores_to_matrix,
    _select_temperature,
    _softmax,
    metrics,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--country", default="Ethiopia")
    p.add_argument("--version", default="v1")
    p.add_argument("--estimators", type=int, default=400)
    p.add_argument("--learning-rate", type=float, default=0.025)
    p.add_argument("--num-leaves", type=int, default=15)
    p.add_argument("--min-child-samples", type=int, default=25)
    p.add_argument("--seed", type=int, default=20260823)
    args = p.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty model directory: {args.output_dir}")

    z = np.load(args.data)
    x = z["x"].astype(np.float32, copy=False)
    f = z["candidate_features"].astype(np.float32, copy=False)
    c = z["candidate_coordinates"].astype(np.float32, copy=False)
    v = z["candidate_valid"]
    y = z["y"].astype(np.float32, copy=False)
    meta = [json.loads(str(q)) for q in z["meta"]]
    n = len(y)
    te = int(0.70 * n)
    ve = int(0.85 * n)
    country_key = args.country.casefold()

    def mask(lo: int, hi: int) -> np.ndarray:
        return np.asarray([
            str(meta[i].get("country", "")).casefold() == country_key
            for i in range(lo, hi)
        ])

    train_mask = mask(0, te)
    val_mask = mask(te, ve)
    dev_mask = mask(ve, n)
    if train_mask.sum() < 100:
        raise RuntimeError(f"too few local training examples: {int(train_mask.sum())}")

    train_x, train_y, train_group, _ = _flatten_features(
        x[:te][train_mask], f[:te][train_mask], v[:te][train_mask],
        c[:te][train_mask], y[:te][train_mask]
    )
    val_x, val_y, val_group, _ = _flatten_features(
        x[te:ve][val_mask], f[te:ve][val_mask], v[te:ve][val_mask],
        c[te:ve][val_mask], y[te:ve][val_mask]
    )
    print(
        f"country={args.country} train_queries={int(train_mask.sum())} "
        f"validation_queries={int(val_mask.sum())} development_queries={int(dev_mask.sum())} "
        f"train_rows={len(train_x):,} features={train_x.shape[1]}", flush=True,
    )

    ranker = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        label_gain=[0, 1, 3, 7, 15],
        n_estimators=args.estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        min_child_samples=args.min_child_samples,
        colsample_bytree=0.80,
        subsample=0.85,
        subsample_freq=1,
        reg_lambda=12.0,
        reg_alpha=0.5,
        max_depth=5,
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

    val_valid = v[te:ve][val_mask]
    val_coordinates = c[te:ve][val_mask]
    val_target = y[te:ve][val_mask]
    best = None
    steps = sorted(set(list(range(25, args.estimators + 1, 25)) + [args.estimators]))
    for iteration in steps:
        flat = ranker.predict(val_x, num_iteration=iteration)
        raw = _scores_to_matrix(flat, val_valid)
        temp, report = _select_temperature(raw, val_coordinates, val_valid, val_target)
        # Checkpoint selection uses top-1 and broad probability coverage together.
        selection = (
            0.45 * report["within_100km"]
            + 0.20 * report["within_200km"]
            + 0.20 * report["top3_within_100km"]
            + 0.15 * report["probability_mass_within_100km"]
        )
        print(json.dumps({"iteration": iteration, "temperature": temp, "selection": selection, "validation": report}), flush=True)
        if best is None or selection > best[0]:
            best = (selection, iteration, temp, report)
    assert best is not None
    _, best_iteration, temperature, validation = best

    dev_x, _, _, _ = _flatten_features(
        x[ve:][dev_mask], f[ve:][dev_mask], v[ve:][dev_mask],
        c[ve:][dev_mask], y[ve:][dev_mask]
    )
    dev_valid = v[ve:][dev_mask]
    dev_raw = _scores_to_matrix(
        ranker.predict(dev_x, num_iteration=best_iteration), dev_valid
    )
    dev_p = _softmax(dev_raw, dev_valid, temperature)
    development = metrics(
        dev_raw, dev_p, c[ve:][dev_mask], dev_valid, y[ve:][dev_mask]
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ranker.booster_.save_model(str(args.output_dir / "geo_lambdarank_local.txt"), num_iteration=best_iteration)
    config = {
        "model_type": "LightGBM LambdaRank local expert",
        "country": args.country,
        "selected_iteration": best_iteration,
        "probability_temperature": temperature,
        "feature_dim": int(train_x.shape[1]),
        "training_queries": int(train_mask.sum()),
    }
    report = {
        "validation": validation,
        "development": development,
        "configuration": config,
        "protocol": "global chronological boundaries; country-only training, validation selection, and development reporting",
        "evaluation_caveat": "Development period has been inspected during architecture research and is not prospective evidence.",
    }
    (args.output_dir / "geo_lambdarank_local_metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output_dir / "ensemble_config.json").write_text(json.dumps(config, indent=2) + "\n")
    write_info(
        args.output_dir,
        ModelInfo(
            subsystem="location/geo_lambdarank_local",
            version=args.version,
            status="research-challenger",
            description=f"Strongly regularized country-local broad-area LambdaRank expert for {args.country}.",
            metrics=report,
            lineage={"dataset": str(args.data), "restore_tag": "geo-supercharge-phase1-2026-08-23"},
            training={
                "country": args.country,
                "train_queries": int(train_mask.sum()),
                "validation_queries": int(val_mask.sum()),
                "development_queries": int(dev_mask.sum()),
                "selected_iteration": best_iteration,
                "seed": args.seed,
            },
            calibration={"temperature": temperature, "selection": "country validation"},
            notes=["Intended only as a diversity/local specialization expert; global models remain preserved."],
        ),
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
