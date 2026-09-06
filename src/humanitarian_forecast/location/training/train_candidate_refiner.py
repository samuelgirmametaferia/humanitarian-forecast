#!/usr/bin/env python3
"""Coarse-to-fine continuous refinement for the strong 32-candidate ranker.

The coarse LambdaRank expert keeps responsibility for *which region* receives
probability.  This module learns a cutoff-safe local residual (east/north, in the
same normalized coordinate system as candidate_coordinates) for each candidate
from global training examples.  Validation chooses only a shrinkage/cap policy;
alpha=0 is always included, so refinement can explicitly fall back to the frozen
coarse model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from lightgbm import Booster, LGBMRegressor

from humanitarian_forecast.core.model_store import ModelInfo, write_info
from humanitarian_forecast.location.training.train_geo_lambdarank_v2 import (
    _flatten_features,
    _scores_to_matrix,
    _softmax,
)

SCALE_KM = 1000.0


def _selected_mask(coords: np.ndarray, valid: np.ndarray, target: np.ndarray, k_near: int, max_km: float, seed: int) -> np.ndarray:
    """Choose supervised local rows without exposing target at inference.

    Target geometry is used only to construct regression supervision.  We retain
    the nearest candidates plus a small random sample from the <=max_km support
    so the refiner sees diverse local origins instead of only the oracle center.
    """
    rng = np.random.default_rng(seed)
    d = np.linalg.norm(coords - target[:, None], axis=-1) * SCALE_KM
    d = np.where(valid, d, np.inf)
    out = np.zeros(valid.shape, dtype=bool)
    for i in range(len(out)):
        order = np.argsort(d[i])
        near = order[np.isfinite(d[i, order])][:k_near]
        out[i, near] = True
        pool = np.flatnonzero((d[i] <= max_km) & ~out[i])
        if len(pool):
            take = min(k_near, len(pool))
            out[i, rng.choice(pool, size=take, replace=False)] = True
    return out


def _build_regression_rows(
    x: np.ndarray,
    candidates: np.ndarray,
    coords: np.ndarray,
    valid: np.ndarray,
    target: np.ndarray,
    meta: list[dict[str, object]],
    lo: int,
    hi: int,
    k_near: int,
    max_km: float,
    seed: int,
    chunk: int = 6000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    ws: list[np.ndarray] = []
    for start in range(lo, hi, chunk):
        stop = min(hi, start + chunk)
        mask = _selected_mask(coords[start:stop], valid[start:stop], target[start:stop], k_near, max_km, seed + start)
        f, _, group, _ = _flatten_features(
            x[start:stop], candidates[start:stop], mask,
            coords[start:stop], target[start:stop],
        )
        residual = (target[start:stop, None, :] - coords[start:stop])[mask].astype(np.float32)
        # Ethiopia receives modest extra weight while global examples teach the
        # transferable residual geometry.  No validation/development labels are
        # used for fitting.
        per_query = np.asarray([
            4.0 if str(meta[i].get("country", "")).casefold() == "ethiopia" else 1.0
            for i in range(start, stop)
        ], dtype=np.float32)
        weight = np.repeat(per_query, group)
        xs.append(f); ys.append(residual); ws.append(weight)
        print(f"refiner rows {stop:,}/{hi:,} accumulated={sum(len(v) for v in xs):,}", flush=True)
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(ws)


def _all_features(
    x: np.ndarray,
    candidates: np.ndarray,
    coords: np.ndarray,
    valid: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    f, _, group, _ = _flatten_features(x, candidates, valid, coords, target)
    return f, group


def _predict_matrices(
    east_model: LGBMRegressor,
    north_model: LGBMRegressor,
    coarse: Booster,
    x: np.ndarray,
    candidates: np.ndarray,
    coords: np.ndarray,
    valid: np.ndarray,
    target: np.ndarray,
    coarse_iteration: int,
) -> tuple[np.ndarray, np.ndarray]:
    f, _ = _all_features(x, candidates, coords, valid, target)
    raw_flat = coarse.predict(f, num_iteration=coarse_iteration)
    east_flat = east_model.predict(f)
    north_flat = north_model.predict(f)
    raw = _scores_to_matrix(raw_flat, valid)
    residual = np.zeros((*valid.shape, 2), dtype=np.float32)
    residual[..., 0][valid] = east_flat.astype(np.float32)
    residual[..., 1][valid] = north_flat.astype(np.float32)
    return raw, residual


def _cap_residual(residual: np.ndarray, cap_km: float) -> np.ndarray:
    cap = float(cap_km) / SCALE_KM
    norm = np.linalg.norm(residual, axis=-1, keepdims=True)
    scale = np.minimum(1.0, cap / np.maximum(norm, 1e-9))
    return residual * scale


def _metrics(
    scores: np.ndarray,
    probability: np.ndarray,
    refined: np.ndarray,
    valid: np.ndarray,
    target: np.ndarray,
) -> dict[str, float]:
    d = np.linalg.norm(refined - target[:, None], axis=-1) * SCALE_KM
    d = np.where(valid, d, np.inf)
    top = np.argmax(scores, axis=1)
    e = d[np.arange(len(d)), top]
    order = np.argsort(-scores, axis=1)
    out = {
        "samples": float(len(e)),
        "mean_error_km": float(np.mean(e)),
        "median_error_km": float(np.median(e)),
        "p90_error_km": float(np.quantile(e, 0.90)),
        "support_oracle_mean_km": float(np.mean(np.min(d, axis=1))),
        "mean_entropy": float(np.mean(-np.sum(probability * np.log(np.maximum(probability, 1e-12)), axis=1))),
    }
    for r in (25, 50, 100, 200):
        inside = d <= r
        out[f"within_{r}km"] = float(np.mean(e <= r))
        out[f"probability_mass_within_{r}km"] = float(np.mean(np.sum(probability * inside, axis=1)))
    chosen3 = np.take_along_axis(d, order[:, :3], axis=1)
    out["top3_within_100km"] = float(np.mean(np.any(chosen3 <= 100, axis=1)))
    out["broad_area_score"] = (
        .55*out["probability_mass_within_100km"] +
        .20*out["probability_mass_within_200km"] +
        .15*out["within_100km"] +
        .10*out["top3_within_100km"]
    )
    return out


def _country_indices(meta: list[dict[str, object]], lo: int, hi: int, country: str) -> np.ndarray:
    c = country.casefold()
    return np.asarray([i-lo for i in range(lo, hi) if str(meta[i].get("country", "")).casefold() == c], dtype=np.int64)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--coarse-model", type=Path, required=True)
    p.add_argument("--coarse-config", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--estimators", type=int, default=500)
    p.add_argument("--k-near", type=int, default=4)
    p.add_argument("--training-radius-km", type=float, default=200.0)
    p.add_argument("--seed", type=int, default=20260824)
    a = p.parse_args()
    if a.output_dir.exists() and any(a.output_dir.iterdir()):
        raise FileExistsError(a.output_dir)

    z = np.load(a.data, allow_pickle=True)
    x = z["x"].astype(np.float32, copy=False)
    candidates = z["candidate_features"].astype(np.float32, copy=False)
    coords = z["candidate_coordinates"].astype(np.float32, copy=False)
    valid = z["candidate_valid"]
    target = z["y"].astype(np.float32, copy=False)
    meta = [json.loads(str(v)) for v in z["meta"]]
    n = len(x); tr = int(.70*n); va = int(.85*n)

    train_x, train_y, train_w = _build_regression_rows(
        x, candidates, coords, valid, target, meta, 0, tr,
        a.k_near, a.training_radius_km, a.seed,
    )
    params = dict(
        objective="huber", n_estimators=a.estimators, learning_rate=.035,
        num_leaves=63, min_child_samples=100, colsample_bytree=.80,
        reg_lambda=8.0, reg_alpha=.3, n_jobs=-1, verbosity=-1,
        random_state=a.seed,
    )
    east = LGBMRegressor(**params)
    north = LGBMRegressor(**{**params, "random_state": a.seed+1})
    east.fit(train_x, train_y[:,0], sample_weight=train_w)
    north.fit(train_x, train_y[:,1], sample_weight=train_w)
    del train_x, train_y, train_w

    coarse = Booster(model_file=str(a.coarse_model))
    config = json.loads(a.coarse_config.read_text())
    iteration = int(config.get("selected_iteration", coarse.num_trees()))
    temperature = float(config.get("probability_temperature", .2))

    val_scores, val_res = _predict_matrices(
        east, north, coarse, x[tr:va], candidates[tr:va], coords[tr:va], valid[tr:va], target[tr:va], iteration
    )
    val_prob = _softmax(val_scores, valid[tr:va], temperature)
    eth_val = _country_indices(meta, tr, va, "Ethiopia")
    best = None
    search=[]
    for cap in (25., 50., 75., 100., 150., 200.):
        capped = _cap_residual(val_res, cap)
        for alpha in np.linspace(0., 1.25, 11):
            refined = coords[tr:va] + float(alpha)*capped
            global_report = _metrics(val_scores, val_prob, refined, valid[tr:va], target[tr:va])
            local_report = _metrics(val_scores[eth_val], val_prob[eth_val], refined[eth_val], valid[tr:va][eth_val], target[tr:va][eth_val])
            row={"cap_km":cap,"alpha":float(alpha),"global":global_report,"ethiopia":local_report}
            search.append(row)
            # Ethiopia broad-area is primary, but require global broad-area not to
            # regress by more than 0.5 percentage points relative to alpha=0.
            if best is None:
                best=row
            else:
                base_global=search[0]["global"]["broad_area_score"]
                admissible=global_report["broad_area_score"] >= base_global-0.005
                best_admissible=best["global"]["broad_area_score"] >= base_global-0.005
                if admissible and (not best_admissible or local_report["broad_area_score"] > best["ethiopia"]["broad_area_score"]):
                    best=row
    assert best is not None

    dev_scores, dev_res = _predict_matrices(
        east, north, coarse, x[va:], candidates[va:], coords[va:], valid[va:], target[va:], iteration
    )
    dev_prob = _softmax(dev_scores, valid[va:], temperature)
    dev_refined = coords[va:] + best["alpha"]*_cap_residual(dev_res, best["cap_km"])
    eth_dev = _country_indices(meta, va, n, "Ethiopia")
    development = _metrics(dev_scores, dev_prob, dev_refined, valid[va:], target[va:])
    development_eth = _metrics(dev_scores[eth_dev], dev_prob[eth_dev], dev_refined[eth_dev], valid[va:][eth_dev], target[va:][eth_dev])

    a.output_dir.mkdir(parents=True, exist_ok=True)
    east.booster_.save_model(str(a.output_dir/"east_refiner.txt"))
    north.booster_.save_model(str(a.output_dir/"north_refiner.txt"))
    report={
        "schema":"candidate-continuous-refiner-v1",
        "selected":{"alpha":best["alpha"],"cap_km":best["cap_km"]},
        "validation_global":best["global"],
        "validation_ethiopia":best["ethiopia"],
        "development_global":development,
        "development_ethiopia":development_eth,
        "baseline_validation_global":search[0]["global"],
        "baseline_validation_ethiopia":search[0]["ethiopia"],
        "training_rows_policy":{"k_near":a.k_near,"training_radius_km":a.training_radius_km,"ethiopia_weight":4.0},
        "coarse":{"model":str(a.coarse_model),"iteration":iteration,"temperature":temperature},
        "protocol":"refiners fit on global first 70%; alpha/cap selected on validation with Ethiopia objective and global non-regression tolerance; development diagnostic only",
    }
    (a.output_dir/"candidate_refiner_metrics.json").write_text(json.dumps(report,indent=2)+"\n")
    write_info(a.output_dir,ModelInfo(
        subsystem="location/candidate_refiner",version="v1",status="research-challenger",
        description="Continuous coarse-to-fine residual refinement of frozen actor-transfer LambdaRank candidate mass.",
        metrics={"validation":best["ethiopia"],"development":development_eth},
        lineage={"dataset":str(a.data),"coarse_model":str(a.coarse_model),"restore_tag":"production-boost-preflight-2026-08-24"},
        training={"estimators":a.estimators,"k_near":a.k_near,"training_radius_km":a.training_radius_km,"selected_alpha":best["alpha"],"selected_cap_km":best["cap_km"],"seed":a.seed},
        notes=["The coarse ranking/order and probabilities are frozen; only candidate coordinates are locally refined.","alpha=0 is included as an explicit validation fallback.","Development is diagnostic, not pristine prospective evaluation."],
    ))
    print(json.dumps(report,indent=2),flush=True)

if __name__ == "__main__":
    main()
