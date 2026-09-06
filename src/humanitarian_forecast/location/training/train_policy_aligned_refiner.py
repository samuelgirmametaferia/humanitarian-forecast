#!/usr/bin/env python3
"""Policy-aligned continuous refiner for the frozen multiradius location ranker.

Unlike the earlier refiner, supervision is drawn from candidates the frozen
parent model itself ranks highly. This removes the oracle-near train/inference
mismatch. Parent scores/ranks/margins are explicit refiner features, and only
parent top-k candidates are eligible for coordinate movement at inference.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from lightgbm import Booster, LGBMRegressor

from humanitarian_forecast.core.model_store import ModelInfo, write_info
from humanitarian_forecast.location.training.train_candidate_refiner import _cap_residual
from humanitarian_forecast.location.training.train_geo_lambdarank_relative import _flatten_features
from humanitarian_forecast.location.training.train_geo_lambdarank_v2 import _scores_to_matrix, _softmax, metrics

RADII = (25, 50, 100, 200)
SCALE_KM = 1000.0


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-5, 1 - 1e-5)
    return np.log(p) - np.log1p(-p)


def _load_parent(model_dir: Path, metrics_path: Path):
    report = json.loads(metrics_path.read_text())
    weights = np.asarray(report["weights"], np.float32)
    temperature = float(report["temperature"])
    models = [Booster(model_file=str(model_dir / f"within_{r}km.txt")) for r in RADII]
    return models, weights, temperature


def _parent_context(flat: np.ndarray, valid: np.ndarray, models: list[Booster], weights: np.ndarray):
    components = []
    for model in models:
        components.append(_logit(np.asarray(model.predict(flat), np.float32)))
    score_flat = np.sum(np.stack(components, axis=1) * weights[None, :], axis=1)
    score = _scores_to_matrix(score_flat, valid)
    rank_score = np.zeros(valid.shape, np.float32)
    gap = np.zeros(valid.shape, np.float32)
    margin = np.zeros(valid.shape, np.float32)
    for i in range(len(valid)):
        ids = np.flatnonzero(valid[i])
        if not len(ids):
            continue
        order = ids[np.argsort(-score[i, ids])]
        denom = max(len(order) - 1, 1)
        rank_score[i, order] = 1.0 - np.arange(len(order), dtype=np.float32) / denom
        top = float(score[i, order[0]])
        second = float(score[i, order[1]]) if len(order) > 1 else top
        gap[i, ids] = top - score[i, ids]
        margin[i, ids] = top - second
    context = np.stack([score.astype(np.float32), rank_score, gap, margin], axis=-1)
    augmented = np.concatenate([flat, context[valid]], axis=1).astype(np.float32, copy=False)
    return augmented, score


def _features(
    x: np.ndarray,
    candidates: np.ndarray,
    coords: np.ndarray,
    valid: np.ndarray,
    target: np.ndarray,
    models: list[Booster],
    weights: np.ndarray,
):
    flat, _, group, distance = _flatten_features(x, candidates, valid, coords, target)
    augmented, score = _parent_context(flat, valid, models, weights)
    return augmented, score, group, distance


def _build_training_rows(
    x: np.ndarray,
    candidates: np.ndarray,
    coords: np.ndarray,
    valid: np.ndarray,
    target: np.ndarray,
    meta: list[dict],
    hi: int,
    models: list[Booster],
    weights: np.ndarray,
    parent_topk: int,
    training_radius_km: float,
    ethiopia_weight: float,
    regional_weight: float,
    chunk: int = 5000,
):
    regional = {"ethiopia", "eritrea", "somalia", "sudan", "south sudan", "djibouti", "kenya"}
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    ws: list[np.ndarray] = []
    diagnostics = {"selected_before_radius": 0, "selected_after_radius": 0, "queries_with_rows": 0}
    for start in range(0, hi, chunk):
        stop = min(hi, start + chunk)
        f, score, group, _ = _features(
            x[start:stop], candidates[start:stop], coords[start:stop], valid[start:stop], target[start:stop], models, weights
        )
        row_ids = np.full(valid[start:stop].shape, -1, np.int64)
        row_ids[valid[start:stop]] = np.arange(len(f), dtype=np.int64)
        d = np.linalg.norm(coords[start:stop] - target[start:stop, None], axis=-1) * SCALE_KM
        selected = np.zeros(valid[start:stop].shape, bool)
        rank_weight = np.zeros(valid[start:stop].shape, np.float32)
        for q in range(stop - start):
            ids = np.flatnonzero(valid[start + q])
            if not len(ids):
                continue
            order = ids[np.argsort(-score[q, ids])][:parent_topk]
            selected[q, order] = True
            rank_weight[q, order] = 1.0 / (1.0 + np.arange(len(order), dtype=np.float32) * 0.5)
        diagnostics["selected_before_radius"] += int(selected.sum())
        selected &= d <= training_radius_km
        diagnostics["selected_after_radius"] += int(selected.sum())
        diagnostics["queries_with_rows"] += int(np.any(selected, axis=1).sum())
        rid = row_ids[selected]
        if not len(rid):
            continue
        residual = (target[start:stop, None, :] - coords[start:stop])[selected].astype(np.float32)
        country_weight = np.ones(stop - start, np.float32)
        for q, gi in enumerate(range(start, stop)):
            country = str(meta[gi].get("country", "")).casefold()
            if country == "ethiopia":
                country_weight[q] = ethiopia_weight
            elif country in regional:
                country_weight[q] = regional_weight
        weight_matrix = country_weight[:, None] * rank_weight
        xs.append(f[rid])
        ys.append(residual)
        ws.append(weight_matrix[selected])
        print(f"policy-refiner rows {stop:,}/{hi:,} accumulated={sum(len(v) for v in xs):,}", flush=True)
    diagnostics["training_rows"] = int(sum(len(v) for v in xs))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(ws), diagnostics


def _predict(
    x: np.ndarray,
    candidates: np.ndarray,
    coords: np.ndarray,
    valid: np.ndarray,
    target: np.ndarray,
    models: list[Booster],
    weights: np.ndarray,
    east: Booster,
    north: Booster,
):
    f, score, _, _ = _features(x, candidates, coords, valid, target, models, weights)
    residual = np.zeros((*valid.shape, 2), np.float32)
    residual[..., 0][valid] = east.predict(f).astype(np.float32)
    residual[..., 1][valid] = north.predict(f).astype(np.float32)
    return score, residual


def _apply_policy(coords, valid, score, residual, topk: int, alpha: float, cap_km: float):
    refined = coords.copy()
    capped = _cap_residual(residual, cap_km)
    for i in range(len(valid)):
        ids = np.flatnonzero(valid[i])
        if not len(ids):
            continue
        chosen = ids[np.argsort(-score[i, ids])][:topk]
        refined[i, chosen] += float(alpha) * capped[i, chosen]
    return refined


def _country_mask(meta, lo, hi, country="Ethiopia"):
    c = country.casefold()
    return np.asarray([str(meta[i].get("country", "")).casefold() == c for i in range(lo, hi)])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--parent-model-dir", type=Path, required=True)
    p.add_argument("--parent-metrics", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--estimators", type=int, default=700)
    p.add_argument("--train-parent-topk", type=int, default=3)
    p.add_argument("--training-radius-km", type=float, default=300.0)
    p.add_argument("--ethiopia-weight", type=float, default=10.0)
    p.add_argument("--regional-weight", type=float, default=2.0)
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
    n = len(x); tr = int(.70 * n); va = int(.85 * n)
    parent_models, parent_weights, parent_temperature = _load_parent(a.parent_model_dir, a.parent_metrics)

    tx, ty, tw, train_diag = _build_training_rows(
        x, candidates, coords, valid, target, meta, tr, parent_models, parent_weights,
        a.train_parent_topk, a.training_radius_km, a.ethiopia_weight, a.regional_weight,
    )
    print(json.dumps({"training": train_diag, "feature_dim": int(tx.shape[1])}), flush=True)
    params = dict(
        objective="huber", n_estimators=a.estimators, learning_rate=.03,
        num_leaves=63, min_child_samples=80, colsample_bytree=.82,
        reg_lambda=8.0, reg_alpha=.35, n_jobs=-1, verbosity=-1,
        random_state=a.seed,
    )
    east_model = LGBMRegressor(**params)
    north_model = LGBMRegressor(**{**params, "random_state": a.seed + 1})
    east_model.fit(tx, ty[:, 0], sample_weight=tw)
    north_model.fit(tx, ty[:, 1], sample_weight=tw)
    del tx, ty, tw

    east = east_model.booster_; north = north_model.booster_
    val_score_all, val_res_all = _predict(x[tr:va], candidates[tr:va], coords[tr:va], valid[tr:va], target[tr:va], parent_models, parent_weights, east, north)
    dev_score_all, dev_res_all = _predict(x[va:], candidates[va:], coords[va:], valid[va:], target[va:], parent_models, parent_weights, east, north)
    vm = _country_mask(meta, tr, va); dm = _country_mask(meta, va, n)
    vs = val_score_all[vm]; ds = dev_score_all[dm]
    vv = valid[tr:va][vm]; dv = valid[va:][dm]
    vc = coords[tr:va][vm]; dc = coords[va:][dm]
    vy = target[tr:va][vm]; dy = target[va:][dm]
    vr = val_res_all[vm]; dr = dev_res_all[dm]
    vp = _softmax(vs, vv, parent_temperature); dp = _softmax(ds, dv, parent_temperature)
    baseline = metrics(vs, vp, vc, vv, vy)

    best_broad = None
    best_mean = None
    search_count = 0
    for topk in (1, 2, 3, 5, 10, 32):
        for cap in (15., 25., 40., 50., 75., 100., 150., 200., 300.):
            for alpha in np.arange(0.0, 2.01, 0.125):
                refined = _apply_policy(vc, vv, vs, vr, topk, float(alpha), cap)
                report = metrics(vs, vp, refined, vv, vy)
                search_count += 1
                row = (report["broad_area_score"], -report["mean_error_km"], topk, cap, float(alpha), report)
                if best_broad is None or row[:2] > best_broad[:2]:
                    best_broad = row
                if report["broad_area_score"] >= baseline["broad_area_score"] - 0.003:
                    rowm = (report["mean_error_km"], topk, cap, float(alpha), report)
                    if best_mean is None or rowm[0] < best_mean[0]:
                        best_mean = rowm
    assert best_broad is not None and best_mean is not None

    def eval_selected(kind, row):
        if kind == "broad":
            _, _, topk, cap, alpha, val_report = row
        else:
            _, topk, cap, alpha, val_report = row
        refined = _apply_policy(dc, dv, ds, dr, int(topk), float(alpha), float(cap))
        return {
            "policy": {"topk": int(topk), "cap_km": float(cap), "alpha": float(alpha)},
            "validation": val_report,
            "development": metrics(ds, dp, refined, dv, dy),
        }

    broad_result = eval_selected("broad", best_broad)
    mean_result = eval_selected("mean", best_mean)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    east.save_model(str(a.output_dir / "east_refiner.txt"))
    north.save_model(str(a.output_dir / "north_refiner.txt"))
    # Save the broad-selected deployment candidate; mean-selected remains fully specified in metrics.
    bpol = broad_result["policy"]
    val_coords_selected = _apply_policy(vc, vv, vs, vr, bpol["topk"], bpol["alpha"], bpol["cap_km"])
    dev_coords_selected = _apply_policy(dc, dv, ds, dr, bpol["topk"], bpol["alpha"], bpol["cap_km"])
    np.savez_compressed(
        a.output_dir / "probabilities.npz",
        validation=vp.astype(np.float32), development=dp.astype(np.float32),
        validation_coordinates=val_coords_selected, development_coordinates=dev_coords_selected,
        validation_logits=vs.astype(np.float32), development_logits=ds.astype(np.float32),
        validation_indices=np.flatnonzero(vm) + tr, development_indices=np.flatnonzero(dm) + va,
    )
    report = {
        "schema": "policy-aligned-continuous-refiner-v1",
        "parent": {"model_dir": str(a.parent_model_dir), "weights": parent_weights.tolist(), "temperature": parent_temperature},
        "training": {"parent_topk": a.train_parent_topk, "training_radius_km": a.training_radius_km, "ethiopia_weight": a.ethiopia_weight, "regional_weight": a.regional_weight, **train_diag},
        "baseline_validation": baseline,
        "broad_selected": broad_result,
        "mean_error_guarded_selected": mean_result,
        "policy_search_count": search_count,
        "protocol": "parent and refiner fit use first70%; validation selects only refinement topk/cap/alpha; development diagnostic only",
        "evaluation_caveat": "2023-2025 development is not pristine prospective evaluation after iterative architecture research.",
    }
    (a.output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    write_info(a.output_dir, ModelInfo(
        subsystem="location/policy_aligned_refiner", version="v1", status="research-challenger",
        description="Continuous residual refiner trained on the frozen multiradius policy's own top-ranked candidates.",
        metrics={"validation": broad_result["validation"], "development": broad_result["development"]},
        lineage={"dataset": str(a.data), "parent": str(a.parent_model_dir), "restore_tag": "production-boost-preflight-2026-08-24"},
        training={"estimators": a.estimators, "train_parent_topk": a.train_parent_topk, "training_radius_km": a.training_radius_km, "ethiopia_weight": a.ethiopia_weight, "regional_weight": a.regional_weight, "seed": a.seed, "broad_policy": broad_result["policy"], "mean_policy": mean_result["policy"]},
        notes=["Training rows are selected by frozen-parent rank before target-distance filtering.", "Four parent policy features are appended: score, rank, top-score gap, and top1/top2 margin.", "Only top-k parent candidates are moved at inference."],
    ))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
