#!/usr/bin/env python3
"""Train causal Ethiopia ReliefWeb spatial specialists and blend with multiradius.

The global replay model is intentionally not retrained with Ethiopia-only report
features.  Instead, this script trains small Ethiopia specialists on publication-
time spatial ReliefWeb context, selects tree counts on an internal chronological
holdout inside the pre-2020 training era, and reserves the normal 2020-2023
validation block for recipe/calibration/blend selection.  The 2023-2025 block is
diagnostic only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier, early_stopping

from humanitarian_forecast.core.model_store import ModelInfo, write_info
from humanitarian_forecast.location.training.train_geo_lambdarank_relative import _flatten_features
from humanitarian_forecast.location.training.train_geo_lambdarank_v2 import _softmax, metrics

RADII = (25, 50, 100, 200)
RECIPES = {
    "balanced": np.asarray([2.0, 1.5, 1.0, 0.5], np.float32),
    "precision": np.asarray([6.0, 3.0, 1.0, 0.25], np.float32),
    "broad": np.asarray([1.0, 1.5, 2.0, 1.0], np.float32),
    "near50": np.asarray([3.0, 3.0, 1.0, 0.4], np.float32),
    "equal": np.ones(4, np.float32),
}


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-5, 1 - 1e-5)
    return np.log(p) - np.log1p(-p)


def _relative_all(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    if values.shape[-1] == 0:
        return np.zeros((*values.shape[:2], 0), np.float32)
    masked = np.where(valid[..., None], values, np.nan)
    mean = np.nanmean(masked, axis=1, keepdims=True)
    std = np.nanstd(masked, axis=1, keepdims=True)
    z = np.nan_to_num((values - mean) / np.maximum(std, 1e-5), nan=0.0, posinf=0.0, neginf=0.0)
    z = np.clip(z, -8.0, 8.0).astype(np.float32)
    sortable = np.where(valid[..., None], values, np.inf)
    order = np.argsort(sortable, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=np.float32)
    ordinal = np.broadcast_to(np.arange(values.shape[1], dtype=np.float32)[None, :, None], order.shape)
    np.put_along_axis(ranks, order, ordinal, axis=1)
    denom = np.maximum(valid.sum(axis=1, keepdims=True) - 1, 1).astype(np.float32)
    percentile = np.where(valid[..., None], ranks / denom[..., None], 0.0).astype(np.float32)
    return np.concatenate([z, percentile], axis=-1)


def _rows(
    events: np.ndarray,
    candidates: np.ndarray,
    valid: np.ndarray,
    coordinates: np.ndarray,
    target: np.ndarray,
    base_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x, _, group, distance = _flatten_features(events, candidates, valid, coordinates, target)
    extra_relative = _relative_all(candidates[..., base_dim:], valid)
    if extra_relative.shape[-1]:
        x = np.concatenate([x, extra_relative[valid]], axis=1).astype(np.float32, copy=False)
    return x, group, distance, valid


def _distance_soft_ce(prob: np.ndarray, coords: np.ndarray, valid: np.ndarray, target: np.ndarray, sigma_km: float = 100.0) -> float:
    distance = np.linalg.norm(coords - target[:, None], axis=-1) * 1000.0
    soft = np.exp(-0.5 * (distance / sigma_km) ** 2) * valid
    soft /= np.maximum(soft.sum(axis=1, keepdims=True), 1e-12)
    return float(-np.mean(np.sum(soft * np.log(np.maximum(prob, 1e-12)), axis=1)))


def _report(prob: np.ndarray, coords: np.ndarray, valid: np.ndarray, target: np.ndarray) -> dict[str, float]:
    out = metrics(np.log(np.maximum(prob, 1e-12)), prob, coords, valid, target)
    out["distance_soft_cross_entropy"] = _distance_soft_ce(prob, coords, valid, target)
    return out


def _make_model(seed: int, radius_index: int, estimators: int, scale_pos_weight: float) -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary",
        n_estimators=estimators,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=35,
        colsample_bytree=0.80,
        subsample=0.90,
        subsample_freq=1,
        reg_lambda=7.0,
        reg_alpha=0.35,
        n_jobs=-1,
        verbosity=-1,
        random_state=seed + radius_index,
        scale_pos_weight=scale_pos_weight,
    )


def _select_recipe(logits: np.ndarray, valid: np.ndarray, coords: np.ndarray, target: np.ndarray):
    best = None
    for name, weights in RECIPES.items():
        raw = np.sum(logits * weights[None, :], axis=1)
        score = np.full(valid.shape, -np.inf, np.float64)
        score[valid] = raw
        for temp in np.geomspace(0.10, 2.0, 20):
            prob = _softmax(score, valid, float(temp))
            report = _report(prob, coords, valid, target)
            key = report["broad_area_score"]
            row = (key, name, float(temp), weights.copy(), score, prob, report)
            if best is None or row[0] > best[0]:
                best = row
    assert best is not None
    return best


def _select_blend(
    baseline: np.ndarray,
    specialist: np.ndarray,
    coords: np.ndarray,
    valid: np.ndarray,
    target: np.ndarray,
):
    best = None
    for alpha in np.linspace(0.0, 1.0, 21):
        p = (1.0 - alpha) * baseline + alpha * specialist
        p /= np.maximum(p.sum(axis=1, keepdims=True), 1e-12)
        report = _report(p, coords, valid, target)
        row = (report["broad_area_score"], float(alpha), p, report)
        if best is None or row[0] > best[0]:
            best = row
    assert best is not None
    return best


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--baseline-probabilities", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--estimators", type=int, default=600)
    p.add_argument("--seed", type=int, default=20260824)
    a = p.parse_args()

    z = np.load(a.data, allow_pickle=True)
    meta = [json.loads(str(v)) for v in z["meta"]]
    n = len(meta)
    train_end = int(0.70 * n)
    valid_end = int(0.85 * n)
    eth_idx = np.asarray([i for i, row in enumerate(meta) if str(row.get("country", "")).casefold() == "ethiopia"], np.int64)
    base_dim = int(np.asarray(z["base_candidate_dim"]).item())
    all_names = [str(v) for v in z["reliefweb_appended_feature_names"]]
    full_candidates = z["candidate_features"].astype(np.float32, copy=False)
    candidates_eth = full_candidates[eth_idx].copy()
    del full_candidates
    events_eth = z["x"][eth_idx].astype(np.float32, copy=True)
    coords_eth = z["candidate_coordinates"][eth_idx].astype(np.float32, copy=True)
    valid_eth = z["candidate_valid"][eth_idx].copy()
    target_eth = z["y"][eth_idx].astype(np.float32, copy=True)

    train_mask = eth_idx < train_end
    val_mask = (eth_idx >= train_end) & (eth_idx < valid_end)
    dev_mask = eth_idx >= valid_end
    print(json.dumps({"ethiopia_queries": int(len(eth_idx)), "train": int(train_mask.sum()), "validation": int(val_mask.sum()), "development": int(dev_mask.sum()), "base_dim": base_dim, "reliefweb_dim": len(all_names)}), flush=True)

    baseline_bundle = np.load(a.baseline_probabilities)
    base_val_logits = baseline_bundle["validation_logits"]
    base_dev_logits = baseline_bundle["development_logits"]
    val_global_eth = np.asarray([str(meta[i].get("country", "")).casefold() == "ethiopia" for i in range(train_end, valid_end)])
    dev_global_eth = np.asarray([str(meta[i].get("country", "")).casefold() == "ethiopia" for i in range(valid_end, n)])
    base_val = _softmax(base_val_logits[val_global_eth], valid_eth[val_mask], 0.15)
    base_dev = _softmax(base_dev_logits[dev_global_eth], valid_eth[dev_mask], 0.15)

    mode_indices = {
        "count": np.asarray([i for i, name in enumerate(all_names) if "mention_count" in name], np.int64),
        "content": np.asarray([i for i, name in enumerate(all_names) if "mention_count" not in name], np.int64),
        "full": np.arange(len(all_names), dtype=np.int64),
    }
    results = {
        "schema": "reliefweb-spatial-specialist-ablation-v1",
        "lineage": {
            "dataset": str(a.data),
            "baseline": str(a.baseline_probabilities),
            "causality": "ReliefWeb date.created <= observation cutoff; tree count selected only inside pre-2020 training era",
        },
        "split": {"global_train_end": train_end, "global_validation_end": valid_end, "ethiopia_train": int(train_mask.sum()), "ethiopia_validation": int(val_mask.sum()), "ethiopia_development": int(dev_mask.sum())},
        "baseline_validation": _report(base_val, coords_eth[val_mask], valid_eth[val_mask], target_eth[val_mask]),
        "baseline_development": _report(base_dev, coords_eth[dev_mask], valid_eth[dev_mask], target_eth[dev_mask]),
        "modes": {},
    }

    for mode, selected in mode_indices.items():
        out_dir = a.output_root / f"{mode}_v1"
        if out_dir.exists() and any(out_dir.iterdir()):
            raise FileExistsError(f"refusing to overwrite {out_dir}")
        keep = np.concatenate([np.arange(base_dim, dtype=np.int64), base_dim + selected])
        c = candidates_eth[..., keep]
        selected_names = [all_names[i] for i in selected]
        print(f"building {mode} specialist rows candidate_dim={c.shape[-1]} reliefweb_dim={len(selected)}", flush=True)
        tx, tg, td, _ = _rows(events_eth[train_mask], c[train_mask], valid_eth[train_mask], coords_eth[train_mask], target_eth[train_mask], base_dim)
        vx, vg, vd, _ = _rows(events_eth[val_mask], c[val_mask], valid_eth[val_mask], coords_eth[val_mask], target_eth[val_mask], base_dim)
        dx, dg, dd, _ = _rows(events_eth[dev_mask], c[dev_mask], valid_eth[dev_mask], coords_eth[dev_mask], target_eth[dev_mask], base_dim)
        inner_q = max(1, int(0.80 * len(tg)))
        inner_row = int(np.sum(tg[:inner_q]))
        val_logits = []
        dev_logits = []
        best_iterations = []
        raw_rw_gain = np.zeros(len(selected), np.float64)
        out_dir.mkdir(parents=True, exist_ok=True)
        for ri, radius in enumerate(RADII):
            y = (td <= radius).astype(np.int8)
            vy = (vd <= radius).astype(np.int8)
            pos = max(1, int(y.sum()))
            neg = max(1, len(y) - pos)
            scale = min(30.0, neg / pos)
            probe = _make_model(a.seed, ri, a.estimators, scale)
            probe.fit(tx[:inner_row], y[:inner_row], eval_set=[(tx[inner_row:], y[inner_row:])], eval_metric="binary_logloss", callbacks=[early_stopping(45, verbose=False)])
            it = int(probe.best_iteration_ or a.estimators)
            best_iterations.append(it)
            model = _make_model(a.seed, ri, it, scale)
            model.fit(tx, y)
            model.booster_.save_model(str(out_dir / f"within_{radius}km.txt"))
            val_logits.append(_logit(model.predict_proba(vx)[:, 1]).astype(np.float32))
            dev_logits.append(_logit(model.predict_proba(dx)[:, 1]).astype(np.float32))
            gain = model.booster_.feature_importance(importance_type="gain")
            if len(selected):
                raw_rw_gain += gain[base_dim:base_dim + len(selected)]
            print(json.dumps({"mode": mode, "radius": radius, "best_iteration": it, "train_positive_rate": float(y.mean()), "validation_positive_rate": float(vy.mean())}), flush=True)
        val_stack = np.stack(val_logits, axis=1)
        dev_stack = np.stack(dev_logits, axis=1)
        best = _select_recipe(val_stack, valid_eth[val_mask], coords_eth[val_mask], target_eth[val_mask])
        _, recipe, temp, weights, _, specialist_val, specialist_val_report = best
        dev_flat = np.sum(dev_stack * weights[None, :], axis=1)
        dev_score = np.full(valid_eth[dev_mask].shape, -np.inf, np.float64)
        dev_score[valid_eth[dev_mask]] = dev_flat
        specialist_dev = _softmax(dev_score, valid_eth[dev_mask], temp)
        specialist_dev_report = _report(specialist_dev, coords_eth[dev_mask], valid_eth[dev_mask], target_eth[dev_mask])
        _, alpha, blended_val, blended_val_report = _select_blend(base_val, specialist_val, coords_eth[val_mask], valid_eth[val_mask], target_eth[val_mask])
        blended_dev = (1.0 - alpha) * base_dev + alpha * specialist_dev
        blended_dev /= np.maximum(blended_dev.sum(axis=1, keepdims=True), 1e-12)
        blended_dev_report = _report(blended_dev, coords_eth[dev_mask], valid_eth[dev_mask], target_eth[dev_mask])
        top_features = []
        if len(selected):
            order = np.argsort(-raw_rw_gain)
            top_features = [{"feature": selected_names[i], "gain": float(raw_rw_gain[i])} for i in order[: min(20, len(order))] if raw_rw_gain[i] > 0]
        mode_report = {
            "selected_channels": len(selected),
            "candidate_dim": int(c.shape[-1]),
            "row_feature_dim": int(tx.shape[-1]),
            "best_iterations": best_iterations,
            "recipe": recipe,
            "radius_weights": weights.tolist(),
            "temperature": temp,
            "baseline_blend_specialist_weight": alpha,
            "specialist_validation": specialist_val_report,
            "specialist_development": specialist_dev_report,
            "blended_validation": blended_val_report,
            "blended_development": blended_dev_report,
            "top_reliefweb_raw_features_by_gain": top_features,
        }
        results["modes"][mode] = mode_report
        np.savez_compressed(out_dir / "ethiopia_probabilities.npz", validation=specialist_val, development=specialist_dev, blended_validation=blended_val, blended_development=blended_dev, validation_global_indices=eth_idx[val_mask], development_global_indices=eth_idx[dev_mask])
        (out_dir / "metrics.json").write_text(json.dumps(mode_report, indent=2) + "\n")
        write_info(out_dir, ModelInfo(
            subsystem="location/reliefweb_spatial_specialist",
            version="v1",
            status="research-challenger",
            description=f"Ethiopia ReliefWeb spatial {mode} specialist blended with the multiradius baseline.",
            metrics={"validation": blended_val_report, "development": blended_dev_report},
            lineage={"dataset": str(a.data), "baseline": str(a.baseline_probabilities), "restore_tag": "production-boost-preflight-2026-08-24"},
            training={"mode": mode, "selected_channels": len(selected), "best_iterations": best_iterations, "internal_holdout": "last 20% of pre-2020 Ethiopia training queries", "recipe": recipe, "radius_weights": weights.tolist(), "temperature": temp, "blend_specialist_weight": alpha, "seed": a.seed},
            notes=["All ReliefWeb signals are publication-time cutoff safe.", "2020-2023 validation selects recipe/calibration/blend only.", "2023-2025 development remains diagnostic and is not pristine prospective evaluation."],
        ))
        print(json.dumps({"mode": mode, "blend_specialist_weight": alpha, "validation_broad": blended_val_report["broad_area_score"], "development_broad": blended_dev_report["broad_area_score"], "validation_mean_km": blended_val_report["mean_error_km"], "development_mean_km": blended_dev_report["mean_error_km"]}, indent=2), flush=True)
        del tx, vx, dx

    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
