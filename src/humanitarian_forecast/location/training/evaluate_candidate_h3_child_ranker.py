#!/usr/bin/env python3
"""Evaluate a frozen H3 child ranker under a frozen multiradius parent model.

No model fitting occurs here. Validation may choose only a small mass-transfer
policy over child predictions. Development is reporting-only. The child scorer
never sees target coordinates at inference; targets are used solely for metrics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np

from humanitarian_forecast.location.training.train_candidate_h3_child_ranker import child_dynamic_features, child_support, parent_matrix
from humanitarian_forecast.location.training.train_geo_lambdarank_v2 import _softmax, metrics


def _eth_mask(meta: list[dict], lo: int, hi: int) -> np.ndarray:
    return np.asarray([str(meta[i].get("country", "")).casefold() == "ethiopia" for i in range(lo, hi)])


def _child_predictions(
    model: lgb.Booster,
    x: np.ndarray,
    candidates: np.ndarray,
    coords: np.ndarray,
    valid: np.ndarray,
    parent_probability: np.ndarray,
    resolution: int,
    ring: int,
    max_parents: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(x)
    parent_features = parent_matrix(x, candidates, valid)
    child_coords = np.zeros((n, max_parents, 2), np.float32)
    child_margin = np.full((n, max_parents), -np.inf, np.float32)
    child_entropy = np.full((n, max_parents), np.inf, np.float32)
    parent_index = np.full((n, max_parents), -1, np.int32)
    for i in range(n):
        order = np.argsort(-parent_probability[i])
        chosen = [int(j) for j in order if valid[i, j]][:max_parents]
        for rank, j in enumerate(chosen):
            plat = float(candidates[i, j, 6] * 90.0)
            plon = float(candidates[i, j, 7] * 180.0)
            support = child_support((plat, plon), coords[i, j], resolution, ring)
            rows = []
            points = []
            for _, _, xy, local in support:
                rows.append(np.concatenate([parent_features[i, j], local]))
                points.append(xy)
            points_array = np.asarray(points, np.float32)
            row_array = np.asarray(rows, np.float32)
            row_array = np.concatenate([row_array, child_dynamic_features(x[i], points_array)], axis=1)
            raw = np.asarray(model.predict(row_array), np.float64)
            k = int(np.argmax(raw))
            child_coords[i, rank] = points_array[k]
            parent_index[i, rank] = j
            if len(raw) > 1:
                top2 = np.partition(raw, -2)[-2:]
                child_margin[i, rank] = float(top2.max() - top2.min())
            else:
                child_margin[i, rank] = np.inf
            shifted = raw - np.max(raw)
            p = np.exp(shifted)
            p /= max(float(p.sum()), 1e-12)
            child_entropy[i, rank] = float(-np.sum(p * np.log(np.maximum(p, 1e-12))))
        if (i + 1) % 250 == 0:
            print(f"child inference {i+1:,}/{n:,}", flush=True)
    return child_coords, parent_index, child_margin, child_entropy


def _expanded(
    base_probability: np.ndarray,
    base_coords: np.ndarray,
    base_valid: np.ndarray,
    child_coords: np.ndarray,
    parent_index: np.ndarray,
    child_margin: np.ndarray,
    top_parents: int,
    transfer: float,
    min_margin: float,
    max_parent_probability: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, base_k = base_probability.shape
    out_k = base_k + top_parents
    prob = np.zeros((n, out_k), np.float64)
    coords = np.zeros((n, out_k, 2), np.float32)
    valid = np.zeros((n, out_k), bool)
    prob[:, :base_k] = base_probability
    coords[:, :base_k] = base_coords
    valid[:, :base_k] = base_valid
    for i in range(n):
        for r in range(top_parents):
            j = int(parent_index[i, r])
            if j < 0 or not base_valid[i, j]:
                continue
            if child_margin[i, r] < min_margin:
                continue
            if base_probability[i, j] > max_parent_probability:
                continue
            mass = float(prob[i, j] * transfer)
            if mass <= 0:
                continue
            prob[i, j] -= mass
            slot = base_k + r
            prob[i, slot] = mass
            coords[i, slot] = child_coords[i, r]
            valid[i, slot] = True
    prob /= np.maximum(prob.sum(axis=1, keepdims=True), 1e-12)
    return prob, coords, valid


def _metric(prob: np.ndarray, coords: np.ndarray, valid: np.ndarray, target: np.ndarray) -> dict[str, float]:
    return metrics(np.log(np.maximum(prob, 1e-12)), prob, coords, valid, target)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--parent-probabilities", type=Path, required=True)
    p.add_argument("--child-model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resolution", type=int, default=5)
    p.add_argument("--ring", type=int, default=4)
    p.add_argument("--max-parents", type=int, default=10)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)

    z = np.load(a.data, allow_pickle=True)
    x = z["x"].astype(np.float32, copy=False)
    candidates = z["candidate_features"].astype(np.float32, copy=False)
    coords = z["candidate_coordinates"].astype(np.float32, copy=False)
    valid = z["candidate_valid"]
    target = z["y"].astype(np.float32, copy=False)
    meta = [json.loads(str(v)) for v in z["meta"]]
    n = len(x)
    tr = int(0.70 * n)
    va = int(0.85 * n)
    val_mask = _eth_mask(meta, tr, va)
    dev_mask = _eth_mask(meta, va, n)
    parent = np.load(a.parent_probabilities)
    val_parent_global = _softmax(parent["validation_logits"], valid[tr:va], 0.15)
    dev_parent_global = _softmax(parent["development_logits"], valid[va:], 0.15)
    val_p = val_parent_global[val_mask]
    dev_p = dev_parent_global[dev_mask]
    val_x = x[tr:va][val_mask]
    dev_x = x[va:][dev_mask]
    val_c = candidates[tr:va][val_mask]
    dev_c = candidates[va:][dev_mask]
    val_coords = coords[tr:va][val_mask]
    dev_coords = coords[va:][dev_mask]
    val_valid = valid[tr:va][val_mask]
    dev_valid = valid[va:][dev_mask]
    val_target = target[tr:va][val_mask]
    dev_target = target[va:][dev_mask]
    model = lgb.Booster(model_file=str(a.child_model))

    print("running child inference on validation", flush=True)
    vc, vpi, vm, ve = _child_predictions(model, val_x, val_c, val_coords, val_valid, val_p, a.resolution, a.ring, a.max_parents)
    print("running child inference on development", flush=True)
    dc, dpi, dm, de = _child_predictions(model, dev_x, dev_c, dev_coords, dev_valid, dev_p, a.resolution, a.ring, a.max_parents)

    base_val = _metric(val_p, val_coords, val_valid, val_target)
    base_dev = _metric(dev_p, dev_coords, dev_valid, dev_target)
    finite_margin = vm[np.isfinite(vm)]
    quantiles = [float(np.quantile(finite_margin, q)) for q in (0.0, 0.25, 0.50, 0.75, 0.90)] if len(finite_margin) else [-np.inf]
    margin_thresholds = sorted(set([-np.inf, *quantiles]))
    top_options = [k for k in (1, 3, 5, 10) if k <= a.max_parents]
    transfer_options = (0.25, 0.50, 0.65, 0.75, 0.90, 1.0)
    parent_caps = (1.01, 0.90, 0.75, 0.60, 0.45, 0.30)
    best_broad = None
    best_guarded_mean = None
    for topk in top_options:
        for transfer in transfer_options:
            for margin in margin_thresholds:
                for cap in parent_caps:
                    pp, cc, vv = _expanded(val_p, val_coords, val_valid, vc, vpi, vm, topk, transfer, margin, cap)
                    report = _metric(pp, cc, vv, val_target)
                    policy = {"top_parents": topk, "transfer": transfer, "min_child_margin": margin, "max_parent_probability": cap}
                    row = (report["broad_area_score"], policy, report)
                    if best_broad is None or row[0] > best_broad[0]:
                        best_broad = row
                    # Mean-error challenger must not give away more than 0.5 percentage points of broad score.
                    if report["broad_area_score"] >= base_val["broad_area_score"] - 0.005:
                        key = report["mean_error_km"]
                        row2 = (key, policy, report)
                        if best_guarded_mean is None or key < best_guarded_mean[0]:
                            best_guarded_mean = row2
    assert best_broad is not None and best_guarded_mean is not None

    def evaluate_policy(selected):
        _, policy, val_report = selected
        dp, dcoords, dvalid = _expanded(dev_p, dev_coords, dev_valid, dc, dpi, dm, int(policy["top_parents"]), float(policy["transfer"]), float(policy["min_child_margin"]), float(policy["max_parent_probability"]))
        return {"policy": policy, "validation": val_report, "development": _metric(dp, dcoords, dvalid, dev_target)}

    broad_result = evaluate_policy(best_broad)
    mean_result = evaluate_policy(best_guarded_mean)
    report = {
        "schema": "candidate-h3-child-evaluation-v1",
        "lineage": {"dataset": str(a.data), "parent_probabilities": str(a.parent_probabilities), "child_model": str(a.child_model)},
        "protocol": "child model frozen from first70%; validation selects mass-transfer policy; development reporting-only",
        "support": {"resolution": a.resolution, "ring": a.ring, "max_parents": a.max_parents},
        "baseline": {"validation": base_val, "development": base_dev},
        "broad_area_selected": broad_result,
        "mean_error_guarded_selected": mean_result,
        "child_diagnostics": {
            "validation_margin_quantiles": {str(q): float(np.quantile(finite_margin, q)) for q in (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)} if len(finite_margin) else {},
            "validation_mean_entropy": float(np.mean(ve[np.isfinite(ve)])),
            "development_mean_entropy": float(np.mean(de[np.isfinite(de)])),
        },
        "evaluation_caveat": "2023-2025 development has been inspected by iterative architecture research and is not pristine prospective evaluation.",
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
