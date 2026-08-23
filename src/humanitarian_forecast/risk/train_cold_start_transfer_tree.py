#!/usr/bin/env python3
"""Train a non-local humanitarian-risk expert selected on unseen PRIO entities.

The expert never consumes the target cell's eight local history channels. Model
selection uses a chronological 60%-70% slice restricted to a deterministic set
of PRIO cells excluded entirely from the first-60% fit set. This makes transfer
to unseen areas an explicit objective instead of a side effect of global fit.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
from sklearn.metrics import average_precision_score, log_loss, mean_absolute_error

from humanitarian_forecast.core.model_store import ModelInfo, write_info
from humanitarian_forecast.risk.train_challenger import chronological_bounds


def parse_meta(raw: np.ndarray) -> list[dict[str, str]]:
    return [json.loads(str(v)) for v in raw]


def heldout_entity(entity: str, modulus: int) -> bool:
    if not entity.startswith("prio:"):
        return False
    value = int(hashlib.sha256(entity.encode("utf-8")).hexdigest()[:8], 16)
    return value % modulus == 0


def nonlocal_features(x: np.ndarray, *, local_dim: int, dynamic_dim: int) -> np.ndarray:
    """Summarize neighbor dynamics and append broad static context once."""
    x = np.asarray(x, dtype=np.float32)
    dynamic = x[..., local_dim:dynamic_dim]
    parts: list[np.ndarray] = []
    for width in (1, 3, 7, 14, 28):
        recent = dynamic[:, -min(width, dynamic.shape[1]):]
        parts.extend([recent.sum(1), recent.mean(1), recent.max(1), recent.std(1)])
    if dynamic.shape[1] >= 14:
        parts.append(dynamic[:, -7:].sum(1) - dynamic[:, -14:-7].sum(1))
    time = np.arange(dynamic.shape[1], dtype=np.float32)
    centered = time - time.mean()
    denom = float((centered * centered).sum()) or 1.0
    parts.append((dynamic * centered[None, :, None]).sum(1) / denom)
    if x.shape[-1] > dynamic_dim:
        parts.append(x[:, -1, dynamic_dim:])
    return np.concatenate(parts, axis=1).astype(np.float32, copy=False)


def local_activity(x: np.ndarray, local_dim: int) -> np.ndarray:
    # activity channel is local channel 0; used only to define evaluation strata,
    # never as a transfer-expert input.
    return np.asarray(x[:, :, 0].sum(1), dtype=np.float32)


def safe_ap(y: np.ndarray, p: np.ndarray) -> float | None:
    if len(y) == 0 or len(np.unique(y)) < 2:
        return None
    return float(average_precision_score(y, p))


def report_slice(y: np.ndarray, p: np.ndarray, intensity_truth: np.ndarray | None = None, intensity_pred: np.ndarray | None = None) -> dict[str, float | int | None]:
    row: dict[str, float | int | None] = {
        "samples": int(len(y)),
        "positive_rate": float(y.mean()) if len(y) else None,
        "average_precision": safe_ap(y, p),
        "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))) if len(y) and len(np.unique(y)) > 1 else None,
    }
    if intensity_truth is not None and intensity_pred is not None and len(intensity_truth):
        row["intensity_mae"] = float(mean_absolute_error(intensity_truth, intensity_pred))
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--version", default="v7-cold-start-transfer")
    ap.add_argument("--local-dim", type=int, default=8)
    ap.add_argument("--dynamic-dim", type=int, default=26)
    ap.add_argument("--holdout-modulus", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260827)
    args = ap.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty model directory: {args.output_dir}")

    z = np.load(args.data, allow_pickle=True)
    x = z["x"].astype(np.float32, copy=False)
    y = z["y"].astype(np.float32, copy=False)
    meta = parse_meta(z["meta"])
    n = len(y)
    inner_end = int(0.60 * n)
    train_end, valid_end = chronological_bounds(n)
    if not inner_end < train_end < valid_end:
        raise ValueError("unexpected chronological bounds")

    print(f"building non-local features for {n:,} samples", flush=True)
    features = nonlocal_features(x, local_dim=args.local_dim, dynamic_dim=args.dynamic_dim)
    prio = np.asarray([m["entity"].startswith("prio:") for m in meta], dtype=bool)
    held = np.asarray([heldout_entity(m["entity"], args.holdout_modulus) for m in meta], dtype=bool)
    inner_fit = prio[:inner_end] & ~held[:inner_end]
    transfer_select = prio[inner_end:train_end] & held[inner_end:train_end]
    if transfer_select.sum() < 100 or y[inner_end:train_end][transfer_select, 1].sum() < 10:
        raise RuntimeError("transfer-selection slice too small")

    x_fit = features[:inner_end][inner_fit]
    y_fit = y[:inner_end][inner_fit]
    x_transfer = features[inner_end:train_end][transfer_select]
    y_transfer = y[inner_end:train_end][transfer_select]
    positives = float(y_fit[:, 1].sum()); negatives = float(len(y_fit) - positives)
    scale_pos_weight = (negatives / max(1.0, positives)) ** 0.6

    candidates = []
    for leaves, child, lr, l2 in itertools.product((7, 15, 31), (60, 140), (0.025, 0.05), (1.0, 3.0)):
        candidates.append((leaves, child, lr, l2))
    best = None
    history = []
    for idx, (leaves, child, lr, l2) in enumerate(candidates):
        model = lgb.LGBMClassifier(
            objective="binary", n_estimators=300, learning_rate=lr,
            num_leaves=leaves, min_child_samples=child,
            subsample=0.90, colsample_bytree=0.85,
            reg_lambda=l2, reg_alpha=0.05,
            scale_pos_weight=scale_pos_weight,
            random_state=args.seed + idx, n_jobs=-1, verbosity=-1,
        )
        model.fit(x_fit, y_fit[:, 1])
        probability = model.predict_proba(x_transfer)[:, 1]
        score = float(average_precision_score(y_transfer[:, 1], probability))
        loss = float(log_loss(y_transfer[:, 1], np.clip(probability, 1e-6, 1 - 1e-6)))
        row = {"num_leaves": leaves, "min_child_samples": child, "learning_rate": lr, "reg_lambda": l2, "transfer_ap": score, "transfer_log_loss": loss}
        history.append(row); print(json.dumps(row), flush=True)
        key = (score, -loss)
        if best is None or key > best[0]:
            best = (key, row)
    assert best is not None
    selected = best[1]

    # Final specialist keeps the held-out entity set excluded to preserve the
    # meaning of an unseen-area expert. It is intended for gated/ensemble use,
    # not as a universal replacement for the ordinary temporal model.
    final_fit = prio[:train_end] & ~held[:train_end]
    x_final = features[:train_end][final_fit]
    y_final = y[:train_end][final_fit]
    pos = float(y_final[:, 1].sum()); neg = float(len(y_final) - pos)
    final_scale = (neg / max(1.0, pos)) ** 0.6
    classifier = lgb.LGBMClassifier(
        objective="binary", n_estimators=300,
        learning_rate=float(selected["learning_rate"]),
        num_leaves=int(selected["num_leaves"]),
        min_child_samples=int(selected["min_child_samples"]),
        subsample=0.90, colsample_bytree=0.85,
        reg_lambda=float(selected["reg_lambda"]), reg_alpha=0.05,
        scale_pos_weight=final_scale,
        random_state=args.seed + 1000, n_jobs=-1, verbosity=-1,
    )
    classifier.fit(x_final, y_final[:, 1])
    regressor = lgb.LGBMRegressor(
        objective="huber", alpha=0.85, n_estimators=300,
        learning_rate=float(selected["learning_rate"]),
        num_leaves=int(selected["num_leaves"]),
        min_child_samples=int(selected["min_child_samples"]),
        subsample=0.90, colsample_bytree=0.85,
        reg_lambda=float(selected["reg_lambda"]), reg_alpha=0.05,
        random_state=args.seed + 1001, n_jobs=-1, verbosity=-1,
    )
    regressor.fit(x_final, y_final[:, 0])

    probs = classifier.predict_proba(features)[:, 1]
    intens = regressor.predict(features)
    low_threshold = float(np.quantile(local_activity(x[:train_end], args.local_dim), 0.25))
    activity = local_activity(x, args.local_dim)

    def period_report(lo: int, hi: int) -> dict[str, object]:
        yy = y[lo:hi]
        pp = probs[lo:hi]
        ii = intens[lo:hi]
        held_mask = held[lo:hi] & prio[lo:hi]
        low_mask = activity[lo:hi] <= low_threshold
        return {
            "all": report_slice(yy[:, 1], pp, yy[:, 0], ii),
            "heldout_entities": report_slice(yy[held_mask, 1], pp[held_mask], yy[held_mask, 0], ii[held_mask]),
            "low_local_activity": report_slice(yy[low_mask, 1], pp[low_mask], yy[low_mask, 0], ii[low_mask]),
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, args.output_dir / "transfer_classifier.joblib")
    joblib.dump(regressor, args.output_dir / "transfer_intensity.joblib")
    np.savez_compressed(
        args.output_dir / "predictions.npz",
        escalation_probability=probs.astype(np.float32),
        intensity_log1p=np.asarray(intens, dtype=np.float32),
    )
    metrics = {
        "transfer_selection": {
            "fit_period_end_fraction": 0.60,
            "selection_period": [0.60, 0.70],
            "heldout_modulus": args.holdout_modulus,
            "fit_rows": int(inner_fit.sum()),
            "selection_rows": int(transfer_select.sum()),
            "selection_positives": int(y_transfer[:, 1].sum()),
            "selected": selected,
            "search": history,
        },
        "validation": period_report(train_end, valid_end),
        "development_holdout": period_report(valid_end, n),
        "low_local_activity_threshold": low_threshold,
        "feature_contract": {
            "local_channels_excluded": list(range(args.local_dim)),
            "nonlocal_dynamic_channels": list(range(args.local_dim, args.dynamic_dim)),
            "static_channels": list(range(args.dynamic_dim, x.shape[-1])),
            "feature_dim": int(features.shape[1]),
        },
        "evaluation_caveat": "The final historical block is a development benchmark; transfer selection occurs entirely inside the first 70% training chronology.",
    }
    (args.output_dir / "ethiopia_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    write_info(args.output_dir, ModelInfo(
        subsystem="risk/ethiopia", version=args.version, status="research-specialist",
        description="Non-local coarse humanitarian-risk expert explicitly selected on chronologically later PRIO entities excluded from fit.",
        metrics={"headline": metrics["development_holdout"], "validation": metrics["validation"], "transfer_selection": metrics["transfer_selection"]},
        lineage={"data": str(args.data)},
        training={"seed": args.seed, "holdout_modulus": args.holdout_modulus, "inner_fit_fraction": 0.60, "transfer_selection_end_fraction": 0.70},
        notes=[
            "Local target-cell history channels are excluded from the expert input.",
            "Hyperparameters are selected on held-out PRIO entities in a later training-period slice.",
            "Specialist is intended for coarse humanitarian cold-start transfer and ensemble use, not exact-event geolocation.",
        ],
    ))
    print(json.dumps({"selected": selected, "validation": metrics["validation"], "development": metrics["development_holdout"]}, indent=2))


if __name__ == "__main__":
    main()
