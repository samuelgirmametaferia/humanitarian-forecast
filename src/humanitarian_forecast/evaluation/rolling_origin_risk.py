#!/usr/bin/env python3
"""Rolling-origin robustness gate for coarse Ethiopia humanitarian-risk features.

The evaluator deliberately uses a fixed lightweight LightGBM probe instead of
retuning the production model on each fold. Its purpose is feature acceptance:
does a candidate representation improve risk/intensity across multiple later
time windows, including PRIO cells not observed in the fold's training period?
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import average_precision_score, mean_absolute_error


@dataclass(frozen=True)
class Fold:
    name: str
    train_end: int
    eval_start: int
    eval_end: int
    train_end_date: str
    eval_end_date: str


def parse_meta(raw: np.ndarray) -> list[dict[str, str]]:
    return [json.loads(str(v)) for v in raw]


def compact_features(x: np.ndarray, dynamic_dim: int) -> np.ndarray:
    """Compact causal summaries; append static channels once, not 28 times."""
    x = np.asarray(x, dtype=np.float32)
    dyn = x[..., :dynamic_dim]
    parts = [dyn[:, -1]]
    for width in (3, 7, 14, 28):
        recent = dyn[:, -min(width, dyn.shape[1]):]
        parts.extend([recent.sum(1), recent.mean(1), recent.max(1), recent.std(1)])
    if dyn.shape[1] >= 14:
        parts.append(dyn[:, -7:].sum(1) - dyn[:, -14:-7].sum(1))
    else:
        parts.append(np.zeros((len(dyn), dynamic_dim), dtype=np.float32))
    if x.shape[-1] > dynamic_dim:
        # Static context builder repeats values across time, so one slice is sufficient.
        parts.append(x[:, -1, dynamic_dim:])
    return np.concatenate(parts, axis=1).astype(np.float32, copy=False)


def build_folds(meta: list[dict[str, str]], train_fracs: tuple[float, ...], eval_frac: float) -> list[Fold]:
    cutoffs = np.asarray([m["cutoff"] for m in meta])
    unique_dates = np.asarray(sorted(set(cutoffs.tolist())))
    folds: list[Fold] = []
    for idx, frac in enumerate(train_fracs, 1):
        train_date_i = min(len(unique_dates) - 2, max(1, int(len(unique_dates) * frac)))
        eval_date_i = min(len(unique_dates) - 1, max(train_date_i + 1, int(len(unique_dates) * (frac + eval_frac))))
        train_date = unique_dates[train_date_i]
        eval_end_date = unique_dates[eval_date_i]
        train_end = int(np.searchsorted(cutoffs, train_date, side="right"))
        eval_end = int(np.searchsorted(cutoffs, eval_end_date, side="right"))
        if eval_end <= train_end:
            continue
        folds.append(Fold(
            name=f"fold_{idx}",
            train_end=train_end,
            eval_start=train_end,
            eval_end=eval_end,
            train_end_date=str(train_date),
            eval_end_date=str(eval_end_date),
        ))
    return folds


def entity_cold_start_mask(meta: list[dict[str, str]], fold: Fold) -> np.ndarray:
    seen = {m["entity"] for m in meta[:fold.train_end] if str(m.get("entity", "")).startswith("prio:")}
    return np.asarray([
        str(m.get("entity", "")).startswith("prio:") and m["entity"] not in seen
        for m in meta[fold.eval_start:fold.eval_end]
    ], dtype=bool)


def _safe_ap(y: np.ndarray, p: np.ndarray) -> float | None:
    if len(y) == 0 or len(np.unique(y)) < 2:
        return None
    return float(average_precision_score(y, p))


def train_fold(features: np.ndarray, y: np.ndarray, meta: list[dict[str, str]], fold: Fold, seed: int) -> dict[str, object]:
    x_train = features[:fold.train_end]
    y_train = y[:fold.train_end]
    x_eval = features[fold.eval_start:fold.eval_end]
    y_eval = y[fold.eval_start:fold.eval_end]
    positive = float(y_train[:, 1].sum())
    negative = float(len(y_train) - positive)
    scale_pos_weight = (negative / max(positive, 1.0)) ** 0.6

    classifier = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=220,
        learning_rate=0.035,
        num_leaves=15,
        max_depth=-1,
        min_child_samples=80,
        subsample=0.90,
        colsample_bytree=0.85,
        reg_lambda=1.5,
        reg_alpha=0.05,
        scale_pos_weight=scale_pos_weight,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )
    classifier.fit(x_train, y_train[:, 1])
    probability = classifier.predict_proba(x_eval)[:, 1]

    regressor = lgb.LGBMRegressor(
        objective="huber",
        alpha=0.85,
        n_estimators=220,
        learning_rate=0.035,
        num_leaves=15,
        max_depth=-1,
        min_child_samples=80,
        subsample=0.90,
        colsample_bytree=0.85,
        reg_lambda=1.5,
        reg_alpha=0.05,
        random_state=seed + 1,
        n_jobs=-1,
        verbosity=-1,
    )
    regressor.fit(x_train, y_train[:, 0])
    intensity = regressor.predict(x_eval)

    cold = entity_cold_start_mask(meta, fold)
    cold_ap = _safe_ap(y_eval[cold, 1], probability[cold]) if cold.any() else None
    cold_mae = float(mean_absolute_error(y_eval[cold, 0], intensity[cold])) if cold.any() else None
    return {
        "train_samples": int(fold.train_end),
        "evaluation_samples": int(len(y_eval)),
        "train_end_date": fold.train_end_date,
        "evaluation_end_date": fold.eval_end_date,
        "average_precision": float(average_precision_score(y_eval[:, 1], probability)),
        "intensity_mae": float(mean_absolute_error(y_eval[:, 0], intensity)),
        "positive_rate": float(y_eval[:, 1].mean()),
        "cold_start": {
            "samples": int(cold.sum()),
            "average_precision": cold_ap,
            "intensity_mae": cold_mae,
            "definition": "PRIO entity absent from every sample before the fold training cutoff",
        },
    }


def summarize(candidate: list[dict[str, object]], baseline: list[dict[str, object]]) -> dict[str, object]:
    ap_delta = np.asarray([c["average_precision"] - b["average_precision"] for c, b in zip(candidate, baseline)], dtype=float)
    mae_delta = np.asarray([c["intensity_mae"] - b["intensity_mae"] for c, b in zip(candidate, baseline)], dtype=float)
    cold_ap_delta = []
    cold_mae_delta = []
    for c, b in zip(candidate, baseline):
        ca, ba = c["cold_start"]["average_precision"], b["cold_start"]["average_precision"]
        cm, bm = c["cold_start"]["intensity_mae"], b["cold_start"]["intensity_mae"]
        if ca is not None and ba is not None:
            cold_ap_delta.append(float(ca - ba))
        if cm is not None and bm is not None:
            cold_mae_delta.append(float(cm - bm))

    classification_accept = bool(ap_delta.mean() > 0 and (ap_delta > 0).sum() >= max(2, len(ap_delta) - 1) and ap_delta.min() > -0.01)
    intensity_accept = bool(mae_delta.mean() < 0 and (mae_delta < 0).sum() >= max(2, len(mae_delta) - 1) and mae_delta.max() < 0.02)
    cold_start_nonregression = True
    if cold_ap_delta:
        cold_start_nonregression &= float(np.mean(cold_ap_delta)) >= -0.005
    if cold_mae_delta:
        cold_start_nonregression &= float(np.mean(cold_mae_delta)) <= 0.01

    return {
        "mean_ap_delta": float(ap_delta.mean()),
        "fold_ap_deltas": ap_delta.tolist(),
        "mean_intensity_mae_delta": float(mae_delta.mean()),
        "fold_intensity_mae_deltas": mae_delta.tolist(),
        "cold_start_mean_ap_delta": float(np.mean(cold_ap_delta)) if cold_ap_delta else None,
        "cold_start_mean_intensity_mae_delta": float(np.mean(cold_mae_delta)) if cold_mae_delta else None,
        "classification_accept": classification_accept,
        "intensity_accept": intensity_accept,
        "cold_start_nonregression": bool(cold_start_nonregression),
        "stable_feature_accept": bool((classification_accept or intensity_accept) and cold_start_nonregression),
        "gate": (
            "Accept only if AP or intensity improves on average and in nearly every rolling fold, "
            "with bounded worst-fold regression and no material cold-start regression."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dynamic-dim", type=int, default=26)
    parser.add_argument("--train-fracs", default="0.55,0.65,0.75,0.85")
    parser.add_argument("--eval-frac", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260823)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output}")
    base = np.load(args.baseline, allow_pickle=True)
    cand = np.load(args.candidate, allow_pickle=True)
    if not np.array_equal(base["y"], cand["y"]) or not np.array_equal(base["meta"], cand["meta"]):
        raise ValueError("baseline/candidate labels or metadata are not aligned")
    if not np.array_equal(base["x"], cand["x"][..., : base["x"].shape[-1]]):
        raise ValueError("candidate does not preserve baseline channels exactly")

    y = base["y"].astype(np.float32)
    meta = parse_meta(base["meta"])
    train_fracs = tuple(float(v) for v in args.train_fracs.split(",") if v.strip())
    folds = build_folds(meta, train_fracs, args.eval_frac)
    if len(folds) < 2:
        raise RuntimeError("rolling-origin configuration produced fewer than two folds")

    print("building compact baseline features", flush=True)
    base_features = compact_features(base["x"], args.dynamic_dim)
    print("building compact candidate features", flush=True)
    candidate_features = compact_features(cand["x"], args.dynamic_dim)
    baseline_reports = []
    candidate_reports = []
    for i, fold in enumerate(folds):
        print(json.dumps({"fold": fold.name, "train_end": fold.train_end_date, "eval_end": fold.eval_end_date}), flush=True)
        baseline_reports.append(train_fold(base_features, y, meta, fold, args.seed + i * 10))
        candidate_reports.append(train_fold(candidate_features, y, meta, fold, args.seed + i * 10))

    report = {
        "schema": "rolling-origin-humanitarian-risk-v1",
        "baseline": str(args.baseline),
        "candidate": str(args.candidate),
        "dynamic_dim": args.dynamic_dim,
        "folds": [fold.__dict__ for fold in folds],
        "baseline_reports": baseline_reports,
        "candidate_reports": candidate_reports,
        "comparison": summarize(candidate_reports, baseline_reports),
        "safety_scope": "coarse humanitarian escalation/intensity and cold-start area transfer; no next-event coordinate ranking",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["comparison"], indent=2))


if __name__ == "__main__":
    main()
