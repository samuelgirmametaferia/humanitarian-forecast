#!/usr/bin/env python3
"""Train and evaluate a multiscale coarse humanitarian-risk challenger.

This module intentionally targets area-level humanitarian risk/intensity rather
than precise next-event coordinates. Hyperparameters and ensemble weights are
selected only on the chronological validation partition; the final partition is
read once for the untouched report.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    mean_absolute_error,
    precision_score,
    recall_score,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from humanitarian_forecast.core.model_store import ModelInfo, write_info
from humanitarian_forecast.risk.features import engineered_temporal_features
from humanitarian_forecast.risk.model import MultiscaleTemporalRiskModel, TemporalRiskTransformer


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(requested)


def chronological_bounds(n: int) -> tuple[int, int]:
    train_end = max(1, int(n * 0.70))
    valid_end = max(train_end + 1, int(n * 0.85))
    if valid_end >= n:
        raise ValueError("dataset is too small for 70/15/15 chronological split")
    return train_end, valid_end


def probabilities(logits: np.ndarray) -> np.ndarray:
    logits = np.clip(logits, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-logits))


def best_balanced_threshold(truth: np.ndarray, probs: np.ndarray) -> float:
    candidates = np.unique(np.quantile(probs, np.linspace(0.01, 0.99, 199)))
    return float(max(candidates, key=lambda t: balanced_accuracy_score(truth, probs >= t)))


def classification_metrics(truth: np.ndarray, probs: np.ndarray, threshold: float) -> dict[str, float]:
    truth_i = truth.astype(np.int64)
    pred = probs >= threshold
    clipped = np.clip(probs, 1e-7, 1 - 1e-7)
    return {
        "average_precision": float(average_precision_score(truth_i, probs)),
        "balanced_accuracy": float(balanced_accuracy_score(truth_i, pred)),
        "precision": float(precision_score(truth_i, pred, zero_division=0)),
        "recall": float(recall_score(truth_i, pred, zero_division=0)),
        "f1": float(f1_score(truth_i, pred, zero_division=0)),
        "brier": float(brier_score_loss(truth_i, probs)),
        "log_loss": float(log_loss(truth_i, clipped, labels=[0, 1])),
        "threshold": float(threshold),
        "positive_rate": float(truth_i.mean()),
    }


def collect_neural(
    model: nn.Module,
    x: torch.Tensor,
    start: int,
    end: int,
    device: torch.device,
    batch_size: int = 1024,
) -> np.ndarray:
    outputs: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for offset in range(start, end, batch_size):
            batch = x[offset:min(end, offset + batch_size)].to(device)
            outputs.append(model(batch).cpu())
    return torch.cat(outputs).numpy()


def load_reference_probabilities(
    checkpoint: Path,
    x: torch.Tensor,
    train_end: int,
    valid_end: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = TemporalRiskTransformer(**state["model_config"])
    model.load_state_dict(state["model_state"])
    model.to(device)
    validation = collect_neural(model, x, train_end, valid_end, device)
    test = collect_neural(model, x, valid_end, len(x), device)
    return validation[:, 0], probabilities(validation[:, 1]), test[:, 0], probabilities(test[:, 1])


def select_ensemble(
    truth: np.ndarray,
    neural: np.ndarray,
    reference: np.ndarray | None,
    tree: np.ndarray | None,
    step: float = 0.05,
) -> dict[str, object]:
    experts: list[tuple[str, np.ndarray]] = [("multiscale", neural)]
    if reference is not None:
        experts.append(("reference", reference))
    if tree is not None:
        experts.append(("tree", tree))

    if len(experts) == 1:
        threshold = best_balanced_threshold(truth, neural)
        return {"weights": {"multiscale": 1.0}, "threshold": threshold}

    increments = int(round(1.0 / step))
    best_ap = -math.inf
    best_weights: dict[str, float] | None = None
    best_blend: np.ndarray | None = None
    if len(experts) == 2:
        grids = [(i, increments - i) for i in range(increments + 1)]
    else:
        grids = [
            (i, j, increments - i - j)
            for i in range(increments + 1)
            for j in range(increments - i + 1)
        ]

    # Average precision is threshold-free, so threshold optimization belongs only
    # after the best blend is selected. Doing it inside this loop multiplies the
    # work by hundreds without changing the primary ranking decision.
    for counts in grids:
        weights = np.asarray(counts, dtype=np.float64) / increments
        blend = sum(float(weight) * values for weight, (_, values) in zip(weights, experts))
        ap = float(average_precision_score(truth, blend))
        if ap > best_ap:
            best_ap = ap
            best_blend = blend
            best_weights = {
                name: float(weight) for weight, (name, _) in zip(weights, experts)
            }

    assert best_weights is not None and best_blend is not None
    threshold = best_balanced_threshold(truth, best_blend)
    return {"weights": best_weights, "threshold": threshold}


def blend_probabilities(weights: dict[str, float], expert_probs: dict[str, np.ndarray]) -> np.ndarray:
    missing = set(weights) - set(expert_probs)
    if missing:
        raise KeyError(f"missing ensemble probabilities for {sorted(missing)}")
    return sum(float(weight) * expert_probs[name] for name, weight in weights.items())


def fit_affine_logit_calibration(truth: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    """Fit a validation-only affine logit calibration (temperature + bias)."""
    clipped = np.clip(probs, 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    calibrator = LogisticRegression(C=1e6, solver="lbfgs", random_state=0)
    calibrator.fit(logits, truth.astype(np.int64))
    scale = float(calibrator.coef_[0, 0])
    bias = float(calibrator.intercept_[0])
    return {
        "scale": scale,
        "temperature": float(1.0 / scale) if scale > 0 else math.inf,
        "bias": bias,
    }


def apply_affine_logit_calibration(probs: np.ndarray, calibration: dict[str, float]) -> np.ndarray:
    clipped = np.clip(probs, 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped))
    calibrated_logits = float(calibration["scale"]) * logits + float(calibration["bias"])
    return probabilities(calibrated_logits)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--reference-checkpoint", type=Path)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=4e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--no-tree", action="store_true")
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty model directory: {args.output_dir}")

    seed_all(args.seed)
    device = choose_device(args.device)
    bundle = np.load(args.data)
    x_np = bundle["x"].astype(np.float32, copy=False)
    y_np = bundle["y"].astype(np.float32, copy=False)
    x = torch.from_numpy(x_np)
    y = torch.from_numpy(y_np)
    train_end, valid_end = chronological_bounds(len(x))

    feature_mean = x[:train_end].mean(dim=(0, 1))
    feature_std = x[:train_end].std(dim=(0, 1)).clamp_min(1e-3)
    model = MultiscaleTemporalRiskModel(
        feature_dim=x.shape[-1],
        sequence_length=x.shape[1],
        feature_mean=feature_mean,
        feature_std=feature_std,
    ).to(device)

    positives = float(y[:train_end, 1].sum())
    negatives = float(train_end - positives)
    # Temper class weighting: full inverse prevalence over-emphasized recall in v4.
    positive_weight = (negatives / max(1.0, positives)) ** 0.6
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight, device=device))
    regression = nn.SmoothL1Loss(beta=0.2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.02)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs), eta_min=5e-5
    )
    loader = DataLoader(
        TensorDataset(x[:train_end], y[:train_end]),
        batch_size=args.batch_size,
        shuffle=True,
    )

    best_ap = -math.inf
    best_mae = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []
    print(
        f"device={device} samples={len(x):,} train={train_end:,} "
        f"validation={valid_end-train_end:,} test={len(x)-valid_end:,} "
        f"positive_weight={positive_weight:.4f}",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(xb)
            loss = 0.6 * regression(output[:, 0], yb[:, 0]) + bce(output[:, 1], yb[:, 1])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach().cpu()) * len(xb)
        scheduler.step()

        validation_out = collect_neural(model, x, train_end, valid_end, device)
        validation_probs = probabilities(validation_out[:, 1])
        validation_ap = float(average_precision_score(y_np[train_end:valid_end, 1], validation_probs))
        validation_mae = float(mean_absolute_error(y_np[train_end:valid_end, 0], validation_out[:, 0]))
        record = {
            "epoch": float(epoch),
            "training_loss": total / train_end,
            "validation_average_precision": validation_ap,
            "validation_intensity_mae": validation_mae,
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)

        improved = validation_ap > best_ap + 1e-8 or (
            abs(validation_ap - best_ap) <= 1e-8 and validation_mae < best_mae
        )
        if improved:
            best_ap = validation_ap
            best_mae = validation_mae
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= args.patience:
            print(f"early_stop epoch={epoch} best_epoch={best_epoch}", flush=True)
            break

    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.to(device)

    validation_out = collect_neural(model, x, train_end, valid_end, device)
    test_out = collect_neural(model, x, valid_end, len(x), device)
    neural_validation = probabilities(validation_out[:, 1])
    neural_test = probabilities(test_out[:, 1])

    tree_model = None
    tree_validation = tree_test = None
    if not args.no_tree:
        engineered = engineered_temporal_features(x_np)
        tree_model = HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.05,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            min_samples_leaf=30,
            early_stopping=False,
            random_state=args.seed,
        )
        tree_model.fit(engineered[:train_end], y_np[:train_end, 1].astype(np.int64))
        tree_validation = tree_model.predict_proba(engineered[train_end:valid_end])[:, 1]
        tree_test = tree_model.predict_proba(engineered[valid_end:])[:, 1]

    reference_validation = reference_test = None
    reference_intensity_validation = reference_intensity_test = None
    if args.reference_checkpoint:
        (
            reference_intensity_validation,
            reference_validation,
            reference_intensity_test,
            reference_test,
        ) = load_reference_probabilities(args.reference_checkpoint, x, train_end, valid_end, device)

    truth_validation = y_np[train_end:valid_end, 1]
    ensemble = select_ensemble(
        truth_validation,
        neural_validation,
        reference_validation,
        tree_validation,
    )
    weights = dict(ensemble["weights"])
    validation_experts = {"multiscale": neural_validation}
    test_experts = {"multiscale": neural_test}
    if reference_validation is not None and reference_test is not None:
        validation_experts["reference"] = reference_validation
        test_experts["reference"] = reference_test
    if tree_validation is not None and tree_test is not None:
        validation_experts["tree"] = tree_validation
        test_experts["tree"] = tree_test

    validation_blend_raw = blend_probabilities(weights, validation_experts)
    test_blend_raw = blend_probabilities(weights, test_experts)
    affine_calibration = fit_affine_logit_calibration(truth_validation, validation_blend_raw)
    validation_blend = apply_affine_logit_calibration(validation_blend_raw, affine_calibration)
    test_blend = apply_affine_logit_calibration(test_blend_raw, affine_calibration)
    threshold = best_balanced_threshold(truth_validation, validation_blend)
    validation_report = classification_metrics(truth_validation, validation_blend, threshold)
    test_report = classification_metrics(y_np[valid_end:, 1], test_blend, threshold)
    validation_report["intensity_mae"] = float(
        mean_absolute_error(y_np[train_end:valid_end, 0], validation_out[:, 0])
    )
    test_report["intensity_mae"] = float(mean_absolute_error(y_np[valid_end:, 0], test_out[:, 0]))

    component_metrics: dict[str, object] = {
        "multiscale_validation_ap": float(average_precision_score(truth_validation, neural_validation)),
        "multiscale_test_ap": float(average_precision_score(y_np[valid_end:, 1], neural_test)),
    }
    if tree_validation is not None and tree_test is not None:
        component_metrics.update(
            tree_validation_ap=float(average_precision_score(truth_validation, tree_validation)),
            tree_test_ap=float(average_precision_score(y_np[valid_end:, 1], tree_test)),
        )
    if reference_validation is not None and reference_test is not None:
        ref_threshold = best_balanced_threshold(truth_validation, reference_validation)
        component_metrics.update(
            reference_validation=classification_metrics(truth_validation, reference_validation, ref_threshold),
            reference_test=classification_metrics(y_np[valid_end:, 1], reference_test, ref_threshold),
            reference_validation_intensity_mae=float(
                mean_absolute_error(y_np[train_end:valid_end, 0], reference_intensity_validation)
            ),
            reference_test_intensity_mae=float(
                mean_absolute_error(y_np[valid_end:, 0], reference_intensity_test)
            ),
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "stage": "ethiopia-humanitarian-risk",
        "model_type": "MultiscaleTemporalRiskModel",
        "model_config": model.config,
        "model_state": best_state,
        "best_epoch": best_epoch,
        "seed": args.seed,
        "chronological_split": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "target_scope": "coarse humanitarian risk/intensity; no precise next-event location output",
    }
    torch.save(checkpoint, args.output_dir / "multiscale_best.pt")
    if tree_model is not None:
        joblib.dump(tree_model, args.output_dir / "tree_classifier.joblib")

    ensemble_payload = {
        "weights": weights,
        "threshold": threshold,
        "reference_checkpoint": str(args.reference_checkpoint) if args.reference_checkpoint else None,
        "selection": "weights maximize validation average precision; threshold maximizes validation balanced accuracy",
        "affine_logit_calibration": affine_calibration,
    }
    (args.output_dir / "ensemble.json").write_text(
        json.dumps(ensemble_payload, indent=2) + "\n", encoding="utf-8"
    )
    metrics_payload = {
        "metric_implementation": "scikit-learn 1.9 standard average_precision_score",
        "best_epoch": best_epoch,
        "training_history": history,
        "ensemble": ensemble_payload,
        "validation": validation_report,
        "untouched_test": test_report,
        "components": component_metrics,
    }
    (args.output_dir / "ethiopia_metrics.json").write_text(
        json.dumps(metrics_payload, indent=2) + "\n", encoding="utf-8"
    )

    reference_ap = None
    reference_mae = None
    reference_ba = None
    ref_test = component_metrics.get("reference_test")
    if isinstance(ref_test, dict):
        reference_ap = float(ref_test["average_precision"])
        reference_ba = float(ref_test["balanced_accuracy"])
        reference_mae = float(component_metrics["reference_test_intensity_mae"])

    promotion = {
        "decision": "challenger",
        "reason": "requires prospective/rolling-origin confirmation before production promotion",
    }
    if reference_ap is not None and reference_mae is not None and reference_ba is not None:
        promotion["offline_comparison"] = {
            "ap_relative_change": test_report["average_precision"] / reference_ap - 1.0,
            "intensity_mae_relative_change": test_report["intensity_mae"] / reference_mae - 1.0,
            "balanced_accuracy_change": test_report["balanced_accuracy"] - reference_ba,
        }

    write_info(
        args.output_dir,
        ModelInfo(
            subsystem="risk/ethiopia",
            version=args.version,
            status="validated-challenger",
            description=(
                "Multiscale coarse humanitarian-risk challenger with featurewise normalization, "
                "local temporal convolutions, Transformer context, explicit multiscale summaries, "
                "and validation-selected expert ensembling."
            ),
            metrics={"headline": test_report, "validation": validation_report, "components": component_metrics},
            lineage={
                "dataset": str(args.data),
                "reference_checkpoint": str(args.reference_checkpoint) if args.reference_checkpoint else None,
            },
            training={
                "samples": len(x),
                "best_epoch": best_epoch,
                "epochs_requested": args.epochs,
                "patience": args.patience,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "positive_weight": positive_weight,
                "seed": args.seed,
                "split": {"train": 0.70, "validation": 0.15, "test": 0.15},
            },
            calibration={"ensemble": ensemble_payload},
            notes=[
                "Untouched test rows were not used for architecture, checkpoint, ensemble-weight, or threshold selection.",
                "Artifact is restricted to coarse humanitarian-risk/intensity forecasting and is not a tactical location predictor.",
                json.dumps(promotion, sort_keys=True),
            ],
        ),
    )
    print(json.dumps({"validation": validation_report, "untouched_test": test_report, "promotion": promotion}, indent=2))


if __name__ == "__main__":
    main()
