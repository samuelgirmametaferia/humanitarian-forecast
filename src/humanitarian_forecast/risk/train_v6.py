#!/usr/bin/env python3
"""Train v6 coarse humanitarian early-warning experts.

v6 keeps the validated v5 local-history predictor as a stable base expert, adds
causal PRIO-grid spillover context through a small spatial neural expert, and
uses a separately optimized spatial model for intensity. All checkpoint,
ensemble-weight, calibration, threshold, and uncertainty choices are made on
the chronological validation partition. The final partition is reported only
as a development holdout because this repository has undergone iterative model
research on the historical corpus.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, mean_absolute_error
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from humanitarian_forecast.core.model_store import ModelInfo, write_info
from humanitarian_forecast.risk.model import MultiscaleTemporalRiskModel
from humanitarian_forecast.risk.predict import HumanitarianRiskPredictor
from humanitarian_forecast.risk.train_challenger import (
    apply_affine_logit_calibration,
    best_balanced_threshold,
    classification_metrics,
    chronological_bounds,
    choose_device,
    fit_affine_logit_calibration,
)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sigmoid(logits: np.ndarray) -> np.ndarray:
    logits = np.clip(logits, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-logits))


def state_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def make_model(x: torch.Tensor, train_end: int) -> MultiscaleTemporalRiskModel:
    mean = x[:train_end].mean(dim=(0, 1))
    std = x[:train_end].std(dim=(0, 1)).clamp_min(1e-3)
    return MultiscaleTemporalRiskModel(
        feature_dim=x.shape[-1],
        sequence_length=x.shape[1],
        feature_mean=mean,
        feature_std=std,
    )


def collect(
    model: nn.Module,
    x: torch.Tensor,
    lo: int,
    hi: int,
    device: torch.device,
    batch_size: int = 1024,
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(lo, hi, batch_size):
            xb = x[start:min(hi, start + batch_size)].to(device)
            outputs.append(model(xb).cpu().numpy())
    return np.concatenate(outputs)


def train_spatial_classifier(
    x: torch.Tensor,
    y: torch.Tensor,
    train_end: int,
    valid_end: int,
    device: torch.device,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> tuple[dict[str, torch.Tensor], dict[str, object], np.ndarray, np.ndarray]:
    seed_all(seed)
    model = make_model(x, train_end).to(device)
    positives = float(y[:train_end, 1].sum())
    negatives = float(train_end - positives)
    positive_weight = (negatives / max(1.0, positives)) ** 0.6
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight, device=device))
    regression = nn.SmoothL1Loss(beta=0.2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.02)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 12), eta_min=5e-5
    )
    loader = DataLoader(
        TensorDataset(x[:train_end], y[:train_end]), batch_size=batch_size, shuffle=True
    )

    best_ap = -math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(xb)
            # A small auxiliary magnitude objective improved classification
            # validation AP in the spatial encoder, while AP remains the sole
            # checkpoint-selection criterion.
            loss = bce(output[:, 1], yb[:, 1]) + 0.6 * regression(output[:, 0], yb[:, 0])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach().cpu()) * len(xb)
        scheduler.step()
        val_output = collect(model, x, train_end, valid_end, device)
        val_probability = sigmoid(val_output[:, 1])
        ap = float(average_precision_score(y[train_end:valid_end, 1].numpy(), val_probability))
        row = {"epoch": float(epoch), "training_loss": total / train_end, "validation_ap": ap}
        history.append(row)
        print(json.dumps({"spatial_classifier": row}), flush=True)
        if ap > best_ap:
            best_ap = ap
            best_epoch = epoch
            best_state = state_cpu(model)

    if best_state is None:
        raise RuntimeError("spatial classifier produced no checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    val_output = collect(model, x, train_end, valid_end, device)
    dev_output = collect(model, x, valid_end, len(x), device)
    return best_state, {
        "seed": seed,
        "best_epoch": best_epoch,
        "validation_ap": best_ap,
        "history": history,
        "positive_weight": positive_weight,
        "model_config": model.config,
    }, val_output, dev_output


def train_spatial_intensity(
    x: torch.Tensor,
    y: torch.Tensor,
    train_end: int,
    valid_end: int,
    device: torch.device,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> tuple[dict[str, torch.Tensor], dict[str, object], np.ndarray, np.ndarray]:
    seed_all(seed)
    model = make_model(x, train_end).to(device)
    loss_fn = nn.SmoothL1Loss(beta=0.2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.02)
    loader = DataLoader(
        TensorDataset(x[:train_end], y[:train_end]), batch_size=batch_size, shuffle=True
    )

    best_mae = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(xb)
            loss = loss_fn(output[:, 0], yb[:, 0])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach().cpu()) * len(xb)
        val_output = collect(model, x, train_end, valid_end, device)
        mae = float(mean_absolute_error(y[train_end:valid_end, 0].numpy(), val_output[:, 0]))
        row = {"epoch": float(epoch), "training_loss": total / train_end, "validation_mae": mae}
        history.append(row)
        print(json.dumps({"spatial_intensity": row}), flush=True)
        if mae < best_mae:
            best_mae = mae
            best_epoch = epoch
            best_state = state_cpu(model)

    if best_state is None:
        raise RuntimeError("spatial intensity model produced no checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    val_output = collect(model, x, train_end, valid_end, device)
    dev_output = collect(model, x, valid_end, len(x), device)
    return best_state, {
        "seed": seed,
        "best_epoch": best_epoch,
        "validation_mae": best_mae,
        "history": history,
        "model_config": model.config,
    }, val_output, dev_output


def select_spatial_weight(
    truth: np.ndarray,
    base_probability: np.ndarray,
    spatial_probability: np.ndarray,
    max_weight: float = 0.50,
    step: float = 0.05,
) -> tuple[float, np.ndarray, float]:
    candidates = np.arange(0.0, max_weight + step / 2.0, step)
    best = (-math.inf, 0.0, base_probability)
    for weight in candidates:
        blend = (1.0 - weight) * base_probability + weight * spatial_probability
        ap = float(average_precision_score(truth, blend))
        if ap > best[0] + 1e-12:
            best = (ap, float(weight), blend)
    return best[1], best[2], best[0]


def conformal_absolute_radii(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    residuals = np.abs(np.asarray(truth) - np.asarray(prediction))
    n = len(residuals)
    result: dict[str, float] = {}
    for coverage in (0.80, 0.90, 0.95):
        level = min(1.0, math.ceil((n + 1) * coverage) / n)
        radius = float(np.quantile(residuals, level, method="higher"))
        result[f"p{int(coverage * 100)}"] = radius
    return result


def assert_aligned(base: np.lib.npyio.NpzFile, spatial: np.lib.npyio.NpzFile) -> None:
    if len(base["x"]) != len(spatial["x"]):
        raise ValueError("base/spatial sample counts differ")
    if not np.array_equal(base["y"], spatial["y"]):
        raise ValueError("base/spatial labels differ")
    if "meta" in base and "meta" in spatial and not np.array_equal(base["meta"], spatial["meta"]):
        raise ValueError("base/spatial metadata order differs")
    if spatial["x"].shape[-1] < base["x"].shape[-1]:
        raise ValueError("spatial feature tensor has fewer channels than base tensor")
    if not np.array_equal(spatial["x"][..., : base["x"].shape[-1]], base["x"]):
        raise ValueError("spatial tensor does not preserve base channels exactly")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-data", type=Path, required=True)
    parser.add_argument("--spatial-data", type=Path, required=True)
    parser.add_argument("--base-model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--version", default="v6")
    parser.add_argument("--classifier-seed", type=int, default=20260823)
    parser.add_argument("--classifier-epochs", type=int, default=3)
    parser.add_argument("--intensity-seed", type=int, default=20260824)
    parser.add_argument("--intensity-epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=4e-4)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty model directory: {args.output_dir}")

    base = np.load(args.base_data, allow_pickle=True)
    spatial = np.load(args.spatial_data, allow_pickle=True)
    assert_aligned(base, spatial)
    x_base_np = base["x"].astype(np.float32, copy=False)
    x_spatial_np = spatial["x"].astype(np.float32, copy=False)
    y_np = base["y"].astype(np.float32, copy=False)
    x_spatial = torch.from_numpy(x_spatial_np)
    y = torch.from_numpy(y_np)
    train_end, valid_end = chronological_bounds(len(y))
    device = choose_device(args.device)
    print(
        f"device={device} samples={len(y):,} train={train_end:,} "
        f"validation={valid_end-train_end:,} development={len(y)-valid_end:,}",
        flush=True,
    )

    base_predictor = HumanitarianRiskPredictor(args.base_model_dir, str(device))
    base_prediction = base_predictor.predict_array(x_base_np)
    base_probability = base_prediction["escalation_probability"]

    cls_state, cls_meta, cls_val_output, cls_dev_output = train_spatial_classifier(
        x_spatial,
        y,
        train_end,
        valid_end,
        device,
        seed=args.classifier_seed,
        epochs=args.classifier_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    spatial_val_probability = sigmoid(cls_val_output[:, 1])
    spatial_dev_probability = sigmoid(cls_dev_output[:, 1])
    truth_val = y_np[train_end:valid_end, 1]
    spatial_weight, raw_val_probability, selected_val_ap = select_spatial_weight(
        truth_val,
        base_probability[train_end:valid_end],
        spatial_val_probability,
    )
    raw_dev_probability = (
        (1.0 - spatial_weight) * base_probability[valid_end:]
        + spatial_weight * spatial_dev_probability
    )
    calibration = fit_affine_logit_calibration(truth_val, raw_val_probability)
    val_probability = apply_affine_logit_calibration(raw_val_probability, calibration)
    dev_probability = apply_affine_logit_calibration(raw_dev_probability, calibration)
    threshold = best_balanced_threshold(truth_val, val_probability)

    intensity_state, intensity_meta, intensity_val_output, intensity_dev_output = train_spatial_intensity(
        x_spatial,
        y,
        train_end,
        valid_end,
        device,
        seed=args.intensity_seed,
        epochs=args.intensity_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    val_intensity = intensity_val_output[:, 0]
    dev_intensity = intensity_dev_output[:, 0]
    conformal = conformal_absolute_radii(y_np[train_end:valid_end, 0], val_intensity)

    validation_report = classification_metrics(truth_val, val_probability, threshold)
    development_report = classification_metrics(y_np[valid_end:, 1], dev_probability, threshold)
    validation_report["intensity_mae"] = float(
        mean_absolute_error(y_np[train_end:valid_end, 0], val_intensity)
    )
    development_report["intensity_mae"] = float(
        mean_absolute_error(y_np[valid_end:, 0], dev_intensity)
    )

    base_val_ap = float(average_precision_score(truth_val, base_probability[train_end:valid_end]))
    base_dev_ap = float(average_precision_score(y_np[valid_end:, 1], base_probability[valid_end:]))
    spatial_val_ap = float(average_precision_score(truth_val, spatial_val_probability))
    spatial_dev_ap = float(average_precision_score(y_np[valid_end:, 1], spatial_dev_probability))
    base_dev_mae = float(mean_absolute_error(y_np[valid_end:, 0], base_prediction["intensity_log1p"][valid_end:]))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_type": "MultiscaleTemporalRiskModel",
            "model_config": cls_meta["model_config"],
            "model_state": cls_state,
            "best_epoch": cls_meta["best_epoch"],
            "seed": args.classifier_seed,
            "target": "coarse spatial humanitarian escalation risk",
        },
        args.output_dir / "spatial_classifier.pt",
    )
    torch.save(
        {
            "model_type": "MultiscaleTemporalRiskModel",
            "model_config": intensity_meta["model_config"],
            "model_state": intensity_state,
            "best_epoch": intensity_meta["best_epoch"],
            "seed": args.intensity_seed,
            "target": "coarse spatial humanitarian intensity",
        },
        args.output_dir / "spatial_intensity.pt",
    )

    ensemble = {
        "base_model_dir": "../v5",
        "probability_weights": {"base_v5": 1.0 - spatial_weight, "spatial": spatial_weight},
        "weight_selection": {
            "partition": "chronological validation only",
            "metric": "average_precision",
            "grid": "spatial weight 0.00..0.50 in 0.05 increments",
            "selected_validation_ap": selected_val_ap,
        },
        "affine_logit_calibration": calibration,
        "threshold": threshold,
        "intensity_model": "spatial_intensity.pt",
        "spatial_classifier": "spatial_classifier.pt",
        "spatial_feature_contract": "local8 + PRIO ring1 sum8 + ring2 sum8 + historical active-neighbor counts2",
        "intensity_conformal_absolute_radii_log1p": conformal,
    }
    (args.output_dir / "ensemble.json").write_text(json.dumps(ensemble, indent=2) + "\n", encoding="utf-8")

    comparison = {
        "ap_relative_change_vs_v5": development_report["average_precision"] / base_dev_ap - 1.0,
        "intensity_mae_relative_change_vs_v5": development_report["intensity_mae"] / base_dev_mae - 1.0,
    }
    metrics = {
        "metric_implementation": "scikit-learn standard metrics",
        "evaluation_caveat": (
            "Architecture research has inspected the final historical block. Treat it as a development "
            "holdout, not a pristine prospective test; promotion requires rolling-origin/prospective confirmation."
        ),
        "validation": validation_report,
        "development_holdout": development_report,
        "components": {
            "base_v5": {"validation_ap": base_val_ap, "development_ap": base_dev_ap, "development_intensity_mae": base_dev_mae},
            "spatial_classifier": {"validation_ap": spatial_val_ap, "development_ap": spatial_dev_ap},
            "spatial_intensity": {
                "validation_mae": validation_report["intensity_mae"],
                "development_mae": development_report["intensity_mae"],
            },
        },
        "ensemble": ensemble,
        "comparison_vs_v5": comparison,
        "classifier_training": cls_meta,
        "intensity_training": intensity_meta,
    }
    (args.output_dir / "ethiopia_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    write_info(
        args.output_dir,
        ModelInfo(
            subsystem="risk/ethiopia",
            version=args.version,
            status="validated-challenger",
            description=(
                "v6 coarse humanitarian early-warning bundle: v5 local-risk expert plus a causally "
                "constructed PRIO-neighborhood spatial expert, separate spatial intensity model, "
                "probability calibration, and split-conformal intensity uncertainty."
            ),
            metrics={
                "headline": development_report,
                "validation": validation_report,
                "comparison_vs_v5": comparison,
                "components": metrics["components"],
            },
            lineage={
                "base_model": str(args.base_model_dir),
                "base_data": str(args.base_data),
                "spatial_data": str(args.spatial_data),
                "spatial_builder": "humanitarian_forecast.data.build_spatial_risk_dataset",
            },
            training={
                "classifier_seed": args.classifier_seed,
                "classifier_epochs": args.classifier_epochs,
                "intensity_seed": args.intensity_seed,
                "intensity_epochs": args.intensity_epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "split": {"train": 0.70, "validation": 0.15, "development_holdout": 0.15},
            },
            calibration={
                "probability": calibration,
                "intensity_conformal_absolute_radii_log1p": conformal,
            },
            notes=[
                "No precise next-event location, route, unit-position, or tactical-output subsystem was modified.",
                "Ensemble weight, probability calibration, threshold, checkpoint epochs, and conformal radii use validation data only.",
                "The historical final block is a development benchmark after iterative research, not a prospective untouched test.",
            ],
        ),
    )
    print(json.dumps({
        "validation": validation_report,
        "development_holdout": development_report,
        "comparison_vs_v5": comparison,
        "spatial_weight": spatial_weight,
        "conformal": conformal,
    }, indent=2))


if __name__ == "__main__":
    main()
