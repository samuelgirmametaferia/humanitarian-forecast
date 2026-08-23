#!/usr/bin/env python3
"""Train a coarse humanitarian-risk model with an explicit unseen-area transfer objective.

The auxiliary objective randomly removes *local* cell history while preserving
neighboring humanitarian context and any broad static population/accessibility
features. The model must still predict coarse escalation/intensity. This is a
cold-start regularizer for newly active PRIO cells, not a next-event location
or route predictor.
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
from humanitarian_forecast.risk.train_v6 import (
    collect,
    conformal_absolute_radii,
    select_spatial_weight,
    sigmoid,
    state_cpu,
)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_model(x: torch.Tensor, train_end: int) -> MultiscaleTemporalRiskModel:
    mean = x[:train_end].mean(dim=(0, 1))
    std = x[:train_end].std(dim=(0, 1)).clamp_min(1e-3)
    return MultiscaleTemporalRiskModel(
        feature_dim=x.shape[-1],
        sequence_length=x.shape[1],
        feature_mean=mean,
        feature_std=std,
    )


def mask_local_history(xb: torch.Tensor, *, local_dim: int, probability: float) -> tuple[torch.Tensor, torch.Tensor]:
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be within [0,1]")
    if local_dim <= 0 or local_dim > xb.shape[-1]:
        raise ValueError(f"invalid local_dim={local_dim} for feature_dim={xb.shape[-1]}")
    chosen = torch.rand(len(xb), device=xb.device) < probability
    if not bool(chosen.any()):
        return xb, chosen
    masked = xb.clone()
    masked[chosen, :, :local_dim] = 0.0
    return masked, chosen


def train_classifier(
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
    local_dim: int,
    transfer_probability: float,
    transfer_weight: float,
) -> tuple[dict[str, torch.Tensor], dict[str, object], np.ndarray, np.ndarray]:
    seed_all(seed)
    model = make_model(x, train_end).to(device)
    positives = float(y[:train_end, 1].sum())
    negatives = float(train_end - positives)
    positive_weight = (negatives / max(1.0, positives)) ** 0.6
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight, device=device))
    regression = nn.SmoothL1Loss(beta=0.2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.02)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 12), eta_min=5e-5)
    loader = DataLoader(TensorDataset(x[:train_end], y[:train_end]), batch_size=batch_size, shuffle=True)

    best_ap = -math.inf
    best_epoch = 0
    best_state = None
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        transfer_total = 0.0
        transfer_examples = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            full = model(xb)
            primary = bce(full[:, 1], yb[:, 1]) + 0.6 * regression(full[:, 0], yb[:, 0])

            masked_x, chosen = mask_local_history(xb, local_dim=local_dim, probability=transfer_probability)
            if bool(chosen.any()):
                transfer = model(masked_x[chosen])
                auxiliary = bce(transfer[:, 1], yb[chosen, 1]) + 0.35 * regression(transfer[:, 0], yb[chosen, 0])
                # Consistency discourages extreme probability changes solely because
                # a cell has no local history, while leaving neighborhood/static context intact.
                consistency = nn.functional.smooth_l1_loss(
                    torch.sigmoid(transfer[:, 1]),
                    torch.sigmoid(full.detach()[chosen, 1]),
                    beta=0.05,
                )
                loss = primary + transfer_weight * (auxiliary + 0.25 * consistency)
                transfer_total += float(auxiliary.detach().cpu()) * int(chosen.sum())
                transfer_examples += int(chosen.sum())
            else:
                loss = primary
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach().cpu()) * len(xb)
        scheduler.step()
        val_output = collect(model, x, train_end, valid_end, device)
        ap = float(average_precision_score(y[train_end:valid_end, 1].numpy(), sigmoid(val_output[:, 1])))
        row = {
            "epoch": float(epoch),
            "training_loss": total / train_end,
            "transfer_loss": transfer_total / max(1, transfer_examples),
            "transfer_examples": float(transfer_examples),
            "validation_ap": ap,
        }
        history.append(row)
        print(json.dumps({"transfer_classifier": row}), flush=True)
        if ap > best_ap:
            best_ap, best_epoch, best_state = ap, epoch, state_cpu(model)

    if best_state is None:
        raise RuntimeError("transfer classifier produced no checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    return best_state, {
        "seed": seed,
        "best_epoch": best_epoch,
        "validation_ap": best_ap,
        "history": history,
        "positive_weight": positive_weight,
        "local_dim": local_dim,
        "transfer_probability": transfer_probability,
        "transfer_weight": transfer_weight,
        "model_config": model.config,
    }, collect(model, x, train_end, valid_end, device), collect(model, x, valid_end, len(x), device)


def train_intensity(
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
    local_dim: int,
    transfer_probability: float,
    transfer_weight: float,
) -> tuple[dict[str, torch.Tensor], dict[str, object], np.ndarray, np.ndarray]:
    seed_all(seed)
    model = make_model(x, train_end).to(device)
    loss_fn = nn.SmoothL1Loss(beta=0.2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.02)
    loader = DataLoader(TensorDataset(x[:train_end], y[:train_end]), batch_size=batch_size, shuffle=True)
    best_mae = math.inf
    best_epoch = 0
    best_state = None
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            full = model(xb)
            primary = loss_fn(full[:, 0], yb[:, 0])
            masked_x, chosen = mask_local_history(xb, local_dim=local_dim, probability=transfer_probability)
            if bool(chosen.any()):
                transfer = model(masked_x[chosen])
                auxiliary = loss_fn(transfer[:, 0], yb[chosen, 0])
                loss = primary + transfer_weight * auxiliary
            else:
                loss = primary
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach().cpu()) * len(xb)
        val_output = collect(model, x, train_end, valid_end, device)
        mae = float(mean_absolute_error(y[train_end:valid_end, 0].numpy(), val_output[:, 0]))
        row = {"epoch": float(epoch), "training_loss": total / train_end, "validation_mae": mae}
        history.append(row)
        print(json.dumps({"transfer_intensity": row}), flush=True)
        if mae < best_mae:
            best_mae, best_epoch, best_state = mae, epoch, state_cpu(model)
    if best_state is None:
        raise RuntimeError("transfer intensity model produced no checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    return best_state, {
        "seed": seed,
        "best_epoch": best_epoch,
        "validation_mae": best_mae,
        "history": history,
        "local_dim": local_dim,
        "transfer_probability": transfer_probability,
        "transfer_weight": transfer_weight,
        "model_config": model.config,
    }, collect(model, x, train_end, valid_end, device), collect(model, x, valid_end, len(x), device)


def masked_evaluation(model: nn.Module, x: torch.Tensor, y: np.ndarray, lo: int, hi: int, device: torch.device, local_dim: int) -> dict[str, float]:
    masked = x[lo:hi].clone()
    masked[..., :local_dim] = 0.0
    output = collect(model, masked, 0, len(masked), device)
    truth = y[lo:hi]
    return {
        "average_precision": float(average_precision_score(truth[:, 1], sigmoid(output[:, 1]))),
        "intensity_mae": float(mean_absolute_error(truth[:, 0], output[:, 0])),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-data", type=Path, required=True)
    p.add_argument("--context-data", type=Path, required=True)
    p.add_argument("--base-model-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--version", default="v7-transfer")
    p.add_argument("--local-dim", type=int, default=8)
    p.add_argument("--transfer-probability", type=float, default=0.25)
    p.add_argument("--transfer-weight", type=float, default=0.35)
    p.add_argument("--classifier-seed", type=int, default=20260823)
    p.add_argument("--classifier-epochs", type=int, default=4)
    p.add_argument("--intensity-seed", type=int, default=20260824)
    p.add_argument("--intensity-epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--learning-rate", type=float, default=4e-4)
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty model directory: {args.output_dir}")

    base = np.load(args.base_data, allow_pickle=True)
    context = np.load(args.context_data, allow_pickle=True)
    if not np.array_equal(base["y"], context["y"]) or not np.array_equal(base["meta"], context["meta"]):
        raise ValueError("base/context labels or metadata differ")
    x_base = base["x"].astype(np.float32, copy=False)
    x_np = context["x"].astype(np.float32, copy=False)
    y_np = base["y"].astype(np.float32, copy=False)
    x = torch.from_numpy(x_np)
    y = torch.from_numpy(y_np)
    train_end, valid_end = chronological_bounds(len(y))
    device = choose_device(args.device)
    print(f"device={device} feature_dim={x_np.shape[-1]} train={train_end:,} validation={valid_end-train_end:,} development={len(y)-valid_end:,}", flush=True)

    base_predictor = HumanitarianRiskPredictor(args.base_model_dir, str(device))
    base_prediction = base_predictor.predict_array(x_base)

    cls_state, cls_meta, cls_val, cls_dev = train_classifier(
        x, y, train_end, valid_end, device,
        seed=args.classifier_seed, epochs=args.classifier_epochs, batch_size=args.batch_size,
        learning_rate=args.learning_rate, local_dim=args.local_dim,
        transfer_probability=args.transfer_probability, transfer_weight=args.transfer_weight,
    )
    cls_val_p, cls_dev_p = sigmoid(cls_val[:, 1]), sigmoid(cls_dev[:, 1])
    truth_val = y_np[train_end:valid_end, 1]
    weight, raw_val, selected_val_ap = select_spatial_weight(
        truth_val, base_prediction["escalation_probability"][train_end:valid_end], cls_val_p,
    )
    raw_dev = (1.0 - weight) * base_prediction["escalation_probability"][valid_end:] + weight * cls_dev_p
    calibration = fit_affine_logit_calibration(truth_val, raw_val)
    val_p = apply_affine_logit_calibration(raw_val, calibration)
    dev_p = apply_affine_logit_calibration(raw_dev, calibration)
    threshold = best_balanced_threshold(truth_val, val_p)

    int_state, int_meta, int_val, int_dev = train_intensity(
        x, y, train_end, valid_end, device,
        seed=args.intensity_seed, epochs=args.intensity_epochs, batch_size=args.batch_size,
        learning_rate=args.learning_rate, local_dim=args.local_dim,
        transfer_probability=args.transfer_probability, transfer_weight=args.transfer_weight,
    )
    val_intensity, dev_intensity = int_val[:, 0], int_dev[:, 0]
    validation = classification_metrics(truth_val, val_p, threshold)
    development = classification_metrics(y_np[valid_end:, 1], dev_p, threshold)
    validation["intensity_mae"] = float(mean_absolute_error(y_np[train_end:valid_end, 0], val_intensity))
    development["intensity_mae"] = float(mean_absolute_error(y_np[valid_end:, 0], dev_intensity))
    conformal = conformal_absolute_radii(y_np[train_end:valid_end, 0], val_intensity)

    # Explicit synthetic cold-start audit on the trained experts.
    cls_model = make_model(x, train_end).to(device); cls_model.load_state_dict(cls_state); cls_model.eval()
    int_model = make_model(x, train_end).to(device); int_model.load_state_dict(int_state); int_model.eval()
    cls_mask_val = masked_evaluation(cls_model, x, y_np, train_end, valid_end, device, args.local_dim)
    cls_mask_dev = masked_evaluation(cls_model, x, y_np, valid_end, len(y_np), device, args.local_dim)
    # Intensity model needs its own masked output; keep AP out of this record because classifier differs.
    masked_val_x = x[train_end:valid_end].clone(); masked_val_x[..., :args.local_dim] = 0.0
    masked_dev_x = x[valid_end:].clone(); masked_dev_x[..., :args.local_dim] = 0.0
    int_mask_val = collect(int_model, masked_val_x, 0, len(masked_val_x), device)[:, 0]
    int_mask_dev = collect(int_model, masked_dev_x, 0, len(masked_dev_x), device)[:, 0]
    cold = {
        "validation": {
            "classifier_average_precision": cls_mask_val["average_precision"],
            "classifier_intensity_mae_aux": cls_mask_val["intensity_mae"],
            "intensity_mae": float(mean_absolute_error(y_np[train_end:valid_end, 0], int_mask_val)),
        },
        "development": {
            "classifier_average_precision": cls_mask_dev["average_precision"],
            "classifier_intensity_mae_aux": cls_mask_dev["intensity_mae"],
            "intensity_mae": float(mean_absolute_error(y_np[valid_end:, 0], int_mask_dev)),
        },
        "simulation": f"first {args.local_dim} local-history channels zeroed; neighborhood/static context preserved",
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_type":"MultiscaleTemporalRiskModel","model_config":cls_meta["model_config"],"model_state":cls_state,"best_epoch":cls_meta["best_epoch"],"target":"coarse humanitarian escalation with unseen-area transfer regularization"}, args.output_dir/"transfer_classifier.pt")
    torch.save({"model_type":"MultiscaleTemporalRiskModel","model_config":int_meta["model_config"],"model_state":int_state,"best_epoch":int_meta["best_epoch"],"target":"coarse humanitarian intensity with unseen-area transfer regularization"}, args.output_dir/"transfer_intensity.pt")
    ensemble = {
        "base_model_dir": str(args.base_model_dir),
        "probability_weights": {"base_v5": 1.0-weight, "transfer": weight},
        "weight_selection": {"partition":"chronological validation only","metric":"average_precision","selected_validation_ap":selected_val_ap},
        "affine_logit_calibration": calibration,
        "threshold": threshold,
        "classifier": "transfer_classifier.pt",
        "intensity_model": "transfer_intensity.pt",
        "intensity_conformal_absolute_radii_log1p": conformal,
        "transfer_objective": {"local_dim":args.local_dim,"mask_probability":args.transfer_probability,"loss_weight":args.transfer_weight},
    }
    (args.output_dir/"ensemble.json").write_text(json.dumps(ensemble,indent=2)+"\n")
    metrics = {
        "validation": validation,
        "development_holdout": development,
        "synthetic_cold_start": cold,
        "classifier_training": cls_meta,
        "intensity_training": int_meta,
        "ensemble": ensemble,
        "evaluation_caveat": "Historical final block is a development holdout. Promotion requires rolling-origin acceptance, including true unseen-entity folds.",
    }
    (args.output_dir/"ethiopia_metrics.json").write_text(json.dumps(metrics,indent=2)+"\n")
    write_info(args.output_dir, ModelInfo(
        subsystem="risk/ethiopia", version=args.version, status="research-challenger",
        description="Coarse humanitarian-risk challenger with explicit unseen-area transfer regularization via local-history masking.",
        metrics={"headline":development,"validation":validation,"synthetic_cold_start":cold},
        lineage={"base_data":str(args.base_data),"context_data":str(args.context_data),"base_model":str(args.base_model_dir)},
        training={"transfer_probability":args.transfer_probability,"transfer_weight":args.transfer_weight,"local_dim":args.local_dim,"split":{"train":0.70,"validation":0.15,"development_holdout":0.15}},
        calibration={"probability":calibration,"intensity_conformal_absolute_radii_log1p":conformal},
        notes=["Coarse humanitarian escalation/intensity only.","No next-event coordinate, route, unit-position, or tactical prediction is produced.","Rolling-origin true cold-start evaluation is required before promotion."],
    ))
    print(json.dumps({"validation":validation,"development":development,"synthetic_cold_start":cold,"transfer_weight_in_ensemble":weight},indent=2))


if __name__ == "__main__":
    main()
