#!/usr/bin/env python3
"""Train a coarse humanitarian-risk challenger with an explicit cold-start objective.

The auxiliary objective masks the local PRIO cell's dynamic history while keeping
neighbor spillover and broad static context. This simulates a newly active or
previously unseen area and forces the encoder to transfer information from its
surroundings/context instead of relying only on local recurrence.
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
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


def state_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def make_model(x: torch.Tensor, train_end: int) -> MultiscaleTemporalRiskModel:
    mean = x[:train_end].mean(dim=(0, 1))
    std = x[:train_end].std(dim=(0, 1)).clamp_min(1e-3)
    return MultiscaleTemporalRiskModel(
        feature_dim=x.shape[-1], sequence_length=x.shape[1],
        feature_mean=mean, feature_std=std,
    )


def mask_local(x: torch.Tensor, local_dim: int) -> torch.Tensor:
    out = x.clone()
    out[..., :local_dim] = 0.0
    return out


def collect(model: nn.Module, x: torch.Tensor, lo: int, hi: int, device: torch.device, *, local_dim: int, masked: bool, batch_size: int = 1024) -> np.ndarray:
    result = []
    model.eval()
    with torch.no_grad():
        for start in range(lo, hi, batch_size):
            xb = x[start:min(hi, start + batch_size)].to(device)
            if masked:
                xb = mask_local(xb, local_dim)
            result.append(model(xb).cpu().numpy())
    return np.concatenate(result)


def select_blend(truth: np.ndarray, base: np.ndarray, expert: np.ndarray) -> tuple[float, np.ndarray, float]:
    best = (-math.inf, 0.0, base)
    for weight in np.arange(0.0, 0.81, 0.05):
        blend = (1.0 - weight) * base + weight * expert
        ap = float(average_precision_score(truth, blend))
        if ap > best[0] + 1e-12:
            best = (ap, float(weight), blend)
    return best[1], best[2], best[0]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-data", type=Path, required=True)
    p.add_argument("--context-data", type=Path, required=True)
    p.add_argument("--base-model-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--version", default="v7-transfer")
    p.add_argument("--local-dim", type=int, default=8)
    p.add_argument("--seed", type=int, default=20260826)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--learning-rate", type=float, default=4e-4)
    p.add_argument("--mask-probability", type=float, default=0.30)
    p.add_argument("--transfer-weight", type=float, default=0.55)
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty model directory: {args.output_dir}")

    base = np.load(args.base_data, allow_pickle=True)
    ctx = np.load(args.context_data, allow_pickle=True)
    if not np.array_equal(base["y"], ctx["y"]) or not np.array_equal(base["meta"], ctx["meta"]):
        raise ValueError("base/context labels or metadata differ")
    if not np.array_equal(ctx["x"][..., :base["x"].shape[-1]], base["x"]):
        raise ValueError("context tensor does not preserve base channels exactly")
    if args.local_dim <= 0 or args.local_dim >= ctx["x"].shape[-1]:
        raise ValueError("local-dim must leave non-local context channels")

    x_np = ctx["x"].astype(np.float32, copy=False)
    y_np = ctx["y"].astype(np.float32, copy=False)
    x = torch.from_numpy(x_np); y = torch.from_numpy(y_np)
    train_end, valid_end = chronological_bounds(len(y))
    device = choose_device(args.device)
    seed_all(args.seed)
    model = make_model(x, train_end).to(device)

    positives = float(y_np[:train_end, 1].sum())
    negatives = float(train_end - positives)
    pos_weight = (negatives / max(1.0, positives)) ** 0.6
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    reg = nn.SmoothL1Loss(beta=0.2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.02)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 12), eta_min=5e-5)
    loader = DataLoader(TensorDataset(x[:train_end], y[:train_end]), batch_size=args.batch_size, shuffle=True)

    best_score = -math.inf
    best_state = None
    best_epoch = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train(); total = 0.0; transfer_rows = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            normal = model(xb)
            normal_loss = bce(normal[:, 1], yb[:, 1]) + 0.6 * reg(normal[:, 0], yb[:, 0])

            row_mask = torch.rand(len(xb), device=device) < args.mask_probability
            if bool(row_mask.any()):
                cold_x = mask_local(xb[row_mask], args.local_dim)
                cold_y = yb[row_mask]
                cold = model(cold_x)
                cold_loss = bce(cold[:, 1], cold_y[:, 1]) + 0.6 * reg(cold[:, 0], cold_y[:, 0])
                # Mild consistency term stops the masked auxiliary task from drifting
                # into an unrelated calibration regime.
                target_prob = torch.sigmoid(normal[row_mask, 1].detach())
                cold_prob = torch.sigmoid(cold[:, 1])
                consistency = torch.mean((cold_prob - target_prob) ** 2)
                loss = normal_loss + args.transfer_weight * cold_loss + 0.10 * consistency
                transfer_rows += int(row_mask.sum().item())
            else:
                loss = normal_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach().cpu()) * len(xb)
        scheduler.step()

        val = collect(model, x, train_end, valid_end, device, local_dim=args.local_dim, masked=False)
        cold_val = collect(model, x, train_end, valid_end, device, local_dim=args.local_dim, masked=True)
        truth = y_np[train_end:valid_end, 1]
        normal_ap = float(average_precision_score(truth, sigmoid(val[:, 1])))
        cold_ap = float(average_precision_score(truth, sigmoid(cold_val[:, 1])))
        normal_mae = float(mean_absolute_error(y_np[train_end:valid_end, 0], val[:, 0]))
        cold_mae = float(mean_absolute_error(y_np[train_end:valid_end, 0], cold_val[:, 0]))
        # Ordinary performance dominates; transfer AP is an explicit secondary objective.
        composite = 0.75 * normal_ap + 0.25 * cold_ap
        row = {
            "epoch": epoch,
            "training_loss": total / train_end,
            "transfer_rows": transfer_rows,
            "validation_ap": normal_ap,
            "cold_start_validation_ap": cold_ap,
            "validation_intensity_mae": normal_mae,
            "cold_start_validation_intensity_mae": cold_mae,
            "selection_score": composite,
        }
        history.append(row); print(json.dumps(row), flush=True)
        if composite > best_score:
            best_score = composite; best_epoch = epoch; best_state = state_cpu(model)

    if best_state is None:
        raise RuntimeError("no transfer checkpoint selected")
    model.load_state_dict(best_state); model.to(device).eval()

    # Persist the selected checkpoint before any downstream prediction/ensemble
    # bookkeeping so a later packaging error can never destroy completed compute.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_type": "MultiscaleTemporalRiskModel",
        "model_config": model.config,
        "model_state": best_state,
        "best_epoch": best_epoch,
        "seed": args.seed,
        "objective": "ordinary humanitarian risk + masked-local cold-start transfer auxiliary loss",
        "local_dim": args.local_dim,
        "status": "selected-checkpoint-before-packaging",
    }, args.output_dir / "transfer_model.pt")
    (args.output_dir / "training_history.json").write_text(json.dumps({"best_epoch": best_epoch, "selection_score": best_score, "history": history}, indent=2) + "\n")

    val = collect(model, x, train_end, valid_end, device, local_dim=args.local_dim, masked=False)
    dev = collect(model, x, valid_end, len(x), device, local_dim=args.local_dim, masked=False)
    cold_val = collect(model, x, train_end, valid_end, device, local_dim=args.local_dim, masked=True)
    cold_dev = collect(model, x, valid_end, len(x), device, local_dim=args.local_dim, masked=True)
    expert_val = sigmoid(val[:, 1]); expert_dev = sigmoid(dev[:, 1])
    cold_val_prob = sigmoid(cold_val[:, 1]); cold_dev_prob = sigmoid(cold_dev[:, 1])

    base_predictor = HumanitarianRiskPredictor(args.base_model_dir, str(device))
    base_x = base["x"].astype(np.float32, copy=False)
    base_pred = base_predictor.predict_array(base_x)
    truth_val = y_np[train_end:valid_end, 1]
    weight, raw_val, selected_ap = select_blend(truth_val, base_pred["escalation_probability"][train_end:valid_end], expert_val)
    raw_dev = (1.0 - weight) * base_pred["escalation_probability"][valid_end:] + weight * expert_dev
    calibration = fit_affine_logit_calibration(truth_val, raw_val)
    val_prob = apply_affine_logit_calibration(raw_val, calibration)
    dev_prob = apply_affine_logit_calibration(raw_dev, calibration)
    threshold = best_balanced_threshold(truth_val, val_prob)

    normal_validation = classification_metrics(truth_val, val_prob, threshold)
    normal_development = classification_metrics(y_np[valid_end:, 1], dev_prob, threshold)
    normal_validation["intensity_mae"] = float(mean_absolute_error(y_np[train_end:valid_end, 0], val[:, 0]))
    normal_development["intensity_mae"] = float(mean_absolute_error(y_np[valid_end:, 0], dev[:, 0]))
    transfer = {
        "validation": {
            "average_precision": float(average_precision_score(truth_val, cold_val_prob)),
            "intensity_mae": float(mean_absolute_error(y_np[train_end:valid_end, 0], cold_val[:, 0])),
        },
        "development": {
            "average_precision": float(average_precision_score(y_np[valid_end:, 1], cold_dev_prob)),
            "intensity_mae": float(mean_absolute_error(y_np[valid_end:, 0], cold_dev[:, 0])),
        },
        "challenge": f"all local channels [0:{args.local_dim}] zeroed; neighbor/static context retained",
    }

    ensemble = {
        "base_model_dir": str(args.base_model_dir),
        "probability_weights": {"base": 1.0 - weight, "transfer": weight},
        "validation_selected_ap": selected_ap,
        "affine_logit_calibration": calibration,
        "threshold": threshold,
        "local_dim": args.local_dim,
        "transfer_mask_probability": args.mask_probability,
        "transfer_weight": args.transfer_weight,
    }
    (args.output_dir / "ensemble.json").write_text(json.dumps(ensemble, indent=2) + "\n")
    metrics = {
        "validation": normal_validation,
        "development_holdout": normal_development,
        "cold_start_transfer": transfer,
        "training": {"best_epoch": best_epoch, "history": history, "selection_score": best_score},
        "ensemble": ensemble,
        "evaluation_caveat": "Historical final block is a development benchmark; promotion requires rolling-origin/prospective confirmation.",
    }
    (args.output_dir / "ethiopia_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    write_info(args.output_dir, ModelInfo(
        subsystem="risk/ethiopia",
        version=args.version,
        status="research-challenger",
        description="Coarse humanitarian-risk challenger with explicit masked-local cold-start transfer objective.",
        metrics={"headline": normal_development, "validation": normal_validation, "cold_start_transfer": transfer},
        lineage={"base_data": str(args.base_data), "context_data": str(args.context_data), "base_model": str(args.base_model_dir)},
        training={"seed": args.seed, "epochs": args.epochs, "mask_probability": args.mask_probability, "transfer_weight": args.transfer_weight, "local_dim": args.local_dim},
        calibration={"probability": calibration},
        notes=[
            "Auxiliary objective masks local-cell history and retains neighboring/static coarse context.",
            "This model predicts coarse humanitarian escalation/intensity, not next-event coordinates or routes.",
            "Normal validation remains the dominant checkpoint-selection component.",
        ],
    ))
    print(json.dumps({"validation": normal_validation, "development": normal_development, "cold_start_transfer": transfer, "expert_weight": weight}, indent=2))


if __name__ == "__main__":
    main()
