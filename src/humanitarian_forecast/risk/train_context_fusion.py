#!/usr/bin/env python3
"""Train a separated static/dynamic context-fusion humanitarian-risk challenger."""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, log_loss, mean_absolute_error
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from humanitarian_forecast.core.model_store import ModelInfo, write_info
from humanitarian_forecast.risk.model import ContextFusionRiskModel
from humanitarian_forecast.risk.predict_v6 import HumanitarianRiskV6Predictor
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


def make_model(x: torch.Tensor, train_end: int, dynamic_dim: int) -> ContextFusionRiskModel:
    dynamic = x[:train_end, :, :dynamic_dim]
    static = x[:train_end, -1, dynamic_dim:]
    return ContextFusionRiskModel(
        feature_dim=x.shape[-1], dynamic_dim=dynamic_dim, sequence_length=x.shape[1],
        dynamic_mean=dynamic.mean(dim=(0, 1)),
        dynamic_std=dynamic.std(dim=(0, 1)).clamp_min(1e-3),
        static_mean=static.mean(dim=0),
        static_std=static.std(dim=0).clamp_min(1e-3),
    )


def collect(model: nn.Module, x: torch.Tensor, lo: int, hi: int, device: torch.device, batch_size: int = 1024) -> np.ndarray:
    rows = []
    model.eval()
    with torch.no_grad():
        for start in range(lo, hi, batch_size):
            rows.append(model(x[start:min(hi, start + batch_size)].to(device)).cpu().numpy())
    return np.concatenate(rows)


def train_classifier(x: torch.Tensor, y: torch.Tensor, train_end: int, valid_end: int, device: torch.device, dynamic_dim: int, seed: int, epochs: int, batch_size: int, learning_rate: float):
    seed_all(seed)
    model = make_model(x, train_end, dynamic_dim).to(device)
    positives = float(y[:train_end, 1].sum()); negatives = float(train_end - positives)
    pos_weight = (negatives / max(1.0, positives)) ** 0.6
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    reg = nn.SmoothL1Loss(beta=0.2)
    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.02)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 12), eta_min=5e-5)
    loader = DataLoader(TensorDataset(x[:train_end], y[:train_end]), batch_size=batch_size, shuffle=True)
    best_ap = -math.inf; best_state = None; best_epoch = 0; history = []
    for epoch in range(1, epochs + 1):
        model.train(); total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            loss = bce(out[:, 1], yb[:, 1]) + 0.45 * reg(out[:, 0], yb[:, 0])
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            total += float(loss.detach().cpu()) * len(xb)
        sched.step()
        val = collect(model, x, train_end, valid_end, device)
        ap = float(average_precision_score(y[train_end:valid_end, 1].numpy(), sigmoid(val[:, 1])))
        row = {"epoch": epoch, "training_loss": total/train_end, "validation_ap": ap}
        history.append(row); print(json.dumps({"context_classifier": row}), flush=True)
        if ap > best_ap:
            best_ap = ap; best_epoch = epoch; best_state = state_cpu(model)
    if best_state is None: raise RuntimeError("no context classifier checkpoint")
    model.load_state_dict(best_state); model.to(device).eval()
    return model, best_state, {"best_epoch": best_epoch, "validation_ap": best_ap, "history": history}


def train_intensity(x: torch.Tensor, y: torch.Tensor, train_end: int, valid_end: int, device: torch.device, dynamic_dim: int, seed: int, epochs: int, batch_size: int, learning_rate: float):
    seed_all(seed)
    model = make_model(x, train_end, dynamic_dim).to(device)
    loss_fn = nn.SmoothL1Loss(beta=0.2)
    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.02)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 12), eta_min=5e-5)
    loader = DataLoader(TensorDataset(x[:train_end], y[:train_end]), batch_size=batch_size, shuffle=True)
    best_mae = math.inf; best_state = None; best_epoch = 0; history = []
    for epoch in range(1, epochs + 1):
        model.train(); total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True); out = model(xb)
            loss = loss_fn(out[:, 0], yb[:, 0]); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            total += float(loss.detach().cpu()) * len(xb)
        sched.step()
        val = collect(model, x, train_end, valid_end, device)
        mae = float(mean_absolute_error(y[train_end:valid_end, 0].numpy(), val[:, 0]))
        row = {"epoch": epoch, "training_loss": total/train_end, "validation_mae": mae}
        history.append(row); print(json.dumps({"context_intensity": row}), flush=True)
        if mae < best_mae:
            best_mae = mae; best_epoch = epoch; best_state = state_cpu(model)
    if best_state is None: raise RuntimeError("no context intensity checkpoint")
    model.load_state_dict(best_state); model.to(device).eval()
    return model, best_state, {"best_epoch": best_epoch, "validation_mae": best_mae, "history": history}


def choose_probability_weight(truth: np.ndarray, base: np.ndarray, expert: np.ndarray, max_weight: float = 0.50) -> tuple[float, np.ndarray, float]:
    best = (math.inf, 0.0, base)
    for w in np.arange(0.0, max_weight + 0.001, 0.05):
        p = np.clip((1-w)*base + w*expert, 1e-6, 1-1e-6)
        score = float(log_loss(truth, p))
        if score < best[0] - 1e-12:
            best = (score, float(w), p)
    return best[1], best[2], best[0]


def choose_intensity_weight(truth: np.ndarray, base: np.ndarray, expert: np.ndarray, max_weight: float = 0.75) -> tuple[float, np.ndarray, float]:
    best = (math.inf, 0.0, base)
    for w in np.arange(0.0, max_weight + 0.001, 0.05):
        pred = (1-w)*base + w*expert
        score = float(mean_absolute_error(truth, pred))
        if score < best[0] - 1e-12:
            best = (score, float(w), pred)
    return best[1], best[2], best[0]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--context-data", type=Path, required=True)
    p.add_argument("--base-v6-model", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--version", default="v7-context-fusion")
    p.add_argument("--dynamic-dim", type=int, default=26)
    p.add_argument("--classifier-seed", type=int, default=20260828)
    p.add_argument("--classifier-epochs", type=int, default=5)
    p.add_argument("--intensity-seed", type=int, default=20260829)
    p.add_argument("--intensity-epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--learning-rate", type=float, default=4e-4)
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty model directory: {args.output_dir}")

    z = np.load(args.context_data, allow_pickle=True)
    x_np = z["x"].astype(np.float32, copy=False); y_np = z["y"].astype(np.float32, copy=False)
    if not args.dynamic_dim < x_np.shape[-1]: raise ValueError("context data has no static channels")
    x = torch.from_numpy(x_np); y = torch.from_numpy(y_np)
    train_end, valid_end = chronological_bounds(len(y)); device = choose_device(args.device)
    print(f"device={device} samples={len(y):,} dynamic={args.dynamic_dim} static={x_np.shape[-1]-args.dynamic_dim}", flush=True)

    cls_model, cls_state, cls_meta = train_classifier(x,y,train_end,valid_end,device,args.dynamic_dim,args.classifier_seed,args.classifier_epochs,args.batch_size,args.learning_rate)
    int_model, int_state, int_meta = train_intensity(x,y,train_end,valid_end,device,args.dynamic_dim,args.intensity_seed,args.intensity_epochs,args.batch_size,args.learning_rate)
    cls_val=collect(cls_model,x,train_end,valid_end,device); cls_dev=collect(cls_model,x,valid_end,len(x),device)
    int_val=collect(int_model,x,train_end,valid_end,device); int_dev=collect(int_model,x,valid_end,len(x),device)

    base_predictor = HumanitarianRiskV6Predictor(args.base_v6_model, str(device))
    base_pred = base_predictor.predict_array(x_np[..., :args.dynamic_dim])
    truth_val=y_np[train_end:valid_end,1]; truth_dev=y_np[valid_end:,1]
    expert_val_raw=sigmoid(cls_val[:,1]); expert_dev_raw=sigmoid(cls_dev[:,1])
    expert_cal=fit_affine_logit_calibration(truth_val,expert_val_raw)
    expert_val=apply_affine_logit_calibration(expert_val_raw,expert_cal)
    expert_dev=apply_affine_logit_calibration(expert_dev_raw,expert_cal)
    prob_w, raw_val, selected_ll=choose_probability_weight(truth_val,base_pred["escalation_probability"][train_end:valid_end],expert_val)
    raw_dev=(1-prob_w)*base_pred["escalation_probability"][valid_end:]+prob_w*expert_dev
    ensemble_cal=fit_affine_logit_calibration(truth_val,raw_val)
    val_prob=apply_affine_logit_calibration(raw_val,ensemble_cal); dev_prob=apply_affine_logit_calibration(raw_dev,ensemble_cal)
    threshold=best_balanced_threshold(truth_val,val_prob)

    int_w,val_int,selected_mae=choose_intensity_weight(y_np[train_end:valid_end,0],base_pred["intensity_log1p"][train_end:valid_end],int_val[:,0])
    dev_int=(1-int_w)*base_pred["intensity_log1p"][valid_end:]+int_w*int_dev[:,0]
    validation=classification_metrics(truth_val,val_prob,threshold); development=classification_metrics(truth_dev,dev_prob,threshold)
    validation["intensity_mae"]=float(mean_absolute_error(y_np[train_end:valid_end,0],val_int)); development["intensity_mae"]=float(mean_absolute_error(y_np[valid_end:,0],dev_int))

    args.output_dir.mkdir(parents=True,exist_ok=True)
    torch.save({"model_type":"ContextFusionRiskModel","model_config":cls_model.config,"model_state":cls_state,"best_epoch":cls_meta["best_epoch"]},args.output_dir/"context_classifier.pt")
    torch.save({"model_type":"ContextFusionRiskModel","model_config":int_model.config,"model_state":int_state,"best_epoch":int_meta["best_epoch"]},args.output_dir/"context_intensity.pt")
    ensemble={
        "base_v6_model":str(args.base_v6_model),
        "probability_weights":{"v6":1-prob_w,"context_fusion":prob_w},
        "probability_selection":"validation binary log loss after expert-only affine calibration",
        "expert_affine_logit_calibration":expert_cal,
        "ensemble_affine_logit_calibration":ensemble_cal,
        "selected_validation_log_loss":selected_ll,
        "threshold":threshold,
        "intensity_weights":{"v6":1-int_w,"context_fusion":int_w},
        "selected_validation_intensity_mae":selected_mae,
        "dynamic_dim":args.dynamic_dim,
    }
    (args.output_dir/"ensemble.json").write_text(json.dumps(ensemble,indent=2)+"\n")
    metrics={
        "validation":validation,"development_holdout":development,
        "components":{
            "v6":{"validation_ap":float(average_precision_score(truth_val,base_pred["escalation_probability"][train_end:valid_end])),"development_ap":float(average_precision_score(truth_dev,base_pred["escalation_probability"][valid_end:])),"validation_intensity_mae":float(mean_absolute_error(y_np[train_end:valid_end,0],base_pred["intensity_log1p"][train_end:valid_end])),"development_intensity_mae":float(mean_absolute_error(y_np[valid_end:,0],base_pred["intensity_log1p"][valid_end:]))},
            "context_fusion":{"validation_ap_raw":float(average_precision_score(truth_val,expert_val_raw)),"development_ap_raw":float(average_precision_score(truth_dev,expert_dev_raw)),"validation_intensity_mae":float(mean_absolute_error(y_np[train_end:valid_end,0],int_val[:,0])),"development_intensity_mae":float(mean_absolute_error(y_np[valid_end:,0],int_dev[:,0]))},
        },
        "ensemble":ensemble,"classifier_training":cls_meta,"intensity_training":int_meta,
        "evaluation_caveat":"Final historical block is a development benchmark. Static representation passed a separate rolling-origin feature gate; prospective confirmation remains required.",
    }
    (args.output_dir/"ethiopia_metrics.json").write_text(json.dumps(metrics,indent=2)+"\n")
    write_info(args.output_dir,ModelInfo(
        subsystem="risk/ethiopia",version=args.version,status="research-challenger",
        description="Separated dynamic/static coarse humanitarian-risk model with bounded FiLM context fusion over spatial history plus population/accessibility.",
        metrics={"headline":development,"validation":validation,"components":metrics["components"]},
        lineage={"context_data":str(args.context_data),"base_v6_model":str(args.base_v6_model)},
        training={"classifier_seed":args.classifier_seed,"classifier_epochs":args.classifier_epochs,"intensity_seed":args.intensity_seed,"intensity_epochs":args.intensity_epochs,"dynamic_dim":args.dynamic_dim},
        calibration={"expert_probability":expert_cal,"ensemble_probability":ensemble_cal},
        notes=["Static context is encoded once, not repeated through temporal convolutions.","Bounded FiLM modulation prevents static geography from replacing dynamic evidence.","Outputs are coarse humanitarian escalation/intensity signals only."],
    ))
    print(json.dumps({"validation":validation,"development_holdout":development,"ensemble":ensemble,"context_component":metrics["components"]["context_fusion"]},indent=2))


if __name__=="__main__": main()
