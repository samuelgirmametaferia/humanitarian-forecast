#!/usr/bin/env python3
"""Train a dense multiresolution H3 GeoState Transformer.

This is the first location model in the repository whose neural output support is
not a bank of historical conflict coordinates. It predicts a probability field
over all H3 cells supplied by the boundary compiler at several resolutions.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from humanitarian_forecast.core.model_store import ModelInfo, write_info
from humanitarian_forecast.location.models.geostate import GeoStateConfig, GeoStateTransformer

PRESETS = {
    "S": dict(d_model=256, heads=8, layers=6, ff_dim=768),
    "M": dict(d_model=384, heads=8, layers=8, ff_dim=1152),
    "L": dict(d_model=768, heads=12, layers=16, ff_dim=3072),
}


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def device_for(name: str) -> torch.device:
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(name)


def _latlon(rows: list[dict[str, object]], prefix: str) -> np.ndarray:
    return np.asarray([[float(row[f"{prefix}_lat"]), float(row[f"{prefix}_lon"])] for row in rows], dtype=np.float32)


def _soft_loss(
    logits: torch.Tensor,
    centroids: torch.Tensor,
    truth_latlon: torch.Tensor,
    radius_km: float,
    expected_distance_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lat = centroids[:, 0][None]
    lon = centroids[:, 1][None]
    tlat = truth_latlon[:, 0:1]
    tlon = truth_latlon[:, 1:2]
    north = (lat - tlat) * 111.32
    east = (lon - tlon) * 111.32 * torch.cos(torch.deg2rad(tlat))
    distance = torch.sqrt(east.square() + north.square() + 1e-6)
    q = torch.softmax(-0.5 * (distance / radius_km).square(), dim=-1)
    logp = torch.log_softmax(logits, dim=-1)
    soft_ce = -(q * logp).sum(dim=-1).mean()
    expected = (torch.softmax(logits, dim=-1) * distance).sum(dim=-1).mean() / 100.0
    return soft_ce + expected_distance_weight * expected, soft_ce, expected


def _batch_loss(
    model: GeoStateTransformer,
    x: torch.Tensor,
    anchor: torch.Tensor,
    truth: torch.Tensor,
    targets: dict[int, torch.Tensor],
    centroids: dict[int, torch.Tensor],
    static_features: dict[int, torch.Tensor] | None,
    hard_weight: float,
    soft_weight: float,
    expected_distance_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    outputs = model(x, anchor, centroids, static_features)
    loss = torch.zeros((), device=x.device)
    details: dict[str, float] = {}
    total_resolution_weight = 0.0
    for resolution, logits in outputs.items():
        target = targets[resolution]
        valid = target >= 0
        if not bool(valid.any()):
            continue
        hard = nn.functional.cross_entropy(logits[valid], target[valid])
        radius = {3: 200.0, 4: 100.0, 5: 50.0}.get(resolution, 100.0)
        soft, soft_ce, expected = _soft_loss(
            logits[valid], centroids[resolution], truth[valid], radius, expected_distance_weight
        )
        resolution_weight = {3: 0.50, 4: 0.75, 5: 1.0}.get(resolution, 1.0)
        loss = loss + resolution_weight * (hard_weight * hard + soft_weight * soft)
        total_resolution_weight += resolution_weight
        details[f"r{resolution}_hard_ce"] = float(hard.detach().cpu())
        details[f"r{resolution}_soft_ce"] = float(soft_ce.detach().cpu())
        details[f"r{resolution}_expected_100km"] = float(expected.detach().cpu())
    return loss / max(total_resolution_weight, 1e-8), details


@torch.no_grad()
def evaluate(
    model: GeoStateTransformer,
    x: torch.Tensor,
    anchor: torch.Tensor,
    truth: torch.Tensor,
    targets: dict[int, torch.Tensor],
    centroids: dict[int, torch.Tensor],
    static_features: dict[int, torch.Tensor] | None,
    batch_size: int,
) -> dict[str, object]:
    model.eval()
    device = next(model.parameters()).device
    reports: dict[str, object] = {}
    for resolution in model.config.resolutions:
        all_error: list[np.ndarray] = []
        all_entropy: list[np.ndarray] = []
        all_ce: list[np.ndarray] = []
        all_top3: list[np.ndarray] = []
        c = centroids[resolution]
        for start in range(0, len(x), batch_size):
            stop = min(len(x), start + batch_size)
            xb = x[start:stop].to(device)
            ab = anchor[start:stop].to(device)
            tb = truth[start:stop].to(device)
            target = targets[resolution][start:stop].to(device)
            valid = target >= 0
            if not bool(valid.any()):
                continue
            history = model.encode_history(xb)
            logits = model.logits_for_resolution(
                history,
                ab,
                c,
                resolution,
                None if static_features is None else static_features[resolution],
            )
            probability = torch.softmax(logits, dim=-1)
            top = logits.argmax(dim=-1)
            chosen = c[top]
            north = (chosen[:, 0] - tb[:, 0]) * 111.32
            east = (chosen[:, 1] - tb[:, 1]) * 111.32 * torch.cos(torch.deg2rad(tb[:, 0]))
            error = torch.sqrt(east.square() + north.square())
            top3_idx = torch.topk(logits, k=min(3, logits.shape[1]), dim=-1).indices
            top3_c = c[top3_idx]
            tlat = tb[:, 0:1]
            tlon = tb[:, 1:2]
            dn = (top3_c[..., 0] - tlat) * 111.32
            de = (top3_c[..., 1] - tlon) * 111.32 * torch.cos(torch.deg2rad(tlat))
            top3_error = torch.sqrt(de.square() + dn.square()).amin(dim=-1)
            entropy = -(probability * torch.log(probability.clamp_min(1e-12))).sum(dim=-1)
            ce = nn.functional.cross_entropy(logits[valid], target[valid], reduction="none")
            all_error.append(error[valid].cpu().numpy())
            all_top3.append(top3_error[valid].cpu().numpy())
            all_entropy.append(entropy[valid].cpu().numpy())
            all_ce.append(ce.cpu().numpy())
        error = np.concatenate(all_error)
        top3 = np.concatenate(all_top3)
        reports[f"r{resolution}"] = {
            "samples": int(len(error)),
            "mean_error_km": float(error.mean()),
            "median_error_km": float(np.median(error)),
            "p90_error_km": float(np.quantile(error, 0.90)),
            "within_25km": float(np.mean(error <= 25)),
            "within_50km": float(np.mean(error <= 50)),
            "within_100km": float(np.mean(error <= 100)),
            "within_200km": float(np.mean(error <= 200)),
            "top3_within_100km": float(np.mean(top3 <= 100)),
            "cross_entropy": float(np.concatenate(all_ce).mean()),
            "mean_entropy": float(np.concatenate(all_entropy).mean()),
        }
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preset", choices=tuple(PRESETS), default="M")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.03)
    parser.add_argument("--hard-weight", type=float, default=1.0)
    parser.add_argument("--soft-weight", type=float, default=0.35)
    parser.add_argument("--expected-distance-weight", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--backbone", type=Path, help="Optional globally pretrained GeoState backbone checkpoint.")
    parser.add_argument("--static-context", type=Path, help="Optional H3 population/accessibility context NPZ.")
    parser.add_argument("--freeze-backbone-epochs", type=int, default=5)
    parser.add_argument("--backbone-lr-scale", type=float, default=0.15)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty directory: {args.output_dir}")
    seed_all(args.seed)
    device = device_for(args.device)

    z = np.load(args.data, allow_pickle=True)
    x_np = z["x"].astype(np.float32, copy=False)
    rows = [json.loads(str(value)) for value in z["meta"]]
    anchor_np = _latlon(rows, "anchor")
    truth_np = _latlon(rows, "target")
    source_indices = z["source_indices"].astype(np.int64)
    train_end = int(z["source_train_end"])
    valid_end = int(z["source_validation_end"])
    train_mask = source_indices < train_end
    val_mask = (source_indices >= train_end) & (source_indices < valid_end)
    dev_mask = source_indices >= valid_end
    resolutions = tuple(int(v) for v in z["resolutions"])

    x = torch.from_numpy(x_np)
    anchor = torch.from_numpy(anchor_np)
    truth = torch.from_numpy(truth_np)
    target_all = {r: torch.from_numpy(z[f"target_index_r{r}"].astype(np.int64)) for r in resolutions}
    centroid_cpu = {r: torch.from_numpy(z[f"centroids_r{r}"].astype(np.float32)) for r in resolutions}
    centroids = {r: value.to(device) for r, value in centroid_cpu.items()}
    static_features: dict[int, torch.Tensor] | None = None
    static_dim = 0
    if args.static_context:
        sz = np.load(args.static_context, allow_pickle=True)
        static_cpu = {r: torch.from_numpy(sz[f"features_r{r}"].astype(np.float32)) for r in resolutions}
        for r in resolutions:
            if len(static_cpu[r]) != len(centroid_cpu[r]):
                raise ValueError(f"static H3 alignment failed at r{r}")
        static_dim = int(static_cpu[resolutions[0]].shape[-1])
        if any(int(static_cpu[r].shape[-1]) != static_dim for r in resolutions):
            raise ValueError("static context dimension differs across H3 resolutions")
        static_features = {r: value.to(device) for r, value in static_cpu.items()}

    preset = dict(PRESETS[args.preset])
    # PyTorch 2.13's MPS scaled-dot-product attention path currently rejects
    # non-zero attention dropout during this dense fine-tuning workload. The
    # pretrained weights remain fully compatible; only fine-tuning regularization
    # changes on MPS. CPU/CUDA keep the configured dropout.
    if device.type == "mps":
        preset["dropout"] = 0.0
    config = GeoStateConfig(
        event_dim=x.shape[-1],
        sequence_length=x.shape[1],
        resolutions=resolutions,
        cells_per_resolution=tuple(len(z[f"cells_r{r}"]) for r in resolutions),
        static_dim=static_dim,
        **preset,
    )
    model = GeoStateTransformer(config).to(device)
    pretrained_backbone = None
    if args.backbone:
        pretrained_backbone = torch.load(args.backbone, map_location="cpu", weights_only=False)
        state = pretrained_backbone["backbone_state_dict"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        # Dense-cell heads are expected to be absent; shared temporal/query weights must align.
        unexpected = [key for key in unexpected if not key.startswith(("cell_embeddings.", "geo_bias.", "temperature."))]
        if unexpected:
            raise ValueError(f"unexpected pretrained backbone keys: {unexpected}")
        print(json.dumps({"loaded_backbone": str(args.backbone), "missing_dense_keys": len(missing)}), flush=True)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(json.dumps({"device": str(device), "preset": args.preset, "parameters": parameter_count, "config": config.__dict__}, default=list), flush=True)

    train_idx = torch.from_numpy(np.flatnonzero(train_mask).astype(np.int64))
    loader = DataLoader(TensorDataset(train_idx), batch_size=args.batch_size, shuffle=True)
    dense_prefixes = ("cell_embeddings.", "geo_bias.", "temperature.", "static_encoder.", "static_bias.")
    backbone_parameters = [parameter for name, parameter in model.named_parameters() if not name.startswith(dense_prefixes)]
    dense_parameters = [parameter for name, parameter in model.named_parameters() if name.startswith(dense_prefixes)]
    if args.backbone:
        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_parameters, "lr": args.learning_rate * args.backbone_lr_scale},
                {"params": dense_parameters, "lr": args.learning_rate},
            ],
            weight_decay=args.weight_decay,
        )
        if args.freeze_backbone_epochs > 0:
            for parameter in backbone_parameters:
                parameter.requires_grad_(False)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    best_state = None
    best_value = float("inf")
    best_epoch = 0
    history: list[dict[str, float]] = []
    val_idx = np.flatnonzero(val_mask)
    dev_idx = np.flatnonzero(dev_mask)

    for epoch in range(1, args.epochs + 1):
        if args.backbone and epoch == args.freeze_backbone_epochs + 1:
            for parameter in backbone_parameters:
                parameter.requires_grad_(True)
            print(json.dumps({"epoch": epoch, "backbone_unfrozen": True, "backbone_lr_scale": args.backbone_lr_scale}), flush=True)
        model.train()
        total = 0.0
        seen = 0
        for (index,) in loader:
            index = index.long()
            xb = x[index].to(device)
            ab = anchor[index].to(device)
            tb = truth[index].to(device)
            targets = {r: target_all[r][index].to(device) for r in resolutions}
            optimizer.zero_grad(set_to_none=True)
            loss, _ = _batch_loss(
                model, xb, ab, tb, targets, centroids, static_features,
                args.hard_weight, args.soft_weight, args.expected_distance_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach().cpu()) * len(index)
            seen += len(index)
        scheduler.step()
        val_report = evaluate(
            model,
            x[val_idx], anchor[val_idx], truth[val_idx],
            {r: target_all[r][val_idx] for r in resolutions}, centroids, static_features, args.batch_size,
        )
        finest = f"r{max(resolutions)}"
        value = float(val_report[finest]["cross_entropy"])
        row = {"epoch": epoch, "train_loss": total / max(1, seen), "validation_finest_ce": value}
        history.append(row)
        print(json.dumps({**row, "validation": val_report}), flush=True)
        if value < best_value:
            best_value = value
            best_epoch = epoch
            best_state = {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}

    assert best_state is not None
    model.load_state_dict(best_state)
    validation = evaluate(
        model, x[val_idx], anchor[val_idx], truth[val_idx],
        {r: target_all[r][val_idx] for r in resolutions}, centroids, static_features, args.batch_size,
    )
    development = evaluate(
        model, x[dev_idx], anchor[dev_idx], truth[dev_idx],
        {r: target_all[r][dev_idx] for r in resolutions}, centroids, static_features, args.batch_size,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": best_state,
        "config": config.__dict__,
        "preset": args.preset,
        "parameter_count": parameter_count,
        "selected_epoch": best_epoch,
    }
    torch.save(checkpoint, args.output_dir / "geostate_best.pt")
    report = {
        "schema": "dense-h3-geostate-v1",
        "parameter_count": parameter_count,
        "preset": args.preset,
        "selected_epoch": best_epoch,
        "validation": validation,
        "development": development,
        "history": history,
        "protocol": "global 70/15/15 chronological boundaries applied to Ethiopia rows; validation selects epoch; final block is development-only",
        "evaluation_caveat": "The historical final block has been repeatedly inspected by prior research and is not a pristine prospective test.",
    }
    (args.output_dir / "geostate_metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    write_info(
        args.output_dir,
        ModelInfo(
            subsystem="location/geostate",
            version="v1",
            status="research-challenger",
            description="Dense multiresolution H3 spatial Transformer with distance-soft supervision.",
            metrics={"validation": validation, "development": development},
            lineage={"dataset": str(args.data), "restore_tag": "production-boost-preflight-2026-08-24"},
            training={
                "preset": args.preset,
                "parameters": parameter_count,
                "epochs": args.epochs,
                "selected_epoch": best_epoch,
                "hard_weight": args.hard_weight,
                "soft_weight": args.soft_weight,
                "expected_distance_weight": args.expected_distance_weight,
            },
            notes=[
                "Dense output support is independent of historically observed candidate coordinates.",
                "Public interpretation remains broad humanitarian geography; exact point errors are diagnostics.",
            ],
        ),
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
