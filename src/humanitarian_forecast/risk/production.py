#!/usr/bin/env python3
"""All-data production training for global and Ethiopia temporal-risk models."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from humanitarian_forecast.core.model_store import ModelInfo, read_info, write_info
from humanitarian_forecast.risk.model import TemporalRiskTransformer


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("base", "ethiopia"), required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--reference-model-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--ranking-weight", type=float, default=0.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260810)
    args = parser.parse_args()

    seed_all(args.seed)
    device = choose_device(args.device)
    bundle = np.load(args.data)
    x = torch.from_numpy(bundle["x"]).float()
    y = torch.from_numpy(bundle["y"]).float()
    loader = DataLoader(TensorDataset(x, y), batch_size=args.batch_size, shuffle=True)

    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        model = TemporalRiskTransformer(**state["model_config"])
        model.load_state_dict(state["model_state"])
    else:
        model = TemporalRiskTransformer(feature_dim=x.shape[-1], sequence_length=x.shape[1])
    model.to(device)

    positives = float(y[:, 1].sum())
    negatives = float(len(y) - positives)
    positive_weight = min(30.0, negatives / max(1.0, positives))
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight, device=device))
    regression = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)

    epoch_losses: list[float] = []
    print(f"production stage={args.stage} device={device} samples={len(x):,}", flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(xb)
            loss = regression(out[:, 0], yb[:, 0]) + bce(out[:, 1], yb[:, 1])
            if args.ranking_weight > 0:
                pos = out[yb[:, 1] >= 0.5, 1]
                neg = out[yb[:, 1] < 0.5, 1]
                pairs = min(len(pos), len(neg))
                if pairs:
                    loss = loss + args.ranking_weight * nn.functional.softplus(
                        -(pos[:pairs] - neg[:pairs])
                    ).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(xb)
        epoch_loss = total / len(x)
        epoch_losses.append(epoch_loss)
        print(f"epoch={epoch:02d}/{args.epochs} train_loss={epoch_loss:.6f}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "model.pt"
    torch.save({
        "stage": args.stage,
        "model_config": model.config,
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "production_all_data": True,
        "training_samples": len(x),
        "epochs": args.epochs,
        "seed": args.seed,
        "ranking_weight": args.ranking_weight,
        "resume": str(args.resume) if args.resume else None,
    }, checkpoint_path)

    training_report = {
        "training_samples": len(x),
        "epochs": args.epochs,
        "final_training_loss": epoch_losses[-1],
        "positive_rate": positives / len(y),
        "positive_weight": positive_weight,
        "held_out_evaluation": False,
    }
    (args.output_dir / "training_metrics.json").write_text(
        json.dumps(training_report, indent=2) + "\n", encoding="utf-8"
    )

    reference: dict[str, object] = {}
    if args.reference_model_dir and (args.reference_model_dir / "info.blt").exists():
        reference = read_info(args.reference_model_dir)

    subsystem = f"risk/{'base' if args.stage == 'base' else 'ethiopia'}"
    write_info(args.output_dir, ModelInfo(
        subsystem=subsystem,
        version=args.version,
        status="production-all-data",
        description=(
            "International temporal-risk model trained on all available rows."
            if args.stage == "base"
            else "Ethiopia temporal-risk fine-tune trained on all available Ethiopia rows."
        ),
        metrics={
            "held_out_evaluation": False,
            "training": training_report,
            "validated_ancestor": reference.get("metrics", {}),
        },
        lineage={
            "resume_checkpoint": str(args.resume) if args.resume else None,
            "validated_ancestor": str(args.reference_model_dir) if args.reference_model_dir else None,
        },
        training={
            "mode": "all-data-production",
            "dataset": str(args.data),
            "samples": len(x),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "ranking_weight": args.ranking_weight,
            "seed": args.seed,
        },
        notes=["No holdout is withheld in this production artifact; use the validated ancestor for generalization metrics."],
    ))


if __name__ == "__main__":
    main()
