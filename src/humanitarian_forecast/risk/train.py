#!/usr/bin/env python3
"""MPS-aware chronological training for base and Ethiopia checkpoints."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

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


def metrics(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> dict[str, float]:
    mae = (logits[:, 0] - targets[:, 0]).abs().mean().item()
    probs = logits[:, 1].sigmoid()
    pred = probs >= threshold
    truth = targets[:, 1] >= 0.5
    accuracy = (pred == truth).float().mean().item()
    positives = truth.sum().item()
    recall = ((pred & truth).sum().item() / positives) if positives else 0.0
    negatives = (~truth).sum().item()
    specificity = (((~pred) & (~truth)).sum().item() / negatives) if negatives else 0.0
    predicted_positives = pred.sum().item()
    precision = ((pred & truth).sum().item() / predicted_positives) if predicted_positives else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    order = torch.argsort(probs, descending=True)
    ranked_truth = truth[order].float()
    cumulative_tp = ranked_truth.cumsum(0)
    precision_curve = cumulative_tp / torch.arange(1, len(truth) + 1)
    average_precision = (
        (precision_curve * ranked_truth).sum().item() / positives if positives else 0.0
    )
    return {
        "intensity_mae": mae, "escalation_accuracy": accuracy,
        "balanced_accuracy": (recall + specificity) / 2,
        "precision": precision, "escalation_recall": recall, "f1": f1,
        "average_precision": average_precision, "threshold": threshold,
        "positive_rate": float(truth.float().mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("base", "ethiopia"), required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--ranking-weight", type=float, default=0.0,
        help="Weight for pairwise positive-vs-negative ranking loss (default: 0)",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260810)
    args = parser.parse_args()
    seed_all(args.seed)
    device = choose_device(args.device)
    print(f"device={device}")

    bundle = np.load(args.data_dir / f"{args.stage}.npz")
    x = torch.from_numpy(bundle["x"]).float()
    y = torch.from_numpy(bundle["y"]).float()
    train_end = max(1, int(len(x) * 0.70))
    valid_end = max(train_end + 1, int(len(x) * 0.85))
    train_ds = TensorDataset(x[:train_end], y[:train_end])
    valid_ds = TensorDataset(x[train_end:valid_end], y[train_end:valid_end])
    test_ds = TensorDataset(x[valid_end:], y[valid_end:])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        model = TemporalRiskTransformer(**state["model_config"])
        model.load_state_dict(state["model_state"])
    else:
        model = TemporalRiskTransformer(feature_dim=x.shape[-1], sequence_length=x.shape[1])
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    positives = float(y[:train_end, 1].sum())
    negatives = float(train_end - positives)
    positive_weight = min(30.0, negatives / max(1.0, positives))
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight, device=device))
    print(f"train_positive_rate={positives/train_end:.4f} positive_weight={positive_weight:.2f}")
    mse = nn.SmoothL1Loss()
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    best_report: dict[str, float] = {}

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(xb)
            loss = mse(out[:, 0], yb[:, 0]) + bce(out[:, 1], yb[:, 1])
            if args.ranking_weight > 0:
                positive_logits = out[yb[:, 1] >= 0.5, 1]
                negative_logits = out[yb[:, 1] < 0.5, 1]
                pairs = min(len(positive_logits), len(negative_logits))
                if pairs:
                    # Directly reward correct ordering without pretending that a
                    # noisy proxy label is an RL environment.
                    ranking = nn.functional.softplus(
                        -(positive_logits[:pairs] - negative_logits[:pairs])
                    ).mean()
                    loss = loss + args.ranking_weight * ranking
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(xb)

        model.eval()
        outputs, targets = [], []
        with torch.no_grad():
            for xb, yb in valid_loader:
                outputs.append(model(xb.to(device)).cpu())
                targets.append(yb)
        out = torch.cat(outputs)
        target = torch.cat(targets)
        val_bce = nn.functional.binary_cross_entropy_with_logits(
            out[:, 1], target[:, 1], pos_weight=torch.tensor(positive_weight)
        )
        val_loss = mse(out[:, 0], target[:, 0]).item() + val_bce.item()
        report = metrics(out, target)
        print(f"epoch={epoch:02d} train_loss={total/len(train_ds):.4f} val_loss={val_loss:.4f} {report}")
        if val_loss < best:
            best = val_loss
            best_report = report
            checkpoint = {
                "stage": args.stage, "model_config": model.config,
                "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "validation": report, "validation_loss": val_loss,
                "seed": args.seed,
                "chronological_split": {"train": 0.70, "validation": 0.15, "test": 0.15},
            }
            torch.save(checkpoint, args.checkpoint_dir / f"{args.stage}_best.pt")

    best_state = torch.load(
        args.checkpoint_dir / f"{args.stage}_best.pt", map_location="cpu", weights_only=False
    )
    model.load_state_dict(best_state["model_state"])
    model.to(device).eval()

    def collect(loader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
        collected_out, collected_y = [], []
        with torch.no_grad():
            for xb, yb in loader:
                collected_out.append(model(xb.to(device)).cpu())
                collected_y.append(yb)
        return torch.cat(collected_out), torch.cat(collected_y)

    validation_out, validation_y = collect(valid_loader)
    candidates = [i / 100 for i in range(10, 91, 2)]
    threshold = max(
        candidates,
        key=lambda candidate: metrics(validation_out, validation_y, candidate)["balanced_accuracy"],
    )
    test_out, test_y = collect(test_loader)
    test_report = metrics(test_out, test_y, threshold)
    print(f"selected_threshold={threshold:.2f} untouched_test={test_report}")
    (args.checkpoint_dir / f"{args.stage}_metrics.json").write_text(
        json.dumps({
            "best_validation_loss": best,
            "validation_at_0_5": best_report,
            "selected_threshold": threshold,
            "untouched_test": test_report,
        }, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
