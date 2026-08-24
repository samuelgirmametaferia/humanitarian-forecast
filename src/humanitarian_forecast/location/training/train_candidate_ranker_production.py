#!/usr/bin/env python3
"""Train or fine-tune the promoted v9 candidate-ranker recipe on all labeled rows."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from humanitarian_forecast.core.model_store import ModelInfo, read_info, write_info
from humanitarian_forecast.location.models.candidate_rank import ConflictCandidateRanker, loss_fn


def _identity_maps(meta: list[dict[str, object]]) -> tuple[dict[str, int], dict[str, int]]:
    countries = {value: i + 1 for i, value in enumerate(sorted({str(m["country"]) for m in meta}))}
    conflicts = {value: i + 1 for i, value in enumerate(sorted({str(m["conflict_id"]) for m in meta}))}
    return countries, conflicts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--reference-model-dir", type=Path)
    parser.add_argument("--resume", type=Path, help="Production checkpoint to fine-tune while preserving its identity maps.")
    parser.add_argument("--country-filter", help="Fine-tune only rows whose country matches this value (case-insensitive).")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--distance-weight", type=float, default=0.0)
    parser.add_argument("--center-weight", type=float, default=50.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--aggregation", default="weighted_geometric_median")
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--trainable-scope", choices=("all", "late"), default="all", help="late freezes the event Transformer and updates only scorer/identity layers.")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data = np.load(args.data)
    x = torch.from_numpy(data["x"]).float()
    features = torch.from_numpy(data["candidate_features"]).float()
    coordinates = torch.from_numpy(data["candidate_coordinates"]).float()
    valid = torch.from_numpy(data["candidate_valid"])
    label = torch.from_numpy(data["label"])
    target = torch.from_numpy(data["y"]).float()
    meta = [json.loads(str(row)) for row in data["meta"]]

    resume_state: dict[str, object] | None = None
    if args.resume:
        resume_state = torch.load(args.resume, map_location="cpu", weights_only=False)
        countries = {str(k): int(v) for k, v in dict(resume_state.get("country_to_id", {})).items()}
        conflicts = {str(k): int(v) for k, v in dict(resume_state.get("conflict_to_id", {})).items()}
        if not countries or not conflicts:
            raise SystemExit(
                "--resume must point to a structured production checkpoint containing country_to_id and conflict_to_id"
            )
    else:
        countries, conflicts = _identity_maps(meta)

    if args.country_filter:
        wanted = args.country_filter.casefold()
        selected = [i for i, row in enumerate(meta) if str(row["country"]).casefold() == wanted]
        if not selected:
            raise SystemExit(f"no rows match --country-filter {args.country_filter!r}")
    else:
        selected = list(range(len(meta)))

    indices = torch.tensor(selected, dtype=torch.long)
    selected_meta = [meta[i] for i in selected]
    country_ids = torch.tensor([countries.get(str(m["country"]), 0) for m in selected_meta])
    conflict_ids = torch.tensor([conflicts.get(str(m["conflict_id"]), 0) for m in selected_meta])
    if (country_ids == 0).any() or (conflict_ids == 0).any():
        raise SystemExit(
            "fine-tune data contains identities missing from the resumed global checkpoint; retrain the global checkpoint first"
        )

    dataset = TensorDataset(
        x.index_select(0, indices),
        features.index_select(0, indices),
        coordinates.index_select(0, indices),
        valid.index_select(0, indices),
        label.index_select(0, indices),
        target.index_select(0, indices),
        country_ids,
        conflict_ids,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    if args.device == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    else:
        device = torch.device(args.device)
    if resume_state is not None:
        model = ConflictCandidateRanker(**resume_state["model_config"]).to(device)
        model.load_state_dict(resume_state["model_state"])
    else:
        model = ConflictCandidateRanker(
            x.shape[-1], features.shape[-1], x.shape[1], len(countries) + 1, len(conflicts) + 1
        ).to(device)
    if args.trainable_scope == "late":
        for parameter in model.parameters():
            parameter.requires_grad = False
        late_prefixes = ("context.", "candidate.", "bias.", "country.", "conflict.")
        for name, parameter in model.named_parameters():
            if name.startswith(late_prefixes):
                parameter.requires_grad = True
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise SystemExit("no trainable parameters selected")
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.01)

    epoch_losses: list[float] = []
    scope = args.country_filter or "global"
    print(
        f"production candidate-ranker scope={scope} device={device} samples={len(dataset):,} "
        f"countries={len(countries)} conflicts={len(conflicts)}",
        flush=True,
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for batch in loader:
            xb, fb, cb, vb, lb, yb, countryb, conflictb = [value.to(device) for value in batch]
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb, fb, vb, countryb, conflictb)
            loss, _, _, _ = loss_fn(logits, cb, yb, lb, args.distance_weight, args.center_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(xb)
        epoch_loss = total / len(dataset)
        epoch_losses.append(epoch_loss)
        print(f"epoch={epoch:02d}/{args.epochs} train_loss={epoch_loss:.6f}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "model_config": model.config,
        "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "selected_epochs": args.epochs,
        "production_all_data": True,
        "training_samples": len(dataset),
        "country_filter": args.country_filter,
        "country_to_id": countries,
        "conflict_to_id": conflicts,
        "center_calibration": {
            "aggregation": args.aggregation,
            "temperature": args.temperature,
            "selected_on": "validated ancestor; reused for all-data production training",
        },
    }
    torch.save(state, args.output_dir / "model.pt")
    torch.save(state, args.output_dir / "candidate_ranker_calibrated.pt")

    training_report = {
        "training_samples": len(dataset),
        "available_global_samples": len(meta),
        "country_filter": args.country_filter,
        "identity_countries": len(countries),
        "identity_conflicts": len(conflicts),
        "epochs": args.epochs,
        "final_training_loss": epoch_losses[-1],
        "distance_weight": args.distance_weight,
        "center_weight": args.center_weight,
        "aggregation": args.aggregation,
        "temperature": args.temperature,
        "held_out_evaluation": False,
        "trainable_scope": args.trainable_scope,
        "trainable_parameters": int(sum(parameter.numel() for parameter in trainable)),
    }
    (args.output_dir / "training_metrics.json").write_text(
        json.dumps(training_report, indent=2) + "\n", encoding="utf-8"
    )

    reference: dict[str, object] = {}
    if args.reference_model_dir and (args.reference_model_dir / "info.blt").exists():
        reference = read_info(args.reference_model_dir)

    is_specialized = bool(args.country_filter)
    subsystem = "location/candidate_ranker_ethiopia" if is_specialized else "location/candidate_ranker"
    status = "production-finetune-all-data" if is_specialized else "production-all-data"
    description = (
        "Ethiopia fine-tune of the all-data spatial-ring candidate Transformer."
        if is_specialized
        else "Promoted spatial-ring candidate Transformer retrained on all available next-event labels."
    )
    write_info(
        args.output_dir,
        ModelInfo(
            subsystem=subsystem,
            version=args.version,
            status=status,
            description=description,
            metrics={
                "held_out_evaluation": False,
                "validated_ancestor": reference.get("metrics", {}),
                "training": training_report,
            },
            lineage={
                "recipe": "conflict_candidate_transformer_spatial_rings_geometric_median_v9",
                "resume_checkpoint": str(args.resume) if args.resume else None,
                "validated_ancestor": str(args.reference_model_dir) if args.reference_model_dir else None,
            },
            training={
                "mode": "all-data-production-finetune" if is_specialized else "all-data-production",
                "dataset": str(args.data),
                "country_filter": args.country_filter,
                "samples": len(dataset),
                "identity_countries": len(countries),
                "identity_conflicts": len(conflicts),
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "distance_weight": args.distance_weight,
                "center_weight": args.center_weight,
                "seed": args.seed,
                "trainable_scope": args.trainable_scope,
                "trainable_parameters": int(sum(parameter.numel() for parameter in trainable)),
                "device": str(device),
            },
            calibration={"aggregation": args.aggregation, "temperature": args.temperature},
            notes=[
                "All selected labels are used for training; no new holdout metric is claimed for this artifact.",
                "Calibration is inherited from the validation-selected v9 recipe.",
                "Ethiopia fine-tuning preserves the global country/conflict identity maps."
                if is_specialized
                else "This checkpoint stores identity maps so later country fine-tunes remain embedding-compatible.",
            ],
        ),
    )


if __name__ == "__main__":
    main()
