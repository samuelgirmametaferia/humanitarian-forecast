#!/usr/bin/env python3
"""Package the all-data two-support v10-prized candidate-ranker.

The package keeps one candidate-ranker architecture but two state views:
* Ethiopia-adapted weights over the validated 32-candidate support.
* Global all-data weights over the expanded 64-candidate support.
The final relative center is a frozen convex blend selected during historical
Ethiopia development. Public inference is intentionally projected to coarse
humanitarian zones by the matching inference module.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from humanitarian_forecast.core.model_store import ModelInfo, write_info


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--global-checkpoint", type=Path, required=True)
    p.add_argument("--ethiopia-checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--version", default="v10-prized")
    p.add_argument("--ethiopia-weight", type=float, default=0.5)
    p.add_argument("--global64-temperature", type=float, default=0.65)
    args = p.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {args.output_dir}")
    if not 0.0 <= args.ethiopia_weight <= 1.0:
        raise SystemExit("--ethiopia-weight must be in [0,1]")

    global_state = torch.load(args.global_checkpoint, map_location="cpu", weights_only=False)
    local_state = torch.load(args.ethiopia_checkpoint, map_location="cpu", weights_only=False)
    for key in ("model_config", "country_to_id", "conflict_to_id"):
        if global_state.get(key) != local_state.get(key):
            raise SystemExit(f"checkpoint mismatch for {key}")

    package = {
        "format": "humanitarian-forecast-location-v10-prized/v1",
        "version": args.version,
        "production_all_data": True,
        "model_config": global_state["model_config"],
        "country_to_id": global_state["country_to_id"],
        "conflict_to_id": global_state["conflict_to_id"],
        "global_model_state": global_state["model_state"],
        "ethiopia_model_state": local_state["model_state"],
        "support_views": {
            "ethiopia32": {
                "candidate_count": 32,
                "feature_dim": 28,
                "aggregation": "weighted_geometric_median",
                "temperature": 1.0,
                "center_weight": float(args.ethiopia_weight),
                "weights": "ethiopia_model_state",
            },
            "global64": {
                "candidate_count": 64,
                "feature_dim": 28,
                "aggregation": "weighted_mean",
                "temperature": float(args.global64_temperature),
                "center_weight": float(1.0 - args.ethiopia_weight),
                "weights": "global_model_state",
            },
        },
        "blend": {
            "ethiopia32_weight": float(args.ethiopia_weight),
            "global64_weight": float(1.0 - args.ethiopia_weight),
            "selection_note": (
                "Historical Ethiopia development recipe: blend chosen after validation/development analysis; "
                "not a pristine prospective estimate."
            ),
        },
        "lineage": {
            "global_checkpoint": str(args.global_checkpoint),
            "ethiopia_checkpoint": str(args.ethiopia_checkpoint),
            "architecture": "ConflictCandidateRanker (v9 family)",
            "global_training_samples": int(global_state.get("training_samples", 0)),
            "ethiopia_adaptation_samples": int(local_state.get("training_samples", 0)),
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.save(package, args.output_dir / "model.pt")

    evaluation = {
        "scope": "historical Ethiopia development; final 15% has been repeatedly inspected and is not prospective",
        "aligned_v9_parent": {
            "validation": {
                "mean_error_km": 166.35638427734375,
                "median_error_km": 115.14903259277344,
                "p90_error_km": 394.1524963378906,
                "within_25km": 0.0904458612203598,
                "within_50km": 0.19745223224163055,
                "within_100km": 0.4331210255622864,
            },
            "development": {
                "mean_error_km": 178.59228515625,
                "median_error_km": 156.82778930664062,
                "p90_error_km": 340.6812438964844,
                "within_25km": 0.03512396663427353,
                "within_50km": 0.10433883965015411,
                "within_100km": 0.30888429284095764,
            },
        },
        "v10_prized_recipe": {
            "validation": {
                "mean_error_km": 163.3533,
                "p90_error_km": 379.2935,
                "within_50km": 0.2013,
                "within_100km": 0.4420,
            },
            "development": {
                "mean_error_km": 177.1967,
                "p90_error_km": 334.1704,
                "within_50km": 0.0961,
                "within_100km": 0.3099,
            },
            "candidate_oracle_mean_km": {
                "ethiopia_32_development": 34.84501266479492,
                "ethiopia_64_development": 23.900867462158203,
            },
        },
        "delta_vs_aligned_v9_development": {
            "mean_error_km": -1.3956,
            "p90_error_km": -6.5108,
            "within_50km_percentage_points": -0.8239,
            "within_100km_percentage_points": 0.1016,
        },
        "production_note": "All-data v10-prized weights have no held-out metric because every historical label is used for final fitting.",
    }
    (args.output_dir / "evaluation_metrics.json").write_text(json.dumps(evaluation, indent=2) + "\n")

    write_info(
        args.output_dir,
        ModelInfo(
            subsystem="location/candidate_ranker",
            version=args.version,
            status="production-all-data-humanitarian-coarse-output",
            description=(
                "Direct v9-family v10-prized: all-data global candidate Transformer plus one-epoch "
                "Ethiopia late-layer adaptation, with frozen 32/64 support-view center blending."
            ),
            metrics={"historical_evaluation": evaluation, "held_out_evaluation": False},
            lineage=package["lineage"],
            training={
                "global_samples": int(global_state.get("training_samples", 0)),
                "ethiopia_adaptation_samples": int(local_state.get("training_samples", 0)),
                "candidate_feature_dim": 28,
                "history_feature_dim": 19,
            },
            calibration=package["support_views"],
            notes=[
                "The historical final-15% block is development data, not a pristine prospective test.",
                "The final package uses all available historical labels, so no new held-out score is claimed for its weights.",
                "Public inference must project the internal center to coarse humanitarian zones and must not expose ranked tactical coordinates.",
            ],
        ),
    )
    print(json.dumps({"output": str(args.output_dir), "version": args.version, "blend": package["blend"]}, indent=2))


if __name__ == "__main__":
    main()
