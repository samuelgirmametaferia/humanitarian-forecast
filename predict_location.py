#!/usr/bin/env python3
"""Render a coarse historical risk circle from a prepared location sequence."""

from __future__ import annotations

import argparse
import json
import math
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import torch

from location_model import ProbabilisticLocationTransformer

KM_PER_DEGREE = 111.32


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--index", type=int, default=-1)
    args = parser.parse_args()

    data = np.load(args.data)
    index = args.index if args.index >= 0 else len(data["x"]) + args.index
    features = torch.from_numpy(data["x"][index:index + 1]).float()
    meta = json.loads(str(data["meta"][index]))
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = ProbabilisticLocationTransformer(**state["model_config"])
    model.load_state_dict(state["model_state"]); model.eval()
    with torch.no_grad():
        center, sigma = model(features)
    east_km, north_km = (center[0].numpy() * 1000.0).tolist()
    anchor_lat = float(meta["anchor_lat"]); anchor_lon = float(meta["anchor_lon"])
    predicted_lat = anchor_lat + north_km / KM_PER_DEGREE
    longitude_scale = KM_PER_DEGREE * max(0.1, math.cos(math.radians(anchor_lat)))
    predicted_lon = anchor_lon + east_km / longitude_scale
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    radius = max(
        float(calibration["minimum_publishable_radius_km"]),
        float(sigma[0]) * 1000.0 * float(calibration["multiplier"]),
    )
    cutoff = date.fromisoformat(meta["target_date"]) - timedelta(days=int(meta["gap_days"]))
    print(json.dumps({
        "observation_cutoff": str(cutoff), "forecast_horizon_days": int(meta["gap_days"]),
        "country": meta["country"], "conflict": meta["conflict"],
        "center_latitude_coarse": round(predicted_lat * 4) / 4,
        "center_longitude_coarse": round(predicted_lon * 4) / 4,
        "uncertainty_radius_km": round(radius),
        "calibrated_target_coverage": calibration["target_coverage"],
        "warning": "Research-only coarse historical risk circle; not a verified front or evacuation order",
    }, indent=2))


if __name__ == "__main__":
    main()

