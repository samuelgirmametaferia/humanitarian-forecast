#!/usr/bin/env python3
"""Run v10-prized and emit a coarse humanitarian zone, not tactical coordinates."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from humanitarian_forecast.location.inference.predict_hackathon import aggregate_center
from humanitarian_forecast.location.models.candidate_rank import ConflictCandidateRanker


def _view(model, x, features, coordinates, valid, country, conflict, temperature, aggregation):
    with torch.no_grad():
        logits = model(x, features, valid, country, conflict)[0]
    probability = (logits / float(temperature)).softmax(-1)
    return aggregate_center(probability, coordinates[0], str(aggregation))


def _snap(value: float, step: float) -> float:
    return round(round(value / step) * step, 4)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data32", type=Path, required=True)
    p.add_argument("--data64", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, default=Path("models/location/candidate_ranker/v10_prized/model.pt"))
    p.add_argument("--index", type=int, default=-1)
    p.add_argument("--zone-degrees", type=float, default=0.25, help="Output grid size; defaults to ~25 km scale.")
    args = p.parse_args()

    if args.zone_degrees < 0.2:
        raise SystemExit("coarse humanitarian output requires --zone-degrees >= 0.2")
    d32 = np.load(args.data32, allow_pickle=True)
    d64 = np.load(args.data64, allow_pickle=True)
    if len(d32["meta"]) != len(d64["meta"]):
        raise SystemExit("32/64 candidate datasets are not aligned")
    meta32 = [json.loads(str(v)) for v in d32["meta"]]
    meta64 = [json.loads(str(v)) for v in d64["meta"]]
    n = len(meta32)
    i = args.index if args.index >= 0 else n + args.index
    if not 0 <= i < n:
        raise SystemExit(f"index must be in [0,{n-1}]")
    if meta32[i] != meta64[i]:
        raise SystemExit("32/64 metadata mismatch at requested row")

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    countries = {str(k): int(v) for k, v in state["country_to_id"].items()}
    conflicts = {str(k): int(v) for k, v in state["conflict_to_id"].items()}
    m = meta32[i]
    country = torch.tensor([countries.get(str(m["country"]), 0)])
    conflict = torch.tensor([conflicts.get(str(m["conflict_id"]), 0)])
    if int(country[0]) == 0 or int(conflict[0]) == 0:
        raise SystemExit("row identity is outside the all-data v10-prized identity map")

    config = state["model_config"]
    global_model = ConflictCandidateRanker(**config)
    global_model.load_state_dict(state["global_model_state"])
    global_model.eval()
    local_model = ConflictCandidateRanker(**config)
    local_model.load_state_dict(state["ethiopia_model_state"])
    local_model.eval()

    x = torch.from_numpy(d32["x"][i:i+1]).float()
    f32 = torch.from_numpy(d32["candidate_features"][i:i+1]).float()
    c32 = torch.from_numpy(d32["candidate_coordinates"][i:i+1]).float()
    v32 = torch.from_numpy(d32["candidate_valid"][i:i+1])
    f64 = torch.from_numpy(d64["candidate_features"][i:i+1]).float()
    c64 = torch.from_numpy(d64["candidate_coordinates"][i:i+1]).float()
    v64 = torch.from_numpy(d64["candidate_valid"][i:i+1])

    s32 = state["support_views"]["ethiopia32"]
    s64 = state["support_views"]["global64"]
    center64 = _view(global_model, x, f64, c64, v64, country, conflict, s64["temperature"], s64["aggregation"])
    if str(m["country"]).casefold() == "ethiopia":
        center32 = _view(local_model, x, f32, c32, v32, country, conflict, s32["temperature"], s32["aggregation"])
        center = float(s32["center_weight"]) * center32 + float(s64["center_weight"]) * center64
        routing = "ethiopia32+global64"
    else:
        center = center64
        routing = "global64-only"

    anchor_lat = float(m["anchor_lat"])
    anchor_lon = float(m["anchor_lon"])
    lat = anchor_lat + float(center[1]) * 1000.0 / 111.32
    lon = anchor_lon + float(center[0]) * 1000.0 / (111.32 * max(0.1, math.cos(math.radians(anchor_lat))))
    cutoff = str(np.datetime64(m["target_date"]) - np.timedelta64(int(m["gap_days"]), "D"))
    print(json.dumps({
        "country": m["country"],
        "observation_cutoff": cutoff,
        "forecast_horizon_days": int(m["gap_days"]),
        "humanitarian_zone": {
            "grid_degrees": args.zone_degrees,
            "center_latitude": _snap(lat, args.zone_degrees),
            "center_longitude": _snap(lon, args.zone_degrees),
        },
        "model": state.get("version", "v10-prized"),
        "routing": routing,
        "warning": "Coarse humanitarian early-warning zone only; not a tactical or targeting coordinate.",
    }, indent=2))


if __name__ == "__main__":
    main()
