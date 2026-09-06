#!/usr/bin/env python3
"""Run the swarm mixture and emit a coarse humanitarian early-warning zone.

Ethiopia rows covered by the packaged swarm blend use the mixture
probabilities over the shared H3 r4 support. Every other row — and any
Ethiopia row outside the packaged chronology — routes through the
v10-prized champion (global64 view, Ethiopia32 view when applicable),
matching the deployment routing described in the model card.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from humanitarian_forecast.location.inference.predict_hackathon import aggregate_center
from humanitarian_forecast.location.models.candidate_rank import ConflictCandidateRanker

EARTH_KM = 111.32


def _snap(value: float, step: float) -> float:
    return round(round(value / step) * step, 4)


def _zone(probability: np.ndarray, cells: np.ndarray, zone_degrees: float) -> dict:
    """Summarize the dense probability field as a coarse humanitarian zone."""
    top = np.argsort(-probability)[:5]
    center = (probability[:, None] * cells).sum(0)
    ranked = [
        {
            "rank": rank + 1,
            "probability": round(float(probability[j]), 6),
            "latitude": round(float(cells[j, 0]), 4),
            "longitude": round(float(cells[j, 1]), 4),
        }
        for rank, j in enumerate(top)
    ]
    return {
        "zone_center": {
            "latitude": _snap(float(center[0]), zone_degrees),
            "longitude": _snap(float(center[1]), zone_degrees),
        },
        "top_cells": ranked,
    }


def _prized_view(state: dict, x, features, coordinates, valid, country, conflict) -> tuple[float, str]:
    calibration = state["support_views"]
    model = ConflictCandidateRanker(**state["model_config"])
    which = "ethiopia32" if "ethiopia_model_state" in state and "ethiopia32" in calibration else "global64"
    key = "ethiopia_model_state" if which == "ethiopia32" else "global_model_state"
    model.load_state_dict(state[key])
    model.eval()
    view = calibration[which]
    with torch.no_grad():
        logits = model(x, features, valid, country, conflict)[0]
    probability = (logits / float(view["temperature"])).softmax(-1)
    center = aggregate_center(probability, coordinates[0], str(view["aggregation"]))
    return float(view["temperature"]), center


def main(default_swarm_dir: str = "models/location/swarm/v1", state_file: str = "swarm.json", model_label: str = "swarm") -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data32", type=Path, default=Path("data/location/conflict_candidates_32_spatial_v5.npz"))
    parser.add_argument("--data64", type=Path, default=Path("data/location/conflict_candidates_64_spatial_v10.npz"))
    parser.add_argument("--swarm-dir", type=Path, default=Path(default_swarm_dir))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/location/candidate_ranker/v10_prized/model.pt"))
    parser.add_argument("--index", type=int, default=-1)
    parser.add_argument("--zone-degrees", type=float, default=0.25, help="Output grid size; defaults to ~25 km scale.")
    args = parser.parse_args()
    if args.zone_degrees < 0.2:
        raise SystemExit("coarse humanitarian output requires --zone-degrees >= 0.2")

    swarm = json.loads((args.swarm_dir / state_file).read_text())
    blend = np.load(args.swarm_dir / "probabilities.npz", allow_pickle=True)
    cells = blend["cells"].astype(np.float64)
    covered = {}
    for split in ("validation", "development"):
        for position, row in enumerate(blend[f"{split}_rows"]):
            covered[int(row)] = (split, position)

    data32 = np.load(args.data32, allow_pickle=True)
    meta = [json.loads(str(v)) for v in data32["meta"]]
    n = len(meta)
    i = args.index if args.index >= 0 else n + args.index
    if not 0 <= i < n:
        raise SystemExit(f"index must be in [0,{n-1}]")
    m = meta[i]

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    countries = {str(k): int(v) for k, v in state["country_to_id"].items()}
    conflicts = {str(k): int(v) for k, v in state["conflict_to_id"].items()}
    country = torch.tensor([countries.get(str(m["country"]), 0)])
    conflict = torch.tensor([conflicts.get(str(m["conflict_id"]), 0)])
    if int(country[0]) == 0 or int(conflict[0]) == 0:
        raise SystemExit("row identity is outside the v10-prized identity map")

    if i in covered:
        split, position = covered[i]
        probability = blend[split][position].astype(np.float64)
        source = model_label
        zone = _zone(probability, cells, args.zone_degrees)
        center = (probability[:, None] * cells).sum(0)
        lat, lon = float(center[0]), float(center[1])
    else:
        data64 = np.load(args.data64, allow_pickle=True)
        meta64 = [json.loads(str(v)) for v in data64["meta"]]
        if meta64[i] != m:
            raise SystemExit("32/64 metadata mismatch at requested row")
        x = torch.from_numpy(data32["x"][i:i + 1]).float()
        f64 = torch.from_numpy(data64["candidate_features"][i:i + 1]).float()
        c64 = torch.from_numpy(data64["candidate_coordinates"][i:i + 1]).float()
        v64 = torch.from_numpy(data64["candidate_valid"][i:i + 1])
        _, center64 = _prized_view(state, x, f64, c64, v64, country, conflict)
        source = "v10_prized_global64"
        anchor_lat, anchor_lon = float(m["anchor_lat"]), float(m["anchor_lon"])
        lat = anchor_lat + float(center64[1]) * 1000 / EARTH_KM
        lon = anchor_lon + float(center64[0]) * 1000 / (EARTH_KM * max(0.1, math.cos(math.radians(anchor_lat))))
        zone = {
            "zone_center": {
                "latitude": _snap(lat, args.zone_degrees),
                "longitude": _snap(lon, args.zone_degrees),
            },
            "top_cells": [],
        }

    cutoff = str(np.datetime64(m["target_date"]) - np.timedelta64(int(m["gap_days"]), "D"))
    print(json.dumps({
        "country": m["country"],
        "conflict": m["conflict"],
        "observation_cutoff": cutoff,
        "forecast_horizon_days": int(m["gap_days"]),
        "model": source,
        "swarm_weights": swarm.get("weights") if source == model_label else None,
        "zone": zone,
        "warning": "Coarse humanitarian early-warning research signal, not a tactical coordinate forecast.",
    }, indent=2))


if __name__ == "__main__":
    main()
