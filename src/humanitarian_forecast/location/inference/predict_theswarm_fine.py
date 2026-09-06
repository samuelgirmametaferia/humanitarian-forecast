#!/usr/bin/env python3
"""Run TheSwarm fine-resolution candidate layer and emit evacuation zones.

Loads the v11 spillover candidate ranker (64 cutoff-safe candidates: conflict
frequency sites plus recent cross-conflict event sites, with ReliefWeb
mention, UCDP activity, and terrain features) and emits the top-k candidates
with calibrated probabilities, each carrying a 20 km-radius advisory zone,
site recency, and local elevation/ruggedness for evacuation planning.

Research signal, not a tactical coordinate forecast: candidate probabilities
come from a chronological-split evaluation lineage and roughly one in eight
Ethiopia validation truths lands within 20 km of the top-ranked candidate
(within 20 km of at least one of the top five: ~three in eight).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from humanitarian_forecast.location.models.candidate_rank import ConflictCandidateRanker

EARTH_KM = 111.32
WARNING = ("Coarse humanitarian early-warning research signal, not a tactical "
           "coordinate forecast. Zone radii are advisory evacuation buffers, "
           "not uncertainty estimates.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/location/conflict_candidates_64_spillover_v6.npz"))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/location/theswarm/fine_v1/model.pt"))
    parser.add_argument("--index", type=int, default=-1, help="Row index into the history dataset; -1 = latest.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--radius-km", type=float, default=20.0)
    args = parser.parse_args()

    data = np.load(args.data, allow_pickle=True)
    meta = [json.loads(str(v)) for v in data["meta"]]
    n = len(meta)
    i = args.index if args.index >= 0 else n + args.index
    if not 0 <= i < n:
        raise SystemExit(f"index must be in [0,{n-1}]")
    m = meta[i]

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    countries = {str(k): int(v) for k, v in state["country_map"].items()} if "country_map" in state else None
    conflicts = {str(k): int(v) for k, v in state["conflict_map"].items()} if "conflict_map" in state else None
    if countries is None:
        ve = int(.85 * n)
        countries = {z: i + 1 for i, z in enumerate(sorted({mm["country"] for mm in meta[:ve]}))}
        conflicts = {z: i + 1 for i, z in enumerate(sorted({mm["conflict_id"] for mm in meta[:ve]}))}
    country = countries.get(str(m["country"]), 0)
    conflict = conflicts.get(str(m["conflict_id"]), 0)
    unseen_identity = country == 0 or conflict == 0

    x = torch.from_numpy(data["x"][i:i + 1]).float()
    f = torch.from_numpy(data["candidate_features"][i:i + 1]).float()
    c = torch.from_numpy(data["candidate_coordinates"][i:i + 1]).float()
    v = torch.from_numpy(data["candidate_valid"][i:i + 1])
    device = torch.device("cpu")  # single-row inference; MPS hits a placeholder-storage bug at batch 1
    cfg = state["model_config"]
    model = ConflictCandidateRanker(cfg["event_dim"], cfg["candidate_dim"], cfg["sequence_length"],
                                    cfg["countries"], cfg["conflicts"])
    model.load_state_dict(state["model_state"])
    model = model.to(device).eval()
    with torch.no_grad():
        logits = model(x.to(device), f.to(device), v.to(device),
                       torch.tensor([country]), torch.tensor([conflict])).cpu()[0]
    valid = v[0].numpy()
    probs = torch.where(v[0], logits.softmax(-1), torch.zeros(())).numpy()
    order = np.argsort(-probs)
    order = [j for j in order if valid[j]][:args.top_k]

    anchor_lat, anchor_lon = float(m["anchor_lat"]), float(m["anchor_lon"])
    feats = f[0].numpy()
    candidates = []
    for rank, j in enumerate(order, 1):
        east, north = float(c[0, j, 0]), float(c[0, j, 1])
        lat = anchor_lat + north * 1000 / EARTH_KM
        lon = anchor_lon + east * 1000 / (EARTH_KM * max(.1, math.cos(math.radians(anchor_lat))))
        candidates.append({
            "rank": rank,
            "probability": round(float(probs[j]), 4),
            "latitude": round(lat, 4),
            "longitude": round(lon, 4),
            "distance_from_last_event_km": round(math.hypot(east, north) * 1000, 1),
            "site_type": "recent_cross_conflict_event" if feats[j, 27] > .5 else "conflict_frequency_site",
            "days_since_last_event_at_site": int(round(math.expm1(float(feats[j, 31]) * 8))),
            "elevation_m": int(round(float(feats[j, 35]) * 1000)),
            "ruggedness_m": round(float(feats[j, 36]) * 1000, 1),
            "advisory_radius_km": args.radius_km,
        })

    cutoff = str(np.datetime64(m["target_date"]) - np.timedelta64(int(m["gap_days"]), "D"))
    print(json.dumps({
        "country": m["country"],
        "conflict": m["conflict"],
        "observation_cutoff": cutoff,
        "forecast_horizon_days": int(m["gap_days"]),
        "model": "theswarm_fine_v1 (v11 spillover candidate ranker)",
        "unseen_identity": unseen_identity,
        "candidates": candidates,
        "combined_top_k_probability": round(float(sum(cc["probability"] for cc in candidates)), 4),
        "warning": WARNING,
    }, indent=2))


if __name__ == "__main__":
    main()
