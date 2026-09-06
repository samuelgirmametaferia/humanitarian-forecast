#!/usr/bin/env python3
"""Run TheSwarm fine-resolution candidate layer and emit evacuation zones.

Loads the packaged fine layer (an ensemble of v11-family candidate rankers
over cutoff-safe candidates) and emits the top-k emission set with
probabilities, 20 km-radius advisory zones, site recency, and local
elevation/ruggedness for evacuation planning. The emission set is chosen
greedily with a spatial-spread discount so advisory zones cover distinct
areas instead of stacking on one cluster.

Research signal, not a tactical coordinate forecast.
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


def greedy_order(p, coords, k, spread_km):
    picked = []
    for _ in range(k):
        best_j, best_s = -1, -1.0
        for j in range(len(p)):
            if p[j] <= 0 or j in picked:
                continue
            if picked:
                dmin = min(math.hypot(coords[j][0] - coords[q][0], coords[j][1] - coords[q][1]) for q in picked)
                s = p[j] * min(1.0, dmin / spread_km)
            else:
                s = p[j]
            if s > best_s:
                best_s, best_j = s, j
        if best_j < 0:
            break
        picked.append(best_j)
    rest = [j for j in np.argsort(-p) if j not in picked and p[j] > 0]
    return (picked + rest)[:k]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/location/conflict_candidates_64_spillover_v6.npz"))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/location/theswarm/fine_v2/model.pt"))
    parser.add_argument("--index", type=int, default=-1, help="Row index into the history dataset; -1 = latest.")
    parser.add_argument("--top-k", type=int, default=None, help="Override the packaged emission-set size.")
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
    countries = {str(k): int(w) for k, w in state["country_map"].items()}
    conflicts = {str(k): int(w) for k, w in state["conflict_map"].items()}
    selection = state.get("selection", {"method": "plain", "spread_km": 30.0, "top_k": 5})
    top_k = args.top_k or int(selection["top_k"])
    spread_km = float(selection.get("spread_km", 30.0))
    country = countries.get(str(m["country"]), 0)
    conflict = conflicts.get(str(m["conflict_id"]), 0)
    unseen_identity = country == 0 or conflict == 0

    x = torch.from_numpy(data["x"][i:i + 1]).float()
    f = torch.from_numpy(data["candidate_features"][i:i + 1]).float()
    c = torch.from_numpy(data["candidate_coordinates"][i:i + 1]).float()
    v = torch.from_numpy(data["candidate_valid"][i:i + 1])
    device = torch.device("cpu")  # single-row inference; MPS hits a placeholder-storage bug at batch 1

    probs = np.zeros(c.shape[1])
    for sub in state["models"]:
        cfg = sub["model_config"]
        model = ConflictCandidateRanker(cfg["event_dim"], cfg["candidate_dim"], cfg["sequence_length"],
                                        cfg["countries"], cfg["conflicts"])
        model.load_state_dict(sub["model_state"])
        model = model.to(device).eval()
        with torch.no_grad():
            logits = model(x, f, v, torch.tensor([country]), torch.tensor([conflict]))[0]
        probs += torch.where(v[0], logits.softmax(-1), torch.zeros(())).numpy()
    probs /= len(state["models"])
    valid = v[0].numpy()

    coords_km = c[0].numpy() * 1000.0
    if selection.get("method") == "greedy_spread":
        order = greedy_order(np.where(valid, probs, 0.0), coords_km, top_k, spread_km)
    else:
        order = [j for j in np.argsort(-probs) if valid[j]][:top_k]

    anchor_lat, anchor_lon = float(m["anchor_lat"]), float(m["anchor_lon"])
    feats = f[0].numpy()
    candidates = []
    for rank, j in enumerate(order, 1):
        east, north = float(c[0, j, 0]), float(c[0, j, 1])
        lat = anchor_lat + north * 1000 / EARTH_KM
        lon = anchor_lon + east * 1000 / (EARTH_KM * max(0.1, math.cos(math.radians(anchor_lat))))
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
        "model": "theswarm_fine_v2 (v11 ensemble, greedy zone selection)",
        "unseen_identity": unseen_identity,
        "candidates": candidates,
        "combined_emission_probability": round(float(sum(cc["probability"] for cc in candidates)), 4),
        "warning": WARNING,
    }, indent=2))


if __name__ == "__main__":
    main()
