#!/usr/bin/env python3
"""Final TheSwarm verification: packaged artifacts vs the swarm-v1 reference."""
import json

import numpy as np

from humanitarian_forecast.location.ensemble.swarm import broad_metrics, distances


def truth(meta, rows):
    return np.asarray([[meta[i]["target_lat"], meta[i]["target_lon"]] for i in rows], dtype=np.float32)


data32 = np.load("data/location/conflict_candidates_32_spatial_v5.npz", allow_pickle=True)
meta = [json.loads(str(v)) for v in data32["meta"]]
n = len(meta)
ethiopia = np.asarray([i for i, m in enumerate(meta) if m["country"] == "Ethiopia"])
validation_rows = ethiopia[(ethiopia >= int(0.70 * n)) & (ethiopia < int(0.85 * n))]
development_rows = ethiopia[ethiopia >= int(0.85 * n)]

for label, path in (("swarm_v1", "models/location/swarm/v1"), ("theswarm_v1", "models/location/theswarm/v1")):
    z = np.load(f"{path}/probabilities.npz", allow_pickle=True)
    assert np.array_equal(z["validation_rows"], validation_rows), label
    assert np.array_equal(z["development_rows"], development_rows), label
    cells = z["cells"].astype(np.float64)
    d_val = distances(cells, truth(meta, validation_rows))
    d_dev = distances(cells, truth(meta, development_rows))
    val = broad_metrics(z["validation"].astype(np.float64), d_val)
    dev = broad_metrics(z["development"].astype(np.float64), d_dev)
    state = json.loads(open(f"{path}/swarm.json" if label == "swarm_v1" else f"{path}/theswarm.json").read())
    print(json.dumps({
        "model": label,
        "validation_broad": round(val["broad_area_score"], 4),
        "validation_median_km": round(val["median_error_km"], 1),
        "development_broad": round(dev["broad_area_score"], 4),
        "development_median_km": round(dev["median_error_km"], 1),
        "mean_entropy_validation": round(val["mean_entropy"], 3),
        "weights": state.get("weights"),
        "pooling": state.get("pooling", "linear (staged)"),
        "smoothing_delta": state.get("guard_smoothing_delta", 0.0),
    }, indent=2))
