#!/usr/bin/env python3
"""Evaluate one or more candidate-ranker checkpoints (mean-of-softmax ensemble)
on Ethiopia validation/development rows: top-k within-20km coverage, median
error, oracle, and probability mass within 20 km."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from humanitarian_forecast.location.models.candidate_rank import ConflictCandidateRanker


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, action="append", required=True)
    p.add_argument("--data", type=Path, default=Path("data/location/conflict_candidates_96_spillover_v7.npz"))
    p.add_argument("--country", default="Ethiopia")
    a = p.parse_args()

    data = np.load(a.data, allow_pickle=True)
    meta = [json.loads(str(v)) for v in data["meta"]]
    n = len(meta)
    te, ve = int(0.7 * n), int(0.85 * n)
    x = torch.from_numpy(data["x"]).float()
    f = torch.from_numpy(data["candidate_features"]).float()
    c = torch.from_numpy(data["candidate_coordinates"]).float()
    v = torch.from_numpy(data["candidate_valid"])
    y = torch.from_numpy(data["y"]).float()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    rows_all = np.asarray([i for i, m in enumerate(meta) if m["country"] == a.country])
    splits = {"validation": rows_all[(rows_all >= te) & (rows_all < ve)],
              "development": rows_all[rows_all >= ve]}

    probs_per_ckpt = []
    for ckpt_path in a.checkpoint:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "country_map" in state:
            countries = {str(k): int(w) for k, w in state["country_map"].items()}
            conflicts = {str(k): int(w) for k, w in state["conflict_map"].items()}
        else:
            countries = {z: i + 1 for i, z in enumerate(sorted({m["country"] for m in meta[:ve]}))}
            conflicts = {z: i + 1 for i, z in enumerate(sorted({m["conflict_id"] for m in meta[:ve]}))}
        cfg = state["model_config"]
        model = ConflictCandidateRanker(cfg["event_dim"], cfg["candidate_dim"], cfg["sequence_length"],
                                        cfg["countries"], cfg["conflicts"])
        model.load_state_dict(state["model_state"])
        model = model.to(device).eval()
        probs = []
        with torch.no_grad():
            for lo in range(0, n, 256):
                sl = np.arange(lo, min(lo + 256, n))
                xb = x[sl].to(device); fb = f[sl].to(device); vb = v[sl].to(device)
                cb = torch.tensor([countries.get(meta[i]["country"], 0) for i in sl]).to(device)
                cf = torch.tensor([conflicts.get(meta[i]["conflict_id"], 0) for i in sl]).to(device)
                probs.append(torch.where(vb, model(xb, fb, vb, cb, cf).softmax(-1), torch.zeros(())).cpu())
        probs_per_ckpt.append(torch.cat(probs))
        print(f"scored {ckpt_path}", flush=True)

    out = {"checkpoints": [str(pth) for pth in a.checkpoint]}
    for name, rows in splits.items():
        if len(rows) == 0:
            continue
        probs = torch.stack([pp[rows] for pp in probs_per_ckpt]).mean(0)
        dist = torch.linalg.vector_norm(c[rows] - y[rows, None, :], dim=-1) * 1000.0
        order = torch.argsort(torch.where(v[rows], probs, torch.full((), -1.0)), dim=-1, descending=True)
        res = {"samples": int(len(rows))}
        for topk in (1, 3, 5, 10):
            e = dist[torch.arange(len(rows))[:, None], order[:, :topk]].min(1).values
            res[f"within_20km_at_top{topk}"] = round(float((e <= 20).float().mean()), 3)
            res[f"within_50km_at_top{topk}"] = round(float((e <= 50).float().mean()), 3)
        e1 = dist[torch.arange(len(rows)), order[:, 0]]
        res["median_error_km"] = round(float(e1.median()), 1)
        res["oracle_within_20km"] = round(float((torch.where(v[rows], dist, torch.full((), 1e9)).min(1).values <= 20).float().mean()), 3)
        mass = torch.where(v[rows] & (dist <= 20), probs, torch.zeros(()))
        res["probability_mass_within_20km"] = round(float(mass.sum(1).mean()), 3)
        out[name] = res
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
