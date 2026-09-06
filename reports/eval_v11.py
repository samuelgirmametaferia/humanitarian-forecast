#!/usr/bin/env python3
"""Evaluate a candidate-ranker checkpoint on Ethiopia validation/development
rows with fine-resolution metrics: top-k within-20km coverage, median error,
and probability mass landing within 20km of the truth."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from humanitarian_forecast.location.models.candidate_rank import ConflictCandidateRanker


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=Path("models/location/candidate_ranker/v11_spillover/candidate_ranker_best.pt"))
    p.add_argument("--data", type=Path, default=Path("data/location/conflict_candidates_64_spillover_v6.npz"))
    a = p.parse_args()

    d = np.load(a.data, allow_pickle=True)
    meta = [json.loads(str(v)) for v in d["meta"]]
    n = len(meta)
    te, ve = int(0.7 * n), int(0.85 * n)
    x = torch.from_numpy(d["x"]).float()
    f = torch.from_numpy(d["candidate_features"]).float()
    c = torch.from_numpy(d["candidate_coordinates"]).float()
    v = torch.from_numpy(d["candidate_valid"])
    y = torch.from_numpy(d["y"]).float()

    ckpt = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    config = ckpt["model_config"]
    # The checkpoint stores the phase-2 country/conflict maps' sizes; rebuild
    # the same index spaces from the rows the trainer saw (first 85%).
    countries = {z: i + 1 for i, z in enumerate(sorted({m["country"] for m in meta[:ve]}))}
    conflicts = {z: i + 1 for i, z in enumerate(sorted({m["conflict_id"] for m in meta[:ve]}))}
    model = ConflictCandidateRanker(config["event_dim"], config["candidate_dim"], config["sequence_length"],
                                    config["countries"], config["conflicts"])
    model.load_state_dict(ckpt["model_state"])
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device).eval()

    eth = np.asarray([i for i, m in enumerate(meta) if m["country"] == "Ethiopia"])
    splits = {
        "validation": eth[(eth >= te) & (eth < ve)],
        "development": eth[eth >= ve],
    }
    out = {}
    with torch.no_grad():
        for name, rows in splits.items():
            logits = []
            for lo in range(0, len(rows), 256):
                sl = rows[lo:lo + 256]
                xb = x[sl].to(device); fb = f[sl].to(device); vb = v[sl].to(device)
                cb = torch.tensor([countries.get(meta[i]["country"], 0) for i in sl]).to(device)
                cf = torch.tensor([conflicts.get(meta[i]["conflict_id"], 0) for i in sl]).to(device)
                logits.append(model(xb, fb, vb, cb, cf).cpu())
            logits = torch.cat(logits)
            probs = torch.where(v[rows], logits.softmax(-1), torch.zeros(()))
            dist = torch.linalg.vector_norm(c[rows] - y[rows, None, :], dim=-1) * 1000.0
            order = torch.argsort(torch.where(v[rows], logits, torch.full((), -1e9)), dim=-1, descending=True)
            res = {"samples": int(len(rows))}
            for topk in (1, 3, 5):
                e = dist[torch.arange(len(rows))[:, None], order[:, :topk]].min(1).values
                res[f"w20@{topk}"] = round(float((e <= 20).float().mean()), 3)
                res[f"w50@{topk}"] = round(float((e <= 50).float().mean()), 3)
            e1 = dist[torch.arange(len(rows)), order[:, 0]]
            res["median_km"] = round(float(e1.median()), 1)
            res["oracle_w20"] = round(float((torch.where(v[rows], dist, torch.full((), 1e9)).min(1).values <= 20).float().mean()), 3)
            # Probability mass on candidates within 20km of the truth.
            mass = torch.where(v[rows] & (dist <= 20), probs, torch.zeros(()))
            res["mass_in_20km"] = round(float(mass.sum(1).mean()), 3)
            out[name] = res
    print(json.dumps({"checkpoint": str(a.checkpoint), **out}, indent=2))


if __name__ == "__main__":
    main()
