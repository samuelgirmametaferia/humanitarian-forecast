#!/usr/bin/env python3
"""Package a v11-family checkpoint as TheSwarm fine-resolution layer.

Attaches the phase-2 country/conflict identity maps to the checkpoint (so
inference does not depend on dataset-internal conventions) and writes an
info.blt card with the honest Ethiopia fine-resolution metrics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from humanitarian_forecast.location.models.candidate_rank import ConflictCandidateRanker


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data", type=Path, default=Path("data/location/conflict_candidates_64_spillover_v6.npz"))
    p.add_argument("--output-dir", type=Path, default=Path("models/location/theswarm/fine_v1"))
    p.add_argument("--label", default="v11_spillover")
    a = p.parse_args()

    data = np.load(a.data, allow_pickle=True)
    meta = [json.loads(str(v)) for v in data["meta"]]
    n = len(meta)
    te, ve = int(.7 * n), int(.85 * n)
    x = torch.from_numpy(data["x"]).float()
    f = torch.from_numpy(data["candidate_features"]).float()
    c = torch.from_numpy(data["candidate_coordinates"]).float()
    v = torch.from_numpy(data["candidate_valid"])
    y = torch.from_numpy(data["y"]).float()

    state = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    if "country_map" not in state:
        state["country_map"] = {z: i + 1 for i, z in enumerate(sorted({m["country"] for m in meta[:ve]}))}
        state["conflict_map"] = {z: i + 1 for i, z in enumerate(sorted({m["conflict_id"] for m in meta[:ve]}))}
    cfg = state["model_config"]
    model = ConflictCandidateRanker(cfg["event_dim"], cfg["candidate_dim"], cfg["sequence_length"],
                                    cfg["countries"], cfg["conflicts"])
    model.load_state_dict(state["model_state"])
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device).eval()

    countries = {str(k): int(w) for k, w in state["country_map"].items()}
    conflicts = {str(k): int(w) for k, w in state["conflict_map"].items()}
    eth = np.asarray([i for i, m in enumerate(meta) if m["country"] == "Ethiopia"])
    splits = {"validation": eth[(eth >= te) & (eth < ve)], "development": eth[eth >= ve]}
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
            for topk in (1, 3, 5, 10):
                e = dist[torch.arange(len(rows))[:, None], order[:, :topk]].min(1).values
                res[f"within_20km_at_top{topk}"] = round(float((e <= 20).float().mean()), 3)
            e1 = dist[torch.arange(len(rows)), order[:, 0]]
            res["median_error_km"] = round(float(e1.median()), 1)
            res["oracle_within_20km"] = round(float((torch.where(v[rows], dist, torch.full((), 1e9)).min(1).values <= 20).float().mean()), 3)
            mass = torch.where(v[rows] & (dist <= 20), probs, torch.zeros(()))
            res["probability_mass_within_20km"] = round(float(mass.sum(1).mean()), 3)
            out[name] = res

    a.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state, a.output_dir / "model.pt")
    card = {
        "model": f"theswarm_fine_v1 ({a.label})",
        "input": "64 cutoff-safe candidates (32 conflict-frequency sites + 32 recent cross-conflict event sites), 37 features incl. ReliefWeb mentions, UCDP windows, spillover activity, elevation/ruggedness",
        "ethiopia_metrics": out,
        "caveats": [
            "oracle_within_20km is the coverage ceiling of the candidate pool, not model accuracy",
            "roughly half of Ethiopia validation targets are geolocated by UCDP to a named-place radius (where_prec>=2), so 20 km hits on those rows are partly coordinate noise",
            "Coarse humanitarian early-warning research signal, not a tactical coordinate forecast",
        ],
    }
    (a.output_dir / "info.blt").write_text(json.dumps(card, indent=2) + "\n")
    print(json.dumps(card, indent=2))


if __name__ == "__main__":
    main()
