#!/usr/bin/env python3
"""Package the TheSwarm fine-resolution layer (v2: ensemble + zone diversity).

Blends one or more v11/v12-family checkpoints (mean of softmax over the
shared dataset), attaches the phase-2 country/conflict identity maps, and
records the zone-selection policy: candidates are emitted greedily with a
spatial-spread discount so advisory zones do not stack on one cluster.
Writes ``model.pt`` plus an ``info.blt`` card with honest Ethiopia metrics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from humanitarian_forecast.location.models.candidate_rank import ConflictCandidateRanker


def greedy_order(p, coords, valid, k, spread_km):
    """Greedy emission set: marginal score = p * min(1, dist-to-picked/spread)."""
    picked = []
    for _ in range(k):
        best_j, best_s = -1, -1.0
        for j in range(len(p)):
            if p[j] <= 0 or j in picked:
                continue
            if picked:
                dmin = min(float(np.hypot(coords[j][0] - coords[q][0], coords[j][1] - coords[q][1])) for q in picked)
                s = p[j] * min(1.0, dmin / spread_km)
            else:
                s = p[j]
            if s > best_s:
                best_s, best_j = s, j
        if best_j < 0:
            break
        picked.append(best_j)
    rest = [j for j in np.argsort(-p) if j not in picked]
    return (picked + rest)[:k]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, action="append", required=True)
    p.add_argument("--data", type=Path, default=Path("data/location/conflict_candidates_64_spillover_v6.npz"))
    p.add_argument("--output-dir", type=Path, default=Path("models/location/theswarm/fine_v2"))
    p.add_argument("--spread-km", type=float, default=30.0)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--label", default="v11_spillover+v11_lambdarank ensemble")
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
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    states, probs_all = [], []
    for ckpt_path in a.checkpoint:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "country_map" not in state:
            state["country_map"] = {z: i + 1 for i, z in enumerate(sorted({m["country"] for m in meta[:ve]}))}
            state["conflict_map"] = {z: i + 1 for i, z in enumerate(sorted({m["conflict_id"] for m in meta[:ve]}))}
        cfg = state["model_config"]
        model = ConflictCandidateRanker(cfg["event_dim"], cfg["candidate_dim"], cfg["sequence_length"],
                                        cfg["countries"], cfg["conflicts"])
        model.load_state_dict(state["model_state"])
        model = model.to(device).eval()
        countries = {str(k): int(w) for k, w in state["country_map"].items()}
        conflicts = {str(k): int(w) for k, w in state["conflict_map"].items()}
        probs = []
        with torch.no_grad():
            for lo in range(0, n, 256):
                sl = np.arange(lo, min(lo + 256, n))
                xb = x[sl].to(device); fb = f[sl].to(device); vb = v[sl].to(device)
                cb = torch.tensor([countries.get(meta[i]["country"], 0) for i in sl]).to(device)
                cf = torch.tensor([conflicts.get(meta[i]["conflict_id"], 0) for i in sl]).to(device)
                probs.append(torch.where(vb, model(xb, fb, vb, cb, cf).softmax(-1), torch.zeros(())).cpu())
        probs_all.append(torch.cat(probs).numpy())
        states.append({k: state[k] for k in ("model_config", "model_state")})
        print(f"scored {ckpt_path}", flush=True)
    probs_all = np.mean(probs_all, axis=0)

    eth = np.asarray([i for i, m in enumerate(meta) if m["country"] == "Ethiopia"])
    out = {}
    for name, rows in (("validation", eth[(eth >= te) & (eth < ve)]), ("development", eth[eth >= ve])):
        if len(rows) == 0:
            continue
        P = probs_all[rows]
        C = c[rows].numpy() * 1000.0
        V = v[rows].numpy()
        Y = y[rows].numpy() * 1000.0
        D = np.linalg.norm(C - Y[:, None, :], axis=2)
        res = {"samples": int(len(rows))}
        plain = np.argsort(-np.where(V, P, -1), axis=1)
        # top-1 from the plain ordering (the greedy set's first pick equals it).
        e1 = D[np.arange(len(rows)), plain[:, 0]]
        res["within_20km_at_top1"] = round(float((e1 <= 20).mean()), 3)
        res["median_error_km"] = round(float(np.median(e1)), 1)
        for topk in (3, 5, 10):
            plain_e = D[np.arange(len(rows))[:, None], plain[:, :topk]].min(1)
            res[f"plain_within_20km_at_top{topk}"] = round(float((plain_e <= 20).mean()), 3)
            if topk >= a.top_k:
                greedy_e = []
                for i in range(len(rows)):
                    order = greedy_order(np.where(V[i], P[i], 0.0), C[i], V[i], topk, a.spread_km)
                    greedy_e.append(D[i, order].min())
                res[f"diverse_within_20km_at_top{topk}"] = round(float(np.mean(np.asarray(greedy_e) <= 20)), 3)
        res["oracle_within_20km"] = round(float((np.where(V, D, 1e9).min(1) <= 20).mean()), 3)
        mass = np.where(V & (D <= 20), P, 0.0).sum(1)
        res["probability_mass_within_20km"] = round(float(mass.mean()), 3)
        out[name] = res

    a.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"models": states,
                "country_map": state["country_map"], "conflict_map": state["conflict_map"],
                "selection": {"method": "greedy_spread", "spread_km": a.spread_km, "top_k": a.top_k},
                "data": str(a.data)}, a.output_dir / "model.pt")
    card = {
        "model": f"theswarm_fine_v2 ({a.label})",
        "input": "64 cutoff-safe candidates (32 conflict-frequency sites + 32 recent cross-conflict event sites), 37 features incl. ReliefWeb mentions, UCDP windows, spillover activity, elevation/ruggedness",
        "blend": "mean of softmax over the listed checkpoints",
        "zone_selection": {"method": "greedy spatial-spread discount", "spread_km": a.spread_km},
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
