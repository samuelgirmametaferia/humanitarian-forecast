#!/usr/bin/env python3
"""Fine-layer performance split by UCDP target geolocation precision.

The headline Ethiopia metrics mix two row populations: exactly-geolocated
targets (where_prec==1) and named-place-radius targets (where_prec>=2,
±25-100 km coordinate noise). This report separates them so the fine
layer's true 20 km competence can be read off the exact subset.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from humanitarian_forecast.location.models.candidate_rank import ConflictCandidateRanker
from humanitarian_forecast.location.training.train_candidate_ranker_precision import target_precision


def main():
    data = np.load("data/location/conflict_candidates_64_spillover_v6.npz", allow_pickle=True)
    meta = [json.loads(str(v)) for v in data["meta"]]
    n = len(meta); te, ve = int(.7 * n), int(.85 * n)
    state = torch.load("models/location/theswarm/fine_v2/model.pt", map_location="cpu", weights_only=False)
    countries = {str(k): int(w) for k, w in state["country_map"].items()}
    conflicts = {str(k): int(w) for k, w in state["conflict_map"].items()}

    eth = [i for i, m in enumerate(meta) if m["country"] == "Ethiopia"]
    val = [i for i in eth if te <= i < ve]
    prec = target_precision([meta[i] for i in val], Path("data/raw/ged261-csv.zip"))

    x = torch.from_numpy(data["x"][val]).float()
    f = torch.from_numpy(data["candidate_features"][val]).float()
    c = torch.from_numpy(data["candidate_coordinates"][val]).float()
    v = torch.from_numpy(data["candidate_valid"][val])
    y = torch.from_numpy(data["y"][val]).float()
    cb = torch.tensor([countries.get(meta[i]["country"], 0) for i in val])
    cf = torch.tensor([conflicts.get(meta[i]["conflict_id"], 0) for i in val])

    probs = None
    for sub in state["models"]:
        cfg = sub["model_config"]
        model = ConflictCandidateRanker(cfg["event_dim"], cfg["candidate_dim"], cfg["sequence_length"],
                                        cfg["countries"], cfg["conflicts"])
        model.load_state_dict(sub["model_state"]); model.eval()
        with torch.no_grad():
            p = model(x, f, v, cb, cf).softmax(-1)
        probs = p if probs is None else probs + p
    probs = (probs / len(state["models"])).numpy()
    D = np.linalg.norm(c.numpy() * 1000 - y.numpy()[:, None] * 1000, axis=2)
    order = np.argsort(-probs, axis=1)

    out = {"samples": len(val),
           "prec_counts": {"exact_1": int((prec == 1).sum()), "named_2": int((prec == 2).sum()),
                           "coarse_3plus": int((prec >= 3).sum()), "unknown_0": int((prec == 0).sum())}}
    for name, mask in (("where_prec_1_exact", prec == 1),
                       ("where_prec_2_named", prec == 2),
                       ("where_prec_3plus", prec >= 3),
                       ("all_rows", np.ones(len(val), bool))):
        if mask.sum() == 0:
            continue
        e1 = D[np.arange(len(val)), order[:, 0]][mask]
        res = {"samples": int(mask.sum()),
               "top1_within_20km": round(float((e1 <= 20).mean()), 3),
               "top1_within_50km": round(float((e1 <= 50).mean()), 3)}
        for topk in (3, 5, 10):
            e = D[np.arange(len(val))[:, None], order[:, :topk]].min(1)[mask]
            res[f"top{topk}_within_20km"] = round(float((e <= 20).mean()), 3)
        res["oracle_within_20km"] = round(float((np.where(v.numpy(), D, 1e9).min(1)[mask] <= 20).mean()), 3)
        out[name] = res
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
