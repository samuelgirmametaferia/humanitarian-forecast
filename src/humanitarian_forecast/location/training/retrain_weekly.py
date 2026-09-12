#!/usr/bin/env python3
"""Weekly live retrain of the TheSwarm fine-resolution layer.

Continues training from the *currently promoted* production package (never
from scratch): each ensemble member is fine-tuned on the Ethiopia training
rows plus every matured live-validated pair, with epoch selection on the
untouched Ethiopia validation split. The output is a drop-in replacement
package (``model.pt`` + ``info.blt``) in the ``fine_v2`` format, ready for
ONNX export, parity validation, and registry promotion.

Live rows carry realized UCDP outcomes only — public preview signals were
already baked into their features at publish time and are never labels.
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from humanitarian_forecast.location.models.candidate_rank import ConflictCandidateRanker, loss_fn
from humanitarian_forecast.location.training.package_theswarm_fine import greedy_order

SCALE = 1000.0


def _load_live(path: Path | None) -> dict[str, np.ndarray]:
    if path is None or not path.exists():
        return {}
    with np.load(path, allow_pickle=False) as data:
        if "x" not in data.files or len(data["meta"]) == 0:
            return {}
        return {key: data[key] for key in data.files}


def _member_model(sub: dict[str, Any]) -> ConflictCandidateRanker:
    config = sub["model_config"]
    model = ConflictCandidateRanker(
        config["event_dim"], config["candidate_dim"], config["sequence_length"],
        config["countries"], config["conflicts"],
        config.get("d_model"), config.get("heads"), config.get("layers"),
        config.get("ff_dim"), config.get("dropout"),
    )
    model.load_state_dict(sub["model_state"])
    return model


def _member_states(package: dict[str, Any]) -> list[dict[str, Any]]:
    if "models" in package:
        return list(package["models"])
    return [{key: package[key] for key in ("model_config", "model_state")}]


def _ensemble_probs(states: list[dict[str, Any]], tensors, rows, device, batch: int = 256):
    """Mean-of-softmax probabilities for the given row indices (serving parity)."""
    x, f, v = tensors["x"], tensors["f"], tensors["v"]
    ctry, cflt = tensors["ctry"], tensors["cflt"]
    prob_sum: np.ndarray | None = None
    for sub in states:
        model = _member_model(sub).to(device).eval()
        probs = []
        with torch.no_grad():
            for lo in range(0, len(rows), batch):
                sl = rows[lo:lo + batch]
                xb = torch.from_numpy(x[sl]).to(device)
                fb = torch.from_numpy(f[sl]).to(device)
                vb = torch.from_numpy(v[sl]).to(device)
                cb = torch.from_numpy(ctry[sl]).to(device)
                fb_conf = torch.from_numpy(cflt[sl]).to(device)
                logits = model(xb, fb, vb, cb, fb_conf)
                probs.append(torch.where(vb, logits.softmax(-1), torch.zeros((), device=device)).cpu().numpy())
        stacked = np.concatenate(probs)
        prob_sum = stacked if prob_sum is None else prob_sum + stacked
        del model
    assert prob_sum is not None
    return prob_sum / len(states)


def _metrics(probs: np.ndarray, tensors, rows: np.ndarray, top_k: int, spread_km: float) -> dict[str, Any]:
    c = tensors["c"][rows]
    v = tensors["v"][rows]
    y = tensors["y"][rows]
    d = np.linalg.norm(c - y[:, None, :], axis=2) * SCALE
    order = np.argsort(-np.where(v, probs, -1), axis=1)
    e1 = d[np.arange(len(rows)), order[:, 0]]
    result: dict[str, Any] = {
        "samples": int(len(rows)),
        "within_20km_at_top1": round(float((e1 <= 20).mean()), 3),
        "median_error_km": round(float(np.median(e1)), 1),
    }
    for k in (3, 5, 10):
        plain = d[np.arange(len(rows))[:, None], order[:, :k]].min(1)
        result[f"plain_within_20km_at_top{k}"] = round(float((plain <= 20).mean()), 3)
        if k >= top_k:
            greedy = [d[i, greedy_order(np.where(v[i], probs[i], 0.0), c[i] * SCALE, v[i], top_k, spread_km)].min() for i in range(len(rows))]
            result[f"diverse_within_20km_at_top{k}"] = round(float((np.asarray(greedy) <= 20).mean()), 3)
    result["oracle_within_20km"] = round(float((np.where(v, d, 1e9).min(1) <= 20).mean()), 3)
    mass = np.where(v & (d <= 20), probs, 0.0).sum(1)
    result["probability_mass_within_20km"] = round(float(mass.mean()), 3)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/location/conflict_candidates_64_spillover_v6.npz"))
    parser.add_argument("--live", type=Path, default=None, help="labeled live pairs npz from label_live_pairs")
    parser.add_argument("--package", type=Path, default=Path("models/location/theswarm/fine_v2/model.pt"),
                        help="current production package to continue from")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--distance-weight", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-live-rows", type=int, default=100000)
    a = parser.parse_args()

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = torch.device("cpu")
    package = torch.load(a.package, map_location="cpu", weights_only=False)
    parent_states = _member_states(package)
    selection = package.get("selection", {"method": "greedy_spread", "spread_km": 30.0, "top_k": 5})
    spread_km = float(selection.get("spread_km", 30.0))
    top_k = int(selection.get("top_k", 5))
    country_map = {str(k): int(v) for k, v in package["country_map"].items()}
    conflict_map = {str(k): int(v) for k, v in package["conflict_map"].items()}

    # Only Ethiopia rows (plus live rows) are ever materialized: the fine-tune
    # set is Ethiopia-scoped, so the 1.5 GB global candidate tensor never
    # needs to exist in memory.
    with np.load(a.data, allow_pickle=False) as d:
        meta = [json.loads(str(z)) for z in d["meta"]]
        n = len(meta)
        te, ve = int(.7 * n), int(.85 * n)
        eth = np.asarray([i for i, m in enumerate(meta) if m["country"] == "Ethiopia"])
        eth_train = eth[eth < te]
        eth_val = eth[(eth >= te) & (eth < ve)]
        eth_dev = eth[eth >= ve]
        keep = np.sort(np.concatenate([eth_train, eth_val, eth_dev]))
        remap = {int(g): i for i, g in enumerate(keep)}
        tensors = {
            "x": d["x"][keep].astype(np.float32),
            "f": d["candidate_features"][keep].astype(np.float32),
            "c": d["candidate_coordinates"][keep].astype(np.float32),
            "v": d["candidate_valid"][keep].astype(bool),
            "y": d["y"][keep].astype(np.float32),
            "label": d["label"][keep].astype(np.int64),
        }
        base_meta = [meta[g] for g in keep]

    live = _load_live(a.live)
    live_rows = 0
    if live:
        live_rows = min(len(live["meta"]), a.max_live_rows)
        live_block = {
            "x": live["x"][:live_rows].astype(np.float32),
            "f": live["candidate_features"][:live_rows].astype(np.float32),
            "c": live["candidate_coordinates"][:live_rows].astype(np.float32),
            "v": live["candidate_valid"][:live_rows].astype(bool),
            "y": live["y"][:live_rows].astype(np.float32),
            "label": live["label"][:live_rows].astype(np.int64),
        }
        tensors = {key: np.concatenate([value, live_block[key]]) for key, value in tensors.items()}
        base_meta.extend(json.loads(str(z)) for z in live["meta"][:live_rows])

    def ids_for(rows_meta):
        return (
            np.asarray([country_map.get(m["country"], 0) for m in rows_meta], dtype=np.int64),
            np.asarray([conflict_map.get(str(m["conflict_id"]), 0) for m in rows_meta], dtype=np.int64),
        )

    ctry_all, cflt_all = ids_for(base_meta)
    tensors["ctry"] = ctry_all
    tensors["cflt"] = cflt_all

    n_total = len(base_meta)
    n_base = n_total - live_rows
    # Row layout after the sort: [eth_train | eth_val | eth_dev | live]
    train_rows = np.arange(0, len(eth_train))
    val_rows = np.arange(len(eth_train), len(eth_train) + len(eth_val))
    dev_rows = np.arange(len(eth_train) + len(eth_val), n_base)
    live_idx = np.arange(n_base, n_total)
    ft_rows = np.concatenate([train_rows, live_idx]) if live_rows else train_rows
    print(json.dumps({
        "device": str(device), "members": len(parent_states),
        "eth_train": int(len(eth_train)), "eth_val": int(len(eth_val)), "eth_dev": int(len(eth_dev)),
        "live_rows": int(live_rows),
    }), flush=True)

    dataset = TensorDataset(
        torch.from_numpy(tensors["x"]), torch.from_numpy(tensors["f"]),
        torch.from_numpy(tensors["c"]), torch.from_numpy(tensors["v"]),
        torch.from_numpy(tensors["label"]), torch.from_numpy(tensors["y"]),
        torch.from_numpy(tensors["ctry"]), torch.from_numpy(tensors["cflt"]),
    )

    def evaluate(model, rows):
        loader = DataLoader(torch.utils.data.Subset(dataset, rows.tolist()), batch_size=256)
        model.eval()
        logits = []
        with torch.no_grad():
            for xb, fb, cb, vb, lb, yb, countryb, conflictb in loader:
                logits.append(model(xb.to(device), fb.to(device), vb.to(device), countryb.to(device), conflictb.to(device)).cpu())
        stacked = torch.cat(logits)
        dd = torch.linalg.vector_norm(
            torch.from_numpy(tensors["c"][rows]) - torch.from_numpy(tensors["y"][rows])[:, None], dim=-1
        ) * SCALE
        e = dd[torch.arange(len(rows)), stacked.argmax(-1)]
        return {"median_km": float(e.median()), "w20": float((e <= 20).float().mean())}

    new_states = []
    for member, sub in enumerate(parent_states):
        model = _member_model(sub).to(device)
        start = evaluate(model, val_rows)
        print(f"member {member} start eth_val={json.dumps(start)}", flush=True)
        best = start["median_km"]
        best_state = {k: z.detach().cpu().clone() for k, z in model.state_dict().items()}
        best_epoch = 0
        loader = DataLoader(torch.utils.data.Subset(dataset, ft_rows.tolist()), batch_size=a.batch_size, shuffle=True)
        for epoch in range(1, a.epochs + 1):
            model.train()
            opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=.01)
            total = 0
            for xb, fb, cb, vb, lb, yb, countryb, conflictb in loader:
                opt.zero_grad(set_to_none=True)
                logits = model(xb, fb, vb, countryb, conflictb)
                loss, _, _, _ = loss_fn(logits, cb, yb, lb, a.distance_weight)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                opt.step()
                total += loss.detach().item() * len(xb)
            metrics = evaluate(model, val_rows)
            print(f"member {member} epoch={epoch:02d} train_loss={total/len(ft_rows):.4f} eth_val={json.dumps(metrics)}", flush=True)
            if metrics["median_km"] < best:
                best = metrics["median_km"]
                best_epoch = epoch
                best_state = {k: z.detach().cpu().clone() for k, z in model.state_dict().items()}
        model.load_state_dict(best_state)
        new_states.append({
            "model_config": sub["model_config"],
            "model_state": {k: z.detach().cpu() for k, z in model.state_dict().items()},
        })
        print(f"member {member} selected epoch={best_epoch} median_km={best:.1f}", flush=True)
        del model

    # Ensemble metrics (mean of softmax, matching serving) for the info card.
    all_rows = np.concatenate([val_rows, dev_rows, live_idx]) if live_rows else np.concatenate([val_rows, dev_rows])
    probs = _ensemble_probs(new_states, tensors, all_rows, device)
    val_probs = probs[:len(val_rows)]
    dev_probs = probs[len(val_rows):len(val_rows) + len(dev_rows)]
    live_probs = probs[len(val_rows) + len(dev_rows):]
    out = {
        "validation": _metrics(val_probs, tensors, val_rows, top_k, spread_km),
        "development": _metrics(dev_probs, tensors, dev_rows, top_k, spread_km),
    }
    live_metrics = _metrics(live_probs, tensors, live_idx, top_k, spread_km) if live_rows else None

    version = f"retrain-{datetime.now(UTC).strftime('%Y%m%d-%H%M')}"
    a.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "models": new_states,
        "country_map": package["country_map"],
        "conflict_map": package["conflict_map"],
        "selection": selection,
        "data": str(a.data),
        "parent": str(a.package),
        "liveRows": int(live_rows),
        "version": version,
    }, a.output_dir / "model.pt")
    card = {
        "model": f"theswarm_fine_v2 ({version})",
        "version": version,
        "parent": str(a.package),
        "input": "64 cutoff-safe candidates (32 conflict-frequency sites + 32 recent cross-conflict event sites), 37 features incl. ReliefWeb mentions, UCDP windows, spillover activity, elevation/ruggedness",
        "blend": "mean of softmax over the member checkpoints",
        "zone_selection": {"method": "greedy spatial-spread discount", "spread_km": spread_km},
        "retrain": {
            "continuedFrom": str(a.package),
            "liveValidatedRows": int(live_rows),
            "epochs": a.epochs,
            "lr": a.lr,
        },
        "ethiopia_metrics": out,
        "live_metrics": live_metrics,
        "caveats": [
            "oracle_within_20km is the coverage ceiling of the candidate pool, not model accuracy",
            "roughly half of Ethiopia validation targets are geolocated by UCDP to a named-place radius (where_prec>=2), so 20 km hits on those rows are partly coordinate noise",
            "Coarse humanitarian early-warning research signal, not a tactical coordinate forecast",
        ],
    }
    (a.output_dir / "info.blt").write_text(json.dumps(card, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"version": version, "validation": out["validation"], "live": live_metrics}), flush=True)


if __name__ == "__main__":
    main()
