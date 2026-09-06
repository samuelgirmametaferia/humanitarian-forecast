#!/usr/bin/env python3
"""Append cutoff-safe spatial ReliefWeb semantic fields to candidate features.

The sparse mention file is generated from publication-time (date.created)
ReliefWeb reports and an auditable Ethiopia gazetteer.  For every Ethiopia
forecast example, candidate-local features summarize only mentions published on
or before the observation cutoff. Non-Ethiopia examples receive zero context so
the same global replay dataset remains usable.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

EARTH_KM = 6371.0088
SPATIAL_KM = (25.0, 50.0, 100.0, 200.0)
HALF_LIFE_DAYS = (7.0, 30.0, 90.0)
MAX_LOOKBACK_DAYS = 365


def haversine_matrix(candidate_latlon: np.ndarray, mention_latlon: np.ndarray) -> np.ndarray:
    """Return candidate x mention great-circle distances in km."""
    lat1 = np.deg2rad(candidate_latlon[:, 0])[:, None]
    lon1 = np.deg2rad(candidate_latlon[:, 1])[:, None]
    lat2 = np.deg2rad(mention_latlon[:, 0])[None, :]
    lon2 = np.deg2rad(mention_latlon[:, 1])[None, :]
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return (2.0 * EARTH_KM * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(1.0 - a, 0.0)))).astype(np.float32)


def feature_names(semantic_names: list[str]) -> list[str]:
    names: list[str] = []
    for radius in SPATIAL_KM:
        for half_life in HALF_LIFE_DAYS:
            for semantic in semantic_names:
                names.append(f"rw_{semantic}_{int(radius)}km_hl{int(half_life)}d")
    # Explicit recent-vs-background changes are useful onset features.
    for semantic in semantic_names:
        names.append(f"rw_{semantic}_100km_recent14_minus_bg90")
    return names


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--mentions", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--country", default="Ethiopia")
    args = p.parse_args()

    z = np.load(args.data, allow_pickle=True)
    base = z["candidate_features"].astype(np.float32, copy=False)
    valid = z["candidate_valid"]
    meta = [json.loads(str(v)) for v in z["meta"]]
    m = np.load(args.mentions, allow_pickle=True)
    days = m["day"].astype(np.int32)
    latlon = m["latlon"].astype(np.float32)
    mf = m["features"].astype(np.float32)
    source_names = [str(v) for v in m["feature_names"]]
    # Keep report-presence + seven semantic lexicons; gazetteer confidence stays
    # in the mention extractor manifest instead of becoming a forecasting signal.
    semantic_dim = min(8, mf.shape[1])
    semantic_names = source_names[:semantic_dim]
    mf = mf[:, :semantic_dim]
    names = feature_names(semantic_names)
    extra = np.zeros((len(base), base.shape[1], len(names)), dtype=np.float32)
    wanted = args.country.casefold()
    matched_examples = 0
    mentions_considered = 0

    for i, row in enumerate(meta):
        if str(row.get("country", "")).casefold() != wanted:
            continue
        cutoff_date = np.datetime64(str(row["target_date"]), "D") - np.timedelta64(int(row["gap_days"]), "D")
        cutoff = int(cutoff_date.astype(int))
        left = int(np.searchsorted(days, cutoff - MAX_LOOKBACK_DAYS, side="left"))
        right = int(np.searchsorted(days, cutoff, side="right"))
        if right <= left:
            continue
        cand_valid = valid[i]
        if not cand_valid.any():
            continue
        cand = np.stack([base[i, cand_valid, 6] * 90.0, base[i, cand_valid, 7] * 180.0], axis=-1).astype(np.float32)
        local_days = days[left:right]
        local_xy = latlon[left:right]
        local_f = mf[left:right]
        age = np.maximum(0.0, cutoff - local_days).astype(np.float32)
        dist = haversine_matrix(cand, local_xy)
        blocks: list[np.ndarray] = []
        for radius in SPATIAL_KM:
            spatial = np.exp(-0.5 * (dist / radius) ** 2).astype(np.float32)
            for half_life in HALF_LIFE_DAYS:
                temporal = np.exp(-np.log(2.0) * age / half_life).astype(np.float32)
                weight = spatial * temporal[None, :]
                value = weight @ local_f
                blocks.append(np.log1p(np.maximum(value, 0.0)).astype(np.float32))
        # Recent-vs-background semantic delta at 100 km. Positive means an
        # unusually strong recent signal compared with the previous local state.
        spatial100 = np.exp(-0.5 * (dist / 100.0) ** 2).astype(np.float32)
        recent_w = spatial100 * np.exp(-np.log(2.0) * age[None, :] / 14.0)
        bg_w = spatial100 * np.exp(-np.log(2.0) * age[None, :] / 90.0)
        recent = np.log1p(np.maximum(recent_w @ local_f, 0.0))
        background = np.log1p(np.maximum(bg_w @ local_f, 0.0))
        blocks.append((recent - background).astype(np.float32))
        context = np.concatenate(blocks, axis=1)
        extra[i, cand_valid] = context
        matched_examples += 1
        mentions_considered += right - left
        if matched_examples % 500 == 0:
            print(f"semantic context {matched_examples} Ethiopia examples", flush=True)

    augmented = np.concatenate([base, extra], axis=-1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: z[k] for k in z.files if k != "candidate_features"}
    payload["candidate_features"] = augmented
    payload["base_candidate_dim"] = np.asarray(base.shape[-1], dtype=np.int32)
    payload["reliefweb_appended_feature_names"] = np.asarray(names)
    payload["reliefweb_context_lineage"] = np.asarray(json.dumps({
        "schema": "candidate-reliefweb-spatial-v1",
        "mentions": str(args.mentions),
        "causality": "date.created mention timestamp <= observation cutoff",
        "spatial_km": SPATIAL_KM,
        "half_life_days": HALF_LIFE_DAYS,
        "max_lookback_days": MAX_LOOKBACK_DAYS,
        "semantic_names": semantic_names,
    }, sort_keys=True))
    np.savez_compressed(args.output, **payload)
    report = {
        "schema": "candidate-reliefweb-spatial-v1",
        "samples": len(base),
        "candidate_dim_before": int(base.shape[-1]),
        "candidate_dim_after": int(augmented.shape[-1]),
        "appended_features": len(names),
        "ethiopia_examples_with_context": matched_examples,
        "mean_lookback_mentions_per_context_example": (mentions_considered / matched_examples if matched_examples else 0.0),
        "mentions": str(args.mentions),
        "output": str(args.output),
        "causality": "all report signals use publication time and are truncated at observation cutoff",
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
