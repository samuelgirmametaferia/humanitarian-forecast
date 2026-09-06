#!/usr/bin/env python3
"""Append materialized static geo context to a candidate-ranking dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-data", type=Path, required=True)
    parser.add_argument("--context", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--country", help="If supplied, keep only this country's examples")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing dataset: {args.output}")

    base = np.load(args.candidate_data)
    meta_text = base["meta"]
    meta = [json.loads(str(value)) for value in meta_text]
    keep = np.ones(len(meta), dtype=bool)
    if args.country:
        keep = np.asarray([str(row.get("country", "")).casefold() == args.country.casefold() for row in meta])
    indices = np.flatnonzero(keep)
    candidate = base["candidate_features"][indices].astype(np.float32, copy=True)
    valid = base["candidate_valid"][indices]
    lat = candidate[:, :, 6] * 90.0
    lon = candidate[:, :, 7] * 180.0

    additions: list[np.ndarray] = []
    all_names: list[str] = []
    context_lineage: list[dict[str, object]] = []
    for context_path in args.context:
        ctx = np.load(context_path)
        points = ctx["points"]
        vectors = ctx["features"].astype(np.float32)
        names = [str(value) for value in ctx["feature_names"]]
        lookup = {
            (round(float(point[0]), 5), round(float(point[1]), 5)): vectors[i]
            for i, point in enumerate(points)
        }
        out = np.zeros((len(indices), candidate.shape[1], vectors.shape[1] + 1), dtype=np.float32)
        for i in range(len(indices)):
            for j in np.flatnonzero(valid[i]):
                vector = lookup.get((round(float(lat[i, j]), 5), round(float(lon[i, j]), 5)))
                if vector is not None:
                    out[i, j, :-1] = vector
                    out[i, j, -1] = 1.0
        additions.append(out)
        all_names.extend(names + [f"{context_path.stem}_available"])
        provenance = str(ctx["provenance"]) if "provenance" in ctx.files else "{}"
        context_lineage.append(json.loads(provenance))

    augmented = np.concatenate([candidate, *additions], axis=-1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        x=base["x"][indices],
        candidate_features=augmented,
        candidate_coordinates=base["candidate_coordinates"][indices],
        candidate_valid=valid,
        label=base["label"][indices],
        y=base["y"][indices],
        meta=meta_text[indices],
        base_candidate_dim=np.asarray(candidate.shape[-1]),
        appended_feature_names=np.asarray(all_names),
        context_lineage=np.asarray(json.dumps(context_lineage, sort_keys=True)),
    )
    print(json.dumps({
        "samples": len(indices),
        "base_candidate_dim": int(candidate.shape[-1]),
        "augmented_candidate_dim": int(augmented.shape[-1]),
        "contexts": [str(path) for path in args.context],
        "country": args.country,
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
