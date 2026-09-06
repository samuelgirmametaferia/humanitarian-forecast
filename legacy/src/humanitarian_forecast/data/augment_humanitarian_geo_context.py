#!/usr/bin/env python3
"""Append coarse static PRIO-grid humanitarian context to a temporal-risk dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--context", type=Path, required=True, action="append")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output}")

    bundle = np.load(args.input, allow_pickle=True)
    x = bundle["x"].astype(np.float32, copy=False)
    y = bundle["y"].astype(np.float32, copy=False)
    raw_meta = bundle["meta"]
    meta = [json.loads(str(v)) for v in raw_meta]
    base_names = [str(v) for v in bundle["feature_names"]] if "feature_names" in bundle else [f"feature_{i}" for i in range(x.shape[-1])]

    additions: list[np.ndarray] = []
    names: list[str] = []
    provenance: list[dict[str, object]] = []
    for path in args.context:
        ctx = np.load(path, allow_pickle=True)
        gids = ctx["gids"].astype(np.int64)
        features = ctx["features"].astype(np.float32)
        if len(gids) != len(features):
            raise ValueError(f"gid/feature mismatch in {path}")
        lookup = {int(gid): features[i] for i, gid in enumerate(gids)}
        static = np.zeros((len(x), features.shape[1] + 1), dtype=np.float32)
        for i, item in enumerate(meta):
            entity = str(item.get("entity", ""))
            vector = None
            if entity.startswith("prio:"):
                try:
                    vector = lookup.get(int(entity.split(":", 1)[1]))
                except ValueError:
                    vector = None
            if vector is None:
                static[i, -1] = 1.0
            else:
                static[i, :-1] = vector
        repeated = np.repeat(static[:, None, :], x.shape[1], axis=1)
        additions.append(repeated)
        ctx_names = [str(v) for v in ctx["feature_names"]]
        stem = path.stem.replace("-", "_")
        names.extend(ctx_names + [f"{stem}_sample_missing"])
        if "provenance" in ctx:
            provenance.append(json.loads(str(ctx["provenance"])))

    augmented = np.concatenate([x, *additions], axis=-1).astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        x=augmented,
        y=y,
        meta=raw_meta,
        feature_names=np.asarray(base_names + names),
        geo_context_version=np.asarray("humanitarian-prio-static-v1"),
        geo_context_provenance=np.asarray(json.dumps(provenance, sort_keys=True)),
    )
    print(json.dumps({
        "input": str(args.input),
        "output": str(args.output),
        "samples": len(x),
        "shape": list(augmented.shape),
        "added_features": names,
    }, indent=2))


if __name__ == "__main__":
    main()
