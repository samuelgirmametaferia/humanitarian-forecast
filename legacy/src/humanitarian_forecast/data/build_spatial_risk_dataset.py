#!/usr/bin/env python3
"""Augment a temporal-risk dataset with cutoff-safe PRIO-GRID spillover channels.

For each sample, this builder looks only at *input histories* from neighboring
PRIO-GRID cells at the same forecast cutoff. It never uses neighboring targets
or the presence of a future-only onset as a signal. The resulting channels are
coarse humanitarian context features, not precise tactical-location outputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


BASE_FEATURE_NAMES = [
    "activity",
    "attack",
    "fatality",
    "displacement",
    "territory",
    "infrastructure",
    "source_diversity",
    "mean_confidence",
]


def gid_to_row_col(gid: int) -> tuple[int, int]:
    """PRIO-GRID ids are row-major over the underlying 360x720 global grid."""
    zero = gid - 1
    return zero // 720 + 1, zero % 720 + 1


def row_col_to_gid(row: int, col: int) -> int | None:
    if row < 1 or row > 360 or col < 1 or col > 720:
        return None
    return (row - 1) * 720 + col


def ring_gids(gid: int, radius: int) -> list[int]:
    row, col = gid_to_row_col(gid)
    out: list[int] = []
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            if max(abs(dr), abs(dc)) != radius:
                continue
            neighbor = row_col_to_gid(row + dr, col + dc)
            if neighbor is not None:
                out.append(neighbor)
    return out


def parse_meta(raw: np.ndarray) -> list[dict[str, str]]:
    return [json.loads(str(value)) for value in raw]


def augment_spatial(
    x: np.ndarray,
    meta: list[dict[str, str]],
) -> tuple[np.ndarray, list[str]]:
    if x.ndim != 3:
        raise ValueError(f"expected x with shape [samples,time,features], got {x.shape}")
    if x.shape[-1] != len(BASE_FEATURE_NAMES):
        raise ValueError(
            f"spatial augmentation currently expects {len(BASE_FEATURE_NAMES)} base features, "
            f"got {x.shape[-1]}"
        )
    if len(x) != len(meta):
        raise ValueError("x/meta length mismatch")

    # Index only PRIO samples. Merely being present in the source dataset can
    # depend on the target in legacy builders, so we never use presence as a
    # feature. Neighbor contributions are numeric input histories only.
    by_cutoff: dict[str, dict[int, int]] = {}
    gids = np.full(len(meta), -1, dtype=np.int64)
    for idx, item in enumerate(meta):
        entity = item.get("entity", "")
        if not entity.startswith("prio:"):
            continue
        try:
            gid = int(entity.split(":", 1)[1])
        except ValueError:
            continue
        gids[idx] = gid
        by_cutoff.setdefault(item["cutoff"], {})[gid] = idx

    ring1 = np.zeros_like(x, dtype=np.float32)
    ring2 = np.zeros_like(x, dtype=np.float32)
    # Active-neighbor counts are defined from the historical activity channel,
    # not from row existence, to avoid legacy sample-selection leakage.
    active1 = np.zeros((len(x), x.shape[1], 1), dtype=np.float32)
    active2 = np.zeros((len(x), x.shape[1], 1), dtype=np.float32)

    for idx, gid in enumerate(gids):
        if gid < 0:
            continue
        cutoff_map = by_cutoff.get(meta[idx]["cutoff"], {})
        for radius, destination, counts in (
            (1, ring1, active1),
            (2, ring2, active2),
        ):
            for neighbor_gid in ring_gids(int(gid), radius):
                neighbor_idx = cutoff_map.get(neighbor_gid)
                if neighbor_idx is None:
                    continue
                history = x[neighbor_idx]
                destination[idx] += history
                counts[idx, :, 0] += (history[:, 0] > 0).astype(np.float32)

    augmented = np.concatenate([x, ring1, ring2, active1, active2], axis=-1).astype(np.float32)
    names = (
        BASE_FEATURE_NAMES
        + [f"ring1_{name}" for name in BASE_FEATURE_NAMES]
        + [f"ring2_{name}" for name in BASE_FEATURE_NAMES]
        + ["ring1_active_neighbors", "ring2_active_neighbors"]
    )
    return augmented, names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    bundle = np.load(args.input, allow_pickle=True)
    x = bundle["x"].astype(np.float32, copy=False)
    y = bundle["y"].astype(np.float32, copy=False)
    raw_meta = bundle["meta"]
    meta = parse_meta(raw_meta)
    augmented, names = augment_spatial(x, meta)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        x=augmented,
        y=y,
        meta=raw_meta,
        feature_names=np.array(names),
        spatial_context_version=np.array("prio-ring-v1"),
    )
    print(
        f"{args.output}: {len(augmented):,} samples, shape={augmented.shape}, "
        f"features={len(names)}"
    )


if __name__ == "__main__":
    main()
