#!/usr/bin/env python3
"""Build a dense multiresolution H3 dataset for coarse humanitarian forecasting.

This builder is intentionally additive: it reuses the existing causal motion-v3
history examples and replaces the historical-candidate output support with every
H3 cell inside a supplied country/admin boundary. It does not expose tactical
routes or unit positions; the intended output is a broad humanitarian-risk field.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h3
import numpy as np

EARTH_KM = 111.32


def _absolute_history(events: np.ndarray, anchor_lat: float, anchor_lon: float) -> np.ndarray:
    """Convert encoded anchor-relative positions back to approximate lat/lon."""
    east_km = events[:, 1] * 1000.0
    north_km = events[:, 2] * 1000.0
    lat = anchor_lat + north_km / EARTH_KM
    lon_scale = EARTH_KM * max(0.1, math.cos(math.radians(anchor_lat)))
    lon = anchor_lon + east_km / lon_scale
    return np.stack([lat, lon], axis=-1)


def _cells_for_boundary(path: Path, resolution: int) -> list[str]:
    geo = json.loads(path.read_text())
    if geo.get("type") == "FeatureCollection":
        geometries = [feature["geometry"] for feature in geo["features"]]
    elif geo.get("type") == "Feature":
        geometries = [geo["geometry"]]
    else:
        geometries = [geo]
    cells: set[str] = set()
    for geometry in geometries:
        cells.update(h3.geo_to_cells(geometry, resolution))
    return sorted(cells)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--boundary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--country", default="Ethiopia")
    parser.add_argument("--resolutions", default="3,4,5")
    args = parser.parse_args()

    source = np.load(args.data, allow_pickle=True)
    rows = [json.loads(str(value)) for value in source["meta"]]
    keep = np.asarray([row["country"] == args.country for row in rows], dtype=bool)
    indices = np.flatnonzero(keep)
    if not len(indices):
        raise ValueError(f"no samples found for country={args.country!r}")

    x = source["x"][indices].astype(np.float32, copy=False)
    y = source["y"][indices].astype(np.float32, copy=False)
    meta = source["meta"][indices]
    selected_rows = [rows[index] for index in indices]
    resolutions = tuple(int(value) for value in args.resolutions.split(",") if value.strip())

    payload: dict[str, np.ndarray] = {
        "x": x,
        "y": y,
        "meta": meta,
        "source_indices": indices.astype(np.int64),
        "source_total_samples": np.asarray(len(rows), dtype=np.int64),
        "source_train_end": np.asarray(int(0.70 * len(rows)), dtype=np.int64),
        "source_validation_end": np.asarray(int(0.85 * len(rows)), dtype=np.int64),
        "resolutions": np.asarray(resolutions, dtype=np.int16),
        "country": np.asarray(args.country),
        "schema": np.asarray("dense-h3-multiscale-v1"),
    }
    report: dict[str, object] = {
        "schema": "dense-h3-multiscale-v1",
        "country": args.country,
        "samples": int(len(indices)),
        "source": str(args.data),
        "boundary": str(args.boundary),
        "resolutions": {},
    }

    for resolution in resolutions:
        cells = _cells_for_boundary(args.boundary, resolution)
        lookup = {cell: idx for idx, cell in enumerate(cells)}
        centroids = np.asarray([h3.cell_to_latlng(cell) for cell in cells], dtype=np.float32)
        target_index = np.full(len(indices), -1, dtype=np.int32)
        history_index = np.full((len(indices), x.shape[1]), -1, dtype=np.int32)
        in_boundary = np.zeros(len(indices), dtype=bool)

        for local_index, (events, row) in enumerate(zip(x, selected_rows)):
            target_cell = h3.latlng_to_cell(float(row["target_lat"]), float(row["target_lon"]), resolution)
            if target_cell in lookup:
                target_index[local_index] = lookup[target_cell]
                in_boundary[local_index] = True
            valid = events[:, 0] > 0.5
            absolute = _absolute_history(events, float(row["anchor_lat"]), float(row["anchor_lon"]))
            for timestep in np.flatnonzero(valid):
                cell = h3.latlng_to_cell(float(absolute[timestep, 0]), float(absolute[timestep, 1]), resolution)
                history_index[local_index, timestep] = lookup.get(cell, -1)

        payload[f"cells_r{resolution}"] = np.asarray(cells)
        payload[f"centroids_r{resolution}"] = centroids
        payload[f"target_index_r{resolution}"] = target_index
        payload[f"history_index_r{resolution}"] = history_index
        payload[f"target_in_boundary_r{resolution}"] = in_boundary
        report["resolutions"][str(resolution)] = {
            "cells": len(cells),
            "target_coverage": float(in_boundary.mean()),
            "covered_samples": int(in_boundary.sum()),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
