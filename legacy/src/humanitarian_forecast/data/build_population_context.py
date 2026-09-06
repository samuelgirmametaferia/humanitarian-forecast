#!/usr/bin/env python3
"""Aggregate a local WorldPop raster around candidate locations."""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import from_bounds


def candidate_points(path: Path, country: str) -> list[tuple[float, float]]:
    z = np.load(path)
    meta = [json.loads(str(v)) for v in z["meta"]]
    selected = np.asarray([str(m.get("country", "")).casefold() == country.casefold() for m in meta])
    f = z["candidate_features"][selected]
    valid = z["candidate_valid"][selected]
    lat = f[:, :, 6] * 90.0
    lon = f[:, :, 7] * 180.0
    return sorted({
        (round(float(a), 5), round(float(b), 5))
        for a, b, ok in zip(lat.reshape(-1), lon.reshape(-1), valid.reshape(-1))
        if bool(ok) and np.isfinite(a) and np.isfinite(b)
    })


def km_to_lat(km: float) -> float:
    return km / 111.32


def km_to_lon(km: float, lat: float) -> float:
    return km / (111.32 * max(0.20, math.cos(math.radians(lat))))


def read_square(ds: rasterio.io.DatasetReader, lat: float, lon: float, radius_km: float) -> np.ndarray:
    dy = km_to_lat(radius_km)
    dx = km_to_lon(radius_km, lat)
    left, bottom, right, top = lon - dx, lat - dy, lon + dx, lat + dy
    if right < ds.bounds.left or left > ds.bounds.right or top < ds.bounds.bottom or bottom > ds.bounds.top:
        return np.empty(0, dtype=np.float32)
    left = max(left, ds.bounds.left)
    right = min(right, ds.bounds.right)
    bottom = max(bottom, ds.bounds.bottom)
    top = min(top, ds.bounds.top)
    if left >= right or bottom >= top:
        return np.empty(0, dtype=np.float32)
    window = from_bounds(left, bottom, right, top, ds.transform)
    data = ds.read(1, window=window, masked=True)
    if np.ma.isMaskedArray(data):
        values = data.compressed().astype(np.float32, copy=False)
    else:
        values = np.asarray(data, dtype=np.float32).reshape(-1)
    values = values[np.isfinite(values)]
    return values[values >= 0]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate-data", type=Path, required=True)
    p.add_argument("--raster", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--country", default="Ethiopia")
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output}")
    points = candidate_points(args.candidate_data, args.country)
    out = np.zeros((len(points), 10), dtype=np.float32)
    with rasterio.open(args.raster) as ds:
        if ds.crs is None or str(ds.crs).upper() not in {"EPSG:4326", "OGC:CRS84"}:
            raise ValueError(f"expected WGS84 population raster, got {ds.crs}")
        for i, (lat, lon) in enumerate(points):
            if not (ds.bounds.left <= lon <= ds.bounds.right and ds.bounds.bottom <= lat <= ds.bounds.top):
                out[i, -1] = 1.0
                continue
            samples: dict[float, np.ndarray] = {r: read_square(ds, lat, lon, r) for r in (2.5, 10.0, 25.0)}
            if any(len(v) == 0 for v in samples.values()):
                out[i, -1] = 1.0
            a, b, c = samples[2.5], samples[10.0], samples[25.0]
            if len(a):
                out[i, 0] = math.log1p(float(a.sum()))
                out[i, 1] = math.log1p(float(a.max()))
            if len(b):
                out[i, 2] = math.log1p(float(b.sum()))
                out[i, 3] = math.log1p(float(b.mean()))
                out[i, 4] = math.log1p(float(b.max()))
                out[i, 5] = math.log1p(float(np.count_nonzero(b > 0)))
            if len(c):
                out[i, 6] = math.log1p(float(c.sum()))
                out[i, 7] = math.log1p(float(c.mean()))
                out[i, 8] = math.log1p(float(np.count_nonzero(c > 0)))
            if i and i % 200 == 0:
                print(f"population context {i}/{len(points)}", flush=True)
        provenance = {
            "schema": "geo-context/worldpop-v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "country_filter": args.country,
            "candidate_data": str(args.candidate_data),
            "raster": str(args.raster),
            "raster_crs": str(ds.crs),
            "raster_bounds": [float(v) for v in ds.bounds],
            "raster_resolution": [float(v) for v in ds.res],
            "source": "WorldPop Global 2015-2030 R2025A, Ethiopia 2026 constrained 100m",
            "doi": "10.5258/SOTON/WP00839",
            "license": "CC BY 4.0",
        }
    names = np.asarray([
        "log_population_5km_square",
        "log_population_max_pixel_5km",
        "log_population_20km_square",
        "log_population_mean_pixel_20km",
        "log_population_max_pixel_20km",
        "log_populated_pixels_20km",
        "log_population_50km_square",
        "log_population_mean_pixel_50km",
        "log_populated_pixels_50km",
        "population_context_missing",
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        points=np.asarray(points, dtype=np.float32),
        features=out,
        feature_names=names,
        provenance=np.asarray(json.dumps(provenance, sort_keys=True)),
    )
    print(json.dumps({**provenance, "points": len(points), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
