#!/usr/bin/env python3
"""Materialize coarse Copernicus GLO-90 terrain features for candidate locations.

For development this uses Open-Meteo's documented bulk elevation endpoint,
which is backed by Copernicus DEM 2021 GLO-90.  The stored artifact is detached
from the API and carries provenance.  Production builds may swap in direct
Copernicus AWS COG sampling without changing the feature contract.
"""
from __future__ import annotations

import argparse
import json
import math
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ENDPOINT = "https://api.open-meteo.com/v1/elevation"


def _fetch(points: list[tuple[float, float]], *, batch_size: int = 100) -> np.ndarray:
    values: list[float] = []
    for start in range(0, len(points), batch_size):
        batch = points[start:start + batch_size]
        query = urllib.parse.urlencode({
            "latitude": ",".join(f"{lat:.5f}" for lat, _ in batch),
            "longitude": ",".join(f"{lon:.5f}" for _, lon in batch),
        })
        request = urllib.request.Request(
            f"{ENDPOINT}?{query}",
            headers={"User-Agent": "humanitarian-forecast/geo-context-v1"},
        )
        for attempt in range(5):
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    payload = json.load(response)
                elevations = payload.get("elevation")
                if not isinstance(elevations, list) or len(elevations) != len(batch):
                    raise RuntimeError(f"unexpected elevation response: {payload}")
                values.extend(float(value) if value is not None else float("nan") for value in elevations)
                break
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(0.5 * (2 ** attempt))
        time.sleep(0.06)
    return np.asarray(values, dtype=np.float32)


def _candidate_points(bundle: np.lib.npyio.NpzFile, country: str | None) -> list[tuple[float, float]]:
    features = bundle["candidate_features"]
    valid = bundle["candidate_valid"]
    meta = [json.loads(str(value)) for value in bundle["meta"]]
    selected = np.ones(len(meta), dtype=bool)
    if country:
        selected = np.asarray([str(row.get("country", "")).casefold() == country.casefold() for row in meta])
    lat = features[selected, :, 6] * 90.0
    lon = features[selected, :, 7] * 180.0
    mask = valid[selected]
    points = {
        (round(float(a), 5), round(float(b), 5))
        for a, b, ok in zip(lat.reshape(-1), lon.reshape(-1), mask.reshape(-1))
        if bool(ok) and np.isfinite(a) and np.isfinite(b)
    }
    return sorted(points)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--country", default="Ethiopia")
    parser.add_argument(
        "--offset-degrees",
        type=float,
        default=0.05,
        help="N/S/E/W sampling offset for coarse local relief (default ~5 km latitude)",
    )
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing context artifact: {args.output}")
    bundle = np.load(args.candidate_data)
    points = _candidate_points(bundle, args.country)
    if not points:
        raise RuntimeError("no candidate points selected")

    delta = float(args.offset_degrees)
    expanded: list[tuple[float, float]] = []
    for lat, lon in points:
        lon_delta = delta / max(0.25, math.cos(math.radians(lat)))
        expanded.extend([
            (lat, lon),
            (lat + delta, lon),
            (lat - delta, lon),
            (lat, lon + lon_delta),
            (lat, lon - lon_delta),
        ])
    elevation = _fetch(expanded).reshape(len(points), 5)
    center = elevation[:, 0]
    local = elevation
    mean = np.nanmean(local, axis=1)
    std = np.nanstd(local, axis=1)
    relief = np.nanmax(local, axis=1) - np.nanmin(local, axis=1)
    # N-S / E-W coarse gradients in metres per kilometre.  The N/S points are
    # separated by ~2*delta*111.32 km; E/W were cosine-adjusted similarly.
    span_km = max(1e-3, 2.0 * delta * 111.32)
    grad_ns = (elevation[:, 1] - elevation[:, 2]) / span_km
    grad_ew = (elevation[:, 3] - elevation[:, 4]) / span_km
    missing = (~np.isfinite(local).all(axis=1)).astype(np.float32)
    features = np.column_stack([center, mean, std, relief, grad_ns, grad_ew, missing]).astype(np.float32)
    names = np.asarray([
        "elevation_m",
        "elevation_local_mean_m",
        "elevation_local_std_m",
        "elevation_local_relief_m",
        "elevation_gradient_ns_m_per_km",
        "elevation_gradient_ew_m_per_km",
        "elevation_missing",
    ])
    provenance = {
        "schema": "geo-context/elevation-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "country_filter": args.country,
        "candidate_data": str(args.candidate_data),
        "source": "Copernicus DEM 2021 GLO-90 via Open-Meteo Elevation API",
        "source_resolution_m": 90,
        "endpoint": ENDPOINT,
        "offset_degrees": delta,
        "points": len(points),
        "production_source": "https://registry.opendata.aws/copernicus-dem/",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        points=np.asarray(points, dtype=np.float32),
        features=features,
        feature_names=names,
        provenance=np.asarray(json.dumps(provenance, sort_keys=True)),
    )
    print(json.dumps({**provenance, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
