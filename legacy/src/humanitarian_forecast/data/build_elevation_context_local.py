#!/usr/bin/env python3
"""Build coarse terrain context from locally cached Copernicus GLO-90 tiles."""
from __future__ import annotations

import argparse
import json
import math
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio


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


def tile_path(tile_dir: Path, lat: float, lon: float) -> Path:
    return tile_dir / f"{math.floor(lat)}_{math.floor(lon)}.tif"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate-data", type=Path, required=True)
    p.add_argument("--tile-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--country", default="Ethiopia")
    p.add_argument("--offset-degrees", type=float, default=0.05)
    p.add_argument("--allow-missing", action="store_true")
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output}")

    points = candidate_points(args.candidate_data, args.country)
    requested_tiles = {
        tile_path(args.tile_dir, lat, lon)
        for lat, lon in points
    }
    missing_tiles = sorted(path for path in requested_tiles if not path.exists())
    if missing_tiles and not args.allow_missing:
        preview = ", ".join(path.name for path in missing_tiles[:8])
        raise FileNotFoundError(
            f"{len(missing_tiles)} required Copernicus tiles are missing ({preview}); "
            "finish the manifest download or pass --allow-missing"
        )

    delta = float(args.offset_degrees)
    features = np.zeros((len(points), 9), dtype=np.float32)
    dataset_cache: dict[Path, rasterio.io.DatasetReader] = {}
    with ExitStack() as stack:
        def sample(lat: float, lon: float) -> float:
            path = tile_path(args.tile_dir, lat, lon)
            if not path.exists():
                return float("nan")
            ds = dataset_cache.get(path)
            if ds is None:
                ds = stack.enter_context(rasterio.open(path))
                dataset_cache[path] = ds
            value = float(next(ds.sample([(lon, lat)]))[0])
            if ds.nodata is not None and value == float(ds.nodata):
                return float("nan")
            return value if math.isfinite(value) else float("nan")

        for i, (lat, lon) in enumerate(points):
            lon_delta = delta / max(0.25, math.cos(math.radians(lat)))
            values = np.asarray([
                sample(lat, lon),
                sample(lat + delta, lon),
                sample(lat - delta, lon),
                sample(lat, lon + lon_delta),
                sample(lat, lon - lon_delta),
            ], dtype=np.float32)
            center = values[0]
            finite = values[np.isfinite(values)]
            if len(finite) == 0:
                features[i, -1] = 1.0
                continue
            mean = float(np.mean(finite))
            std = float(np.std(finite))
            relief = float(np.max(finite) - np.min(finite))
            span_km = max(1e-3, 2.0 * delta * 111.32)
            grad_ns = (values[1] - values[2]) / span_km if np.isfinite(values[1:3]).all() else 0.0
            grad_ew = (values[3] - values[4]) / span_km if np.isfinite(values[3:5]).all() else 0.0
            gradient_mag = math.hypot(float(grad_ns), float(grad_ew))
            # Compact ruggedness proxy: local elevation standard deviation plus
            # gradient magnitude.  This is broad terrain context, not a
            # traversability/route model.
            ruggedness = std + 5.0 * gradient_mag
            features[i] = [
                center if math.isfinite(float(center)) else mean,
                mean,
                std,
                relief,
                float(grad_ns),
                float(grad_ew),
                gradient_mag,
                ruggedness,
                float(len(finite) < len(values)),
            ]
            if i and i % 250 == 0:
                print(f"terrain context {i}/{len(points)}", flush=True)

    names = np.asarray([
        "elevation_m",
        "elevation_local_mean_m",
        "elevation_local_std_m",
        "elevation_local_relief_m",
        "elevation_gradient_ns_m_per_km",
        "elevation_gradient_ew_m_per_km",
        "elevation_gradient_magnitude_m_per_km",
        "terrain_ruggedness_proxy",
        "elevation_context_partial",
    ])
    provenance = {
        "schema": "geo-context/copernicus-glo90-local-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate_data": str(args.candidate_data),
        "country_filter": args.country,
        "tile_dir": str(args.tile_dir),
        "source": "Copernicus DEM 2021 GLO-90 public AWS COGs",
        "source_resolution_m": 90,
        "offset_degrees": delta,
        "requested_tiles": len(requested_tiles),
        "missing_tiles": len(missing_tiles),
        "broad_area_use": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        points=np.asarray(points, dtype=np.float32),
        features=features,
        feature_names=names,
        provenance=np.asarray(json.dumps(provenance, sort_keys=True)),
    )
    print(json.dumps({**provenance, "points": len(points), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
