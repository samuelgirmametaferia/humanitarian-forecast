#!/usr/bin/env python3
"""Fetch a terrarium-tile elevation grid and derive per-cell ruggedness.

Downloads Mapzen/AWS terrain tiles (terrarium encoding: elevation =
256*R + G + B/256 metres), stitches them into a single lat/lon grid for a
named region, and computes local ruggedness (standard deviation in a
~5 km window). The grid feeds candidate-level terrain features in the
candidate-rank dataset builder.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import urllib.request
from pathlib import Path

import numpy as np

REGIONS = {
    "ethiopia": {"lat": (3.0, 15.5), "lon": (32.8, 48.6)},
}
TILE_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"


def tile_y(lat: float, z: int) -> int:
    lat_rad = math.radians(lat)
    y = (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0
    return int(y * (2 ** z))


def fetch(z: int, x: int, y: int, retries: int = 3) -> np.ndarray | None:
    url = TILE_URL.format(z=z, x=x, y=y)
    for _ in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                data = response.read()
        except Exception:
            continue
        png = np.frombuffer(data, dtype=np.uint8)
        try:
            import PIL.Image

            image = PIL.Image.open(io.BytesIO(png.tobytes()))
            rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
        except ImportError:
            # Fallback: decode PNG via matplotlib if PIL is unavailable.
            import matplotlib.image

            rgb = matplotlib.image.imread(io.BytesIO(data)).astype(np.float32)[:, :, :3] * 255.0
        return 256.0 * rgb[:, :, 0] + rgb[:, :, 1] + rgb[:, :, 2] / 256.0 - 32768.0
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", choices=sorted(REGIONS), default="ethiopia")
    parser.add_argument("--zoom", type=int, default=8)
    parser.add_argument("--window", type=int, default=5, help="Ruggedness window in pixels.")
    parser.add_argument("--output", type=Path, default=Path("data/geo/elevation_ethiopia_z8.npz"))
    args = parser.parse_args()

    box = REGIONS[args.region]
    z = args.zoom
    scale = 2 ** z
    x0, x1 = int((box["lon"][0] + 180) / 360 * scale), int((box["lon"][1] + 180) / 360 * scale)
    y0, y1 = tile_y(box["lat"][1], z), tile_y(box["lat"][0], z)
    print(json.dumps({"tiles_x": [x0, x1], "tiles_y": [y0, y1], "count": (x1 - x0 + 1) * (y1 - y0 + 1)}))

    rows = []
    for ty in range(y0, y1 + 1):
        row_tiles = []
        for tx in range(x0, x1 + 1):
            tile = fetch(z, tx, ty)
            if tile is None:
                tile = np.zeros((256, 256), dtype=np.float32)
            row_tiles.append(tile)
        rows.append(np.concatenate(row_tiles, axis=1))
    elevation = np.concatenate(rows, axis=0).astype(np.float32)

    # Local ruggedness: rolling standard deviation over a window.
    padded = np.pad(elevation, args.window // 2, mode="edge")
    stacks = np.stack([padded[i:i + elevation.shape[0], j:j + elevation.shape[1]]
                       for i in range(args.window) for j in range(args.window)])
    rugged = stacks.std(axis=0).astype(np.float32)

    nlat, nlon = elevation.shape
    # Flip rows so latitude ascends (row 0 = south), then clip implausible
    # void/bathymetry outliers; land candidates never need values beyond
    # the physical East-Africa range.
    elevation = np.clip(elevation[::-1], -500.0, 5000.0).astype(np.float32)
    rugged = rugged[::-1].astype(np.float32)
    lats = np.linspace(box["lat"][0], box["lat"][1], nlat, dtype=np.float64)
    lons = np.linspace(box["lon"][0], box["lon"][1], nlon, dtype=np.float64)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, lat=lats, lon=lons, elevation=elevation, rugged=rugged,
                        region=args.region, zoom=z)
    print(json.dumps({"output": str(args.output), "shape": list(elevation.shape),
                      "elevation_range_m": [float(elevation.min()), float(elevation.max())],
                      "rugged_range_m": [float(rugged.min()), float(rugged.max())]}))


if __name__ == "__main__":
    main()
