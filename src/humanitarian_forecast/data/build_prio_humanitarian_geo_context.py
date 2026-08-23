#!/usr/bin/env python3
"""Materialize coarse static humanitarian context for PRIO-GRID cells.

This builder intentionally aggregates population and accessibility at broad
0.5-degree / tens-of-kilometres scales. It does not construct routes or expose
turn-by-turn infrastructure. Outputs are designed for coarse humanitarian-risk,
displacement, and access-disruption models.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def gid_to_center(gid: int) -> tuple[float, float]:
    if gid < 1 or gid > 360 * 720:
        raise ValueError(f"invalid PRIO-GRID gid: {gid}")
    zero = gid - 1
    row = zero // 720 + 1
    col = zero % 720 + 1
    lat = -90.0 + (row - 0.5) * 0.5
    lon = -180.0 + (col - 0.5) * 0.5
    return lat, lon


def risk_gids(path: Path) -> list[int]:
    bundle = np.load(path, allow_pickle=True)
    meta = [json.loads(str(v)) for v in bundle["meta"]]
    gids: set[int] = set()
    for item in meta:
        entity = str(item.get("entity", ""))
        if entity.startswith("prio:"):
            try:
                gids.add(int(entity.split(":", 1)[1]))
            except ValueError:
                continue
    if not gids:
        raise ValueError(f"no PRIO-GRID entities found in {path}")
    return sorted(gids)


def km_to_lat(km: float) -> float:
    return km / 111.32


def km_to_lon(km: float, lat: float) -> float:
    return km / (111.32 * max(0.20, math.cos(math.radians(lat))))


def population_features(gids: list[int], raster: Path) -> tuple[np.ndarray, list[str], dict[str, object]]:
    import rasterio
    from rasterio.windows import from_bounds

    def read_square(ds, lat: float, lon: float, radius_km: float) -> np.ndarray:
        dy = km_to_lat(radius_km)
        dx = km_to_lon(radius_km, lat)
        left, bottom, right, top = lon - dx, lat - dy, lon + dx, lat + dy
        if right < ds.bounds.left or left > ds.bounds.right or top < ds.bounds.bottom or bottom > ds.bounds.top:
            return np.empty(0, dtype=np.float32)
        left, right = max(left, ds.bounds.left), min(right, ds.bounds.right)
        bottom, top = max(bottom, ds.bounds.bottom), min(top, ds.bounds.top)
        if left >= right or bottom >= top:
            return np.empty(0, dtype=np.float32)
        window = from_bounds(left, bottom, right, top, ds.transform)
        data = ds.read(1, window=window, masked=True)
        values = data.compressed() if np.ma.isMaskedArray(data) else np.asarray(data).reshape(-1)
        values = np.asarray(values, dtype=np.float32)
        values = values[np.isfinite(values) & (values >= 0)]
        return values

    out = np.zeros((len(gids), 10), dtype=np.float32)
    with rasterio.open(raster) as ds:
        if ds.crs is None or str(ds.crs).upper() not in {"EPSG:4326", "OGC:CRS84"}:
            raise ValueError(f"expected WGS84 population raster, got {ds.crs}")
        for i, gid in enumerate(gids):
            lat, lon = gid_to_center(gid)
            samples = {r: read_square(ds, lat, lon, r) for r in (5.0, 15.0, 30.0)}
            if any(len(v) == 0 for v in samples.values()):
                out[i, -1] = 1.0
            a, b, c = samples[5.0], samples[15.0], samples[30.0]
            if len(a):
                out[i, 0] = math.log1p(float(a.sum()))
                out[i, 1] = math.log1p(float(a.max()))
            if len(b):
                out[i, 2] = math.log1p(float(b.sum()))
                out[i, 3] = math.log1p(float(b.mean()))
                out[i, 4] = math.log1p(float(np.count_nonzero(b > 0)))
            if len(c):
                out[i, 5] = math.log1p(float(c.sum()))
                out[i, 6] = math.log1p(float(c.mean()))
                out[i, 7] = math.log1p(float(c.max()))
                out[i, 8] = math.log1p(float(np.count_nonzero(c > 0)))
            if i and i % 50 == 0:
                print(f"population context {i}/{len(gids)}", flush=True)
        is_1km = "1km" in raster.name.casefold()
        provenance = {
            "source": (
                "WorldPop Global 2015-2030 R2025A, Ethiopia 2026 constrained 1km"
                if is_1km else
                "WorldPop Global 2015-2030 R2025A, Ethiopia 2026 constrained 100m"
            ),
            "doi": "10.5258/SOTON/WP00845" if is_1km else "10.5258/SOTON/WP00839",
            "license": "CC BY 4.0",
            "raster": str(raster),
            "raster_crs": str(ds.crs),
            "raster_bounds": [float(v) for v in ds.bounds],
            "raster_resolution": [float(v) for v in ds.res],
        }
    names = [
        "population_log_sum_10km_square",
        "population_log_max_pixel_10km",
        "population_log_sum_30km_square",
        "population_log_mean_pixel_30km",
        "population_log_populated_pixels_30km",
        "population_log_sum_60km_square",
        "population_log_mean_pixel_60km",
        "population_log_max_pixel_60km",
        "population_log_populated_pixels_60km",
        "population_context_missing",
    ]
    return out, names, provenance


MAJOR = {"motorway", "trunk", "primary", "secondary"}
LOCAL = {"tertiary", "residential", "unclassified", "service", "living_street"}
TRACK = {"track", "path", "footway", "bridleway"}
SETTLEMENT = {"city", "town", "village", "hamlet", "suburb", "neighbourhood"}


def _ensure_osm_extracts(pbf: Path, cache_dir: Path) -> tuple[Path, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    roads_pbf = cache_dir / "roads.osm.pbf"
    places_pbf = cache_dir / "places.osm.pbf"
    roads_json = cache_dir / "roads.geojsonseq"
    places_json = cache_dir / "places.geojsonseq"
    if not roads_pbf.exists():
        subprocess.run(["osmium", "tags-filter", str(pbf), "w/highway", "-o", str(roads_pbf)], check=True)
    if not places_pbf.exists():
        subprocess.run(["osmium", "tags-filter", str(pbf), "n/place", "-o", str(places_pbf)], check=True)
    if not roads_json.exists():
        subprocess.run(["osmium", "export", str(roads_pbf), "-f", "geojsonseq", "--geometry-types", "linestring", "-o", str(roads_json)], check=True)
    if not places_json.exists():
        subprocess.run(["osmium", "export", str(places_pbf), "-f", "geojsonseq", "--geometry-types", "point", "-o", str(places_json)], check=True)
    return roads_json, places_json


def accessibility_features(
    gids: list[int], pbf: Path, cache_dir: Path, snapshot: str
) -> tuple[np.ndarray, list[str], dict[str, object]]:
    from pyproj import Transformer
    from shapely.geometry import Point, shape
    from shapely.ops import transform as transform_geometry
    from shapely.strtree import STRtree

    roads_json, places_json = _ensure_osm_extracts(pbf, cache_dir)
    to_metric = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

    def load_geojsonseq(path: Path, tag: str, allowed: set[str]):
        geometries = []
        categories: list[str] = []
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line.lstrip("\x1e"))
                props = row.get("properties") or {}
                value = props.get(tag)
                if not value or value not in allowed:
                    continue
                geometry = row.get("geometry")
                if not geometry:
                    continue
                try:
                    g = shape(geometry)
                    if g.is_empty:
                        continue
                    g = transform_geometry(to_metric.transform, g)
                except Exception:
                    continue
                geometries.append(g)
                categories.append(str(value))
        return geometries, categories

    roads, road_class = load_geojsonseq(roads_json, "highway", MAJOR | LOCAL | TRACK)
    places, place_class = load_geojsonseq(places_json, "place", SETTLEMENT)
    road_tree = STRtree(roads) if roads else None
    place_tree = STRtree(places) if places else None
    road_major = np.asarray([v in MAJOR for v in road_class])
    road_local = np.asarray([v in LOCAL for v in road_class])
    road_track = np.asarray([v in TRACK for v in road_class])
    place_city = np.asarray([v in {"city", "town"} for v in place_class])

    out = np.zeros((len(gids), 11), dtype=np.float32)
    for i, gid in enumerate(gids):
        lat, lon = gid_to_center(gid)
        x, y = to_metric.transform(lon, lat)
        point = Point(x, y)
        for r_idx, radius_km in enumerate((25.0, 50.0)):
            buffer = point.buffer(radius_km * 1000.0)
            if road_tree is not None:
                idx = np.asarray(road_tree.query(buffer, predicate="intersects"), dtype=np.int64)
                if len(idx):
                    clipped = np.asarray([roads[j].intersection(buffer).length / 1000.0 for j in idx], dtype=np.float64)
                    offset = 0 if radius_km == 25.0 else 4
                    out[i, offset + 0] = math.log1p(float(clipped.sum()))
                    out[i, offset + 1] = math.log1p(float(clipped[road_major[idx]].sum()))
                    out[i, offset + 2] = math.log1p(float(clipped[road_local[idx]].sum()))
                    out[i, offset + 3] = math.log1p(float(clipped[road_track[idx]].sum()))
            if place_tree is not None:
                pidx = np.asarray(place_tree.query(buffer, predicate="intersects"), dtype=np.int64)
                if radius_km == 25.0:
                    out[i, 8] = math.log1p(float(len(pidx)))
                    out[i, 9] = math.log1p(float(place_city[pidx].sum())) if len(pidx) else 0.0
        if not roads or not places:
            out[i, 10] = 1.0
        if i and i % 50 == 0:
            print(f"accessibility context {i}/{len(gids)}", flush=True)

    names = [
        "access_log_road_km_25km",
        "access_log_major_road_km_25km",
        "access_log_local_road_km_25km",
        "access_log_track_path_km_25km",
        "access_log_road_km_50km",
        "access_log_major_road_km_50km",
        "access_log_local_road_km_50km",
        "access_log_track_path_km_50km",
        "access_log_settlements_25km",
        "access_log_city_town_25km",
        "accessibility_context_missing",
    ]
    provenance = {
        "source": "OpenStreetMap via Geofabrik",
        "license": "ODbL 1.0",
        "pbf": str(pbf),
        "snapshot": snapshot,
        "roads_loaded": len(roads),
        "settlements_loaded": len(places),
        "projection_for_aggregation": "EPSG:3857",
        "interpretation": "coarse density/accessibility context only; no route reconstruction",
    }
    return out, names, provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--risk-data", type=Path, required=True)
    parser.add_argument("--kind", choices=("population", "accessibility"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raster", type=Path)
    parser.add_argument("--pbf", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--snapshot", default="")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output}")

    gids = risk_gids(args.risk_data)
    centers = np.asarray([gid_to_center(gid) for gid in gids], dtype=np.float32)
    if args.kind == "population":
        if args.raster is None or not args.raster.exists():
            raise FileNotFoundError(args.raster)
        features, names, source = population_features(gids, args.raster)
    else:
        if args.pbf is None or not args.pbf.exists():
            raise FileNotFoundError(args.pbf)
        if args.cache_dir is None:
            raise ValueError("--cache-dir is required for accessibility")
        features, names, source = accessibility_features(gids, args.pbf, args.cache_dir, args.snapshot)

    provenance = {
        "schema": f"humanitarian-prio-context/{args.kind}-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "risk_data": str(args.risk_data),
        "grid_scale": "PRIO-GRID 0.5 degree cell centers; features aggregated over tens of kilometres",
        "purpose": "coarse humanitarian risk / displacement / access-disruption modelling",
        **source,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        gids=np.asarray(gids, dtype=np.int64),
        centers=centers,
        features=features.astype(np.float32),
        feature_names=np.asarray(names),
        provenance=np.asarray(json.dumps(provenance, sort_keys=True)),
    )
    print(json.dumps({**provenance, "cells": len(gids), "features": len(names), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
