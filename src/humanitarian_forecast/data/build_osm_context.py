#!/usr/bin/env python3
"""Build coarse road/settlement accessibility context from a pinned OSM PBF.

The output is a candidate-point feature table.  It intentionally contains
aggregate accessibility/density signals, not turn-by-turn routes.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from pyproj import Transformer
from shapely.geometry import Point, shape
from shapely.ops import transform as transform_geometry
from shapely.strtree import STRtree

MAJOR = {"motorway", "trunk", "primary", "secondary"}
LOCAL = {"tertiary", "residential", "unclassified", "service", "living_street"}
TRACK = {"track", "path", "footway", "bridleway"}
SETTLEMENT = {"city", "town", "village", "hamlet", "suburb", "neighbourhood"}


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


def ensure_extracts(pbf: Path, cache_dir: Path) -> tuple[Path, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    roads_pbf = cache_dir / "roads.osm.pbf"
    places_pbf = cache_dir / "places.osm.pbf"
    roads_json = cache_dir / "roads.geojsonseq"
    places_json = cache_dir / "places.geojsonseq"
    if not roads_pbf.exists():
        subprocess.run(
            ["osmium", "tags-filter", str(pbf), "w/highway", "-o", str(roads_pbf)],
            check=True,
        )
    if not places_pbf.exists():
        subprocess.run(
            ["osmium", "tags-filter", str(pbf), "n/place", "-o", str(places_pbf)],
            check=True,
        )
    if not roads_json.exists():
        subprocess.run(
            ["osmium", "export", str(roads_pbf), "-f", "geojsonseq", "--geometry-types", "linestring", "-o", str(roads_json)],
            check=True,
        )
    if not places_json.exists():
        subprocess.run(
            ["osmium", "export", str(places_pbf), "-f", "geojsonseq", "--geometry-types", "point", "-o", str(places_json)],
            check=True,
        )
    return roads_json, places_json


def load_geojsonseq(path: Path, tag: str, allowed: set[str] | None, transformer: Transformer):
    geometries = []
    categories: list[str] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            props = row.get("properties") or {}
            value = props.get(tag)
            if not value or (allowed is not None and value not in allowed):
                continue
            geometry = row.get("geometry")
            if not geometry:
                continue
            try:
                g = shape(geometry)
                if g.is_empty:
                    continue
                g = transform_geometry(transformer.transform, g)
            except Exception:
                continue
            geometries.append(g)
            categories.append(str(value))
    return geometries, categories


def _indices(result) -> list[int]:
    # Shapely 2 STRtree returns integer indices for query/nearest.
    arr = np.asarray(result)
    if arr.ndim == 0:
        return [int(arr)]
    return [int(v) for v in arr.tolist()]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate-data", type=Path, required=True)
    p.add_argument("--pbf", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--country", default="Ethiopia")
    p.add_argument("--snapshot", required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output}")
    if not args.pbf.exists():
        raise FileNotFoundError(args.pbf)

    points = candidate_points(args.candidate_data, args.country)
    roads_json, places_json = ensure_extracts(args.pbf, args.cache_dir)
    to_metric = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    roads, road_class = load_geojsonseq(roads_json, "highway", MAJOR | LOCAL | TRACK, to_metric)
    places, place_class = load_geojsonseq(places_json, "place", SETTLEMENT, to_metric)
    road_tree = STRtree(roads) if roads else None
    place_tree = STRtree(places) if places else None
    road_major = np.asarray([v in MAJOR for v in road_class])
    road_local = np.asarray([v in LOCAL for v in road_class])
    road_track = np.asarray([v in TRACK for v in road_class])
    place_city = np.asarray([v in {"city", "town"} for v in place_class])

    features = np.zeros((len(points), 12), dtype=np.float32)
    for i, (lat, lon) in enumerate(points):
        x, y = to_metric.transform(lon, lat)
        point = Point(x, y)
        if road_tree is not None:
            nearest_i = int(road_tree.nearest(point))
            features[i, 0] = math.log1p(point.distance(roads[nearest_i]) / 1000.0)
        else:
            features[i, 0] = math.log1p(1000.0)
            features[i, -1] = 1.0

        for r_i, radius_km in enumerate((10.0, 25.0, 50.0)):
            buffer = point.buffer(radius_km * 1000.0)
            if road_tree is not None:
                idx = np.asarray(road_tree.query(buffer, predicate="intersects"), dtype=np.int64)
                if len(idx):
                    clipped = [roads[j].intersection(buffer).length / 1000.0 for j in idx]
                    clipped = np.asarray(clipped, dtype=np.float64)
                    features[i, 1 + r_i] = math.log1p(float(clipped.sum()))
                    if radius_km == 25.0:
                        features[i, 4] = math.log1p(float(clipped[road_major[idx]].sum()))
                        features[i, 5] = math.log1p(float(clipped[road_local[idx]].sum()))
                        features[i, 6] = math.log1p(float(clipped[road_track[idx]].sum()))
            if place_tree is not None:
                pidx = np.asarray(place_tree.query(buffer, predicate="intersects"), dtype=np.int64)
                if radius_km == 10.0:
                    features[i, 7] = math.log1p(float(len(pidx)))
                elif radius_km == 25.0:
                    features[i, 8] = math.log1p(float(len(pidx)))
                    features[i, 10] = math.log1p(float(place_city[pidx].sum())) if len(pidx) else 0.0
                else:
                    features[i, 9] = math.log1p(float(len(pidx)))
        if i and i % 200 == 0:
            print(f"OSM context {i}/{len(points)}", flush=True)

    names = np.asarray([
        "log_nearest_road_km",
        "log_road_km_10km",
        "log_road_km_25km",
        "log_road_km_50km",
        "log_major_road_km_25km",
        "log_local_road_km_25km",
        "log_track_path_km_25km",
        "log_settlements_10km",
        "log_settlements_25km",
        "log_settlements_50km",
        "log_city_town_25km",
        "osm_context_missing",
    ])
    provenance = {
        "schema": "geo-context/osm-accessibility-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "country_filter": args.country,
        "candidate_data": str(args.candidate_data),
        "pbf": str(args.pbf),
        "snapshot": args.snapshot,
        "source": "OpenStreetMap via Geofabrik",
        "license": "ODbL 1.0",
        "roads_loaded": len(roads),
        "settlements_loaded": len(places),
        "projection_for_aggregation": "EPSG:3857",
        "interpretation": "coarse accessibility/density only; no route reconstruction",
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
