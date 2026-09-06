#!/usr/bin/env python3
"""Build cutoff-safe candidates with spillover + ReliefWeb coverage (v7).

Extends the v6 recipe toward the 20 km scale:

* ``--spillover N`` recent cross-conflict UCDP event candidates (any conflict
  in the country, most recent first, within 300 km of the anchor);
* ``--mentions N`` recent ReliefWeb mention-site candidates (0.25 degree
  cells with the freshest mentions) — report geography moves before event
  databases do;
* per-candidate self-exciting (Hawkes-style) kernel features at two scales
  (30 day / 50 km and 365 day / 100 km) computed from recent any-conflict
  events — continuous versions of the ring-count features;
* ``target_precision`` stores each target's UCDP ``where_prec`` so trainers
  can weight exactly-geolocated rows more heavily (never a model feature).

The label is the nearest candidate under the project's planar kilometre
approximation.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import io
import json
import math
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

EARTH_KM = 111.32
SPILLOVER_RADIUS_KM = 300.0
SPILLOVER_SCAN_EVENTS = 900  # recent events scanned per row before dedup
HAWKES_WINDOW_DAYS = 365
HAWKES_SHORT = (30.0, 50.0)   # time decay (days), distance decay (km)
HAWKES_LONG = (365.0, 100.0)
FAR_AWAY_DAYS = 3650
MENTION_CELL = 0.25


def pkey(lat, lon):
    return round(float(lat), 5), round(float(lon), 5)


def offset(point, anchor):
    north = (point[0] - anchor[0]) * EARTH_KM
    east = (point[1] - anchor[1]) * EARTH_KM * math.cos(math.radians(anchor[0]))
    return east, north


def load_ucdp(path: Path):
    conflict_sites: dict[tuple, list[int]] = defaultdict(list)
    country_sites: dict[str, dict[tuple, list[int]]] = defaultdict(lambda: defaultdict(list))
    target_prec: dict[tuple, int] = {}
    with zipfile.ZipFile(path) as archive:
        name = next(value for value in archive.namelist() if value.endswith(".csv"))
        with archive.open(name) as raw:
            for row in csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig")):
                try:
                    if int(row["where_prec"]) > 4 or int(row["date_prec"]) > 3:
                        continue
                    day = int(np.datetime64(row["date_start"][:10], "D").astype(int))
                    key = pkey(float(row["latitude"]), float(row["longitude"]))
                except (ValueError, KeyError):
                    continue
                conflict_sites[(str(row["conflict_new_id"]), key)].append(day)
                country_sites[str(row["country"])][key].append(day)
                target_prec.setdefault((row["date_start"][:10], key), int(row["where_prec"]))
    for days in conflict_sites.values():
        days.sort()
    for sites in country_sites.values():
        for days in sites.values():
            days.sort()
    flat: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for country, sites in country_sites.items():
        events = [(day, key) for key, days in sites.items() for day in days]
        events.sort()
        if events:
            keys = np.empty(len(events), dtype=object)
            keys[:] = [e[1] for e in events]
            coords = np.asarray([e[1] for e in events], dtype=np.float64)
            flat[country] = (
                np.asarray([e[0] for e in events], dtype=np.int64),
                keys,
                coords,
            )
    return conflict_sites, country_sites, flat, target_prec


def load_mentions(path: Path):
    """0.25 degree mention cells -> (sorted days, representative point)."""
    cells: dict[tuple, list[int]] = defaultdict(list)
    points: dict[tuple, list[tuple]] = defaultdict(list)
    z = np.load(path, allow_pickle=True)
    days = np.asarray(z["day"]).astype(np.int64)
    latlon = np.asarray(z["latlon"], dtype=np.float64)
    for (lat, lon), day in zip(latlon, days):
        key = (round(float(lat) / MENTION_CELL), round(float(lon) / MENTION_CELL))
        cells[key].append(int(day))
        points[key].append((float(lat), float(lon)))
    for key in cells:
        cells[key].sort()
    site = {key: (float(np.mean([p[0] for p in ps])), float(np.mean([p[1] for p in ps])))
            for key, ps in points.items()}
    return cells, site


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidates", type=int, default=32, help="Frequency-site candidates.")
    parser.add_argument("--spillover", type=int, default=48, help="Recent cross-conflict event candidates.")
    parser.add_argument("--mentions", type=int, default=16, help="Recent ReliefWeb mention-site candidates.")
    parser.add_argument("--reliefweb-mentions", type=Path, help="lat/lon-keyed ReliefWeb mention npz.")
    parser.add_argument("--ucdp-events", type=Path, required=True)
    parser.add_argument("--elevation", type=Path, help="Elevation grid npz from build_elevation_grid.")
    a = parser.parse_args()

    d = np.load(a.data)
    x = d["x"]
    y = d["y"]
    rows = [json.loads(str(v)) for v in d["meta"]]
    n = len(rows)
    k_freq = a.candidates
    k_spill = a.spillover
    k_mention = a.mentions
    k = k_freq + k_spill + k_mention

    mention_cells, mention_site = load_mentions(a.reliefweb_mentions) if a.reliefweb_mentions and a.reliefweb_mentions.exists() else ({}, {})
    # Flat mention events (day, key) for recency ordering, Ethiopia corpus only.
    mention_events = sorted((day, key) for key, days in mention_cells.items() for day in days)
    mention_event_days = np.asarray([e[0] for e in mention_events], dtype=np.int64) if mention_events else np.zeros(0, np.int64)
    mention_event_keys = [e[1] for e in mention_events]

    mention_days: dict[tuple, list[int]] = defaultdict(list)
    for key, days in mention_cells.items():
        mention_days[(round(mention_site[key][0] * 4), round(mention_site[key][1] * 4))] = days
    for days_at in mention_days.values():
        days_at.sort()

    elevation = rugged = None
    grid_lat = grid_lon = None
    if a.elevation and a.elevation.exists():
        g = np.load(a.elevation)
        grid_lat = np.asarray(g["lat"], dtype=np.float64)
        grid_lon = np.asarray(g["lon"], dtype=np.float64)
        elevation = np.asarray(g["elevation"], dtype=np.float32)
        rugged = np.asarray(g["rugged"], dtype=np.float32)

    def terrain(point) -> tuple[float, float]:
        if elevation is None:
            return 0.0, 0.0
        i = int(np.clip(np.searchsorted(grid_lat, point[0]), 0, len(grid_lat) - 1))
        j = int(np.clip(np.searchsorted(grid_lon, point[1]), 0, len(grid_lon) - 1))
        if abs(grid_lat[i] - point[0]) > 0.2 or abs(grid_lon[j] - point[1]) > 0.2:
            return 0.0, 0.0
        return float(elevation[i, j]) / 1000.0, float(rugged[i, j]) / 1000.0

    conflict_sites, country_sites, country_flat, target_prec = load_ucdp(a.ucdp_events)

    # 11 base + 4 same-conflict activity + 8 rings + 4 raw same-conflict
    # windows + 5 spillover block + 3 mentions + 2 terrain + 2 hawkes + 1 mention-flag.
    candidate_dim = 37 + 3
    features = np.zeros((n, k, candidate_dim), np.float32)
    coordinates = np.zeros((n, k, 2), np.float32)
    valid = np.zeros((n, k), bool)
    labels = np.zeros(n, np.int64)
    oracle = np.zeros(n, np.float32)
    target_precision = np.zeros(n, np.int64)

    locations = defaultdict(dict)
    transitions = defaultdict(lambda: defaultdict(Counter))
    cursor = 0
    while cursor < n:
        date = rows[cursor]["target_date"]
        stop = cursor
        while stop < n and rows[stop]["target_date"] == date:
            stop += 1
        day = int(np.datetime64(date, "D").astype(int))
        for index in range(cursor, stop):
            row = rows[index]
            conflict = str(row["conflict_id"])
            country = str(row["country"])
            anchor = np.array([row["anchor_lat"], row["anchor_lon"]], float)
            source = pkey(*anchor)
            cutoff = day - int(row["gap_days"])
            if index % 20000 == 0:
                print(json.dumps({"progress_rows": index, "total": n}), flush=True)
            history_valid = x[index, :, 0] > 0.5
            history_positions = x[index, history_valid, 1:3]
            history_days = np.maximum(0.0, np.expm1(x[index, history_valid, 3] * 6) - float(row["gap_days"]))
            history_fatalities = np.maximum(0.0, np.expm1(x[index, history_valid, 4] * 6))

            ranked = sorted(locations[conflict].items(), key=lambda item: (-item[1]["count"], -item[1]["last"], item[0]))
            chosen: list[tuple[tuple, np.ndarray, bool]] = [(source, anchor, False)]
            chosen.extend((key, value["point"], False) for key, value in ranked if key != source)
            chosen = chosen[:k_freq]
            chosen_keys = {key for key, _, _ in chosen}

            # Spillover candidates: most recent cross-conflict events near the anchor.
            if k_spill and country in country_flat:
                days_arr, keys_arr, coords_arr = country_flat[country]
                hi = int(np.searchsorted(days_arr, cutoff, "right"))
                lo = max(0, hi - SPILLOVER_SCAN_EVENTS)
                seen: dict[tuple, int] = {}
                for pos in range(hi - 1, lo - 1, -1):
                    key = keys_arr[pos]
                    if key in seen:
                        continue
                    seen[key] = int(days_arr[pos])
                    if len(seen) >= 4 * k_spill:
                        break
                for key, last_day in sorted(seen.items(), key=lambda item: -item[1]):
                    if len(chosen) >= k_freq + k_spill:
                        break
                    if key in chosen_keys:
                        continue
                    point = np.array([key[0], key[1]], float)
                    east, north = offset(point, anchor)
                    if math.hypot(east, north) > SPILLOVER_RADIUS_KM:
                        continue
                    chosen.append((key, point, True))
                    chosen_keys.add(key)

            # ReliefWeb mention-site candidates: freshest mention cells.
            # The mention corpus is Ethiopia-only, so gate on the anchor country
            # to keep Ethiopian sites out of other countries' candidate pools.
            if k_mention and mention_events and country == "Ethiopia":
                hi = int(np.searchsorted(mention_event_days, cutoff, "right"))
                seen_cells: dict[tuple, int] = {}
                for pos in range(hi - 1, -1, -1):
                    key = mention_event_keys[pos]
                    if key not in seen_cells:
                        seen_cells[key] = int(mention_event_days[pos])
                    if len(seen_cells) >= 4 * k_mention:
                        break
                for key, last_day in sorted(seen_cells.items(), key=lambda item: -item[1]):
                    if len(chosen) >= k:
                        break
                    if key in chosen_keys:
                        continue
                    point = np.array(mention_site[key], float)
                    chosen.append((key, point, True))
                    chosen_keys.add(key)

            # Recent any-conflict events for the Hawkes kernel features.
            recent_days = recent_lat = recent_lon = None
            if country in country_flat:
                days_arr, _, coords_arr = country_flat[country]
                hi = int(np.searchsorted(days_arr, cutoff, "right"))
                lo = int(np.searchsorted(days_arr, cutoff - HAWKES_WINDOW_DAYS, "left"))
                if hi - lo > 1500:
                    lo = hi - 1500
                if hi > lo:
                    recent_days = days_arr[lo:hi].astype(np.float64)
                    recent_lat = coords_arr[lo:hi, 0]
                    recent_lon = coords_arr[lo:hi, 1]

            sites_any = country_sites.get(country, {})
            for j, (key, point, is_spillover) in enumerate(chosen):
                east, north = offset(point, anchor)
                site = locations[conflict].get(key)
                count = site["count"] if site else 0
                age = max(1, cutoff - site["last"]) if site else FAR_AWAY_DAYS
                transition = transitions[conflict][source][key]
                values = [
                    1, east / 1000, north / 1000, math.hypot(east, north) / 1000,
                    math.log1p(count) / 6, math.log1p(age) / 6,
                    point[0] / 90, point[1] / 180, float(key == source),
                    math.log1p(transition) / 5, math.log1p(row["gap_days"]) / 5,
                ]
                # Same-conflict site activity from the dataset-internal history.
                site_days = site["days"] if site else []
                right = bisect.bisect_right(site_days, cutoff)
                values.extend(math.log1p(right - bisect.bisect_left(site_days, cutoff - window)) / 4 for window in (7, 30, 90, 365))
                history_distance = np.linalg.norm(history_positions - np.asarray([east / 1000, north / 1000]), axis=1) * 1000
                rings = ((25, 7), (50, 30), (100, 90), (250, 90))
                ring_masks = [(history_distance <= radius) & (history_days <= window) for radius, window in rings]
                values.extend(math.log1p(int(mask.sum())) / 4 for mask in ring_masks)
                values.extend(math.log1p(float(history_fatalities[mask].sum())) / 6 for mask in ring_masks)
                # Raw UCDP same-conflict windows.
                raw_days = conflict_sites.get((conflict, key), [])
                raw_right = bisect.bisect_right(raw_days, cutoff)
                values.extend(math.log1p(raw_right - bisect.bisect_left(raw_days, cutoff - window)) / 4 for window in (7, 30, 90, 365))
                # Spillover block: any-conflict site activity.
                any_days = sites_any.get(key, [])
                any_right = bisect.bisect_right(any_days, cutoff)
                values.append(float(is_spillover))
                values.extend(math.log1p(any_right - bisect.bisect_left(any_days, cutoff - window)) / 4 for window in (7, 30, 90))
                values.append(math.log1p(max(1, cutoff - any_days[any_right - 1])) / 8 if any_right else 1.0)
                if mention_days:
                    days_at = mention_days.get((round(float(point[0]) * 4), round(float(point[1]) * 4)), [])
                    values.extend(math.log1p(bisect.bisect_right(days_at, cutoff) - bisect.bisect_left(days_at, cutoff - window)) / 4 for window in (7, 30, 90))
                if elevation is not None:
                    elev_m, rugged_m = terrain(point)
                    values.extend([elev_m, rugged_m])
                # Hawkes-style kernels over recent any-conflict events.
                if recent_days is not None:
                    dlat = recent_lat - point[0]
                    dlon = (recent_lon - point[1]) * math.cos(math.radians(point[0]))
                    dist_km = np.hypot(dlat, dlon) * EARTH_KM
                    dt_days = cutoff - recent_days
                    h_short = float(np.sum(np.exp(-dist_km / HAWKES_SHORT[1] - dt_days / HAWKES_SHORT[0])))
                    h_long = float(np.sum(np.exp(-dist_km / HAWKES_LONG[1] - dt_days / HAWKES_LONG[0])))
                    values.extend([math.log1p(h_short) / 4, math.log1p(h_long) / 4])
                else:
                    values.extend([0.0, 0.0])
                values.append(float(key in mention_site))
                features[index, j] = values
                coordinates[index, j] = [east / 1000, north / 1000]
                valid[index, j] = True
            distance = np.linalg.norm(coordinates[index, :len(chosen)] - y[index, None], axis=1) * 1000
            labels[index] = int(distance.argmin())
            oracle[index] = float(distance.min())
            target_precision[index] = target_prec.get(
                (row["target_date"], pkey(row["target_lat"], row["target_lon"])), 0)
        for index in range(cursor, stop):
            row = rows[index]
            conflict = str(row["conflict_id"])
            source = pkey(row["anchor_lat"], row["anchor_lon"])
            target = pkey(row["target_lat"], row["target_lon"])
            day = int(np.datetime64(row["target_date"], "D").astype(int))
            value = locations[conflict].get(target)
            if value is None:
                value = {"point": np.array([row["target_lat"], row["target_lon"]], float), "count": 0, "last": day, "days": []}
                locations[conflict][target] = value
            value["count"] += 1
            value["last"] = day
            value["days"].append(day)
            transitions[conflict][source][target] += 1
        cursor = stop

    a.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.output, x=x, candidate_features=features, candidate_coordinates=coordinates,
                        candidate_valid=valid, label=labels, y=y, meta=d["meta"],
                        target_precision=target_precision)
    te = int(.7 * n)
    ve = int(.85 * n)
    eth_val = [i for i in range(te, ve) if rows[i]["country"] == "Ethiopia"]
    eth_test = [i for i in range(ve, n) if rows[i]["country"] == "Ethiopia"]
    print(json.dumps({
        "samples": n, "shape": list(features.shape),
        "oracle_mean_km": {"train": float(oracle[:te].mean()), "validation": float(oracle[te:ve].mean()), "test": float(oracle[ve:].mean())},
        "ethiopia_oracle_within_20km": {
            "validation": float(np.mean(oracle[eth_val] <= 20)),
            "test": float(np.mean(oracle[eth_test] <= 20)),
        },
    }, indent=2))


if __name__ == "__main__":
    main()
