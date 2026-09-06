#!/usr/bin/env python3
"""Build cutoff-safe candidates with spillover coverage for fine-resolution ranking.

Extends the v5 recipe (top-frequency conflict sites + spatial ring features)
with the coverage and terrain signals needed for ~20 km scale ranking:

* ``--spillover N`` adds N recent-event candidates — locations of events by
  *any* conflict in the same country within 300 km of the anchor —
  deduplicated against the frequency sites. Half of the validation rows
  whose truth is >20 km from every frequency site sit within 20 km of a
  recent cross-conflict event, so coverage, not only ranking, is the binding
  constraint at this scale.
* ``--reliefweb-mentions`` (lat/lon-keyed mention file) adds 7/30/90-day
  mention counts at each candidate's ~0.25 degree cell.
* ``--elevation`` adds candidate elevation and local ruggedness from a
  prebuilt terrain grid.
* every candidate carries any-conflict site activity features (event counts
  by window and days since the last event at that site).

The label is the nearest candidate under the project's planar kilometre
approximation, so the oracle distance shrinks as coverage grows.
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
SPILLOVER_SCAN_EVENTS = 600  # recent events scanned per row before dedup
FAR_AWAY_DAYS = 3650


def pkey(lat, lon):
    return round(float(lat), 5), round(float(lon), 5)


def offset(point, anchor):
    north = (point[0] - anchor[0]) * EARTH_KM
    east = (point[1] - anchor[1]) * EARTH_KM * math.cos(math.radians(anchor[0]))
    return east, north


def load_ucdp(path: Path):
    """Two cutoff-safe pools from the raw GED export:

    * ``conflict_sites``: (conflict_id, site) -> sorted event days — the v5
      raw same-conflict activity features;
    * ``country_sites``: country -> site -> sorted event days — any-conflict
      site activity plus per-country flat event lists for spillover
      candidate selection.
    """
    conflict_sites: dict[tuple, list[int]] = defaultdict(list)
    country_sites: dict[str, dict[tuple, list[int]]] = defaultdict(lambda: defaultdict(list))
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
    for days in conflict_sites.values():
        days.sort()
    for sites in country_sites.values():
        for days in sites.values():
            days.sort()
    flat: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for country, sites in country_sites.items():
        events = [(day, key) for key, days in sites.items() for day in days]
        events.sort()
        if events:
            keys = np.empty(len(events), dtype=object)
            keys[:] = [e[1] for e in events]
            flat[country] = (
                np.asarray([e[0] for e in events], dtype=np.int64),
                keys,
            )
    return conflict_sites, country_sites, flat


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidates", type=int, default=32, help="Frequency-site candidates.")
    parser.add_argument("--spillover", type=int, default=32, help="Recent cross-conflict event candidates.")
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
    k = k_freq + k_spill

    # ReliefWeb mentions keyed by ~0.25 degree cell.
    mention_days: dict[tuple, list[int]] = defaultdict(list)
    if a.reliefweb_mentions and a.reliefweb_mentions.exists():
        z = np.load(a.reliefweb_mentions, allow_pickle=True)
        days = np.asarray(z["day"]).astype(np.int64)
        latlon = np.asarray(z["latlon"], dtype=np.float64)
        for (lat, lon), day in zip(latlon, days):
            mention_days[(round(float(lat) * 4), round(float(lon) * 4))].append(int(day))
        for days_at in mention_days.values():
            days_at.sort()

    # Elevation grid lookup.
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

    conflict_sites, country_sites, country_flat = load_ucdp(a.ucdp_events)

    # 11 base + 4 same-conflict activity + 8 rings + 4 raw same-conflict
    # windows + 5 spillover block (+3 mentions) (+2 terrain).
    candidate_dim = 32 + (3 if mention_days else 0) + (2 if elevation is not None else 0)
    features = np.zeros((n, k, candidate_dim), np.float32)
    coordinates = np.zeros((n, k, 2), np.float32)
    valid = np.zeros((n, k), bool)
    labels = np.zeros(n, np.int64)
    oracle = np.zeros(n, np.float32)

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
                days_arr, keys_arr = country_flat[country]
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
                    if len(chosen) >= k:
                        break
                    if key in chosen_keys:
                        continue
                    point = np.array([key[0], key[1]], float)
                    east, north = offset(point, anchor)
                    if math.hypot(east, north) > SPILLOVER_RADIUS_KM:
                        continue
                    chosen.append((key, point, True))
                    chosen_keys.add(key)

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
                    days_at = mention_days[(round(float(point[0]) * 4), round(float(point[1]) * 4))]
                    values.extend(math.log1p(bisect.bisect_right(days_at, cutoff) - bisect.bisect_left(days_at, cutoff - window)) / 4 for window in (7, 30, 90))
                if elevation is not None:
                    elev_m, rugged_m = terrain(point)
                    values.extend([elev_m, rugged_m])
                features[index, j] = values
                coordinates[index, j] = [east / 1000, north / 1000]
                valid[index, j] = True
            distance = np.linalg.norm(coordinates[index, :len(chosen)] - y[index, None], axis=1) * 1000
            labels[index] = int(distance.argmin())
            oracle[index] = float(distance.min())
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
                        candidate_valid=valid, label=labels, y=y, meta=d["meta"])
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
