#!/usr/bin/env python3
"""Build a Telegram-free UCDP+ReliefWeb next-event location dataset."""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import math
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

EARTH_KM_PER_DEGREE = 111.32


def parse_date(value: str) -> datetime:
    return datetime.fromisoformat(value[:10])


def reliefweb_context(path: Path | None) -> dict[tuple[str, str], np.ndarray]:
    """Country/day context: activity, attack, harm, displacement report signals."""
    if path is None or not path.exists():
        return {}
    values: dict[tuple[str, str], np.ndarray] = defaultdict(lambda: np.zeros(4, np.float32))
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            dt = (row.get("date") or {}).get("original", "")[:10]
            country = (row.get("primary_country") or {}).get("name", "").casefold()
            if not dt or not country:
                continue
            text = f"{row.get('title', '')} {row.get('body', '')}".casefold()
            vector = values[(country, dt)]
            vector[0] += 1
            vector[1] += sum(text.count(term) for term in ("attack", "clash", "fight", "shell", "drone"))
            vector[2] += sum(text.count(term) for term in ("kill", "death", "casualt", "wound", "injur"))
            vector[3] += sum(text.count(term) for term in ("displac", "evacuat", "flee", "refugee"))
    return {key: np.log1p(value) for key, value in values.items()}


def load_ucdp(path: Path) -> dict[str, list[dict[str, object]]]:
    conflicts: dict[str, list[dict[str, object]]] = defaultdict(list)
    with zipfile.ZipFile(path) as archive:
        name = next(n for n in archive.namelist() if n.lower().endswith(".csv"))
        with archive.open(name) as raw, io.TextIOWrapper(raw, encoding="utf-8-sig", newline="") as text:
            for row in csv.DictReader(text):
                try:
                    where_precision = int(row["where_prec"])
                    date_precision = int(row["date_prec"])
                    latitude = float(row["latitude"])
                    longitude = float(row["longitude"])
                    dt = parse_date(row["date_start"])
                    conflict_id = row["conflict_new_id"].strip()
                except (KeyError, TypeError, ValueError):
                    continue
                if not conflict_id or where_precision > 4 or date_precision > 3:
                    continue
                conflicts[conflict_id].append({
                    "id": int(row["id"]), "date": dt, "lat": latitude, "lon": longitude,
                    "country": row["country"], "conflict": row["conflict_name"],
                    "violence": int(row["type_of_violence"]),
                    "fatalities": max(0.0, float(row.get("best") or 0)),
                    "civilian_fatalities": max(0.0, float(row.get("deaths_civilians") or 0)),
                    "sources": max(0.0, float(row.get("number_of_sources") or 0)),
                    "where_precision": where_precision, "date_precision": date_precision,
                })
    for events in conflicts.values():
        events.sort(key=lambda event: (event["date"], event["id"]))
    return conflicts


def offset_km(lat: float, lon: float, anchor_lat: float, anchor_lon: float) -> tuple[float, float]:
    north = (lat - anchor_lat) * EARTH_KM_PER_DEGREE
    east = (lon - anchor_lon) * EARTH_KM_PER_DEGREE * math.cos(math.radians(anchor_lat))
    return east, north


def event_features(
    event: dict[str, object], previous_event: dict[str, object],
    anchor: dict[str, object], cutoff: datetime,
    rw: dict[tuple[str, str], np.ndarray],
) -> np.ndarray:
    east, north = offset_km(
        float(event["lat"]), float(event["lon"]), float(anchor["lat"]), float(anchor["lon"])
    )
    days_ago = max(0, (cutoff - event["date"]).days)
    violence = int(event["violence"])
    context = rw.get((str(event["country"]).casefold(), str(event["date"].date())), np.zeros(4))
    day_of_year = cutoff.timetuple().tm_yday
    season_angle = 2.0 * math.pi * day_of_year / 365.25
    step_east, step_north = offset_km(
        float(event["lat"]), float(event["lon"]),
        float(previous_event["lat"]), float(previous_event["lon"]),
    )
    step_days = max(1, (event["date"] - previous_event["date"]).days)
    step_distance = math.hypot(step_east, step_north)
    bearing = math.atan2(step_east, step_north) if step_distance else 0.0
    return np.array([
        1.0,
        east / 1000.0, north / 1000.0,
        math.log1p(days_ago) / 6.0,
        math.log1p(float(event["fatalities"])) / 6.0,
        math.log1p(float(event["civilian_fatalities"])) / 6.0,
        float(violence == 1), float(violence == 2), float(violence == 3),
        1.0 - (float(event["where_precision"]) - 1.0) / 5.0,
        math.log1p(float(event["sources"])) / 5.0,
        *np.clip(context / 8.0, 0, 2),
        float(anchor["lat"]) / 90.0,
        float(anchor["lon"]) / 180.0,
        math.sin(season_angle), math.cos(season_angle),
        step_east / 1000.0, step_north / 1000.0,
        math.log1p(step_days) / 6.0,
        min(step_distance / step_days, 250.0) / 250.0,
        math.sin(bearing), math.cos(bearing),
    ], dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ucdp", type=Path, required=True)
    parser.add_argument("--reliefweb", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/location/next_location.npz"))
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--minimum-history", type=int, default=4)
    parser.add_argument("--minimum-gap-days", type=int, default=1)
    parser.add_argument("--maximum-gap-days", type=int, default=90)
    args = parser.parse_args()

    rw = reliefweb_context(args.reliefweb)
    conflicts = load_ucdp(args.ucdp)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    metadata: list[str] = []

    for conflict_id, events in conflicts.items():
        for target_index in range(args.minimum_history, len(events)):
            target = events[target_index]
            previous = events[target_index - 1]
            gap = (target["date"] - previous["date"]).days
            if gap < args.minimum_gap_days or gap > args.maximum_gap_days:
                continue
            history = events[max(0, target_index - args.sequence_length):target_index]
            anchor = history[-1]
            tensor = np.zeros((args.sequence_length, 25), dtype=np.float32)
            encoded = [
                event_features(event, history[max(0, index - 1)], anchor, target["date"], rw)
                for index, event in enumerate(history)
            ]
            tensor[-len(encoded):] = np.stack(encoded)
            east, north = offset_km(
                float(target["lat"]), float(target["lon"]),
                float(anchor["lat"]), float(anchor["lon"]),
            )
            xs.append(tensor)
            ys.append(np.array([east / 1000.0, north / 1000.0], dtype=np.float32))
            metadata.append(json.dumps({
                "target_date": str(target["date"].date()), "country": target["country"],
                "conflict_id": conflict_id, "conflict": target["conflict"],
                "anchor_lat": anchor["lat"], "anchor_lon": anchor["lon"],
                "target_lat": target["lat"], "target_lon": target["lon"], "gap_days": gap,
            }, separators=(",", ":")))

    order = sorted(range(len(xs)), key=lambda index: json.loads(metadata[index])["target_date"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        x=np.stack([xs[index] for index in order]),
        y=np.stack([ys[index] for index in order]),
        meta=np.array([metadata[index] for index in order]),
    )
    print(f"conflicts={len(conflicts):,} samples={len(xs):,} shape={(len(xs), args.sequence_length, 25)}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
