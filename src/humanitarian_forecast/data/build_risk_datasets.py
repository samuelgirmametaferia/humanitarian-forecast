#!/usr/bin/env python3
"""Build aligned international and Ethiopia temporal-risk tensors."""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import math
import re
import sqlite3
import zipfile
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

FEATURE_NAMES = [
    "activity", "attack", "fatality", "displacement", "territory",
    "infrastructure", "source_diversity", "mean_confidence",
]
LEXICONS = {
    "attack": ("attack", "clash", "fight", "shell", "airstrike", "drone", "ambush", "raid"),
    "fatality": ("kill", "death", "dead", "casualt", "wound", "injur", "massacre"),
    "displacement": ("displac", "evacuat", "flee", "refugee", "returnee"),
    "territory": ("capture", "control", "advance", "retreat", "front line", "frontline", "siege"),
    "infrastructure": ("road", "bridge", "hospital", "school", "water", "electric", "infrastructure"),
}


def day(value: str) -> date:
    return datetime.fromisoformat(value[:10]).date()


def text_flags(text: str) -> list[float]:
    lower = text.casefold()
    return [float(sum(lower.count(term) for term in LEXICONS[name])) for name in list(LEXICONS)]


def make_samples(
    daily: dict[str, dict[date, np.ndarray]], sequence: int, horizon: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, str]]]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    meta: list[dict[str, str]] = []
    for entity, values in sorted(daily.items()):
        if len(values) < 3:
            continue
        first, last = min(values), max(values)
        span = (last - first).days + 1
        matrix = np.zeros((span, len(FEATURE_NAMES)), dtype=np.float32)
        for dt, vector in values.items():
            matrix[(dt - first).days] = vector
        for idx in range(sequence - 1, span - horizon):
            history = matrix[idx - sequence + 1:idx + 1]
            future = matrix[idx + 1:idx + 1 + horizon, 0].sum()
            recent = history[:, 0].sum()
            # Completely quiet windows provide little information and explode the
            # dataset size for sparse global grids. Cessation windows (recent>0,
            # future=0) remain as meaningful negatives; onsets remain positives.
            if future == 0 and recent == 0:
                continue
            xs.append(history)
            ys.append(np.array([math.log1p(float(future)), float(future > max(1.0, recent * 0.5))], dtype=np.float32))
            meta.append({"entity": entity, "cutoff": str(first + timedelta(days=idx))})
    if not xs:
        raise RuntimeError("No usable temporal samples were produced")
    order = sorted(range(len(meta)), key=lambda i: (meta[i]["cutoff"], meta[i]["entity"]))
    return np.stack([xs[i] for i in order]), np.stack([ys[i] for i in order]), [meta[i] for i in order]


def reliefweb_daily(path: Path) -> dict[str, dict[date, np.ndarray]]:
    accum: dict[str, dict[date, dict[str, object]]] = defaultdict(lambda: defaultdict(lambda: {
        "count": 0, "flags": np.zeros(5, dtype=np.float32), "sources": set(),
    }))
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            dates = row.get("date") or {}
            # Publication time is the causal availability timestamp. ``original``
            # may refer to an earlier event and cannot be used to place a report
            # into a historical feature window.
            dt_raw = dates.get("created") or dates.get("original")
            country = row.get("primary_country") or {}
            entity = country.get("iso3") or country.get("name")
            if not dt_raw or not entity:
                continue
            bucket = accum[str(entity)][day(dt_raw)]
            bucket["count"] = int(bucket["count"]) + 1
            bucket["flags"] += np.array(text_flags(f"{row.get('title', '')} {row.get('body', '')}"), dtype=np.float32)
            for source in row.get("source") or []:
                if source.get("name"):
                    bucket["sources"].add(source["name"])
    result: dict[str, dict[date, np.ndarray]] = defaultdict(dict)
    for entity, days in accum.items():
        for dt, item in days.items():
            count = float(item["count"])
            result[entity][dt] = np.array([
                math.log1p(count), *np.log1p(item["flags"]),
                math.log1p(len(item["sources"])), 1.0,
            ], dtype=np.float32)
    return result


def ucdp_daily(path: Path, country_filter: str | None = None) -> dict[str, dict[date, np.ndarray]]:
    """Read UCDP GED directly from its ZIP and aggregate precise PRIO-grid days."""
    buckets: dict[str, dict[date, dict[str, object]]] = defaultdict(lambda: defaultdict(lambda: {
        "count": 0, "deaths": 0.0, "civilian": 0.0, "sources": 0.0,
        "confidence": [],
    }))
    with zipfile.ZipFile(path) as archive:
        csv_name = next(name for name in archive.namelist() if name.lower().endswith(".csv"))
        with archive.open(csv_name) as raw, io.TextIOWrapper(raw, encoding="utf-8-sig", newline="") as text:
            for row in csv.DictReader(text):
                if country_filter and row.get("country") != country_filter:
                    continue
                try:
                    where_prec = int(row["where_prec"])
                    date_prec = int(row["date_prec"])
                    grid = row["priogrid_gid"].strip()
                    dt = day(row["date_start"])
                except (KeyError, TypeError, ValueError):
                    continue
                # Country-only and broad-region geocodes are not suitable targets
                # for a general-area forecaster.
                if not grid or where_prec > 4:
                    continue
                bucket = buckets[f"prio:{grid}"][dt]
                bucket["count"] = int(bucket["count"]) + 1
                bucket["deaths"] = float(bucket["deaths"]) + float(row.get("best") or 0)
                bucket["civilian"] = float(bucket["civilian"]) + float(row.get("deaths_civilians") or 0)
                bucket["sources"] = float(bucket["sources"]) + max(0.0, float(row.get("number_of_sources") or 0))
                bucket["confidence"].append(max(0.0, 1.0 - ((where_prec - 1) + (date_prec - 1)) / 10.0))
    result: dict[str, dict[date, np.ndarray]] = defaultdict(dict)
    for grid, days in buckets.items():
        for dt, item in days.items():
            count = float(item["count"])
            result[grid][dt] = np.array([
                math.log1p(count), math.log1p(count), math.log1p(float(item["deaths"])),
                0.0, 0.0, 0.0,
                math.log1p(float(item["sources"])), float(np.mean(item["confidence"])),
            ], dtype=np.float32)
    return result


def ethiopia_daily(db_path: Path) -> dict[str, dict[date, np.ndarray]]:
    conn = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro&immutable=1", uri=True)
    conn.row_factory = sqlite3.Row
    sql = """
        SELECT e.event_id,e.post_day,e.event_type,e.event_extraction_confidence,
               e.source_name,l.raw_name,l.normalized_name_en,l.region,l.zone,l.woreda,l.locality,
               l.admin_level,l.confidence
        FROM ai_events e
        JOIN ai_event_locations l USING(event_id)
        JOIN ai_post_analysis a USING(analysis_id)
        JOIN posts p USING(post_id)
        WHERE e.event_extraction_confidence >= 0.55 AND l.confidence >= 0.55
          AND (COALESCE(p.ethiopic_ratio, 0) = 0 OR a.translation_used = 1)
    """
    broad = {"ethiopia", "amhara", "amhara region", "oromia", "oromia region", "addis ababa"}
    buckets: dict[str, dict[date, dict[str, object]]] = defaultdict(lambda: defaultdict(lambda: {
        "events": set(), "flags": np.zeros(5, dtype=np.float32), "sources": set(), "conf": [],
    }))
    for row in conn.execute(sql):
        area = next((row[k] for k in ("woreda", "locality", "zone", "normalized_name_en", "raw_name") if row[k]), "")
        area = re.sub(r"\s+", " ", area).strip()
        if not area or area.casefold() in broad:
            continue
        bucket = buckets[area][day(row["post_day"])]
        if row["event_id"] in bucket["events"]:
            continue
        bucket["events"].add(row["event_id"])
        bucket["sources"].add(row["source_name"])
        bucket["conf"].append(min(float(row["event_extraction_confidence"]), float(row["confidence"])))
        event = row["event_type"]
        mapping = [
            event in {"armed_clash", "ambush", "raid", "ground_attack", "drone_or_airstrike", "shelling_or_bombardment"},
            event in {"civilian_killing", "civilian_injury"},
            event == "displacement",
            event in {"territorial_control_change", "checkpoint_or_road_control"},
            event == "infrastructure_damage",
        ]
        bucket["flags"] += np.array(mapping, dtype=np.float32)
    conn.close()
    result: dict[str, dict[date, np.ndarray]] = defaultdict(dict)
    for area, days in buckets.items():
        for dt, item in days.items():
            count = len(item["events"])
            result[area][dt] = np.array([
                math.log1p(count), *np.log1p(item["flags"]),
                math.log1p(len(item["sources"])), float(np.mean(item["conf"])),
            ], dtype=np.float32)
    return result


def save(path: Path, x: np.ndarray, y: np.ndarray, meta: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, x=x, y=y, meta=np.array([json.dumps(v) for v in meta]))
    print(f"{path}: {len(x):,} samples, shape={x.shape}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reliefweb", type=Path, required=True)
    parser.add_argument("--ucdp", type=Path)
    parser.add_argument("--ethiopia-db", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--sequence", type=int, default=28)
    parser.add_argument("--horizon", type=int, default=7)
    args = parser.parse_args()
    rx, ry, rm = make_samples(reliefweb_daily(args.reliefweb), args.sequence, args.horizon)
    save(args.output_dir / "reliefweb.npz", rx, ry, rm)
    if args.ucdp:
        ux, uy, um = make_samples(ucdp_daily(args.ucdp), args.sequence, args.horizon)
        # Both sources share the same generic feature contract. UCDP supplies
        # georeferenced event truth; ReliefWeb adds broader humanitarian reporting
        # dynamics. Sort the combined samples chronologically after concatenation.
        records = list(zip(ux, uy, um)) + list(zip(rx, ry, rm))
        records.sort(key=lambda item: (item[2]["cutoff"], item[2]["entity"]))
        bx = np.stack([item[0] for item in records])
        by = np.stack([item[1] for item in records])
        bm = [item[2] for item in records]
    else:
        bx, by, bm = rx, ry, rm
    ex, ey, em = make_samples(ethiopia_daily(args.ethiopia_db), args.sequence, args.horizon)
    if args.ucdp:
        hx, hy, hm = make_samples(
            ucdp_daily(args.ucdp, country_filter="Ethiopia"), args.sequence, args.horizon
        )
        ethiopia_records = list(zip(hx, hy, hm)) + list(zip(ex, ey, em))
        ethiopia_records.sort(key=lambda item: (item[2]["cutoff"], item[2]["entity"]))
        ex = np.stack([item[0] for item in ethiopia_records])
        ey = np.stack([item[1] for item in ethiopia_records])
        em = [item[2] for item in ethiopia_records]
    save(args.output_dir / "base.npz", bx, by, bm)
    save(args.output_dir / "ethiopia.npz", ex, ey, em)


if __name__ == "__main__":
    main()
