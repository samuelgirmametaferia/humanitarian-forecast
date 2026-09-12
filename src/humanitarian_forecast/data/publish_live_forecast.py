#!/usr/bin/env python3
"""Build and publish the live Ethiopia production forecast snapshot.

Turns the freshest public sources — UCDP GED 26.1 plus the candidate GED
extension, ReliefWeb reports, and extracted Telegram public-web-preview
signals — into a ``feature-contract.v1`` payload, signs it with the ingest
HMAC secret, and posts it to the deployed ingest endpoint. The server runs
the ONNX ensemble and persists a ``forecast-snapshot.v1`` for the map.

Feature parity with training:

* events: the 19-dim geo-v2 encoding from ``build_next_location_dataset``
  (imported directly, never re-implemented);
* candidates: the 37-dim v6 recipe from ``build_candidate_rank_dataset_v6``
  (32 base + 3 mention windows + 2 terrain).

Live-time reference frames (documented deviation from training rows):

* Training rows measure recency from the target event date ``T`` and compute
  activity windows from the anchor date ``A = T - gap``. Live, the next event
  is unknown, so the target date is assumed to fall at the end of the
  horizon: ``T = cutoff + horizon_days`` and ``gap = T - A``.
* All candidate activity/mention windows and site ages are computed from the
  observation cutoff (today). In training the window cutoff equals the anchor
  date; using today is what lets fresh ReliefWeb/Telegram signals reach the
  model, at the cost of a small distribution shift when the anchor is stale.

Extracted public signals enter only as mention-window features and source
health entries — they are model input signals, never labels or verified
events. Publication never trains anything; weights change only through the
validated reconcile gate.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import gzip
import hashlib
import hmac
import json
import math
import os
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from humanitarian_forecast.data.build_candidate_rank_dataset_v6 import (
    EARTH_KM,
    FAR_AWAY_DAYS,
    SPILLOVER_RADIUS_KM,
    SPILLOVER_SCAN_EVENTS,
    offset,
    pkey,
)
from humanitarian_forecast.data.build_next_location_dataset import (
    event_features,
    load_ucdp,
    reliefweb_context,
)
from humanitarian_forecast.data.reliefweb_mentions import (
    build_aliases,
    extract_mentions,
    iter_reliefweb_rows,
    report_day,
)

SEQUENCE_LENGTH = 16
CANDIDATE_FREQ = 32
CANDIDATE_SPILL = 32
CANDIDATE_TOTAL = CANDIDATE_FREQ + CANDIDATE_SPILL
EVENT_DIM = 19
CANDIDATE_DIM = 37
CONFLICT_ID = "16069"  # Ethiopia: Government/Amhara
INGEST_TIMEOUT_SECONDS = 60


def day_int(value: str) -> int:
    return int(np.datetime64(value[:10], "D").astype(int))


def as_utc(dt: datetime) -> datetime:
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def load_candidate_ged(path: Path) -> list[dict[str, Any]]:
    """Rows from the plain-CSV candidate GED extension (2026 events), carrying
    every field the training event encoding needs."""
    out: list[dict[str, Any]] = []
    with open(path, encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            try:
                if int(row["where_prec"]) > 4 or int(row["date_prec"]) > 3:
                    continue
                out.append({
                    "id": int(row["id"]),
                    "day": day_int(row["date_start"]),
                    "date": datetime.fromisoformat(row["date_start"][:10]),
                    "lat": float(row["latitude"]),
                    "lon": float(row["longitude"]),
                    "country": row["country"],
                    "conflict": row["conflict_new_id"].strip(),
                    "conflict_name": row["conflict_name"],
                    "violence": int(row["type_of_violence"]),
                    "fatalities": max(0.0, float(row.get("best") or 0)),
                    "civilian_fatalities": max(0.0, float(row.get("deaths_civilians") or 0)),
                    "sources": max(0.0, float(row.get("number_of_sources") or 0)),
                    "where_precision": int(row["where_prec"]),
                    "date_precision": int(row["date_prec"]),
                })
            except (KeyError, TypeError, ValueError):
                continue
    return out


def build_pools(
    events: dict[str, list[dict[str, Any]]],
    extension: list[dict[str, Any]],
) -> tuple[dict[tuple, list[int]], dict[str, dict[tuple, list[int]]], dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Cutoff-safe activity pools: same-conflict site days, any-conflict site
    days per country, and flat per-country event lists for spillover selection.

    Mirrors ``load_ucdp`` in the v6 builder, extended with the deduplicated
    candidate-GED rows so 2026 activity is visible.
    """
    conflict_sites: dict[tuple, list[int]] = defaultdict(list)
    country_sites: dict[str, dict[tuple, list[int]]] = defaultdict(lambda: defaultdict(list))
    for conflict_id, rows in events.items():
        for event in rows:
            key = pkey(event["lat"], event["lon"])
            day = day_int(event["date"].date().isoformat())
            conflict_sites[(str(conflict_id), key)].append(day)
            country_sites[str(event["country"])][key].append(day)
    seen: set[int] = {int(event["id"]) for rows in events.values() for event in rows}
    for row in extension:
        if row["id"] in seen:
            continue
        seen.add(row["id"])
        key = pkey(row["lat"], row["lon"])
        conflict_sites[(row["conflict"], key)].append(row["day"])
        country_sites[row["country"]][key].append(row["day"])
    for days in conflict_sites.values():
        days.sort()
    for sites in country_sites.values():
        for days in sites.values():
            days.sort()
    flat: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for country, sites in country_sites.items():
        pairs = sorted((day, key) for key, days in sites.items() for day in days)
        if pairs:
            keys = np.empty(len(pairs), dtype=object)
            keys[:] = [pair[1] for pair in pairs]
            flat[country] = (np.asarray([pair[0] for pair in pairs], dtype=np.int64), keys)
    return conflict_sites, country_sites, flat


def merged_conflict_events(
    events: dict[str, list[dict[str, Any]]],
    extension: list[dict[str, Any]],
    conflict_id: str,
    cutoff_day: int,
) -> list[dict[str, Any]]:
    """Same-conflict event stream (training schema) extended with deduplicated
    candidate-GED rows up to the cutoff."""
    stream = [dict(event) for event in events.get(conflict_id, [])]
    stream_ids = {int(event["id"]) for event in stream}
    for row in extension:
        if row["conflict"] != conflict_id or row["id"] in stream_ids or row["day"] > cutoff_day:
            continue
        stream.append({key: row[key] for key in (
            "id", "date", "lat", "lon", "country", "violence", "fatalities",
            "civilian_fatalities", "sources", "where_precision", "date_precision",
        )} | {"conflict": row["conflict_name"]})
        stream_ids.add(row["id"])
    stream.sort(key=lambda event: (event["date"], int(event["id"])))
    return stream


def mention_index(
    reliefweb_path: Path | None,
    gazetteer_path: Path | None,
    signals_path: Path | None,
    mention_lookback_days: int,
    cutoff_day: int,
) -> dict[tuple, list[int]]:
    """ReliefWeb place mentions plus extracted signal mentions, keyed by the
    ~0.25 degree cell used by the v6 mention features."""
    cells: dict[tuple, list[int]] = defaultdict(list)
    first_day = cutoff_day - mention_lookback_days
    if reliefweb_path and gazetteer_path and Path(reliefweb_path).exists():
        aliases, max_words = build_aliases(Path(gazetteer_path))
        rows = [
            row for row in iter_reliefweb_rows(Path(reliefweb_path))
            if (day := report_day(row)) is not None and first_day <= day <= cutoff_day
        ]
        for day, latitude, longitude in extract_mentions(rows, aliases, max_words):
            if first_day <= day <= cutoff_day:
                cells[(round(latitude * 4), round(longitude * 4))].append(day)
    if signals_path and Path(signals_path).exists():
        with open(signals_path, encoding="utf-8") as stream:
            for line in stream:
                try:
                    signal = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if signal.get("schemaVersion") != "source-signal.v1":
                    continue
                latitude = signal.get("latitude")
                longitude = signal.get("longitude")
                posted = str(signal.get("postedAt") or "")[:10]
                if latitude is None or longitude is None or not posted:
                    continue
                try:
                    day = day_int(posted)
                except ValueError:
                    continue
                if first_day <= day <= cutoff_day:
                    cells[(round(float(latitude) * 4), round(float(longitude) * 4))].append(day)
    for days in cells.values():
        days.sort()
    return cells


def terrain_lookup(elevation_path: Path | None):
    """Closure replicating the v6 terrain() lookup (metres / 1000)."""
    if not elevation_path or not Path(elevation_path).exists():
        return None
    grid = np.load(Path(elevation_path))
    grid_lat = np.asarray(grid["lat"], dtype=np.float64)
    grid_lon = np.asarray(grid["lon"], dtype=np.float64)
    elevation = np.asarray(grid["elevation"], dtype=np.float32)
    rugged = np.asarray(grid["rugged"], dtype=np.float32)

    def terrain(point: tuple[float, float]) -> tuple[float, float]:
        i = int(np.clip(np.searchsorted(grid_lat, point[0]), 0, len(grid_lat) - 1))
        j = int(np.clip(np.searchsorted(grid_lon, point[1]), 0, len(grid_lon) - 1))
        if abs(grid_lat[i] - point[0]) > 0.2 or abs(grid_lon[j] - point[1]) > 0.2:
            return 0.0, 0.0
        return float(elevation[i, j]) / 1000.0, float(rugged[i, j]) / 1000.0

    return terrain


def build_candidate_features(
    *, anchor: tuple[float, float], cutoff_day: int, gap_days: int,
    history_events: list[dict[str, Any]],
    conflict_sites: dict[tuple, list[int]],
    country_sites: dict[str, dict[tuple, list[int]]],
    flat: dict[str, tuple[np.ndarray, np.ndarray]],
    site_stats: dict[tuple, dict[str, Any]],
    transitions: Counter,
    mention_cells: dict[tuple, list[int]],
    terrain,
    conflict_id: str,
    country: str,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Assemble the 64-candidate set and its 37-dim features (v6 recipe)."""
    source = pkey(*anchor)
    # History geometry for the ring features: positions relative to the anchor
    # (thousands of km) and ages relative to the cutoff — the same quantities
    # the v6 builder recovers from the encoded history block.
    history_positions = np.asarray([
        list(offset((float(event["lat"]), float(event["lon"])), anchor))
        for event in history_events
    ], dtype=np.float64) / 1000.0
    history_days = np.asarray([
        max(0, cutoff_day - day_int(event["date"].date().isoformat())) for event in history_events
    ], dtype=np.float64)
    history_fatalities = np.asarray([float(event["fatalities"]) for event in history_events])
    ranked = sorted(
        site_stats.items(),
        key=lambda item: (-item[1]["count"], -item[1]["last"], item[0]),
    )
    chosen: list[tuple[tuple, np.ndarray, bool]] = [(source, np.array(anchor, float), False)]
    chosen.extend((key, value["point"], False) for key, value in ranked if key != source)
    chosen = chosen[:CANDIDATE_FREQ]
    chosen_keys = {key for key, _, _ in chosen}

    if country in flat:
        days_arr, keys_arr = flat[country]
        hi = int(np.searchsorted(days_arr, cutoff_day, "right"))
        lo = max(0, hi - SPILLOVER_SCAN_EVENTS)
        seen: dict[tuple, int] = {}
        for pos in range(hi - 1, lo - 1, -1):
            key = keys_arr[pos]
            if key in seen:
                continue
            seen[key] = int(days_arr[pos])
            if len(seen) >= 4 * CANDIDATE_SPILL:
                break
        for key, last_day in sorted(seen.items(), key=lambda item: -item[1]):
            if len(chosen) >= CANDIDATE_TOTAL:
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
    features = np.zeros((CANDIDATE_TOTAL, CANDIDATE_DIM), np.float32)
    valid = np.zeros(CANDIDATE_TOTAL, bool)
    ids: list[str] = [""] * CANDIDATE_TOTAL
    for j, (key, point, is_spillover) in enumerate(chosen):
        east, north = offset(point, anchor)
        site = site_stats.get(key)
        count = site["count"] if site else 0
        age = max(1, cutoff_day - site["last"]) if site else FAR_AWAY_DAYS
        transition = transitions[(source, key)]
        values = [
            1, east / 1000, north / 1000, math.hypot(east, north) / 1000,
            math.log1p(count) / 6, math.log1p(age) / 6,
            point[0] / 90, point[1] / 180, float(key == source),
            math.log1p(transition) / 5, math.log1p(gap_days) / 5,
        ]
        site_days = site["days"] if site else []
        right = bisect.bisect_right(site_days, cutoff_day)
        values.extend(
            math.log1p(right - bisect.bisect_left(site_days, cutoff_day - window)) / 4
            for window in (7, 30, 90, 365)
        )
        rings = ((25, 7), (50, 30), (100, 90), (250, 90))
        history_distance = np.linalg.norm(
            history_positions - np.asarray([east / 1000, north / 1000]), axis=1
        ) * 1000
        ring_masks = [
            (history_distance <= radius) & (history_days <= window) for radius, window in rings
        ]
        values.extend(math.log1p(int(mask.sum())) / 4 for mask in ring_masks)
        values.extend(math.log1p(float(history_fatalities[mask].sum())) / 6 for mask in ring_masks)
        raw_days = conflict_sites.get((conflict_id, key), [])
        raw_right = bisect.bisect_right(raw_days, cutoff_day)
        values.extend(
            math.log1p(raw_right - bisect.bisect_left(raw_days, cutoff_day - window)) / 4
            for window in (7, 30, 90, 365)
        )
        any_days = sites_any.get(key, [])
        any_right = bisect.bisect_right(any_days, cutoff_day)
        values.append(float(is_spillover))
        values.extend(
            math.log1p(any_right - bisect.bisect_left(any_days, cutoff_day - window)) / 4
            for window in (7, 30, 90)
        )
        values.append(math.log1p(max(1, cutoff_day - any_days[any_right - 1])) / 8 if any_right else 1.0)
        days_at = mention_cells.get((round(float(point[0]) * 4), round(float(point[1]) * 4)), [])
        mention_right = bisect.bisect_right(days_at, cutoff_day)
        values.extend(
            math.log1p(mention_right - bisect.bisect_left(days_at, cutoff_day - window)) / 4
            for window in (7, 30, 90)
        )
        if terrain is not None:
            values.extend(terrain((float(point[0]), float(point[1]))))
        features[j] = values
        valid[j] = True
        ids[j] = f"{key[0]:.5f},{key[1]:.5f}"
    for j in range(len(chosen), CANDIDATE_TOTAL):
        ids[j] = f"unfilled-{j:02d}"
    return ids, features, valid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ged-zip", type=Path, required=True, help="UCDP GED 26.1 csv zip (history through 2025).")
    parser.add_argument("--candidate-ged", type=Path, help="Candidate GED extension CSV (2026 events).")
    parser.add_argument("--reliefweb", type=Path, help="ReliefWeb jsonl.gz export.")
    parser.add_argument("--gazetteer", type=Path, help="Ethiopia gazetteer json for mention geocoding.")
    parser.add_argument("--elevation", type=Path, help="Elevation grid npz (terrain features).")
    parser.add_argument("--signals", type=Path, help="source-signal.v1 jsonl from groq_source_processor.")
    parser.add_argument("--ingest-url", default=os.environ.get("HF_INGEST_URL", ""))
    parser.add_argument("--hmac-secret", default=os.environ.get("HF_INGEST_HMAC_SECRET", ""))
    parser.add_argument("--horizon-days", type=int, default=7)
    parser.add_argument("--mention-lookback-days", type=int, default=400)
    parser.add_argument("--dry-run", action="store_true", help="Print the envelope summary instead of posting.")
    parser.add_argument("--output", type=Path, help="Optionally write the envelope json here (dry-run inspection).")
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    cutoff = now.replace(hour=0, minute=0, second=0, microsecond=0)
    cutoff_day = int(np.datetime64(cutoff.date().isoformat(), "D").astype(int))
    target_day = cutoff_day + args.horizon_days

    events = load_ucdp(args.ged_zip)
    extension = load_candidate_ged(args.candidate_ged) if args.candidate_ged else []
    conflict_events = merged_conflict_events(events, extension, CONFLICT_ID, cutoff_day)
    if len(conflict_events) < 2:
        raise SystemExit(f"conflict {CONFLICT_ID}: only {len(conflict_events)} events available")
    history = conflict_events[-SEQUENCE_LENGTH:]
    anchor_event = history[-1]
    anchor = (float(anchor_event["lat"]), float(anchor_event["lon"]))
    anchor_day = day_int(anchor_event["date"].date().isoformat())
    gap_days = max(1, target_day - anchor_day)
    conflict_label = str(anchor_event["conflict"]) or f"UCDP conflict {CONFLICT_ID}"
    country = str(anchor_event["country"]) or "Ethiopia"
    if gap_days > 90:
        print(json.dumps({"warning": "anchor_stale", "gap_days": gap_days}), file=sys.stderr)

    rw = reliefweb_context(args.reliefweb)

    # History events: the training encoding, measured from the assumed target
    # date (cutoff + horizon) exactly as training rows measure from theirs.
    rows_out = np.zeros((SEQUENCE_LENGTH, EVENT_DIM), np.float32)
    for index, event in enumerate(history):
        previous = history[index - 1] if index else history[0]
        rows_out[index] = event_features(
            event, previous, anchor_event,
            (cutoff + timedelta(days=args.horizon_days)).replace(tzinfo=None),
            rw, include_motion=False,
        )

    conflict_sites, country_sites, flat = build_pools(events, extension)
    site_stats: dict[tuple, dict[str, Any]] = {}
    for event in conflict_events:
        key = pkey(event["lat"], event["lon"])
        day = day_int(event["date"].date().isoformat())
        stats = site_stats.setdefault(
            key, {"point": np.array([event["lat"], event["lon"]], float), "count": 0, "last": day, "days": []}
        )
        stats["count"] += 1
        stats["last"] = day
        stats["days"].append(day)
    transitions: Counter = Counter()
    for before, after in zip(conflict_events, conflict_events[1:]):
        transitions[(pkey(before["lat"], before["lon"]), pkey(after["lat"], after["lon"]))] += 1

    mention_cells = mention_index(
        args.reliefweb, args.gazetteer, args.signals, args.mention_lookback_days, cutoff_day
    )
    terrain = terrain_lookup(args.elevation)
    if terrain is None:
        print(json.dumps({"warning": "elevation grid missing; terrain features zero"}), file=sys.stderr)

    ids, features, valid = build_candidate_features(
        anchor=anchor, cutoff_day=cutoff_day, gap_days=gap_days,
        history_events=history,
        conflict_sites=conflict_sites, country_sites=country_sites, flat=flat,
        site_stats=site_stats, transitions=transitions, mention_cells=mention_cells,
        terrain=terrain, conflict_id=CONFLICT_ID, country=country,
    )

    source_summary = [
        {
            "id": "ucdp-ged-26.1",
            "label": "UCDP GED 26.1 (through 2025-12-31)",
            "status": "healthy",
            "lastSuccessfulAt": now.isoformat(),
            "articlesProcessed": sum(
                len(rows) for rows in events.values() if rows and str(rows[0]["country"]) == "Ethiopia"
            ),
        }
    ]
    if args.candidate_ged:
        source_summary.append({
            "id": "ucdp-candidate-ged",
            "label": "UCDP candidate GED extension (2026)",
            "status": "healthy",
            "lastSuccessfulAt": now.isoformat(),
            "articlesProcessed": len([row for row in extension if row["country"] == "Ethiopia"]),
        })
    if args.reliefweb and Path(args.reliefweb).exists():
        with gzip.open(args.reliefweb, "rb") as stream:
            reliefweb_count = sum(1 for _ in stream)
        source_summary.append({
            "id": "reliefweb",
            "label": "ReliefWeb API (mentions + context)",
            "status": "healthy",
            "lastSuccessfulAt": now.isoformat(),
            "articlesProcessed": reliefweb_count,
        })
    if args.signals and Path(args.signals).exists():
        per_source: Counter = Counter()
        with open(args.signals, encoding="utf-8") as stream:
            for line in stream:
                try:
                    signal = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if signal.get("schemaVersion") == "source-signal.v1":
                    per_source[str(signal.get("sourceId", "telegram"))] += 1
        for source_id, count in sorted(per_source.items())[:12 - len(source_summary)]:
            source_summary.append({
                "id": source_id[:40],
                "label": f"Telegram public web preview: {source_id}"[:80],
                "status": "healthy" if count else "unavailable",
                "lastSuccessfulAt": now.isoformat(),
                "articlesProcessed": int(count),
            })

    run_id = f"live-{cutoff.strftime('%Y%m%d')}-{now.strftime('%H%M')}"
    payload = {
        "schemaVersion": "feature-contract.v1",
        "country": "Ethiopia",
        "conflictId": CONFLICT_ID,
        "conflictLabel": conflict_label[:100],
        "anchor": {"latitude": round(anchor[0], 5), "longitude": round(anchor[1], 5)},
        "observationCutoff": cutoff.isoformat(),
        "horizonDays": args.horizon_days,
        "events": [[float(value) for value in row] for row in rows_out],
        "candidates": [
            {
                "id": ids[j],
                "valid": bool(valid[j]),
                "eastKm": float(features[j, 1] * 1000),
                "northKm": float(features[j, 2] * 1000),
                "features": [float(value) for value in features[j]],
            }
            for j in range(CANDIDATE_TOTAL)
        ],
    }
    envelope = {
        "schemaVersion": "ingest-envelope.v1",
        "runId": run_id,
        "generatedAt": now.isoformat(),
        "featurePayload": payload,
        "sourceSummary": source_summary[:12],
        "historyEvents": [
            {
                "id": f"dataset-update-{cutoff.strftime('%Y%m%d')}",
                "kind": "dataset_update",
                "occurredAt": now.isoformat(),
                "mode": "production",
                "modelName": None,
                "modelVersion": None,
                "sourceSnapshotId": run_id,
                "metrics": {},
                "changes": ["public-source refresh: GED 26.1 + candidate GED + ReliefWeb + telegram signals"],
            },
            {
                "id": f"model-run-{run_id}",
                "kind": "model_run",
                "occurredAt": now.isoformat(),
                "mode": "production",
                "modelName": "theswarm_fine_v2",
                "modelVersion": "fine-v2",
                "sourceSnapshotId": run_id,
                "metrics": {},
                "changes": ["production forecast published from live public-source features"],
            },
        ],
    }
    body = json.dumps(envelope, separators=(",", ":")).encode()

    summary = {
        "runId": run_id,
        "anchor": {"lat": anchor[0], "lon": anchor[1], "date": anchor_event["date"].date().isoformat()},
        "gapDays": gap_days,
        "validCandidates": int(valid.sum()),
        "mentionCells": len(mention_cells),
        "signalSources": [entry["id"] for entry in source_summary],
        "bodyBytes": len(body),
        "observationCutoff": cutoff.isoformat(),
    }
    print(json.dumps(summary, indent=2))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(body)

    if args.dry_run:
        return
    if not args.ingest_url or not args.hmac_secret:
        raise SystemExit("posting requires --ingest-url and --hmac-secret (or their env vars)")

    timestamp = str(int(time.time()))
    digest = hashlib.sha256(body).hexdigest()
    signature = hmac.new(
        args.hmac_secret.encode(), f"{timestamp}.{run_id}.{digest}".encode(), hashlib.sha256
    ).hexdigest()
    request = urllib.request.Request(
        args.ingest_url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "X-HF-Timestamp": timestamp,
            "X-HF-Run-ID": run_id,
            "X-HF-Signature": signature,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=INGEST_TIMEOUT_SECONDS) as response:
            print(json.dumps({"posted": True, "status": response.status, "response": response.read().decode()[:2000]}))
    except urllib.error.HTTPError as error:
        print(json.dumps({"posted": False, "status": error.code, "response": error.read().decode()[:2000]}))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
