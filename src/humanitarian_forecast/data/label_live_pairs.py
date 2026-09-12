#!/usr/bin/env python3
"""Label matured live forecast payloads with realized UCDP outcomes.

Each published forecast's feature payload is a complete, cutoff-safe snapshot
of the model inputs. Once the predicted window has matured and UCDP has had
time to record what actually happened, this script turns those payloads into
training rows in exactly the ``conflict_candidates_64_spillover_v6`` schema:
the realized event for the target conflict inside the horizon window becomes
the ``y`` target, and the nearest candidate becomes the ``label``.

Realized outcomes come only from UCDP GED (verified event data). Public
preview signals are features, never labels.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import zipfile
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from humanitarian_forecast.data.build_candidate_rank_dataset_v6 import offset

EVENT_DIM = 19
CANDIDATE_DIM = 37
SEQUENCE_LENGTH = 16
_UNIX_EPOCH_DAY = int(np.datetime64("1970-01-01", "D").astype(int))


def _day_int(value: str) -> int:
    return int(np.datetime64(value[:10], "D").astype(int))


def _day_to_date(day_int: int) -> date:
    """np day number (days since 0000-01-01) to a calendar date."""
    return date(1970, 1, 1) + timedelta(days=int(day_int) - _UNIX_EPOCH_DAY)


def load_conflict_events(ged_zip: Path, candidate_ged: Path, conflict_id: str) -> list[dict[str, Any]]:
    """Merged, deduplicated event stream for one conflict across the yearly
    GED release and the candidate extension."""
    events: dict[int, dict[str, Any]] = {}
    with zipfile.ZipFile(ged_zip) as archive:
        name = next(n for n in archive.namelist() if n.lower().endswith(".csv"))
        with archive.open(name) as raw, io.TextIOWrapper(raw, encoding="utf-8-sig", newline="") as text:
            for row in csv.DictReader(text):
                try:
                    if row["conflict_new_id"].strip() != conflict_id:
                        continue
                    if int(row["where_prec"]) > 4 or int(row["date_prec"]) > 3:
                        continue
                    events[int(row["id"])] = {
                        "id": int(row["id"]),
                        "day": _day_int(row["date_start"]),
                        "lat": float(row["latitude"]),
                        "lon": float(row["longitude"]),
                    }
                except (KeyError, TypeError, ValueError):
                    continue
    with open(candidate_ged, encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            try:
                if row["conflict_new_id"].strip() != conflict_id:
                    continue
                if int(row["where_prec"]) > 4 or int(row["date_prec"]) > 3:
                    continue
                events[int(row["id"])] = {
                    "id": int(row["id"]),
                    "day": _day_int(row["date_start"]),
                    "lat": float(row["latitude"]),
                    "lon": float(row["longitude"]),
                }
            except (KeyError, TypeError, ValueError):
                continue
    return sorted(events.values(), key=lambda event: (event["day"], event["id"]))


def _payload_rows(
    payload: dict[str, Any],
    events: list[dict[str, Any]],
    today: date,
    grace_days: int,
) -> tuple[list[dict[str, Any]], bool] | None:
    """Rows for one payload, or None when the window has not matured yet.

    Returns (rows, had_event): rows are v6-schema entries, had_event is False
    when the matured window contained no conflict events (a permanent skip).
    """
    features = payload["featurePayload"]
    cutoff = datetime.fromisoformat(features["observationCutoff"].replace("Z", "+00:00"))
    horizon = int(features["horizonDays"])
    window_end = cutoff.date() + timedelta(days=horizon)
    if today <= window_end + timedelta(days=grace_days):
        return None
    anchor = features["anchor"]
    realized = [
        event for event in events
        if cutoff.date() < _day_to_date(event["day"]) <= window_end
    ]
    if not realized:
        return [], False
    events_matrix = np.asarray(features["events"], dtype=np.float32)
    candidate_features = np.asarray(
        [[float(v) for v in candidate["features"]] for candidate in features["candidates"]],
        dtype=np.float32,
    )
    coordinates = np.asarray(
        [[candidate["eastKm"] / 1000.0, candidate["northKm"] / 1000.0] for candidate in features["candidates"]],
        dtype=np.float32,
    )
    valid = np.asarray([candidate["valid"] for candidate in features["candidates"]], dtype=bool)
    history_alive = events_matrix[:, 0] > 0.5
    gap_days = max(1, int(round(float(np.expm1(events_matrix[history_alive, 3].max() * 6))))) if history_alive.any() else 1
    rows: list[dict[str, Any]] = []
    for event in realized:
        east, north = offset((event["lat"], event["lon"]), (float(anchor["latitude"]), float(anchor["longitude"])))
        target = np.asarray([east / 1000.0, north / 1000.0], dtype=np.float32)
        distance = np.linalg.norm(coordinates - target, axis=1) * 1000.0
        masked = np.where(valid, distance, np.inf)
        rows.append({
            "run_id": payload["runId"],
            "x": events_matrix,
            "candidate_features": candidate_features,
            "candidate_coordinates": coordinates,
            "candidate_valid": valid,
            "y": target,
            "label": int(masked.argmin()),
            "oracle_km": float(masked.min()),
            "meta": {
                "run_id": payload["runId"],
                "country": features["country"],
                "conflict_id": features["conflictId"],
                "anchor_lat": float(anchor["latitude"]),
                "anchor_lon": float(anchor["longitude"]),
                "target_date": str(_day_to_date(event["day"])),
                "target_lat": event["lat"],
                "target_lon": event["lon"],
                "gap_days": gap_days,
            },
        })
    return rows, True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payloads", type=Path, required=True, help="JSONL of stored ingest payloads")
    parser.add_argument("--ged-zip", type=Path, required=True)
    parser.add_argument("--candidate-ged", type=Path, required=True)
    parser.add_argument("--live", type=Path, default=None, help="existing live-pairs npz to extend")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grace-days", type=int, default=45)
    args = parser.parse_args()

    payloads = [json.loads(line) for line in args.payloads.read_text(encoding="utf-8").splitlines() if line.strip()]
    existing_ids: set[str] = set()
    live: dict[str, Any] = {}
    if args.live is not None and args.live.exists():
        with np.load(args.live, allow_pickle=False) as previous:
            live = {key: previous[key] for key in previous.files}
            existing_ids = {str(row["run_id"]) for row in (live["meta"] if "meta" in live else [])}
            existing_ids |= {str(value) for value in (live.get("no_event", []) if isinstance(live.get("no_event", []), np.ndarray) else [])}

    conflict_ids = {payload["featurePayload"]["conflictId"] for payload in payloads}
    event_streams = {
        conflict_id: load_conflict_events(args.ged_zip, args.candidate_ged, conflict_id)
        for conflict_id in conflict_ids
    }

    today = datetime.now(UTC).date()
    new_rows: list[dict[str, Any]] = []
    new_no_event: list[str] = []
    labeled = skipped = 0
    for payload in payloads:
        run_id = str(payload["runId"])
        if run_id in existing_ids:
            continue
        outcome = _payload_rows(payload, event_streams[str(payload["featurePayload"]["conflictId"])], today, args.grace_days)
        if outcome is None:
            continue
        rows, had_event = outcome
        if not had_event:
            new_no_event.append(run_id)
            skipped += 1
            continue
        new_rows.extend(rows)
        labeled += 1

    if live:
        live["x"] = np.concatenate([live["x"], np.asarray([row["x"] for row in new_rows], dtype=np.float32)]) if new_rows else live["x"]
        live["candidate_features"] = np.concatenate([live["candidate_features"], np.asarray([row["candidate_features"] for row in new_rows], dtype=np.float32)]) if new_rows else live["candidate_features"]
        live["candidate_coordinates"] = np.concatenate([live["candidate_coordinates"], np.asarray([row["candidate_coordinates"] for row in new_rows], dtype=np.float32)]) if new_rows else live["candidate_coordinates"]
        live["candidate_valid"] = np.concatenate([live["candidate_valid"], np.asarray([row["candidate_valid"] for row in new_rows], dtype=bool)]) if new_rows else live["candidate_valid"]
        live["y"] = np.concatenate([live["y"], np.asarray([row["y"] for row in new_rows], dtype=np.float32)]) if new_rows else live["y"]
        live["label"] = np.concatenate([live["label"], np.asarray([row["label"] for row in new_rows], dtype=np.int64)]) if new_rows else live["label"]
        live["oracle"] = np.concatenate([live["oracle"], np.asarray([row["oracle_km"] for row in new_rows], dtype=np.float32)]) if new_rows else live["oracle"]
        live["meta"] = np.concatenate([live["meta"], np.asarray([json.dumps(row["meta"]) for row in new_rows])]) if new_rows else live["meta"]
        no_event = [str(value) for value in live.get("no_event", []).tolist()] if "no_event" in live else []
        live["no_event"] = np.asarray(sorted(set(no_event + new_no_event)))
    else:
        live = {
            "x": np.asarray([row["x"] for row in new_rows], dtype=np.float32),
            "candidate_features": np.asarray([row["candidate_features"] for row in new_rows], dtype=np.float32),
            "candidate_coordinates": np.asarray([row["candidate_coordinates"] for row in new_rows], dtype=np.float32),
            "candidate_valid": np.asarray([row["candidate_valid"] for row in new_rows], dtype=bool),
            "y": np.asarray([row["y"] for row in new_rows], dtype=np.float32),
            "label": np.asarray([row["label"] for row in new_rows], dtype=np.int64),
            "oracle": np.asarray([row["oracle_km"] for row in new_rows], dtype=np.float32),
            "meta": np.asarray([json.dumps(row["meta"]) for row in new_rows]),
            "no_event": np.asarray(sorted(set(new_no_event))),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **live)
    total = len(live["meta"])
    print(json.dumps({
        "payloads": len(payloads), "labeled": labeled, "skipped_no_event": skipped,
        "new_rows": len(new_rows), "total_rows": int(total),
    }))


if __name__ == "__main__":
    main()
