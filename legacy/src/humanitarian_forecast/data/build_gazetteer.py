#!/usr/bin/env python3
"""Build an Ethiopia place→PRIO-grid gazetteer and audit Telegram mappings."""

from __future__ import annotations

import argparse
import csv
import difflib
import io
import json
import re
import sqlite3
import unicodedata
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

BROAD_NAMES = {
    "ethiopia", "amhara", "amhara region", "oromia", "oromia region",
    "addis ababa", "southern nations nationalities and peoples", "afar",
    "somali region", "tigray", "benishangul gumuz", "gambela",
}
ADMIN_SUFFIXES = re.compile(
    r"\b(zone|woreda|wereda|district|town|city|region|province|commune|kebele)\b",
    re.IGNORECASE,
)
NON_WORD = re.compile(r"[^\w\u1200-\u137f]+", re.UNICODE)


def normalize_place(value: Any, strip_admin: bool = False) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold().strip()
    text = text.replace("’", "'").replace("`", "'")
    if strip_admin:
        text = ADMIN_SUFFIXES.sub(" ", text)
    return re.sub(r"\s+", " ", NON_WORD.sub(" ", text)).strip()


def priogrid_from_coordinates(latitude: float, longitude: float) -> int:
    if not (-90 <= latitude < 90 and -180 <= longitude < 180):
        raise ValueError("coordinates outside PRIO grid bounds")
    row = int((latitude + 90.0) * 2) + 1
    column = int((longitude + 180.0) * 2) + 1
    return (row - 1) * 720 + column


def ucdp_rows(path: Path) -> Iterable[dict[str, str]]:
    with zipfile.ZipFile(path) as archive:
        name = next(n for n in archive.namelist() if n.lower().endswith(".csv"))
        with archive.open(name) as raw:
            text = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
            yield from csv.DictReader(text)


def build_gazetteer(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    observations: dict[str, Counter[int]] = defaultdict(Counter)
    displays: dict[str, Counter[str]] = defaultdict(Counter)
    coordinates: dict[tuple[str, int], list[tuple[float, float]]] = defaultdict(list)
    stats = Counter()

    for row in ucdp_rows(path):
        if row.get("country") != "Ethiopia":
            continue
        stats["ethiopia_rows"] += 1
        try:
            latitude = float(row["latitude"])
            longitude = float(row["longitude"])
            recorded_grid = int(row["priogrid_gid"])
            precision = int(row["where_prec"])
        except (KeyError, TypeError, ValueError):
            stats["invalid_coordinate_rows"] += 1
            continue

        try:
            calculated = priogrid_from_coordinates(latitude, longitude)
            stats["grid_formula_checked"] += 1
            if calculated == recorded_grid:
                stats["grid_formula_exact"] += 1
            else:
                stats["grid_formula_mismatch"] += 1
        except ValueError:
            stats["grid_formula_invalid_coordinate"] += 1

        if precision > 4:
            stats["excluded_broad_precision"] += 1
            continue

        for field in ("where_coordinates", "adm_2", "adm_1"):
            display = (row.get(field) or "").strip()
            if not display:
                continue
            for key in {normalize_place(display), normalize_place(display, strip_admin=True)}:
                if len(key) < 3 or key in BROAD_NAMES:
                    continue
                observations[key][recorded_grid] += 1
                displays[key][display] += 1
                coordinates[(key, recorded_grid)].append((latitude, longitude))

    gazetteer: dict[str, dict[str, Any]] = {}
    for key, grid_counts in observations.items():
        total = sum(grid_counts.values())
        ranked = grid_counts.most_common()
        best_grid, best_count = ranked[0]
        coords = coordinates[(key, best_grid)]
        gazetteer[key] = {
            "canonical_name": displays[key].most_common(1)[0][0],
            "priogrid_gid": best_grid,
            "modal_share": best_count / total,
            "observation_count": total,
            "latitude_mean": sum(v[0] for v in coords) / len(coords),
            "longitude_mean": sum(v[1] for v in coords) / len(coords),
            "alternatives": [
                {"priogrid_gid": grid, "count": count, "share": count / total}
                for grid, count in ranked[:8]
            ],
        }
    stats["gazetteer_aliases"] = len(gazetteer)
    stats["ambiguous_aliases_below_80pct"] = sum(
        entry["modal_share"] < 0.80 for entry in gazetteer.values()
    )
    return gazetteer, dict(stats)


def candidate_names(row: sqlite3.Row) -> Iterable[tuple[str, str]]:
    for field in ("woreda", "locality", "zone", "normalized_name_en", "raw_name"):
        value = (row[field] or "").strip()
        if value:
            yield field, value


def map_location(
    row: sqlite3.Row,
    gazetteer: dict[str, dict[str, Any]],
    keys: list[str],
) -> dict[str, Any]:
    attempted: list[str] = []
    for field, original in candidate_names(row):
        for stripped in (False, True):
            key = normalize_place(original, strip_admin=stripped)
            if not key or key in attempted:
                continue
            attempted.append(key)
            if key in BROAD_NAMES:
                continue
            entry = gazetteer.get(key)
            if entry and entry["modal_share"] >= 0.80:
                return {
                    "status": "accepted", "method": "exact" if not stripped else "admin_alias",
                    "matched_field": field, "matched_input": original, "matched_key": key,
                    "similarity": 1.0, **entry,
                }

    # Fuzzy matching is deliberately conservative: high score, unique winner,
    # and an internally unambiguous gazetteer entry are all required.
    for field, original in candidate_names(row):
        key = normalize_place(original, strip_admin=True)
        if len(key) < 4 or key in BROAD_NAMES:
            continue
        matches = difflib.get_close_matches(key, keys, n=2, cutoff=0.94)
        if not matches:
            continue
        best = matches[0]
        score = difflib.SequenceMatcher(None, key, best).ratio()
        second_score = (
            difflib.SequenceMatcher(None, key, matches[1]).ratio() if len(matches) > 1 else 0.0
        )
        entry = gazetteer[best]
        if score >= 0.94 and score - second_score >= 0.03 and entry["modal_share"] >= 0.80:
            return {
                "status": "accepted", "method": "fuzzy", "matched_field": field,
                "matched_input": original, "matched_key": best, "similarity": score,
                **entry,
            }

    original = next(candidate_names(row), ("", ""))[1]
    broad = normalize_place(original) in BROAD_NAMES
    return {
        "status": "rejected_broad" if broad else "unmapped",
        "method": "none", "matched_field": "", "matched_input": original,
        "matched_key": "", "similarity": 0.0,
    }


def audit_telegram(
    db_path: Path,
    gazetteer: dict[str, dict[str, Any]],
    output_csv: Path,
) -> dict[str, int]:
    conn = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro&immutable=1", uri=True)
    conn.row_factory = sqlite3.Row
    sql = """
        SELECT
            l.location_id,l.event_id,e.post_id,e.post_day,e.event_type,e.source_name,
            l.raw_name,l.normalized_name_en,l.region,l.zone,l.woreda,l.locality,
            l.admin_level,l.confidence,e.event_extraction_confidence,
            a.translation_used,a.original_report_cluster_id,p.ethiopic_ratio
        FROM ai_event_locations l
        JOIN ai_events e USING(event_id)
        JOIN ai_post_analysis a USING(analysis_id)
        JOIN posts p USING(post_id)
        WHERE l.confidence >= 0.55 AND e.event_extraction_confidence >= 0.55
          AND (COALESCE(p.ethiopic_ratio, 0) = 0 OR a.translation_used = 1)
        ORDER BY e.post_day,e.post_id,l.location_id
    """
    keys = sorted(gazetteer)
    stats = Counter()
    seen_clusters: set[tuple[int, str, str]] = set()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "location_id", "event_id", "post_id", "post_day", "event_type", "source_name",
        "raw_name", "normalized_name_en", "zone", "woreda", "locality", "admin_level",
        "status", "method", "matched_field", "matched_input", "matched_key",
        "canonical_name", "priogrid_gid", "similarity", "modal_share",
        "observation_count", "latitude_mean", "longitude_mean",
        "report_cluster_id", "is_duplicate_cluster",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in conn.execute(sql):
            stats["eligible_location_rows"] += 1
            mapped = map_location(row, gazetteer, keys)
            stats[mapped["status"]] += 1
            stats[f"method_{mapped['method']}"] += 1
            grid = mapped.get("priogrid_gid")
            cluster = row["original_report_cluster_id"]
            duplicate = False
            if mapped["status"] == "accepted":
                dedup_id = str(cluster) if cluster is not None else f"post:{row['post_id']}"
                dedup_key = (int(grid), row["post_day"], dedup_id)
                duplicate = dedup_key in seen_clusters
                seen_clusters.add(dedup_key)
                stats["accepted_duplicate_clusters" if duplicate else "accepted_unique_clusters"] += 1
            record = {field: row[field] if field in row.keys() else "" for field in fields}
            record.update(mapped)
            record["report_cluster_id"] = cluster if cluster is not None else ""
            record["is_duplicate_cluster"] = int(duplicate)
            writer.writerow({field: record.get(field, "") for field in fields})
    conn.close()
    return dict(stats)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ucdp", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed_v4"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    gazetteer, build_stats = build_gazetteer(args.ucdp)
    gazetteer_path = args.output_dir / "ethiopia_gazetteer.json"
    gazetteer_path.write_text(
        json.dumps(gazetteer, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    audit_path = args.output_dir / "telegram_priogrid_mappings.csv"
    mapping_stats = audit_telegram(args.db, gazetteer, audit_path)
    summary = {"gazetteer": build_stats, "telegram_mapping": mapping_stats}
    summary_path = args.output_dir / "mapping_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Gazetteer: {gazetteer_path}")
    print(f"Mapping audit: {audit_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
