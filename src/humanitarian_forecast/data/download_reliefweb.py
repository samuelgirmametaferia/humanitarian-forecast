#!/usr/bin/env python3
"""Resumable ReliefWeb conflict-report downloader using the approved appname."""

from __future__ import annotations

import argparse
import gzip
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

API_URL = "https://api.reliefweb.int/v2/reports"
DEFAULT_APPNAME = "HMA-research-W5F2P"
DEFAULT_QUERY = (
    'conflict OR "armed violence" OR attack OR clashes OR displacement OR '
    'airstrike OR shelling OR fighting OR war'
)
FIELDS = [
    "title", "body", "date.original", "date.created", "country",
    "primary_country", "source.name", "theme.name", "disaster_type.name",
    "language.code", "format.name", "url",
]


def request_page(appname: str, payload: dict[str, Any], attempts: int = 6) -> dict[str, Any]:
    url = API_URL + "?" + urllib.parse.urlencode({"appname": appname})
    body = json.dumps(payload).encode("utf-8")
    for attempt in range(attempts):
        req = urllib.request.Request(
            url, data=body, method="POST",
            headers={"Content-Type": "application/json", "User-Agent": appname},
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                return json.load(response)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            if attempt + 1 == attempts:
                raise
            time.sleep(min(60, 2 ** attempt))
    raise RuntimeError("unreachable")


def existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            try:
                ids.add(str(json.loads(line)["id"]))
            except (KeyError, json.JSONDecodeError):
                continue
    return ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--appname", default=DEFAULT_APPNAME)
    parser.add_argument("--output", type=Path, default=Path("data/reliefweb_conflict.jsonl.gz"))
    parser.add_argument("--start-year", type=int, default=1980)
    parser.add_argument("--end-year", type=int, default=datetime.now(timezone.utc).year)
    parser.add_argument("--page-size", type=int, default=1000)
    parser.add_argument("--max-records", type=int)
    parser.add_argument(
        "--max-per-year", type=int,
        help="Optional sampling cap per year; useful for broad temporal coverage",
    )
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument(
        "--date-field", choices=("created", "original"), default="created",
        help="Partition/sort by publication time by default. Use original only for legacy reproduction.",
    )
    args = parser.parse_args()
    if not (1 <= args.page_size <= 1000):
        parser.error("--page-size must be between 1 and 1000")
    if args.start_year > args.end_year:
        parser.error("--start-year must not exceed --end-year")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    seen = existing_ids(args.output)
    saved = len(seen)
    state_path = args.output.with_suffix(args.output.suffix + ".state.json")
    try:
        offsets = {int(k): int(v) for k, v in json.loads(state_path.read_text()).items()}
    except (FileNotFoundError, ValueError, json.JSONDecodeError):
        offsets = {}
    print(f"Resuming with {saved:,} existing reports")

    with gzip.open(args.output, "at", encoding="utf-8") as out:
        for year in range(args.start_year, args.end_year + 1):
            offset = offsets.get(year, 0)
            fetched_this_run = 0
            while True:
                remaining = None if args.max_records is None else args.max_records - saved
                if remaining is not None and remaining <= 0:
                    print(f"Reached --max-records={args.max_records:,}")
                    return
                year_remaining = None if args.max_per_year is None else args.max_per_year - fetched_this_run
                if year_remaining is not None and year_remaining <= 0:
                    break
                limits = [args.page_size]
                if remaining is not None:
                    limits.append(remaining)
                if year_remaining is not None:
                    limits.append(year_remaining)
                limit = min(limits)
                payload = {
                    "limit": limit,
                    "offset": offset,
                    "query": {"value": args.query, "fields": ["title", "body"]},
                    "filter": {
                        "field": f"date.{args.date_field}",
                        "value": {
                            "from": f"{year}-01-01T00:00:00+00:00",
                            "to": f"{year}-12-31T23:59:59+00:00",
                        },
                    },
                    "sort": [f"date.{args.date_field}:asc", "id:asc"],
                    "fields": {"include": FIELDS},
                }
                page = request_page(args.appname, payload)
                rows = page.get("data") or []
                if not rows:
                    break
                for row in rows:
                    rid = str(row.get("id", ""))
                    if not rid or rid in seen:
                        continue
                    record = {"id": rid, **(row.get("fields") or {})}
                    out.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                    seen.add(rid)
                    saved += 1
                out.flush()
                offset += len(rows)
                fetched_this_run += len(rows)
                offsets[year] = offset
                state_path.write_text(json.dumps(offsets, sort_keys=True), encoding="utf-8")
                total = int(page.get("totalCount") or 0)
                print(f"{year}: {min(offset, total):,}/{total:,}; saved total={saved:,}")
                if len(rows) < limit or offset >= total:
                    break


if __name__ == "__main__":
    main()
