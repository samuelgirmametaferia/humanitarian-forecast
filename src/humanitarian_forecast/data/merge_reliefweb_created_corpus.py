#!/usr/bin/env python3
"""Merge publication-time ReliefWeb shards into one deduplicated causal corpus.

The merge contract is intentionally strict:
- records are assigned to years by date.created (date.original is never used),
- report IDs are globally deduplicated,
- the legacy/serial source may be restricted to a year range so partial later
  years cannot contaminate complete per-year shards,
- output is sorted by (date.created, id), and a manifest records per-year counts
  and the SHA-256 of the final gzip payload.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path


def created_year(row: dict) -> int | None:
    stamp = ((row.get("date") or {}).get("created"))
    if not stamp or len(stamp) < 4:
        return None
    try:
        return int(stamp[:4])
    except ValueError:
        return None


def load_file(path: Path, allowed_years: set[int] | None, rows: dict[str, dict], counts: Counter, source_counts: Counter, *, tolerate_truncated: bool = False) -> bool:
    """Load one gzip member and return whether its stream ended truncated."""
    truncated = False
    stream = gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    try:
        while True:
            try:
                line = stream.readline()
            except EOFError:
                if not tolerate_truncated:
                    raise
                truncated = True
                break
            if not line:
                break
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = str(row.get("id", ""))
            year = created_year(row)
            if not rid or year is None:
                continue
            if allowed_years is not None and year not in allowed_years:
                continue
            previous = rows.get(rid)
            if previous is None:
                rows[rid] = row
                counts[year] += 1
                source_counts[str(path)] += 1
                continue
            old_body = str(previous.get("body") or "")
            new_body = str(row.get("body") or "")
            if len(new_body) > len(old_body):
                rows[rid] = row
    finally:
        stream.close()
    return truncated


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--serial-source", type=Path, required=True, help="Corrected serial corpus; normally used only for complete early years.")
    p.add_argument("--serial-through-year", type=int, default=2008)
    p.add_argument("--shard-dir", type=Path, required=True)
    p.add_argument("--start-year", type=int, default=2000)
    p.add_argument("--end-year", type=int, default=2026)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.start_year > args.end_year:
        p.error("--start-year must be <= --end-year")

    rows: dict[str, dict] = {}
    counts: Counter[int] = Counter()
    source_counts: Counter[str] = Counter()
    early = set(range(args.start_year, min(args.serial_through_year, args.end_year) + 1))
    serial_truncated = load_file(args.serial_source, early, rows, counts, source_counts, tolerate_truncated=True)
    truncated_shards: list[int] = []
    for year in range(max(args.start_year, args.serial_through_year + 1), args.end_year + 1):
        shard = args.shard_dir / f"{year}.jsonl.gz"
        if not shard.exists():
            raise FileNotFoundError(f"missing ReliefWeb shard for publication year {year}: {shard}")
        if load_file(shard, {year}, rows, counts, source_counts, tolerate_truncated=False):
            truncated_shards.append(year)

    ordered = sorted(rows.values(), key=lambda r: (((r.get("date") or {}).get("created") or ""), str(r.get("id", ""))))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output, "wt", encoding="utf-8", compresslevel=6) as out:
        for row in ordered:
            out.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    sha = hashlib.sha256(args.output.read_bytes()).hexdigest()
    missing = [y for y in range(args.start_year, args.end_year + 1) if counts[y] == 0]
    manifest = {
        "schema": "reliefweb-created-corpus-v1",
        "causality": "all partitioning and ordering use date.created only",
        "serial_source": str(args.serial_source),
        "serial_through_year": args.serial_through_year,
        "serial_source_truncated_but_tolerated": serial_truncated,
        "truncated_shards": truncated_shards,
        "shard_dir": str(args.shard_dir),
        "start_year": args.start_year,
        "end_year": args.end_year,
        "reports": len(ordered),
        "per_year": {str(y): int(counts[y]) for y in range(args.start_year, args.end_year + 1)},
        "missing_years": missing,
        "source_rows_accepted": dict(sorted(source_counts.items())),
        "sha256": sha,
        "output": str(args.output),
    }
    args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
