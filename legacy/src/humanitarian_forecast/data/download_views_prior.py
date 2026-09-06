#!/usr/bin/env python3
"""Download one immutable VIEWS production forecast prior from the public API.

The run id is mandatory. Historical experiments must pass the historical run that
actually existed at the forecast cutoff; ``current`` is intentionally rejected by
default to prevent accidental retrospective substitution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

BASE = "https://api.viewsforecasting.org"


def priogrid_centroid(gid: int) -> tuple[float, float]:
    zero = int(gid) - 1
    row, col = divmod(zero, 720)
    return -89.75 + 0.5 * row, -179.75 + 0.5 * col


def fetch_json(url: str) -> dict[str, object]:
    request = urllib.request.Request(url, headers={"User-Agent": "humanitarian-forecast/0.1 research"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Immutable VIEWS run id, e.g. fatalities003_2026_06_t01")
    parser.add_argument("--iso", default="ETH")
    parser.add_argument("--violence", choices=("sb", "ns", "os"), default="sb")
    parser.add_argument("--date-start")
    parser.add_argument("--date-end")
    parser.add_argument("--pagesize", type=int, default=10000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-current", action="store_true")
    args = parser.parse_args()
    if args.run == "current" and not args.allow_current:
        raise ValueError("refusing non-immutable run='current'; pass an explicit run id or --allow-current")

    params: list[tuple[str, str]] = [("iso", args.iso), ("pagesize", str(args.pagesize))]
    if args.date_start:
        params.append(("date_start", args.date_start))
    if args.date_end:
        params.append(("date_end", args.date_end))
    root = f"{BASE}/{args.run}/pgm/{args.violence}?{urllib.parse.urlencode(params)}"
    page_url = root
    data_rows: list[dict[str, object]] = []
    models: list[str] = []
    model_tree: list[dict[str, object]] = []
    page_count = None
    while page_url:
        payload = fetch_json(page_url)
        if not models:
            models = [str(value) for value in payload.get("models", [])]
            model_tree = list(payload.get("model_tree", []))
            page_count = int(payload.get("page_count", 1))
        data_rows.extend(payload.get("data", []))
        page_url = str(payload.get("next_page") or "")
        print(f"downloaded rows={len(data_rows):,} pages={payload.get('page_cur')}/{page_count}", flush=True)

    if not data_rows:
        raise RuntimeError("VIEWS query returned no rows")
    gids = np.asarray([int(row["pg_id"]) for row in data_rows], dtype=np.int64)
    months = np.asarray([int(row["month_id"]) for row in data_rows], dtype=np.int32)
    centers = np.asarray([priogrid_centroid(int(gid)) for gid in gids], dtype=np.float32)
    values = np.full((len(data_rows), len(models)), np.nan, dtype=np.float32)
    for i, row in enumerate(data_rows):
        for j, model in enumerate(models):
            value = row.get(model)
            if value is not None:
                values[i, j] = float(value)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        pg_id=gids,
        month_id=months,
        centroids=centers,
        values=values,
        model_names=np.asarray(models),
        run_id=np.asarray(args.run),
        iso=np.asarray(args.iso),
        violence=np.asarray(args.violence),
    )
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    manifest = {
        "schema": "views-prior-v1",
        "provider": "VIEWS / Violence & Impacts Early-Warning System",
        "run_id": args.run,
        "iso": args.iso,
        "violence": args.violence,
        "query": root,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "rows": len(data_rows),
        "models": models,
        "model_tree": model_tree,
        "month_id_min": int(months.min()),
        "month_id_max": int(months.max()),
        "sha256": digest,
        "causality_rule": "Historical features must use the production run that existed by that historical cutoff; never replace it with a later run.",
    }
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
