"""Cutoff-safe Ethiopia place mentions from ReliefWeb reports.

Deterministic, auditable mention extraction using the repository's historical
gazetteer rather than an online geocoder. Every mention is timestamped by
publication time (``date.created``, falling back to ``date.original`` only
when created is absent) so no later-published information can leak into a
historical feature window. Ported from the legacy
``build_reliefweb_spatial_mentions`` recipe with the same matching semantics.

The output is a sparse set of ``(day, lat, lon)`` mention rows. Downstream
models aggregate them into space/time windows; the mention channel is also
where extracted public Telegram web-preview signals enter as model input
signals, never as outcome labels.
"""

from __future__ import annotations

import gzip
import json
import re
from pathlib import Path
from typing import Any, Iterator

import numpy as np

TOKEN_RE = re.compile(r"[a-z0-9]+")


def norm_tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.casefold())


def build_aliases(path: Path, min_observations: int = 2) -> tuple[dict[tuple[str, ...], dict[str, Any]], int]:
    """Load the gazetteer into a token-tuple -> place lookup, longest alias wins."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    aliases: dict[tuple[str, ...], dict[str, Any]] = {}
    max_words = 1
    for alias, value in raw.items():
        tokens = tuple(norm_tokens(alias))
        if not tokens or len(" ".join(tokens)) < 4:
            continue
        # One-off aliases are retained only when sufficiently specific; repeated
        # historical names are much safer for short-token matching.
        if int(value.get("observation_count", 0)) < min_observations and len(" ".join(tokens)) < 8:
            continue
        aliases[tokens] = value
        max_words = max(max_words, len(tokens))
    return aliases, max_words


def iter_reliefweb_rows(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as stream:
        for line in stream:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def report_countries(row: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    primary = row.get("primary_country") or {}
    if isinstance(primary, dict):
        out.add(str(primary.get("name", "")).casefold())
    for value in row.get("country") or []:
        if isinstance(value, dict):
            out.add(str(value.get("name", "")).casefold())
    return out


def report_day(row: dict[str, Any]) -> int | None:
    """Publication day as a numpy datetime64 day integer (causal contract)."""
    dates = row.get("date") or {}
    stamp = dates.get("created") or dates.get("original")
    if not stamp:
        return None
    try:
        return int(np.datetime64(str(stamp)[:10], "D").astype(int))
    except ValueError:
        return None


def extract_mentions(
    rows: list[dict[str, Any]],
    aliases: dict[tuple[str, ...], dict[str, Any]],
    max_words: int,
    *,
    country: str = "Ethiopia",
    max_body_chars: int = 12000,
) -> list[tuple[int, float, float]]:
    """Geocode place mentions per report; returns (day, lat, lon) rows."""
    name = country.casefold()
    mentions: list[tuple[int, float, float]] = []
    for row in rows:
        if name not in report_countries(row):
            continue
        day = report_day(row)
        if day is None:
            continue
        text = f"{row.get('title', '')} {(row.get('body') or '')[:max_body_chars]}"
        tokens = norm_tokens(text)
        # Longest-first n-gram lookup; one mention per unique place per report.
        found: dict[tuple[float, float], tuple[float, float]] = {}
        occupied: set[int] = set()
        for width in range(min(max_words, len(tokens)), 0, -1):
            for start in range(0, len(tokens) - width + 1):
                if any(j in occupied for j in range(start, start + width)):
                    continue
                place = aliases.get(tuple(tokens[start : start + width]))
                if place is None:
                    continue
                for j in range(start, start + width):
                    occupied.add(j)
                latitude = float(place.get("latitude_mean", 0) or 0)
                longitude = float(place.get("longitude_mean", 0) or 0)
                if latitude or longitude:
                    found[(round(latitude, 5), round(longitude, 5))] = (latitude, longitude)
        mentions.extend((day, latitude, longitude) for latitude, longitude in found.values())
    return mentions
