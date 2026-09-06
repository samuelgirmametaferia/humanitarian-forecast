#!/usr/bin/env python3
"""Append cutoff-safe cross-conflict actor recurrence features to candidate data.

The purpose is transfer to unseen conflict IDs.  If an actor has appeared in a
previous conflict, its historical geographic recurrence can inform a new
conflict without using any event after the forecast cutoff.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import io
import json
import math
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np


def pkey(lat: float, lon: float) -> tuple[float, float]:
    return round(float(lat), 5), round(float(lon), 5)


def load_actor_index(path: Path):
    conflict_actors: dict[str, tuple[str, str]] = {}
    raw: dict[tuple[str, tuple[float, float]], list[tuple[int, float]]] = defaultdict(list)
    with zipfile.ZipFile(path) as archive:
        name = next(value for value in archive.namelist() if value.lower().endswith('.csv'))
        with archive.open(name) as binary, io.TextIOWrapper(binary, encoding='utf-8-sig') as text:
            for row in csv.DictReader(text):
                try:
                    conflict = str(row['conflict_new_id']).strip()
                    day = int(np.datetime64(row['date_start'][:10], 'D').astype(int))
                    point = pkey(float(row['latitude']), float(row['longitude']))
                    fatalities = max(0.0, float(row.get('best') or 0.0))
                except (KeyError, ValueError, TypeError):
                    continue
                side_a = str(row.get('side_a') or '').strip()
                side_b = str(row.get('side_b') or '').strip()
                if conflict and conflict not in conflict_actors:
                    conflict_actors[conflict] = (side_a, side_b)
                for actor in {side_a, side_b}:
                    if actor:
                        raw[(actor, point)].append((day, fatalities))
    index = {}
    for key, values in raw.items():
        values.sort()
        days = [d for d, _ in values]
        prefix = [0.0]
        for _, fatalities in values:
            prefix.append(prefix[-1] + fatalities)
        index[key] = (days, prefix)
    return conflict_actors, index


def actor_point_features(index, actor: str, point, cutoff: int) -> list[float]:
    if not actor:
        return [0.0] * 7
    days, prefix = index.get((actor, point), ([], [0.0]))
    right = bisect.bisect_right(days, cutoff)
    if right == 0:
        return [0.0] * 7
    age = max(1, cutoff - days[right - 1])
    counts = []
    for window in (30, 90, 365):
        left = bisect.bisect_left(days, cutoff - window)
        counts.append(right - left)
    left90 = bisect.bisect_left(days, cutoff - 90)
    fatal90 = prefix[right] - prefix[left90]
    lifetime = right
    # Transform scales mirror the existing candidate feature magnitudes.
    return [
        math.log1p(lifetime) / 6.0,
        math.log1p(age) / 6.0,
        math.log1p(counts[0]) / 4.0,
        math.log1p(counts[1]) / 4.0,
        math.log1p(counts[2]) / 4.0,
        math.log1p(fatal90) / 6.0,
        1.0,
    ]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate-data', type=Path, required=True)
    p.add_argument('--ucdp-events', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(f'refusing to overwrite: {args.output}')

    z = np.load(args.candidate_data)
    features = z['candidate_features'].astype(np.float32, copy=False)
    valid = z['candidate_valid']
    meta = [json.loads(str(value)) for value in z['meta']]
    conflict_actors, actor_index = load_actor_index(args.ucdp_events)
    n, k, _ = features.shape
    # side A (7), side B (7), plus six symmetric combination features.
    actor_features = np.zeros((n, k, 20), dtype=np.float32)

    for i, row in enumerate(meta):
        conflict = str(row['conflict_id'])
        side_a, side_b = conflict_actors.get(conflict, ('', ''))
        target_day = int(np.datetime64(row['target_date'], 'D').astype(int))
        cutoff = target_day - int(row['gap_days'])
        for j in np.flatnonzero(valid[i]):
            lat = float(features[i, j, 6]) * 90.0
            lon = float(features[i, j, 7]) * 180.0
            point = pkey(lat, lon)
            fa = actor_point_features(actor_index, side_a, point, cutoff)
            fb = actor_point_features(actor_index, side_b, point, cutoff)
            combined = [
                max(fa[0], fb[0]),                 # strongest lifetime recurrence
                fa[0] + fb[0],                    # total actor recurrence
                max(fa[2], fb[2]),                 # strongest 30d recurrence
                max(fa[3], fb[3]),                 # strongest 90d recurrence
                max(fa[4], fb[4]),                 # strongest 365d recurrence
                float(fa[6] > 0 and fb[6] > 0),    # both actors seen here before
            ]
            actor_features[i, j] = np.asarray([*fa, *fb, *combined], dtype=np.float32)
        if i and i % 20000 == 0:
            print(f'actor context {i:,}/{n:,}', flush=True)

    names = np.asarray([
        'side_a_lifetime_at_candidate','side_a_recency_at_candidate','side_a_count30_at_candidate',
        'side_a_count90_at_candidate','side_a_count365_at_candidate','side_a_fatal90_at_candidate','side_a_seen_candidate',
        'side_b_lifetime_at_candidate','side_b_recency_at_candidate','side_b_count30_at_candidate',
        'side_b_count90_at_candidate','side_b_count365_at_candidate','side_b_fatal90_at_candidate','side_b_seen_candidate',
        'actor_max_lifetime_at_candidate','actor_sum_lifetime_at_candidate','actor_max_count30_at_candidate',
        'actor_max_count90_at_candidate','actor_max_count365_at_candidate','both_actors_seen_candidate',
    ])
    augmented = np.concatenate([features, actor_features], axis=-1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        x=z['x'], candidate_features=augmented,
        candidate_coordinates=z['candidate_coordinates'], candidate_valid=valid,
        label=z['label'], y=z['y'], meta=z['meta'],
        base_candidate_dim=np.asarray(features.shape[-1]),
        appended_feature_names=names,
        context_lineage=np.asarray(json.dumps({
            'schema':'candidate-actor-transfer/v1',
            'source':str(args.ucdp_events),
            'cutoff':'target_day - gap_days (same pre-target event cutoff as base candidate builder)',
            'purpose':'cross-conflict actor geographic recurrence for unseen conflict transfer',
        }, sort_keys=True)),
    )
    # Coverage summary, especially useful for detecting whether actor transfer can
    # help new conflict IDs at all.
    side_b_seen = actor_features[:, :, 13]
    print(json.dumps({
        'samples': n,
        'candidate_dim_before': int(features.shape[-1]),
        'candidate_dim_after': int(augmented.shape[-1]),
        'valid_candidates': int(valid.sum()),
        'valid_with_side_a_memory': int(((actor_features[:,:,6] > 0) & valid).sum()),
        'valid_with_side_b_memory': int(((side_b_seen > 0) & valid).sum()),
        'output': str(args.output),
    }, indent=2))

if __name__ == '__main__':
    main()
