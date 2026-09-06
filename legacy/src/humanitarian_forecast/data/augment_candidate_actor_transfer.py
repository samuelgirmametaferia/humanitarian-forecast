#!/usr/bin/env python3
"""Append cross-conflict actor geography at exact, ~50 km, and ~100 km scales.

All counts are computed strictly before the sample cutoff and explicitly subtract
activity belonging to the current conflict ID.  This makes the features useful
for transferring actor geography into genuinely new conflicts instead of merely
duplicating the current conflict's own recurrence features.
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


def point_key(lat: float, lon: float) -> tuple[float, float]:
    return round(float(lat), 5), round(float(lon), 5)


def cell_key(lat: float, lon: float, scale: int) -> tuple[int, int]:
    return math.floor(float(lat) * scale), math.floor(float(lon) * scale)


def _add(index, key, day: int) -> None:
    index[key].append(day)


def load_indices(path: Path):
    conflict_actors: dict[str, tuple[str, str]] = {}
    actor_point = defaultdict(list)
    actor_half = defaultdict(list)
    actor_one = defaultdict(list)
    conflict_point = defaultdict(list)
    conflict_half = defaultdict(list)
    conflict_one = defaultdict(list)
    with zipfile.ZipFile(path) as archive:
        name = next(v for v in archive.namelist() if v.lower().endswith('.csv'))
        with archive.open(name) as raw, io.TextIOWrapper(raw, encoding='utf-8-sig') as text:
            for row in csv.DictReader(text):
                try:
                    conflict = str(row['conflict_new_id']).strip()
                    day = int(np.datetime64(row['date_start'][:10], 'D').astype(int))
                    lat = float(row['latitude']); lon = float(row['longitude'])
                except (KeyError, ValueError, TypeError):
                    continue
                a = str(row.get('side_a') or '').strip()
                b = str(row.get('side_b') or '').strip()
                if conflict and conflict not in conflict_actors:
                    conflict_actors[conflict] = (a, b)
                pk = point_key(lat, lon); h = cell_key(lat, lon, 2); o = cell_key(lat, lon, 1)
                _add(conflict_point, (conflict, pk), day)
                _add(conflict_half, (conflict, h), day)
                _add(conflict_one, (conflict, o), day)
                for actor in {a, b}:
                    if not actor: continue
                    _add(actor_point, (actor, pk), day)
                    _add(actor_half, (actor, h), day)
                    _add(actor_one, (actor, o), day)
    for index in (actor_point, actor_half, actor_one, conflict_point, conflict_half, conflict_one):
        for days in index.values(): days.sort()
    return conflict_actors, actor_point, actor_half, actor_one, conflict_point, conflict_half, conflict_one


def count_before(days: list[int], cutoff: int, window: int | None = None) -> int:
    right = bisect.bisect_right(days, cutoff)
    if window is None: return right
    return right - bisect.bisect_left(days, cutoff - window, 0, right)


def other_counts(actor_index, conflict_index, actor: str, conflict: str, key, cutoff: int) -> tuple[int,int,int,int]:
    if not actor: return 0,0,0,0
    ad = actor_index.get((actor, key), [])
    cd = conflict_index.get((conflict, key), [])
    result=[]
    for window in (None, 30, 90, 365):
        result.append(max(0, count_before(ad, cutoff, window) - count_before(cd, cutoff, window)))
    return tuple(result)


def encode(counts: tuple[int,int,int,int]) -> list[float]:
    life,c30,c90,c365=counts
    return [math.log1p(life)/6.0, math.log1p(c30)/4.0, math.log1p(c90)/4.0, math.log1p(c365)/4.0]


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate-data',type=Path,required=True)
    p.add_argument('--ucdp-events',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists(): raise FileExistsError(f'refusing to overwrite: {args.output}')
    z=np.load(args.candidate_data)
    base=z['candidate_features'].astype(np.float32,copy=False);valid=z['candidate_valid']
    meta=[json.loads(str(q)) for q in z['meta']]
    idx=load_indices(args.ucdp_events)
    conflict_actors,ap,ah,ao,cp,ch,co=idx
    n,k,_=base.shape
    # Each actor: 4 exact + 4 half-degree + 4 one-degree = 12. Plus 8 symmetric summaries.
    extra=np.zeros((n,k,32),np.float32)
    for i,row in enumerate(meta):
        conflict=str(row['conflict_id']); actor_a,actor_b=conflict_actors.get(conflict,('',''))
        target_day=int(np.datetime64(row['target_date'],'D').astype(int)); cutoff=target_day-int(row['gap_days'])
        for j in np.flatnonzero(valid[i]):
            lat=float(base[i,j,6])*90.;lon=float(base[i,j,7])*180.
            pk=point_key(lat,lon); hk=cell_key(lat,lon,2); ok=cell_key(lat,lon,1)
            a_exact=other_counts(ap,cp,actor_a,conflict,pk,cutoff); b_exact=other_counts(ap,cp,actor_b,conflict,pk,cutoff)
            a_half=other_counts(ah,ch,actor_a,conflict,hk,cutoff); b_half=other_counts(ah,ch,actor_b,conflict,hk,cutoff)
            a_one=other_counts(ao,co,actor_a,conflict,ok,cutoff); b_one=other_counts(ao,co,actor_b,conflict,ok,cutoff)
            ea=encode(a_exact)+encode(a_half)+encode(a_one)
            eb=encode(b_exact)+encode(b_half)+encode(b_one)
            # Symmetric summaries focus on broad transfer recurrence.
            combo=[
                max(ea[0],eb[0]), ea[0]+eb[0],
                max(ea[6],eb[6]), ea[6]+eb[6],      # ~50km 90d
                max(ea[7],eb[7]), ea[7]+eb[7],      # ~50km 365d
                max(ea[10],eb[10]), ea[10]+eb[10],  # ~100km 90d
            ]
            extra[i,j]=np.asarray([*ea,*eb,*combo],np.float32)
        if i and i%20000==0: print(f'actor transfer {i:,}/{n:,}',flush=True)
    names=[]
    for side in ('side_a','side_b'):
        for scale in ('exact','cell50km','cell100km'):
            for stat in ('lifetime','count30','count90','count365'):
                names.append(f'{side}_other_conflict_{scale}_{stat}')
    names += ['actor_max_exact_lifetime','actor_sum_exact_lifetime','actor_max_50km_count90','actor_sum_50km_count90','actor_max_50km_count365','actor_sum_50km_count365','actor_max_100km_count90','actor_sum_100km_count90']
    augmented=np.concatenate([base,extra],axis=-1)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,x=z['x'],candidate_features=augmented,candidate_coordinates=z['candidate_coordinates'],candidate_valid=valid,label=z['label'],y=z['y'],meta=z['meta'],base_candidate_dim=np.asarray(base.shape[-1]),appended_feature_names=np.asarray(names),context_lineage=np.asarray(json.dumps({'schema':'candidate-actor-transfer/v2','source':str(args.ucdp_events),'current_conflict_activity_subtracted':True,'cell_scales_degrees':[0.5,1.0]},sort_keys=True)))
    active=(extra[:,:,24:] > 0).any(-1)&valid
    print(json.dumps({'samples':n,'candidate_dim_before':int(base.shape[-1]),'candidate_dim_after':int(augmented.shape[-1]),'valid_candidates':int(valid.sum()),'valid_candidates_with_cross_conflict_actor_transfer':int(active.sum()),'output':str(args.output)},indent=2))

if __name__=='__main__':main()
