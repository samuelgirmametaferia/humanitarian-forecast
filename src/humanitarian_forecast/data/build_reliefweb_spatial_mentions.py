#!/usr/bin/env python3
"""Extract cutoff-safe Ethiopia place mentions from ReliefWeb reports.

This is deliberately deterministic and auditable: it uses the repository's
historical gazetteer rather than an online geocoder, and timestamps every mention
by publication time (date.created). Output is sparse; downstream models may build
space/time kernels without materializing an N x H3 x feature tensor.
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

TOKEN_RE = re.compile(r"[a-z0-9]+")
LEXICONS = {
    "attack": ("attack","clash","fight","fighting","shell","airstrike","drone","ambush","raid","offensive"),
    "harm": ("kill","killed","death","dead","casualt","wound","injur","massacre"),
    "displacement": ("displac","evacuat","flee","fled","refugee","returnee"),
    "territory": ("capture","captured","control","advance","retreat","frontline","siege","encircle"),
    "mobilization": ("mobiliz","reinforcement","troops","fighters","militia","armed group","forces"),
    "infrastructure": ("road","bridge","hospital","school","water","electric","infrastructure","airport"),
    "peace": ("ceasefire","truce","agreement","negotiat","peace talk","disarm","demobil"),
}


def norm_tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.casefold())


def build_aliases(path: Path, min_observations: int) -> tuple[dict[tuple[str,...], dict], int]:
    raw=json.loads(path.read_text());aliases={};max_words=1
    for alias,value in raw.items():
        tokens=tuple(norm_tokens(alias))
        if not tokens or len(" ".join(tokens))<4:continue
        # One-off aliases are retained only when sufficiently specific; repeated
        # historical names are much safer for short-token matching.
        if int(value.get('observation_count',0))<min_observations and len(" ".join(tokens))<8:continue
        aliases[tokens]=value;max_words=max(max_words,len(tokens))
    return aliases,max_words


def countries(row: dict) -> set[str]:
    out=set()
    primary=row.get('primary_country') or {}
    if isinstance(primary,dict):out.add(str(primary.get('name','')).casefold())
    for value in row.get('country') or []:
        if isinstance(value,dict):out.add(str(value.get('name','')).casefold())
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reliefweb',type=Path,required=True);p.add_argument('--gazetteer',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--country',default='Ethiopia');p.add_argument('--min-observations',type=int,default=2);p.add_argument('--max-body-chars',type=int,default=12000);a=p.parse_args()
    aliases,max_words=build_aliases(a.gazetteer,a.min_observations);country=a.country.casefold()
    days=[];latlon=[];features=[];report_ids=[];names=[];created_rows=0;fallback_original=0;reports=0;matched_reports=0
    with gzip.open(a.reliefweb,'rt',encoding='utf-8',errors='ignore') as f:
        for line in f:
            row=json.loads(line)
            if country not in countries(row):continue
            dates=row.get('date') or {};created=dates.get('created')
            stamp=created or dates.get('original')
            if not stamp:continue
            if created:created_rows+=1
            else:fallback_original+=1
            try:day=int(np.datetime64(stamp[:10],'D').astype(int))
            except ValueError:continue
            text=f"{row.get('title','')} {(row.get('body') or '')[:a.max_body_chars]}";lower=text.casefold();tokens=norm_tokens(text);reports+=1
            semantic=np.asarray([sum(lower.count(term) for term in terms) for terms in LEXICONS.values()],np.float32)
            found={}
            # Longest-first n-gram lookup suppresses duplicate aliases such as
            # 'abi adi' and 'abi adi town' within the same report/location.
            occupied=set()
            for width in range(min(max_words,len(tokens)),0,-1):
                for i in range(0,len(tokens)-width+1):
                    if any(j in occupied for j in range(i,i+width)):continue
                    value=aliases.get(tuple(tokens[i:i+width]))
                    if value is None:continue
                    key=(round(float(value['latitude_mean']),5),round(float(value['longitude_mean']),5))
                    found[key]=value;occupied.update(range(i,i+width))
            if not found:continue
            matched_reports+=1
            for value in found.values():
                days.append(day);latlon.append([float(value['latitude_mean']),float(value['longitude_mean'])]);features.append(np.concatenate([[1.0],np.log1p(semantic),[float(value.get('modal_share',1.0)),math_log1p(value.get('observation_count',0))]]).astype(np.float32));report_ids.append(str(row.get('id','')));names.append(str(value.get('canonical_name','')))
    if not days:raise RuntimeError('no spatial ReliefWeb mentions found')
    order=np.argsort(np.asarray(days),kind='stable');feature_names=['mention_count',*[f'log_{k}' for k in LEXICONS],'gazetteer_modal_share','log_gazetteer_observations']
    a.output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.output,day=np.asarray(days,np.int32)[order],latlon=np.asarray(latlon,np.float32)[order],features=np.asarray(features,np.float32)[order],feature_names=np.asarray(feature_names),report_id=np.asarray(report_ids)[order],place_name=np.asarray(names)[order])
    manifest={'schema':'reliefweb-spatial-mentions-v1','source':str(a.reliefweb),'gazetteer':str(a.gazetteer),'reports_considered':reports,'matched_reports':matched_reports,'mentions':len(days),'aliases':len(aliases),'date_created_rows':created_rows,'date_original_fallback_rows':fallback_original,'causality':'mentions are timestamped by publication time; original date is fallback only when created is absent'};a.output.with_suffix('.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest,indent=2))

def math_log1p(value):
    return float(np.log1p(float(value or 0)))

if __name__=='__main__':main()
