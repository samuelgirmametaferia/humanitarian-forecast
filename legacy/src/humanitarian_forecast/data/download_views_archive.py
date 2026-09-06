#!/usr/bin/env python3
"""Probe and cache monthly immutable VIEWS production runs for one country."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date
from pathlib import Path


def months(start: str, end: str):
    sy,sm=map(int,start.split('-'));ey,em=map(int,end.split('-'))
    y,m=sy,sm
    while (y,m)<=(ey,em):
        yield y,m
        m+=1
        if m==13:y+=1;m=1


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',default='fatalities003');p.add_argument('--start',default='2025-10');p.add_argument('--end',default='2026-06');p.add_argument('--iso',default='ETH');p.add_argument('--violence',default='sb');p.add_argument('--output-dir',type=Path,default=Path('data/geo/raw/views/archive'));a=p.parse_args()
    a.output_dir.mkdir(parents=True,exist_ok=True);entries=[]
    for y,m in months(a.start,a.end):
        run=f'{a.model}_{y:04d}_{m:02d}_t01';out=a.output_dir/f'{run}_{a.iso}_{a.violence}.npz'
        if not out.exists():
            cmd=[sys.executable,'-m','humanitarian_forecast.data.download_views_prior','--run',run,'--iso',a.iso,'--violence',a.violence,'--output',str(out)]
            print(' '.join(cmd),flush=True)
            result=subprocess.run(cmd)
            if result.returncode!=0:
                entries.append({'run_id':run,'status':'unavailable'});continue
        manifest=json.loads(out.with_suffix('.json').read_text());entries.append({'run_id':run,'status':'cached','rows':manifest['rows'],'month_id_min':manifest['month_id_min'],'month_id_max':manifest['month_id_max'],'sha256':manifest['sha256']})
    payload={'schema':'views-archive-v1','model':a.model,'iso':a.iso,'violence':a.violence,'runs':entries,'causality_rule':'For forecast cutoff YYYY-MM, use only a production run whose information month is <= cutoff; never backfill with a later run.'}
    (a.output_dir/f'manifest_{a.model}_{a.iso}_{a.violence}.json').write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload,indent=2))
if __name__=='__main__':main()
