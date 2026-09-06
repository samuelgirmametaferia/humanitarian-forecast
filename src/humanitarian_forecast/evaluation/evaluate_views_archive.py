#!/usr/bin/env python3
"""Causally evaluate archived VIEWS PRIO-grid forecasts on Ethiopia next events."""
from __future__ import annotations
import argparse,glob,json,re
from datetime import date,timedelta
from pathlib import Path
import numpy as np

EARTH_KM=111.32
RUN_RE=re.compile(r'_(\d{4})_(\d{2})_t\d+')

def month_id(dt: date)->int:return (dt.year-1980)*12+dt.month

def add_months_first(y:int,m:int,n:int)->date:
    z=y*12+(m-1)+n;return date(z//12,z%12+1,1)

def load_runs(paths):
    runs=[]
    for root in paths:
        for fn in glob.glob(str(Path(root)/'*.npz')):
            z=np.load(fn,allow_pickle=True);run=str(z['run_id']);match=RUN_RE.search(run)
            if not match:continue
            y,m=map(int,match.groups())
            # Conservative availability: information through M is treated as
            # usable only starting the first day two calendar months later.
            available=add_months_first(y,m,2)
            names=[str(v) for v in z['model_names']]
            runs.append({'run':run,'available':available,'month':(y,m),'pg_id':z['pg_id'].astype(np.int64),'months':z['month_id'].astype(np.int32),'centers':z['centroids'].astype(np.float32),'values':z['values'].astype(np.float32),'names':names})
    return sorted(runs,key=lambda r:r['available'])

def distance(cells,truth):
    north=(cells[:,0]-truth[0])*EARTH_KM;east=(cells[:,1]-truth[1])*EARTH_KM*np.cos(np.deg2rad(truth[0]));return np.sqrt(north*north+east*east)

def metrics(records,model):
    errors=[];masses={r:[] for r in (25,50,100,200)};ent=[];run_counts={}
    for rec in records:
        run=rec['run'];mid=rec['mid'];idx=np.flatnonzero(run['months']==mid)
        if not len(idx):continue
        try:j=run['names'].index(model)
        except ValueError:continue
        raw=np.nan_to_num(run['values'][idx,j].astype(np.float64),nan=0.,posinf=0.,neginf=0.)
        # All VIEWS outputs are used only as a relative spatial prior here.
        raw=np.maximum(raw,0.)
        if model.endswith('_ln'):
            raw=np.expm1(np.clip(raw,0,20))
        if raw.sum()<=0:continue
        p=raw/raw.sum();cells=run['centers'][idx];d=distance(cells,rec['truth']);e=float(d[np.argmax(p)]);errors.append(e);ent.append(float(-(p*np.log(np.maximum(p,1e-12))).sum()))
        for r in masses:masses[r].append(float(p[d<=r].sum()))
        run_counts[run['run']]=run_counts.get(run['run'],0)+1
    if not errors:return {'samples':0}
    e=np.asarray(errors);out={'samples':len(e),'mean_error_km':float(e.mean()),'median_error_km':float(np.median(e)),'p90_error_km':float(np.quantile(e,.9)),'mean_entropy':float(np.mean(ent)),'runs_used':run_counts}
    for r in masses:out[f'within_{r}km']=float(np.mean(e<=r));out[f'probability_mass_within_{r}km']=float(np.mean(masses[r]))
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--archive',action='append',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    runs=load_runs(a.archive);z=np.load(a.data,allow_pickle=True);rows=[json.loads(str(v)) for v in z['meta']];source=z['source_indices'].astype(np.int64);va=int(z['source_validation_end'])
    records=[]
    for i,row in enumerate(rows):
        target=date.fromisoformat(row['target_date']);cutoff=target-timedelta(days=int(row['gap_days']));eligible=[r for r in runs if r['available']<=cutoff]
        if not eligible:continue
        run=eligible[-1];records.append({'i':i,'run':run,'mid':month_id(target),'truth':np.asarray([row['target_lat'],row['target_lon']],np.float32),'cutoff':cutoff,'development':bool(source[i]>=va)})
    dev=[r for r in records if r['development']];pre=[r for r in records if not r['development']]
    models=sorted(set(n for r in runs for n in r['names']));report={'schema':'views-archive-next-event-evaluation-v1','availability_rule':'run information month M becomes eligible on first day of M+2 (conservative)','archives':a.archive,'runs':len(runs),'covered_examples':len(records),'predevelopment_covered':len(pre),'development_covered':len(dev),'models':{}}
    for model in models:report['models'][model]={'all_covered':metrics(records,model),'development':metrics(dev,model),'predevelopment':metrics(pre,model)}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
