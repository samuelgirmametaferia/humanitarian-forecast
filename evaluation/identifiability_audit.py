#!/usr/bin/env python3
"""Optimistic hindsight risks for repeated observable forecast states."""
from __future__ import annotations
import argparse,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from full_history_forecaster import haversine,summarize

def geometric_median(points,iterations=30):
    center=points.mean(0)
    for _ in range(iterations):
        distance=haversine(points,np.broadcast_to(center,points.shape));weight=1/np.maximum(distance,.1);updated=np.average(points,axis=0,weights=weight)
        if np.linalg.norm(updated-center)<1e-7:break
        center=updated
    return center
def risk(rows,key_fn):
    groups=defaultdict(list)
    for i,r in enumerate(rows):groups[key_fn(r)].append(i)
    errors=np.zeros(len(rows));sizes=np.zeros(len(rows),int)
    for indices in groups.values():
        points=np.asarray([[rows[i]['target_lat'],rows[i]['target_lon']] for i in indices]);center=geometric_median(points);value=haversine(points,np.broadcast_to(center,points.shape))
        errors[indices]=value;sizes[indices]=len(indices)
    result=summarize(errors);result.update({'groups':len(groups),'singleton_rate':float((sizes==1).mean()),'mean_group_size':float(sizes.mean())});return result
def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();d=np.load(a.data);n=len(d['meta']);start=int(.85*n);rows=[json.loads(str(v)) for v in d['meta'][start:]]
    specs={'conflict':lambda r:(r['conflict_id'],),'conflict_anchor_1deg':lambda r:(r['conflict_id'],round(r['anchor_lat']),round(r['anchor_lon'])),'conflict_anchor_exact':lambda r:(r['conflict_id'],round(r['anchor_lat'],5),round(r['anchor_lon'],5)),'conflict_anchor_exact_horizon_bucket':lambda r:(r['conflict_id'],round(r['anchor_lat'],5),round(r['anchor_lon'],5),min(4,int(np.log2(max(1,r['gap_days'])))))}
    report={'warning':'Uses test targets to fit centers; deliberately optimistic diagnostic, never a forecast','test_hindsight_risk':{name:risk(rows,fn) for name,fn in specs.items()}};a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
