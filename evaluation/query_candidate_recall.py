#!/usr/bin/env python3
"""Oracle error among the N historical candidates nearest a forecast query."""
from __future__ import annotations
import argparse,json
from collections import defaultdict
from pathlib import Path
import sys
import numpy as np,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from evaluate_candidate_snap import model_queries,absolute_query
from full_history_forecaster import haversine,summarize

def evaluate(rows,queries,seed_end,start,end,ks,scope):
    banks=defaultdict(list)
    field='conflict_id' if scope=='conflict' else 'country'
    for r in rows[:seed_end]:banks[r[field]].append(np.array([r['target_lat'],r['target_lon']]))
    errors={k:[] for k in ks};cursor=start
    while cursor<end:
        date=rows[cursor]['target_date'];stop=cursor
        while stop<end and rows[stop]['target_date']==date:stop+=1
        for i in range(cursor,stop):
            r=rows[i];target=np.array([r['target_lat'],r['target_lon']]);anchor=np.array([r['anchor_lat'],r['anchor_lon']])
            points=np.asarray([anchor,*banks[r[field]]]);qd=haversine(points,np.broadcast_to(queries[i],points.shape));td=haversine(points,np.broadcast_to(target,points.shape));order=np.argsort(qd)
            for k in ks:errors[k].append(float(td[order[:k]].min()))
        for r in rows[cursor:stop]:banks[r[field]].append(np.array([r['target_lat'],r['target_lon']]))
        cursor=stop
    return {str(k):summarize(np.asarray(v)) for k,v in errors.items()}
def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    d=np.load(a.data);rows=[json.loads(str(v)) for v in d['meta']];top,_=model_queries(d,a.checkpoint);hist=np.asarray([.95*r[r[:,0]>.5,1:3].mean(0) for r in d['x']]);off=.8*top+.2*hist;queries=np.asarray([absolute_query(r,o) for r,o in zip(rows,off)])
    n=len(rows);te=int(.7*n);ve=int(.85*n);ks=(1,2,4,8,16,32,64,128,256)
    report={scope:{'validation':evaluate(rows,queries,te,te,ve,ks,scope),'untouched_test':evaluate(rows,queries,ve,ve,n,ks,scope)} for scope in ('conflict','country')}
    report['interpretation']='retrospective candidate-set recall only'
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
