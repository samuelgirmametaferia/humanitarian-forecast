#!/usr/bin/env python3
"""Forecast and oracle recall for the most frequent conflict locations."""
from __future__ import annotations
import argparse,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from full_history_forecaster import haversine,summarize

def evaluate(rows,seed_end,start,end,ks):
    banks=defaultdict(lambda:defaultdict(lambda:[0,None]));out={k:[] for k in ks};top=[]
    def add(r):
        key=(round(r['target_lat'],5),round(r['target_lon'],5));v=banks[r['conflict_id']][key];v[0]+=1;v[1]=np.array([r['target_lat'],r['target_lon']])
    for r in rows[:seed_end]:add(r)
    cursor=start
    while cursor<end:
        date=rows[cursor]['target_date'];stop=cursor
        while stop<end and rows[stop]['target_date']==date:stop+=1
        for r in rows[cursor:stop]:
            target=np.array([r['target_lat'],r['target_lon']]);anchor=np.array([r['anchor_lat'],r['anchor_lon']]);ranked=sorted(banks[r['conflict_id']].values(),key=lambda v:v[0],reverse=True);points=np.asarray([anchor,*[v[1] for v in ranked]])
            dist=haversine(points,np.broadcast_to(target,points.shape));top.append(float(dist[0 if len(ranked)==0 else 1]))
            for k in ks:out[k].append(float(dist[:k+1].min()))
        for r in rows[cursor:stop]:add(r)
        cursor=stop
    return {'top1_historical_mode':summarize(np.asarray(top)),'oracle':{str(k):summarize(np.asarray(v)) for k,v in out.items()}}
def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();d=np.load(a.data);rows=[json.loads(str(v)) for v in d['meta']];n=len(rows);te=int(.7*n);ve=int(.85*n);ks=(1,2,4,8,16,32,64,128,256)
    report={'validation':evaluate(rows,te,te,ve,ks),'untouched_test':evaluate(rows,ve,ve,n,ks)};a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
