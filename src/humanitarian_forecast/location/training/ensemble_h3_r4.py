#!/usr/bin/env python3
"""Validation-only convex ensemble of dense r4 probability experts."""
from __future__ import annotations
import argparse,itertools,json
from pathlib import Path
import numpy as np
from humanitarian_forecast.core.model_store import ModelInfo,write_info
from humanitarian_forecast.location.training import train_h3_lambdarank as rank

def distances(cells,truth):
    out=np.empty((len(truth),len(cells)),np.float32)
    for i,t in enumerate(truth):out[i]=rank._distance(cells,t)
    return out

def metrics(p,d):return rank._metrics_from_scores(np.log(np.maximum(p,1e-12)),d,1.0)[0]
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--expert',action='append',required=True,help='name=probabilities.npz');p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--step',type=float,default=.05);a=p.parse_args()
    if a.output_dir.exists() and any(a.output_dir.iterdir()):raise FileExistsError(a.output_dir)
    z=np.load(a.data,allow_pickle=True);rows=[json.loads(str(v)) for v in z['meta']];source=z['source_indices'].astype(np.int64);tr=int(z['source_train_end']);va=int(z['source_validation_end']);vi=np.flatnonzero((source>=tr)&(source<va));di=np.flatnonzero(source>=va);truth=np.asarray([[r['target_lat'],r['target_lon']] for r in rows],np.float32);cells=z['centroids_r4'].astype(np.float32);dv=distances(cells,truth[vi]);dd=distances(cells,truth[di]);names=[];pv=[];pd=[]
    for spec in a.expert:
        name,path=spec.split('=',1);q=np.load(path);names.append(name);pv.append(q['validation'].astype(np.float64));pd.append(q['development'].astype(np.float64))
    pv=np.stack(pv);pd=np.stack(pd);units=int(round(1/a.step));best=None;trials=[]
    # Integer simplex enumeration avoids floating point sum drift.
    def comps(total,n,prefix=()):
        if n==1:yield prefix+(total,);return
        for i in range(total+1):yield from comps(total-i,n-1,prefix+(i,))
    for parts in comps(units,len(names)):
        w=np.asarray(parts,np.float64)/units;q=np.tensordot(w,pv,axes=(0,0));q/=q.sum(axis=1,keepdims=True);rep=metrics(q,dv);trials.append({'weights':dict(zip(names,map(float,w))),'validation_broad_area_score':rep['broad_area_score']})
        if best is None or rep['broad_area_score']>best[0]:best=(rep['broad_area_score'],w,rep,q)
    _,w,validation,vprob=best;dprob=np.tensordot(w,pd,axes=(0,0));dprob/=dprob.sum(axis=1,keepdims=True);development=metrics(dprob,dd);weights=dict(zip(names,map(float,w)));a.output_dir.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.output_dir/'probabilities.npz',validation=vprob.astype(np.float32),development=dprob.astype(np.float32),cells=cells);report={'schema':'h3-r4-ensemble-v1','experts':names,'weights':weights,'validation':validation,'development':development,'step':a.step,'protocol':'convex weights selected on validation broad-area score only; development diagnostic'};(a.output_dir/'metrics.json').write_text(json.dumps(report,indent=2)+'\n');write_info(a.output_dir,ModelInfo(subsystem='location/h3_r4_ensemble',version='v1',status='research-challenger',description='Validation-selected convex ensemble of dense r4 Ethiopia experts.',metrics={'validation':validation,'development':development},lineage={'experts':a.expert,'restore_tag':'production-boost-preflight-2026-08-24'},calibration={'weights':weights,'selection':'validation broad-area score'},notes=['All experts share the same r4 H3 support.','Development remains diagnostic.']));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
