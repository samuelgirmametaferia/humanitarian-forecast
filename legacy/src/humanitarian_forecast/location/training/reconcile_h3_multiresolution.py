#!/usr/bin/env python3
"""Hierarchically reconcile dense H3 r5 probabilities to an r4 parent forecast."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import h3
import numpy as np
from humanitarian_forecast.location.training import train_h3_lambdarank as rank


def distances(cells,truth):
    out=np.empty((len(truth),len(cells)),np.float32)
    for i,t in enumerate(truth):out[i]=rank._distance(cells,t)
    return out

def reconcile(p5,p4,parent_index,strength):
    eps=1e-12
    current=np.zeros((len(p5),p4.shape[1]),np.float64)
    for child,parent in enumerate(parent_index):
        if parent>=0:current[:,parent]+=p5[:,child]
    out=p5.astype(np.float64,copy=True)
    for child,parent in enumerate(parent_index):
        if parent<0:continue
        ratio=(p4[:,parent]+eps)/(current[:,parent]+eps)
        out[:,child]*=np.power(ratio,strength)
    out/=np.maximum(out.sum(axis=1,keepdims=True),eps)
    return out.astype(np.float32)
def report_from_prob(p,d):return rank._metrics_from_scores(np.log(np.maximum(p,1e-12)),d,1.0)[0]
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--r4',type=Path,required=True);p.add_argument('--r5',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    if a.output_dir.exists() and any(a.output_dir.iterdir()):raise FileExistsError(a.output_dir)
    z=np.load(a.data,allow_pickle=True);r4=np.load(a.r4);r5=np.load(a.r5);rows=[json.loads(str(v)) for v in z['meta']];source=z['source_indices'].astype(np.int64);tr=int(z['source_train_end']);va=int(z['source_validation_end']);vi=np.flatnonzero((source>=tr)&(source<va));di=np.flatnonzero(source>=va);truth=np.asarray([[r['target_lat'],r['target_lon']] for r in rows],np.float32);c4=[str(v) for v in z['cells_r4']];c5=[str(v) for v in z['cells_r5']];xy5=z['centroids_r5'].astype(np.float32);lookup={c:i for i,c in enumerate(c4)};parent=np.asarray([lookup.get(h3.cell_to_parent(c,4),-1) for c in c5],np.int32);missing=int((parent<0).sum())
    pv4=r4['validation'];pd4=r4['development'];pv5=r5['validation'];pd5=r5['development'];dv=distances(xy5,truth[vi]);dd=distances(xy5,truth[di]);trials=[];best=None
    for strength in np.linspace(0,1.5,31):
        q=reconcile(pv5,pv4,parent,float(strength));rep=report_from_prob(q,dv);trials.append({'strength':float(strength),'validation':rep})
        if best is None or rep['broad_area_score']>best[0]:best=(rep['broad_area_score'],float(strength),rep,q)
    _,strength,validation,vprob=best;dprob=reconcile(pd5,pd4,parent,strength);development=report_from_prob(dprob,dd)
    a.output_dir.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.output_dir/'reconciled_probabilities.npz',validation=vprob,development=dprob,parent_r4_index=parent,cells_r5=np.asarray(c5));report={'schema':'h3-multiresolution-reconciliation-v1','r4':str(a.r4),'r5':str(a.r5),'r5_cells':len(c5),'missing_parent_cells':missing,'selected_strength':strength,'validation':validation,'development':development,'trials':trials,'protocol':'reconciliation strength selected only on validation; development diagnostic only'};(a.output_dir/'metrics.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='trials'},indent=2))
if __name__=='__main__':main()
