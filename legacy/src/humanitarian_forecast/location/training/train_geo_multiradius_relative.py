#!/usr/bin/env python3
"""Train nested-radius candidate classifiers as a heterogeneous ranking expert.

Four binary LightGBM models estimate P(candidate within 25/50/100/200 km).
Their logits are combined using a small validation-only recipe set, yielding a
ranking signal with a different objective from LambdaRank while using the same
cutoff-safe candidate/history representation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

from humanitarian_forecast.core.model_store import ModelInfo, write_info
from humanitarian_forecast.location.training.train_geo_lambdarank_relative import _flatten_features
from humanitarian_forecast.location.training.train_geo_lambdarank_v2 import (
    _scores_to_matrix,
    _softmax,
    metrics,
)

RADII=(25,50,100,200)
RECIPES={
    "balanced":np.asarray([2.0,1.5,1.0,.5],np.float32),
    "precision":np.asarray([6.0,3.0,1.0,.25],np.float32),
    "broad":np.asarray([1.0,1.5,2.0,1.0],np.float32),
    "near50":np.asarray([3.0,3.0,1.0,.4],np.float32),
    "equal":np.ones(4,np.float32),
}


def _logit(p:np.ndarray)->np.ndarray:
    p=np.clip(p,1e-5,1-1e-5);return np.log(p)-np.log1p(-p)


def _country_weights(meta,lo,hi,groups,ethiopia_weight,regional_weight):
    regional={"ethiopia","eritrea","somalia","sudan","south sudan","djibouti","kenya"}
    q=np.ones(hi-lo,np.float32)
    for j,i in enumerate(range(lo,hi)):
        country=str(meta[i].get('country','')).casefold()
        if country=='ethiopia':q[j]=ethiopia_weight
        elif country in regional:q[j]=regional_weight
    return np.repeat(q,groups).astype(np.float32)


def _eth_mask(meta,lo,hi):
    return np.asarray([str(meta[i].get('country','')).casefold()=='ethiopia' for i in range(lo,hi)])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--estimators',type=int,default=450);p.add_argument('--ethiopia-weight',type=float,default=12.0);p.add_argument('--regional-weight',type=float,default=2.0);p.add_argument('--seed',type=int,default=20260824)
    a=p.parse_args()
    if a.output_dir.exists() and any(a.output_dir.iterdir()):raise FileExistsError(a.output_dir)
    z=np.load(a.data,allow_pickle=True);x=z['x'].astype(np.float32);c=z['candidate_features'].astype(np.float32);coords=z['candidate_coordinates'].astype(np.float32);valid=z['candidate_valid'];target=z['y'].astype(np.float32);meta=[json.loads(str(v)) for v in z['meta']]
    n=len(x);tr=int(.7*n);va=int(.85*n)
    print('building multiradius training rows',flush=True)
    tx,_,tg,td=_flatten_features(x[:tr],c[:tr],valid[:tr],coords[:tr],target[:tr])
    print('building multiradius validation rows',flush=True)
    vx,_,vg,vd=_flatten_features(x[tr:va],c[tr:va],valid[tr:va],coords[tr:va],target[tr:va])
    tw=_country_weights(meta,0,tr,tg,a.ethiopia_weight,a.regional_weight)
    val_probs=[];models=[];best_iters=[]
    for ri,radius in enumerate(RADII):
        y=(td<=radius).astype(np.int8);vy=(vd<=radius).astype(np.int8)
        pos=max(1,int(y.sum()));neg=max(1,len(y)-pos);scale=min(25.0,neg/pos)
        model=LGBMClassifier(objective='binary',n_estimators=a.estimators,learning_rate=.035,num_leaves=47,min_child_samples=100,colsample_bytree=.80,reg_lambda=6.,reg_alpha=.25,n_jobs=-1,verbosity=-1,random_state=a.seed+ri,scale_pos_weight=scale)
        model.fit(tx,y,sample_weight=tw,eval_set=[(vx,vy)],eval_metric='binary_logloss',callbacks=[early_stopping(35,verbose=False),log_evaluation(50)])
        it=int(model.best_iteration_ or a.estimators);best_iters.append(it);models.append(model)
        val_probs.append(model.predict_proba(vx,num_iteration=it)[:,1].astype(np.float32))
        print(json.dumps({'radius':radius,'positive_rate':float(y.mean()),'scale_pos_weight':scale,'best_iteration':it}),flush=True)
    val_valid=valid[tr:va];eth=_eth_mask(meta,tr,va);flat=np.stack(val_probs,axis=1)
    best=None
    for name,w in RECIPES.items():
        score_flat=(_logit(flat)*w[None,:]).sum(1);raw=_scores_to_matrix(score_flat,val_valid)
        for temp in np.geomspace(.15,3.,16):
            prob=_softmax(raw,val_valid,float(temp));global_report=metrics(raw,prob,coords[tr:va],val_valid,target[tr:va]);local=metrics(raw[eth],prob[eth],coords[tr:va][eth],val_valid[eth],target[tr:va][eth]);row=(local['broad_area_score'],name,float(temp),global_report,local,w.copy())
            if best is None or row[0]>best[0]:best=row
    assert best is not None
    _,recipe,temp,val_global,val_eth,w=best
    print(json.dumps({'selected_recipe':recipe,'weights':w.tolist(),'temperature':temp,'validation_ethiopia':val_eth},indent=2),flush=True)

    print('building multiradius development rows',flush=True)
    dx,_,dg,dd=_flatten_features(x[va:],c[va:],valid[va:],coords[va:],target[va:])
    devp=[]
    for model,it in zip(models,best_iters):devp.append(model.predict_proba(dx,num_iteration=it)[:,1].astype(np.float32))
    dflat=np.stack(devp,1);draw=_scores_to_matrix((_logit(dflat)*w[None,:]).sum(1),valid[va:]);dprob=_softmax(draw,valid[va:],temp);dev_global=metrics(draw,dprob,coords[va:],valid[va:],target[va:]);deth=_eth_mask(meta,va,n);dev_eth=metrics(draw[deth],dprob[deth],coords[va:][deth],valid[va:][deth],target[va:][deth])
    a.output_dir.mkdir(parents=True,exist_ok=True)
    for r,m,it in zip(RADII,models,best_iters):m.booster_.save_model(str(a.output_dir/f'within_{r}km.txt'),num_iteration=it)
    np.savez_compressed(a.output_dir/'multiradius_probabilities.npz',validation_logits=_scores_to_matrix((_logit(flat)*w[None,:]).sum(1),val_valid),development_logits=draw,validation_indices=np.arange(tr,va),development_indices=np.arange(va,n))
    report={'schema':'geo-multiradius-relative-ranker-v2','radii':list(RADII),'best_iterations':best_iters,'selected_recipe':recipe,'weights':w.tolist(),'temperature':temp,'validation':val_global,'validation_ethiopia':val_eth,'development':dev_global,'development_ethiopia':dev_eth,'protocol':'global first 70% fit; validation chooses one of five fixed radius-weight recipes and temperature by Ethiopia broad-area score; development diagnostic only'}
    (a.output_dir/'multiradius_metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    write_info(a.output_dir,ModelInfo(subsystem='location/geo_multiradius_relative',version='v2',status='research-challenger',description='Nested 25/50/100/200 km proximity classifiers with explicit within-query relative candidate features.',metrics={'validation':val_global,'validation_ethiopia':val_eth,'development':dev_global,'development_ethiopia':dev_eth},lineage={'dataset':str(a.data),'restore_tag':'production-boost-preflight-2026-08-24'},training={'radii':list(RADII),'best_iterations':best_iters,'recipe':recipe,'weights':w.tolist(),'temperature':temp,'ethiopia_weight':a.ethiopia_weight,'regional_weight':a.regional_weight,'seed':a.seed},notes=['Uses a different supervision objective from LambdaRank.','Adds within-query z-score/percentile features for recurrence, transition, raw-event and actor-transfer signals.','Development is diagnostic, not pristine prospective evaluation.']))
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
