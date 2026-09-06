#!/usr/bin/env python3
"""Globally pretrain the dense-H3 propagation scoring rule on candidate points.

The objective is transfer, not a new global production model. We learn the same
103-dimensional geometry/propagation feature contract used by dense Ethiopia H3,
but over >100k globally distributed pre-cutoff training queries. Static Ethiopia
channels are held at zero so subsequent local continuation can learn them without
feature-contract mismatch.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from lightgbm import LGBMRanker, early_stopping, log_evaluation

from humanitarian_forecast.core.model_store import ModelInfo, write_info
from humanitarian_forecast.location.training import train_h3_lambdarank as dense

STATIC_DIM = 22


def _absolute_candidates(candidate_features: np.ndarray) -> np.ndarray:
    return np.stack([candidate_features[:, 6] * 90.0, candidate_features[:, 7] * 180.0], axis=-1).astype(np.float32)


def _choose(rng: np.random.Generator, valid: np.ndarray, cells: np.ndarray, truth: np.ndarray, count: int) -> np.ndarray:
    ids=np.flatnonzero(valid)
    if len(ids)<=count:return ids
    d=dense._distance(cells[ids],truth)
    selected={int(ids[np.argmin(d)])}
    # Preserve several relevance levels so every rank group carries geography.
    for radius,cap in ((25,2),(50,2),(100,2),(200,2)):
        pool=ids[d<=radius]
        pool=np.asarray([v for v in pool if int(v) not in selected],np.int64)
        if len(pool):selected.update(map(int,rng.choice(pool,size=min(cap,len(pool)),replace=False)))
    remaining=count-len(selected)
    if remaining>0:
        pool=np.asarray([v for v in ids if int(v) not in selected],np.int64)
        if len(pool):selected.update(map(int,rng.choice(pool,size=min(remaining,len(pool)),replace=False)))
    arr=np.fromiter(selected,np.int64)
    if len(arr)>count:
        nearest=int(ids[np.argmin(d)]);rest=arr[arr!=nearest];arr=np.concatenate([[nearest],rng.choice(rest,size=count-1,replace=False)])
    return arr


def build_rows(x,candidate_features,valid,truth,anchors,indices,samples,seed,label='train'):
    rng=np.random.default_rng(seed);nrows=sum(min(samples,int(valid[i].sum())) for i in indices)
    # Infer feature dimension from one valid query without building Python lists
    # of hundreds of MB worth of arrays.
    probe=int(indices[0]);pcells=_absolute_candidates(candidate_features[probe]);pidx=_choose(rng,valid[probe],pcells,truth[probe],samples)
    pdim=dense._features_one(x[probe],anchors[probe],pcells[pidx],np.zeros((len(pidx),STATIC_DIM),np.float32)).shape[1]
    features=np.empty((nrows,pdim),np.float32);labels=np.empty(nrows,np.int32);groups=np.empty(len(indices),np.int32)
    cursor=0
    # Reset for deterministic selection independent of the probe.
    rng=np.random.default_rng(seed)
    for pos,i in enumerate(indices):
        cells=_absolute_candidates(candidate_features[i]);chosen=_choose(rng,valid[i],cells,truth[i],samples);m=len(chosen)
        features[cursor:cursor+m]=dense._features_one(x[i],anchors[i],cells[chosen],np.zeros((m,STATIC_DIM),np.float32));labels[cursor:cursor+m]=dense._relevance(dense._distance(cells[chosen],truth[i]));groups[pos]=m;cursor+=m
        if (pos+1)%10000==0:print(f'{label} queries {pos+1:,}/{len(indices):,}',flush=True)
    return features,labels,groups


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--samples-per-query',type=int,default=10);p.add_argument('--estimators',type=int,default=500);p.add_argument('--seed',type=int,default=20260824);p.add_argument('--exclude-country',action='append',default=[],help='Exclude all rows from this country from global pretraining/validation (repeatable).');a=p.parse_args()
    if a.output_dir.exists() and any(a.output_dir.iterdir()):raise FileExistsError(a.output_dir)
    z=np.load(a.data,allow_pickle=True);x=z['x'].astype(np.float32,copy=False);cf=z['candidate_features'].astype(np.float32,copy=False);valid=z['candidate_valid'];truth_rel=z['y'].astype(np.float32,copy=False);rows=[json.loads(str(v)) for v in z['meta']]
    # Absolute truth/anchor avoids converting relative labels back to lat/lon.
    anchors=np.asarray([[r['anchor_lat'],r['anchor_lon']] for r in rows],np.float32);truth=np.asarray([[r['target_lat'],r['target_lon']] for r in rows],np.float32);n=len(x);tr=int(.70*n);va=int(.85*n)
    excluded={value.casefold() for value in a.exclude_country}
    allowed=np.asarray([str(r.get('country','')).casefold() not in excluded for r in rows],bool)
    # Use a capped validation sample for early stopping to control memory while
    # retaining chronological ordering across the entire validation era.
    train_idx=np.flatnonzero((np.arange(n)<tr)&allowed);val_full=np.flatnonzero((np.arange(n)>=tr)&(np.arange(n)<va)&allowed);stride=max(1,len(val_full)//6000);val_idx=val_full[::stride]
    print(json.dumps({'excluded_countries':sorted(excluded),'excluded_training_rows':int(tr-len(train_idx)),'excluded_validation_rows':int((va-tr)-len(val_full))}),flush=True)
    tx,ty,tg=build_rows(x,cf,valid,truth,anchors,train_idx,a.samples_per_query,a.seed,'train');print(json.dumps({'train_rows':len(tx),'train_queries':len(train_idx),'feature_dim':tx.shape[1]}),flush=True)
    vx,vy,vg=build_rows(x,cf,valid,truth,anchors,val_idx,a.samples_per_query,a.seed+1,'validation');print(json.dumps({'validation_rows':len(vx),'validation_queries':len(val_idx)}),flush=True)
    ranker=LGBMRanker(objective='lambdarank',metric='ndcg',label_gain=[0,1,3,7,15],n_estimators=a.estimators,learning_rate=.025,num_leaves=63,min_child_samples=100,colsample_bytree=.82,reg_lambda=6.,reg_alpha=.2,n_jobs=-1,verbosity=-1,random_state=a.seed)
    ranker.fit(tx,ty,group=tg,eval_set=[(vx,vy)],eval_group=[vg],eval_at=[1,3,5],callbacks=[early_stopping(40,verbose=True),log_evaluation(25)])
    best=int(ranker.best_iteration_ or a.estimators);a.output_dir.mkdir(parents=True,exist_ok=True);ranker.booster_.save_model(str(a.output_dir/'global_h3_scoring_pretrain.txt'),num_iteration=best)
    report={'schema':'global-h3-scoring-pretrain-v1','train_queries':len(train_idx),'train_rows':len(tx),'validation_queries':len(val_idx),'validation_rows':len(vx),'feature_dim':int(tx.shape[1]),'selected_iteration':best,'samples_per_query':a.samples_per_query,'excluded_countries':sorted(a.exclude_country),'protocol':'global first 70% training; chronological 70-85% validation subsample for NDCG early stopping; excluded countries removed from both; no development labels used','purpose':'initialize/furnish a universal propagation scoring rule for later dense-H3 Ethiopia continuation'};(a.output_dir/'pretrain_metrics.json').write_text(json.dumps(report,indent=2)+'\n');write_info(a.output_dir,ModelInfo(subsystem='location/h3_lambdarank_global_pretrain',version='v1',status='research-pretrain',description='Global invariant propagation scoring rule using the same dense-H3 feature contract.',metrics={},lineage={'dataset':str(a.data),'restore_tag':'production-boost-preflight-2026-08-24'},training=report,notes=['Static Ethiopia H3 context channels are zero during global pretraining and become learnable during local continuation.','Excluded countries are never used in global pretraining or its early-stopping validation.','No development labels are used.']));print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
