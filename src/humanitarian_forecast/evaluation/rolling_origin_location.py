#!/usr/bin/env python3
"""Rolling-origin Ethiopia gate for broad-area candidate feature ablations.

This evaluator compares two *aligned* candidate datasets using the same fixed
LambdaRank recipe. Models train only on examples preceding each origin, while
metrics are reported on Ethiopia examples in the following chronological
window. It is intended to reject terrain/static-feature wins that do not persist
across regimes. Exact point errors are diagnostics; the acceptance target is
broad 100/200 km ranking stability.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np

from humanitarian_forecast.location.training.train_geo_lambdarank_v2 import (
    _flatten_features,
    _scores_to_matrix,
)

SCALE_KM = 1000.0


def parse_meta(raw: np.ndarray) -> list[dict[str,str]]:
    return [json.loads(str(v)) for v in raw]


def broad_metrics(scores: np.ndarray, coordinates: np.ndarray, valid: np.ndarray, target: np.ndarray) -> dict[str,float]:
    distance=np.linalg.norm(coordinates-target[:,None],axis=-1)*SCALE_KM
    distance=np.where(valid,distance,np.inf)
    top=np.argmax(scores,axis=1);err=distance[np.arange(len(distance)),top]
    order=np.argsort(-scores,axis=1)
    out={
        'samples':float(len(err)),
        'mean_error_km':float(np.mean(err)),
        'median_error_km':float(np.median(err)),
        'p90_error_km':float(np.quantile(err,.90)),
        'within_50km':float(np.mean(err<=50)),
        'within_100km':float(np.mean(err<=100)),
        'within_200km':float(np.mean(err<=200)),
        'top3_within_100km':float(np.mean(np.any(np.take_along_axis(distance,order[:,:3],axis=1)<=100,axis=1))),
        'top5_within_200km':float(np.mean(np.any(np.take_along_axis(distance,order[:,:5],axis=1)<=200,axis=1))),
        'candidate_oracle_mean_km':float(np.mean(np.min(distance,axis=1))),
    }
    out['rolling_broad_score']=.45*out['within_100km']+.25*out['within_200km']+.20*out['top3_within_100km']+.10*out['top5_within_200km']
    return out


def fit_predict(z,train_end:int,eval_start:int,eval_end:int,eval_mask:np.ndarray,seed:int,estimators:int):
    train_x,train_y,train_group,_=_flatten_features(
        z['x'][:train_end].astype(np.float32),z['candidate_features'][:train_end].astype(np.float32),z['candidate_valid'][:train_end],z['candidate_coordinates'][:train_end].astype(np.float32),z['y'][:train_end].astype(np.float32)
    )
    model=lgb.LGBMRanker(objective='lambdarank',metric='ndcg',label_gain=[0,1,3,7,15],n_estimators=estimators,learning_rate=.04,num_leaves=31,min_child_samples=100,colsample_bytree=.75,reg_lambda=4.0,reg_alpha=.15,verbosity=-1,n_jobs=-1,random_state=seed)
    model.fit(train_x,train_y,group=train_group)
    del train_x,train_y
    x=z['x'][eval_start:eval_end][eval_mask].astype(np.float32); f=z['candidate_features'][eval_start:eval_end][eval_mask].astype(np.float32);v=z['candidate_valid'][eval_start:eval_end][eval_mask];c=z['candidate_coordinates'][eval_start:eval_end][eval_mask].astype(np.float32);y=z['y'][eval_start:eval_end][eval_mask].astype(np.float32)
    flat,_,_,_=_flatten_features(x,f,v,c,y);raw=_scores_to_matrix(model.predict(flat),v)
    return broad_metrics(raw,c,v,y)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--train-fracs',default='.60,.70,.80');p.add_argument('--eval-frac',type=float,default=.08);p.add_argument('--estimators',type=int,default=100);p.add_argument('--seed',type=int,default=20260830);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    b=np.load(a.baseline,allow_pickle=True);c=np.load(a.candidate,allow_pickle=True)
    for key in ('y','candidate_coordinates','candidate_valid','meta','label'):
        if not np.array_equal(b[key],c[key]):raise ValueError(f'alignment failure: {key}')
    meta=parse_meta(b['meta']); n=len(meta);rows=[]
    for i,frac in enumerate(float(x) for x in a.train_fracs.split(',')):
        tr=int(frac*n);end=min(n,int((frac+a.eval_frac)*n));mask=np.asarray([m.get('country')=='Ethiopia' for m in meta[tr:end]])
        if not mask.any():continue
        print(json.dumps({'fold':i+1,'train_end_index':tr,'eval_end_index':end,'ethiopia_eval':int(mask.sum()),'train_end_date':meta[tr-1]['target_date'],'eval_end_date':meta[end-1]['target_date']}),flush=True)
        bm=fit_predict(b,tr,tr,end,mask,a.seed+i*10,a.estimators);cm=fit_predict(c,tr,tr,end,mask,a.seed+i*10,a.estimators)
        rows.append({'fold':i+1,'train_fraction':frac,'train_end_date':meta[tr-1]['target_date'],'eval_end_date':meta[end-1]['target_date'],'baseline':bm,'candidate':cm,'delta':{'rolling_broad_score':cm['rolling_broad_score']-bm['rolling_broad_score'],'mean_error_km':cm['mean_error_km']-bm['mean_error_km'],'within_100km':cm['within_100km']-bm['within_100km']}})
        # Persist completed fold before any later training/reporting step.
        progress=a.output.with_suffix(a.output.suffix+'.progress')
        progress.parent.mkdir(parents=True,exist_ok=True)
        progress.write_text(json.dumps({'folds':rows},indent=2)+'\n')
        print(json.dumps(rows[-1]),flush=True)
    score=np.asarray([r['delta']['rolling_broad_score'] for r in rows]);mean=np.asarray([r['delta']['mean_error_km'] for r in rows]);w100=np.asarray([r['delta']['within_100km'] for r in rows])
    summary={'mean_broad_score_delta':float(score.mean()),'fold_broad_score_deltas':score.tolist(),'mean_error_delta_km':float(mean.mean()),'mean_within_100km_delta':float(w100.mean()),'positive_broad_score_folds':int((score>0).sum()),'stable_accept':bool(score.mean()>0 and (score>0).sum()>=max(2,len(score)-1) and score.min()>-.02),'rule':'positive mean broad-score delta, positive in nearly every fold, no fold worse than -0.02'}
    report={'schema':'rolling-origin-ethiopia-broad-area-v1','baseline':str(a.baseline),'candidate':str(a.candidate),'fixed_ranker':{'estimators':a.estimators,'learning_rate':.04,'num_leaves':31},'folds':rows,'summary':summary,'evaluation_scope':'Ethiopia broad-area ranking; exact point errors diagnostic only'}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
