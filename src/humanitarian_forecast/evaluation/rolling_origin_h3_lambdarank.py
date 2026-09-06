#!/usr/bin/env python3
"""Rolling-origin stability gate for dense H3 LambdaRank recipes."""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
from lightgbm import LGBMRanker

from humanitarian_forecast.location.training import train_h3_lambdarank as h3rank

FOLDS = [
    ("2020", date(2020,1,1), date(2020,12,31)),
    ("2021", date(2021,1,1), date(2021,12,31)),
    ("2022", date(2022,1,1), date(2022,12,31)),
    ("2023_pre_split", date(2023,1,1), date(2023,5,31)),
]


def train_one(x,anchors,truth,cells,static,train_idx,samples,seed,estimators,init_model):
    tx,ty,group=h3rank._build_train(x,anchors,truth,cells,static,train_idx,samples,seed)
    model=LGBMRanker(objective='lambdarank',metric='ndcg',label_gain=[0,1,3,7,15],n_estimators=estimators,learning_rate=.025,num_leaves=63,min_child_samples=60,colsample_bytree=.8,reg_lambda=6.,reg_alpha=.2,n_jobs=-1,verbosity=-1,random_state=seed)
    model.fit(tx,ty,group=group,init_model=str(init_model) if init_model else None)
    return model


def evaluate(model,x,anchors,truth,cells,static,idx,temp):
    sc,dd=h3rank._score_matrix(model,x,anchors,truth,cells,static,idx,model.booster_.num_trees())
    report,_=h3rank._metrics_from_scores(sc,dd,temp)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--static-context',type=Path,required=True);p.add_argument('--global-pretrain',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--resolution',type=int,default=4);p.add_argument('--samples-per-query',type=int,default=256);p.add_argument('--local-estimators',type=int,default=150);p.add_argument('--temperature',type=float,default=.12);p.add_argument('--seed',type=int,default=20260824);a=p.parse_args()
    z=np.load(a.data,allow_pickle=True);s=np.load(a.static_context,allow_pickle=True);x=z['x'].astype(np.float32);rows=[json.loads(str(v)) for v in z['meta']];dates=np.asarray([date.fromisoformat(r['target_date']) for r in rows],dtype=object);anchors=np.asarray([[r['anchor_lat'],r['anchor_lon']] for r in rows],np.float32);truth=np.asarray([[r['target_lat'],r['target_lon']] for r in rows],np.float32);cells=z[f'centroids_r{a.resolution}'].astype(np.float32);static=s[f'features_r{a.resolution}'].astype(np.float32)
    result={'schema':'rolling-origin-h3-lambdarank-v1','resolution':a.resolution,'local_estimators':a.local_estimators,'temperature':a.temperature,'global_pretrain':str(a.global_pretrain),'folds':[]}
    for fold,(name,start,end) in enumerate(FOLDS):
        train_idx=np.flatnonzero(dates<start);eval_idx=np.flatnonzero((dates>=start)&(dates<=end))
        if len(train_idx)<100 or len(eval_idx)<20:continue
        print(json.dumps({'fold':name,'train':len(train_idx),'eval':len(eval_idx)}),flush=True)
        scratch=train_one(x,anchors,truth,cells,static,train_idx,a.samples_per_query,a.seed+fold,a.local_estimators,None)
        scratch_report=evaluate(scratch,x,anchors,truth,cells,static,eval_idx,a.temperature)
        transfer=train_one(x,anchors,truth,cells,static,train_idx,a.samples_per_query,a.seed+fold,a.local_estimators,a.global_pretrain)
        transfer_report=evaluate(transfer,x,anchors,truth,cells,static,eval_idx,a.temperature)
        row={'fold':name,'train_samples':len(train_idx),'eval_samples':len(eval_idx),'scratch':scratch_report,'global_transfer':transfer_report,'delta_broad_area':transfer_report['broad_area_score']-scratch_report['broad_area_score'],'delta_mean_error_km':transfer_report['mean_error_km']-scratch_report['mean_error_km']}
        result['folds'].append(row);print(json.dumps(row),flush=True)
    deltas=np.asarray([r['delta_broad_area'] for r in result['folds']],np.float64)
    result['summary']={'folds':len(deltas),'mean_delta_broad_area':float(deltas.mean()) if len(deltas) else None,'median_delta_broad_area':float(np.median(deltas)) if len(deltas) else None,'improved_folds':int((deltas>0).sum()),'nonregressed_folds_0_01':int((deltas>=-.01).sum()),'promotion_gate':bool(len(deltas)>=3 and (deltas>0).sum()>=3 and deltas.mean()>0)}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['summary'],indent=2))
if __name__=='__main__':main()
