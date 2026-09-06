#!/usr/bin/env python3
"""Compose the relative-feature LambdaRank with the continuous coordinate refiner.

This is a deliberately simple heterogeneous composition: ranking probabilities
come from the relative-feature LambdaRank expert; candidate support coordinates
are shifted by the frozen continuous refiner learned on the first 70% global
training era.  No development labels participate in composition.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from lightgbm import Booster

from humanitarian_forecast.core.model_store import ModelInfo, write_info
from humanitarian_forecast.location.training.train_geo_lambdarank_v2 import (
    _scores_to_matrix,
    _softmax,
    metrics,
)
from humanitarian_forecast.location.training.train_geo_lambdarank_relative import _flatten_features as relative_features
from humanitarian_forecast.location.training.train_geo_lambdarank_v2 import _flatten_features as base_features
from humanitarian_forecast.location.training.train_candidate_refiner import _cap_residual


def _country_indices(meta: list[dict[str, object]], lo: int, hi: int, country: str) -> np.ndarray:
    c=country.casefold()
    return np.asarray([i for i in range(lo,hi) if str(meta[i].get('country','')).casefold()==c],dtype=np.int64)


def _evaluate_split(z,meta,ids,ranker,rank_iteration,temperature,east,north,alpha,cap):
    x=z['x'][ids].astype(np.float32);candidate=z['candidate_features'][ids].astype(np.float32);coords=z['candidate_coordinates'][ids].astype(np.float32);valid=z['candidate_valid'][ids];target=z['y'][ids].astype(np.float32)
    rf,_,_,_=relative_features(x,candidate,valid,coords,target)
    raw=_scores_to_matrix(ranker.predict(rf,num_iteration=rank_iteration),valid)
    probability=_softmax(raw,valid,temperature)
    bf,_,_,_=base_features(x,candidate,valid,coords,target)
    residual=np.zeros((*valid.shape,2),np.float32)
    residual[...,0][valid]=east.predict(bf).astype(np.float32)
    residual[...,1][valid]=north.predict(bf).astype(np.float32)
    refined=coords+float(alpha)*_cap_residual(residual,float(cap))
    report=metrics(raw,probability,refined,valid,target)
    return report,probability.astype(np.float32),refined.astype(np.float32),raw.astype(np.float32)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--ranker',type=Path,required=True)
    p.add_argument('--rank-config',type=Path,required=True)
    p.add_argument('--east-refiner',type=Path,required=True)
    p.add_argument('--north-refiner',type=Path,required=True)
    p.add_argument('--refiner-metrics',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--country',default='Ethiopia')
    a=p.parse_args()
    if a.output_dir.exists() and any(a.output_dir.iterdir()):raise FileExistsError(a.output_dir)
    z=np.load(a.data,allow_pickle=True);meta=[json.loads(str(v)) for v in z['meta']];n=len(z['x']);tr=int(.7*n);va=int(.85*n)
    rcfg=json.loads(a.rank_config.read_text());rank_iteration=int(rcfg['selected_iteration']);temperature=float(rcfg['probability_temperature'])
    rmetrics=json.loads(a.refiner_metrics.read_text());selected=rmetrics['selected'];alpha=float(selected['alpha']);cap=float(selected['cap_km'])
    ranker=Booster(model_file=str(a.ranker));east=Booster(model_file=str(a.east_refiner));north=Booster(model_file=str(a.north_refiner))
    vi=_country_indices(meta,tr,va,a.country);di=_country_indices(meta,va,n,a.country)
    val,pv,cv,lv=_evaluate_split(z,meta,vi,ranker,rank_iteration,temperature,east,north,alpha,cap)
    dev,pd,cd,ld=_evaluate_split(z,meta,di,ranker,rank_iteration,temperature,east,north,alpha,cap)
    a.output_dir.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.output_dir/'relative_refiner_probabilities.npz',validation=pv,development=pd,validation_coordinates=cv,development_coordinates=cd,validation_logits=lv,development_logits=ld,validation_indices=vi,development_indices=di)
    report={'schema':'relative-ranker-continuous-refiner-v1','country':a.country,'validation':val,'development':dev,'composition':{'ranker':str(a.ranker),'rank_iteration':rank_iteration,'temperature':temperature,'east_refiner':str(a.east_refiner),'north_refiner':str(a.north_refiner),'alpha':alpha,'cap_km':cap},'protocol':'all learned components fit on first 70% training era; component hyperparameters selected on validation; development diagnostic only'}
    (a.output_dir/'relative_refiner_metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    write_info(a.output_dir,ModelInfo(subsystem='location/relative_refiner',version='v1',status='research-champion-ethiopia-coarse-to-fine',description='Relative-feature candidate rank probabilities composed with continuous coarse-to-fine coordinate refinement.',metrics={'validation':val,'development':dev},lineage={'dataset':str(a.data),'ranker':str(a.ranker),'refiner_metrics':str(a.refiner_metrics),'restore_tag':'production-boost-preflight-2026-08-24'},training={'rank_iteration':rank_iteration,'temperature':temperature,'alpha':alpha,'cap_km':cap},notes=['No production champion is changed by this artifact.','Development is diagnostic and has been repeatedly inspected; prospective gates remain mandatory.','Composition improves broad-area placement but does not yet recover the coarse model’s <=25 km hit rate.']))
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
