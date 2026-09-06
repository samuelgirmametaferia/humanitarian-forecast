#!/usr/bin/env python3
"""Compose the nested-radius ranker with the frozen continuous coordinate refiner."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from lightgbm import Booster
from humanitarian_forecast.core.model_store import ModelInfo,write_info
from humanitarian_forecast.location.training.train_geo_lambdarank_v2 import _flatten_features,_softmax,metrics
from humanitarian_forecast.location.training.train_candidate_refiner import _cap_residual

def country_mask(meta,lo,hi,country):
 c=country.casefold();return np.asarray([str(meta[i].get('country','')).casefold()==c for i in range(lo,hi)])

def refine_subset(z,ids,east,north,alpha,cap):
 x=z['x'][ids].astype(np.float32);c=z['candidate_features'][ids].astype(np.float32);co=z['candidate_coordinates'][ids].astype(np.float32);v=z['candidate_valid'][ids];y=z['y'][ids].astype(np.float32)
 f,_,_,_=_flatten_features(x,c,v,co,y);res=np.zeros((*v.shape,2),np.float32);res[...,0][v]=east.predict(f).astype(np.float32);res[...,1][v]=north.predict(f).astype(np.float32);return co+alpha*_cap_residual(res,cap),v,y

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--multiradius',type=Path,required=True);p.add_argument('--multiradius-metrics',type=Path,required=True);p.add_argument('--east-refiner',type=Path,required=True);p.add_argument('--north-refiner',type=Path,required=True);p.add_argument('--refiner-metrics',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--country',default='Ethiopia');a=p.parse_args()
 if a.output_dir.exists() and any(a.output_dir.iterdir()):raise FileExistsError(a.output_dir)
 z=np.load(a.data,allow_pickle=True);meta=[json.loads(str(q)) for q in z['meta']];n=len(z['x']);tr=int(.7*n);va=int(.85*n);m=np.load(a.multiradius);mm=json.loads(a.multiradius_metrics.read_text());temperature=float(mm['temperature']);rm=json.loads(a.refiner_metrics.read_text());alpha=float(rm['selected']['alpha']);cap=float(rm['selected']['cap_km']);east=Booster(model_file=str(a.east_refiner));north=Booster(model_file=str(a.north_refiner))
 vm=country_mask(meta,tr,va,a.country);dm=country_mask(meta,va,n,a.country);vi=np.flatnonzero(vm)+tr;di=np.flatnonzero(dm)+va
 rv=m['validation_logits'][vm];rd=m['development_logits'][dm]
 cv,vv,yv=refine_subset(z,vi,east,north,alpha,cap);cd,vd,yd=refine_subset(z,di,east,north,alpha,cap)
 pv=_softmax(rv,vv,temperature);pd=_softmax(rd,vd,temperature);val=metrics(rv,pv,cv,vv,yv);dev=metrics(rd,pd,cd,vd,yd)
 a.output_dir.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.output_dir/'multiradius_refiner_probabilities.npz',validation=pv.astype(np.float32),development=pd.astype(np.float32),validation_coordinates=cv,development_coordinates=cd,validation_logits=rv.astype(np.float32),development_logits=rd.astype(np.float32),validation_indices=vi,development_indices=di)
 report={'schema':'multiradius-continuous-refiner-v1','country':a.country,'validation':val,'development':dev,'composition':{'multiradius':str(a.multiradius),'temperature':temperature,'east_refiner':str(a.east_refiner),'north_refiner':str(a.north_refiner),'alpha':alpha,'cap_km':cap},'protocol':'all learned components use first-70% training; validation-only component selection; development diagnostic only'};(a.output_dir/'multiradius_refiner_metrics.json').write_text(json.dumps(report,indent=2)+'\n');write_info(a.output_dir,ModelInfo(subsystem='location/multiradius_refiner',version='v1',status='research-champion-ethiopia-broad-area',description='Nested-radius candidate probability ranker composed with continuous local coordinate refinement.',metrics={'validation':val,'development':dev},lineage={'dataset':str(a.data),'multiradius':str(a.multiradius),'refiner':str(a.refiner_metrics),'restore_tag':'production-boost-preflight-2026-08-24'},training={'temperature':temperature,'alpha':alpha,'cap_km':cap},notes=['Current strongest Ethiopia broad-area research composition in the production-boost branch.','Keep unrefined multiradius output available because it preserves more <=25 km top-1 hits.','No production promotion without rolling/prospective gates.']))
 print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
