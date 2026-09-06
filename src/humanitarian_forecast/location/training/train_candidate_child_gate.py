#!/usr/bin/env python3
"""Chronological gate for parent+refined-child candidate support.

A residual model is trained on an early training-era slice.  Its genuinely
out-of-sample predictions on the later training-era slice define whether a
refined child beats the original parent.  The gate is fit only on that held-out
training-era slice.  Validation then selects child-spawn calibration while the
final development block remains diagnostic only.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from lightgbm import Booster,LGBMClassifier,LGBMRegressor
from humanitarian_forecast.core.model_store import ModelInfo,write_info
from humanitarian_forecast.location.training.train_candidate_refiner import _build_regression_rows,_cap_residual,_country_indices
from humanitarian_forecast.location.training.train_geo_lambdarank_v2 import _flatten_inference_features,_scores_to_matrix,_softmax
SCALE_KM=1000.0

def _matrix(flat,valid,dtype=np.float32):
 out=np.zeros(valid.shape,dtype=dtype);out[valid]=flat;return out

def _residual_matrix(east,north,features,valid):
 r=np.zeros((*valid.shape,2),np.float32);r[...,0][valid]=east.predict(features).astype(np.float32);r[...,1][valid]=north.predict(features).astype(np.float32);return r

def _coarse_context(coarse,features,valid,iteration,temp):
 raw=_scores_to_matrix(coarse.predict(features,num_iteration=iteration),valid);p=_softmax(raw,valid,temp);ent=-(p*np.log(np.maximum(p,1e-12))).sum(1);order=np.argsort(-raw,axis=1);rank=np.empty_like(order);rank[np.arange(len(order))[:,None],order]=np.arange(order.shape[1])[None,:];return raw,p,ent,rank

def _gate_features(base,residual,raw,p,ent,rank,valid):
 rn=np.linalg.norm(residual,axis=-1)*SCALE_KM
 extra=np.stack([residual[...,0],residual[...,1],rn/200.0,raw,p,np.repeat(ent[:,None],p.shape[1],1),rank/np.maximum(valid.sum(1,keepdims=True)-1,1)],axis=-1).astype(np.float32)
 return np.concatenate([base,extra[valid]],axis=1).astype(np.float32)

def _metrics(prob,xy,valid,target):
 d=np.linalg.norm(xy-target[:,None],axis=-1)*SCALE_KM;d=np.where(valid,d,np.inf);top=np.argmax(prob,1);e=d[np.arange(len(d)),top];order=np.argsort(-prob,axis=1)
 o={'samples':float(len(e)),'mean_error_km':float(e.mean()),'median_error_km':float(np.median(e)),'p90_error_km':float(np.quantile(e,.9)),'within_25km':float((e<=25).mean()),'within_50km':float((e<=50).mean()),'within_100km':float((e<=100).mean()),'within_200km':float((e<=200).mean())}
 for r in (25,50,100,200):o[f'probability_mass_within_{r}km']=float((prob*(d<=r)).sum(1).mean())
 o['top3_within_100km']=float(np.any(np.take_along_axis(d,order[:,:3],axis=1)<=100,axis=1).mean());o['broad_area_score']=.55*o['probability_mass_within_100km']+.20*o['probability_mass_within_200km']+.15*o['within_100km']+.10*o['top3_within_100km'];return o

def _predict_bundle(x,candidates,coords,valid,east,north,gate,coarse,it,temp):
 f,_=_flatten_inference_features(x,candidates,valid);raw,p,ent,rank=_coarse_context(coarse,f,valid,it,temp);res=_residual_matrix(east,north,f,valid);gf=_gate_features(f,res,raw,p,ent,rank,valid);gp=_matrix(gate.predict_proba(gf)[:,1],valid);return p,res,gp

def _expand(p,res,coords,valid,cap,alpha,gatep,gate_threshold,parent_max,child_fraction):
 refined=coords+alpha*_cap_residual(res,cap);eligible=(gatep>=gate_threshold)&(p<=parent_max)&valid
 parent=np.where(eligible,p*(1-child_fraction),p);child=np.where(eligible,p*child_fraction,0.0)
 return np.concatenate([parent,child],1),np.concatenate([coords,refined],1),np.concatenate([valid,eligible],1)

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--data',type=Path,required=True);ap.add_argument('--coarse-model',type=Path,required=True);ap.add_argument('--coarse-config',type=Path,required=True);ap.add_argument('--full-refiner-dir',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True);ap.add_argument('--early-fraction',type=float,default=.55);ap.add_argument('--estimators',type=int,default=350);ap.add_argument('--seed',type=int,default=20260824);a=ap.parse_args()
 if a.output_dir.exists() and any(a.output_dir.iterdir()):raise FileExistsError(a.output_dir)
 z=np.load(a.data,allow_pickle=True);x=z['x'].astype(np.float32);c=z['candidate_features'].astype(np.float32);coords=z['candidate_coordinates'].astype(np.float32);valid=z['candidate_valid'];y=z['y'].astype(np.float32);meta=[json.loads(str(v)) for v in z['meta']];n=len(x);tr=int(.70*n);va=int(.85*n);early=int(a.early_fraction*n)
 # early residual model: gate labels must be based on out-of-sample residual predictions
 tx,ty,tw=_build_regression_rows(x,c,coords,valid,y,meta,0,early,4,200.0,a.seed)
 rp=dict(objective='huber',n_estimators=a.estimators,learning_rate=.04,num_leaves=63,min_child_samples=100,colsample_bytree=.8,reg_lambda=8.,reg_alpha=.3,n_jobs=-1,verbosity=-1,random_state=a.seed)
 ee=LGBMRegressor(**rp);nn=LGBMRegressor(**{**rp,'random_state':a.seed+1});ee.fit(tx,ty[:,0],sample_weight=tw);nn.fit(tx,ty[:,1],sample_weight=tw);del tx,ty,tw
 coarse=Booster(model_file=str(a.coarse_model));cfg=json.loads(a.coarse_config.read_text());it=int(cfg.get('selected_iteration',coarse.num_trees()));temp=float(cfg.get('probability_temperature',.2))
 gf,_=_flatten_inference_features(x[early:tr],c[early:tr],valid[early:tr]);raw,p,ent,rank=_coarse_context(coarse,gf,valid[early:tr],it,temp);res=_residual_matrix(ee,nn,gf,valid[early:tr]);proposal=coords[early:tr]+1.25*_cap_residual(res,75.)
 old=np.linalg.norm(coords[early:tr]-y[early:tr,None],axis=-1)*SCALE_KM;new=np.linalg.norm(proposal-y[early:tr,None],axis=-1)*SCALE_KM;benefit=(old-new)[valid[early:tr]]
 gate_x=_gate_features(gf,res,raw,p,ent,rank,valid[early:tr]);gate_y=(benefit>2.0).astype(np.int32);improve_w=np.clip(np.abs(benefit)/25.,.5,4.);qweight=np.repeat(np.asarray([4.0 if str(meta[i].get('country','')).casefold()=='ethiopia' else 1. for i in range(early,tr)],np.float32),valid[early:tr].sum(1));gw=improve_w*qweight
 gate=LGBMClassifier(objective='binary',n_estimators=450,learning_rate=.035,num_leaves=47,min_child_samples=150,colsample_bytree=.82,reg_lambda=7.,reg_alpha=.3,n_jobs=-1,verbosity=-1,random_state=a.seed+2);gate.fit(gate_x,gate_y,sample_weight=gw);print(json.dumps({'gate_rows':len(gate_x),'positive_rate':float(gate_y.mean()),'early_end':early,'train_end':tr}),flush=True)
 # final refiner was trained on all 0:70% rows by the prior stage
 east=Booster(model_file=str(a.full_refiner_dir/'east_refiner.txt'));north=Booster(model_file=str(a.full_refiner_dir/'north_refiner.txt'))
 vp,vr,vg=_predict_bundle(x[tr:va],c[tr:va],coords[tr:va],valid[tr:va],east,north,gate,coarse,it,temp);ei=_country_indices(meta,tr,va,'Ethiopia');baseg=_metrics(vp,coords[tr:va],valid[tr:va],y[tr:va]);basee=_metrics(vp[ei],coords[tr:va][ei],valid[tr:va][ei],y[tr:va][ei]);best=None
 for cap in (50.,75.):
  for alpha in (1.,1.25):
   for gt in (.5,.6,.7):
    for pm in (.2,.25,.35):
     for cf in (.65,.8,.9):
      pp,xy,vv=_expand(vp,vr,coords[tr:va],valid[tr:va],cap,alpha,vg,gt,pm,cf);g=_metrics(pp,xy,vv,y[tr:va]);e=_metrics(pp[ei],xy[ei],vv[ei],y[tr:va][ei])
      if g['broad_area_score']<baseg['broad_area_score']-.001 or e['within_25km']<basee['within_25km']-.002:continue
      score=(e['broad_area_score']-basee['broad_area_score'])+.0005*(basee['mean_error_km']-e['mean_error_km'])+.15*(e['within_25km']-basee['within_25km'])
      row=(score,cap,alpha,gt,pm,cf,e,g)
      if best is None or row[0]>best[0]:best=row
 if best is None:raise RuntimeError('no admissible child policy')
 _,cap,alpha,gt,pm,cf,ve,vgm=best
 a.output_dir.mkdir(parents=True,exist_ok=True);gate.booster_.save_model(str(a.output_dir/'child_gate.txt'));policy={'cap_km':cap,'alpha':alpha,'gate_threshold':gt,'parent_probability_max':pm,'child_mass_fraction':cf};(a.output_dir/'child_policy.json').write_text(json.dumps(policy,indent=2)+'\n')
 dp,dr,dg=_predict_bundle(x[va:],c[va:],coords[va:],valid[va:],east,north,gate,coarse,it,temp);dei=_country_indices(meta,va,n,'Ethiopia');dpp,dxy,dvv=_expand(dp,dr,coords[va:],valid[va:],cap,alpha,dg,gt,pm,cf);devg=_metrics(dpp,dxy,dvv,y[va:]);deve=_metrics(dpp[dei],dxy[dei],dvv[dei],y[va:][dei]);base_dg=_metrics(dp,coords[va:],valid[va:],y[va:]);base_de=_metrics(dp[dei],coords[va:][dei],valid[va:][dei],y[va:][dei])
 report={'schema':'candidate-child-gate-v1','policy':policy,'validation_global':vgm,'validation_ethiopia':ve,'baseline_validation_global':baseg,'baseline_validation_ethiopia':basee,'development_global':devg,'development_ethiopia':deve,'baseline_development_global':base_dg,'baseline_development_ethiopia':base_de,'protocol':'early residual 0:55%; child-benefit gate trained on out-of-sample 55:70%; child policy selected on 70:85%; 85:100 development diagnostic only'};(a.output_dir/'metrics.json').write_text(json.dumps(report,indent=2)+'\n');write_info(a.output_dir,ModelInfo(subsystem='location/candidate_child_gate',version='v1',status='research-challenger',description='Parent-preserving refined-child support with chronological out-of-sample benefit gate.',metrics={'validation':ve,'development':deve},lineage={'dataset':str(a.data),'coarse_model':str(a.coarse_model),'full_refiner':str(a.full_refiner_dir),'restore_tag':'production-boost-preflight-2026-08-24'},training={'early_fraction':a.early_fraction,'gate_rows':len(gate_x),'policy':policy,'seed':a.seed},notes=['Original candidate support is never removed; refined children only receive split probability mass.','Gate labels are generated from residual predictions out-of-sample in the training era.','Development remains diagnostic, not pristine prospective evaluation.']));print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
