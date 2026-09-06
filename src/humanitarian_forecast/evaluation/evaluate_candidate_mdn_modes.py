#!/usr/bin/env python3
"""Evaluate pre-exported MDN modes in a LightGBM-only process."""
from __future__ import annotations
import argparse,gc,json
from pathlib import Path
import numpy as np
from lightgbm import Booster
from humanitarian_forecast.location.training.train_candidate_refiner import _country_indices
from humanitarian_forecast.location.training.train_geo_lambdarank_v2 import _flatten_inference_features,_scores_to_matrix,_softmax
SCALE_KM=1000.0

def metrics(prob,xy,valid,target):
 d=np.linalg.norm(xy-target[:,None],axis=-1)*SCALE_KM;d=np.where(valid,d,np.inf);top=np.argmax(prob,1);e=d[np.arange(len(d)),top];order=np.argsort(-prob,axis=1);o={'samples':float(len(e)),'mean_error_km':float(e.mean()),'median_error_km':float(np.median(e)),'p90_error_km':float(np.quantile(e,.9)),'within_25km':float((e<=25).mean()),'within_50km':float((e<=50).mean()),'within_100km':float((e<=100).mean()),'within_200km':float((e<=200).mean())}
 for r in (25,50,100,200):o[f'probability_mass_within_{r}km']=float((prob*(d<=r)).sum(1).mean())
 o['top3_within_100km']=float(np.any(np.take_along_axis(d,order[:,:3],axis=1)<=100,axis=1).mean());o['broad_area_score']=.55*o['probability_mass_within_100km']+.2*o['probability_mass_within_200km']+.15*o['within_100km']+.1*o['top3_within_100km'];return o

def expand(parent_p,mode_pi,mode_mu,coords,valid,keep,alpha,max_parent,mixture_temp):
 q=np.power(np.maximum(mode_pi,1e-8),1/max(float(mixture_temp),1e-4));q/=np.maximum(q.sum(-1,keepdims=True),1e-12);eligible=valid&(parent_p<=max_parent);pp=np.where(eligible,parent_p*keep,parent_p);children=parent_p[...,None]*(1-keep)*q*eligible[...,None];child_xy=coords[:,:,None,:]+alpha*mode_mu;prob=np.concatenate([pp,children.reshape(len(pp),-1)],1);xy=np.concatenate([coords,child_xy.reshape(len(coords),-1,2)],1);vv=np.concatenate([valid,np.repeat(eligible[...,None],mode_pi.shape[-1],axis=-1).reshape(len(valid),-1)],1);return prob,xy,vv

def coarse_probability(model,it,temp,x,c,valid):
 flat,_=_flatten_inference_features(x,c,valid);raw=_scores_to_matrix(model.predict(flat,num_iteration=it),valid);return _softmax(raw,valid,temp)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--modes',type=Path,required=True);p.add_argument('--coarse-model',type=Path,required=True);p.add_argument('--coarse-config',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();z=np.load(a.data,allow_pickle=True);m=np.load(a.modes);x=z['x'].astype(np.float32);c=z['candidate_features'].astype(np.float32);coords=z['candidate_coordinates'].astype(np.float32);valid=z['candidate_valid'];y=z['y'].astype(np.float32);meta=[json.loads(str(v)) for v in z['meta']];n=len(x);tr=int(.70*n);va=int(.85*n);model=Booster(model_file=str(a.coarse_model));cfg=json.loads(a.coarse_config.read_text());it=int(cfg.get('selected_iteration',model.num_trees()));temp=float(cfg.get('probability_temperature',.2));parent=coarse_probability(model,it,temp,x[tr:va],c[tr:va],valid[tr:va]);mpi=m['validation_pi'];mmu=m['validation_mu'];ei=_country_indices(meta,tr,va,'Ethiopia');baseg=metrics(parent,coords[tr:va],valid[tr:va],y[tr:va]);basee=metrics(parent[ei],coords[tr:va][ei],valid[tr:va][ei],y[tr:va][ei]);local=[];ep=parent[ei];empi=mpi[ei];emmu=mmu[ei];eco=coords[tr:va][ei];ev=valid[tr:va][ei];ey=y[tr:va][ei]
 for keepmass in (.2,.35,.5,.65,.8):
  for alpha in (.75,1.,1.25):
   for pm in (.2,.25,.35,.5):
    for mt in (.6,1.,1.6):
     pp,xy,vv=expand(ep,empi,emmu,eco,ev,keepmass,alpha,pm,mt);e=metrics(pp,xy,vv,ey);del pp,xy,vv
     if e['within_25km']<basee['within_25km']-.003:continue
     score=(e['broad_area_score']-basee['broad_area_score'])+.0005*(basee['mean_error_km']-e['mean_error_km'])+.1*(e['within_25km']-basee['within_25km']);local.append((score,keepmass,alpha,pm,mt,e))
 local.sort(key=lambda q:q[0],reverse=True);best=None
 for score,km,alpha,pm,mt,e in local[:20]:
  pp,xy,vv=expand(parent,mpi,mmu,coords[tr:va],valid[tr:va],km,alpha,pm,mt);g=metrics(pp,xy,vv,y[tr:va]);del pp,xy,vv;gc.collect()
  if g['broad_area_score']>=baseg['broad_area_score']-.001:best=(score,km,alpha,pm,mt,e,g);break
 if best is None:raise RuntimeError('no globally admissible MDN policy')
 _,km,alpha,pm,mt,ve,vg=best;del parent,mpi,mmu;gc.collect();dp=coarse_probability(model,it,temp,x[va:],c[va:],valid[va:]);dpi=m['development_pi'];dmu=m['development_mu'];dpp,dxy,dvv=expand(dp,dpi,dmu,coords[va:],valid[va:],km,alpha,pm,mt);dei=_country_indices(meta,va,n,'Ethiopia');dg=metrics(dpp,dxy,dvv,y[va:]);de=metrics(dpp[dei],dxy[dei],dvv[dei],y[va:][dei]);bdg=metrics(dp,coords[va:],valid[va:],y[va:]);bde=metrics(dp[dei],coords[va:][dei],valid[va:][dei],y[va:][dei]);policy={'parent_keep_mass':km,'mean_scale':alpha,'parent_probability_max':pm,'mixture_temperature':mt};report={'schema':'candidate-mdn-refiner-evaluation-v1','selected_epoch':int(m['selected_epoch']),'components':int(m['components']),'policy':policy,'validation_global':vg,'validation_ethiopia':ve,'baseline_validation_global':baseg,'baseline_validation_ethiopia':basee,'development_global':dg,'development_ethiopia':de,'baseline_development_global':bdg,'baseline_development_ethiopia':bde,'protocol':'MDN trained on first70; modes exported Torch-only; validation selects support policy; final block diagnostic only'};a.output_dir.mkdir(parents=True,exist_ok=True);(a.output_dir/'policy.json').write_text(json.dumps(policy,indent=2)+'\n');(a.output_dir/'metrics.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
