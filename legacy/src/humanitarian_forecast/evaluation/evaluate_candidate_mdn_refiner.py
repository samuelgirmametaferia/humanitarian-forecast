#!/usr/bin/env python3
"""Evaluate a saved candidate MDN refiner without retaining training tensors."""
from __future__ import annotations
import argparse,gc,json
from pathlib import Path
import numpy as np
import torch
from lightgbm import Booster
from humanitarian_forecast.location.training.train_candidate_mdn_refiner import MDN,expand,matrix_modes,metrics,predict_mdn
from humanitarian_forecast.location.training.train_candidate_refiner import _country_indices
from humanitarian_forecast.location.training.train_geo_lambdarank_v2 import _flatten_inference_features,_scores_to_matrix,_softmax

def evaluate_split(model,device,coarse,it,temp,x,c,coords,valid,y,batch=8192):
 flat,_=_flatten_inference_features(x,c,valid);raw=_scores_to_matrix(coarse.predict(flat,num_iteration=it),valid);parent=_softmax(raw,valid,temp);fpi,fmu,_=predict_mdn(model,flat,batch,device);mpi,mmu=matrix_modes(fpi,fmu,valid,model.k);return parent,mpi,mmu

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--coarse-model',type=Path,required=True);p.add_argument('--coarse-config',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--device',default='cpu');a=p.parse_args()
 z=np.load(a.data,allow_pickle=True);x=z['x'].astype(np.float32);c=z['candidate_features'].astype(np.float32);coords=z['candidate_coordinates'].astype(np.float32);valid=z['candidate_valid'];y=z['y'].astype(np.float32);meta=[json.loads(str(v)) for v in z['meta']];n=len(x);tr=int(.70*n);va=int(.85*n)
 ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False);model=MDN(int(ck['feature_dim']),int(ck['components']),int(ck['width']));model.load_state_dict(ck['state_dict']);device=torch.device(a.device);model.to(device).eval();coarse=Booster(model_file=str(a.coarse_model));cfg=json.loads(a.coarse_config.read_text());it=int(cfg.get('selected_iteration',coarse.num_trees()));temp=float(cfg.get('probability_temperature',.2))
 parent,mpi,mmu=evaluate_split(model,device,coarse,it,temp,x[tr:va],c[tr:va],coords[tr:va],valid[tr:va],y[tr:va]);ei=_country_indices(meta,tr,va,'Ethiopia');baseg=metrics(parent,coords[tr:va],valid[tr:va],y[tr:va]);basee=metrics(parent[ei],coords[tr:va][ei],valid[tr:va][ei],y[tr:va][ei]);local=[]
 ep=parent[ei];empi=mpi[ei];emmu=mmu[ei];eco=coords[tr:va][ei];ev=valid[tr:va][ei];ey=y[tr:va][ei]
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
 _,km,alpha,pm,mt,ve,vg=best;del parent,mpi,mmu;gc.collect()
 dp,dpi,dmu=evaluate_split(model,device,coarse,it,temp,x[va:],c[va:],coords[va:],valid[va:],y[va:]);dpp,dxy,dvv=expand(dp,dpi,dmu,coords[va:],valid[va:],km,alpha,pm,mt);dei=_country_indices(meta,va,n,'Ethiopia');dg=metrics(dpp,dxy,dvv,y[va:]);de=metrics(dpp[dei],dxy[dei],dvv[dei],y[va:][dei]);bdg=metrics(dp,coords[va:],valid[va:],y[va:]);bde=metrics(dp[dei],coords[va:][dei],valid[va:][dei],y[va:][dei]);policy={'parent_keep_mass':km,'mean_scale':alpha,'parent_probability_max':pm,'mixture_temperature':mt};report={'schema':'candidate-mdn-refiner-evaluation-v1','selected_epoch':int(ck['selected_epoch']),'parameter_count':sum(p.numel() for p in model.parameters()),'policy':policy,'validation_global':vg,'validation_ethiopia':ve,'baseline_validation_global':baseg,'baseline_validation_ethiopia':basee,'development_global':dg,'development_ethiopia':de,'baseline_development_global':bdg,'baseline_development_ethiopia':bde,'protocol':'saved MDN trained on first70; support policy selected on 70-85 validation; 85-100 development diagnostic only'}
 a.output_dir.mkdir(parents=True,exist_ok=True);(a.output_dir/'policy.json').write_text(json.dumps(policy,indent=2)+'\n');(a.output_dir/'metrics.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
