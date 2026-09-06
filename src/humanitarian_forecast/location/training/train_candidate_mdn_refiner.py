#!/usr/bin/env python3
"""Train a multimodal local residual density over frozen coarse candidates.

The strong candidate LambdaRank remains responsible for regional probability.
This model learns p(target_offset | candidate/history) as a small Gaussian
mixture from cutoff-safe first-70% rows.  Validation selects epoch and the
parent/child mass policy; development remains diagnostic only.
"""
from __future__ import annotations
import argparse,gc,json,math,random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from lightgbm import Booster
from humanitarian_forecast.core.model_store import ModelInfo,write_info
from humanitarian_forecast.location.training.train_candidate_refiner import _build_regression_rows,_country_indices
from humanitarian_forecast.location.training.train_geo_lambdarank_v2 import _flatten_inference_features,_scores_to_matrix,_softmax
SCALE_KM=1000.0

class MDN(nn.Module):
 def __init__(self,din:int,k:int=5,width:int=384):
  super().__init__();self.k=k;self.net=nn.Sequential(nn.LayerNorm(din),nn.Linear(din,width),nn.GELU(),nn.Linear(width,width),nn.GELU(),nn.Linear(width,256),nn.GELU());self.out=nn.Linear(256,k*5)
 def forward(self,x):
  z=self.out(self.net(x)).view(len(x),self.k,5);logits=z[...,0];mu=z[...,1:3];logsig=z[...,3:5].clamp(-5.0,-1.0);return logits,mu,logsig

def nll(model,x,y,w):
 logits,mu,ls=model(x);inv=torch.exp(-ls);q=(y[:,None,:]-mu)*inv;lp=-.5*(q.square().sum(-1)+2*ls.sum(-1)+2*math.log(2*math.pi));mix=torch.log_softmax(logits,-1)+lp;loss=-torch.logsumexp(mix,-1);return (loss*w).sum()/w.sum().clamp_min(1e-6)

def device_for(name):
 if name=='auto':
  if torch.backends.mps.is_available():return torch.device('mps')
  if torch.cuda.is_available():return torch.device('cuda')
  return torch.device('cpu')
 return torch.device(name)

def predict_mdn(model,flat,batch,device):
 model.eval();pi=[];mu=[];sig=[]
 with torch.no_grad():
  for s in range(0,len(flat),batch):
   xb=torch.from_numpy(flat[s:s+batch]).to(device);l,m,ls=model(xb);pi.append(torch.softmax(l,-1).cpu().numpy());mu.append(m.cpu().numpy());sig.append(torch.exp(ls).cpu().numpy())
 return np.concatenate(pi),np.concatenate(mu),np.concatenate(sig)

def matrix_modes(flat_pi,flat_mu,valid,k):
 pi=np.zeros((*valid.shape,k),np.float32);mu=np.zeros((*valid.shape,k,2),np.float32);pi[valid]=flat_pi.astype(np.float32);mu[valid]=flat_mu.astype(np.float32);return pi,mu

def metrics(prob,xy,valid,target):
 d=np.linalg.norm(xy-target[:,None],axis=-1)*SCALE_KM;d=np.where(valid,d,np.inf);top=np.argmax(prob,1);e=d[np.arange(len(d)),top];order=np.argsort(-prob,axis=1);o={'samples':float(len(e)),'mean_error_km':float(e.mean()),'median_error_km':float(np.median(e)),'p90_error_km':float(np.quantile(e,.9)),'within_25km':float((e<=25).mean()),'within_50km':float((e<=50).mean()),'within_100km':float((e<=100).mean()),'within_200km':float((e<=200).mean())}
 for r in (25,50,100,200):o[f'probability_mass_within_{r}km']=float((prob*(d<=r)).sum(1).mean())
 o['top3_within_100km']=float(np.any(np.take_along_axis(d,order[:,:3],axis=1)<=100,axis=1).mean());o['broad_area_score']=.55*o['probability_mass_within_100km']+.2*o['probability_mass_within_200km']+.15*o['within_100km']+.1*o['top3_within_100km'];return o

def expand(parent_p,mode_pi,mode_mu,coords,valid,keep,alpha,max_parent,mixture_temp):
 # mixture temperature operates within each parent only
 q=np.power(np.maximum(mode_pi,1e-8),1/max(mixture_temp,1e-4));q/=np.maximum(q.sum(-1,keepdims=True),1e-12)
 eligible=valid&(parent_p<=max_parent);pp=np.where(eligible,parent_p*keep,parent_p);children=parent_p[...,None]*(1-keep)*q*eligible[...,None];child_xy=coords[:,:,None,:]+alpha*mode_mu
 prob=np.concatenate([pp,children.reshape(len(pp),-1)],1);xy=np.concatenate([coords,child_xy.reshape(len(coords),-1,2)],1);vv=np.concatenate([valid,np.repeat(eligible[...,None],mode_pi.shape[-1],axis=-1).reshape(len(valid),-1)],1);return prob,xy,vv

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--coarse-model',type=Path,required=True);p.add_argument('--coarse-config',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--components',type=int,default=5);p.add_argument('--width',type=int,default=384);p.add_argument('--epochs',type=int,default=12);p.add_argument('--batch-size',type=int,default=2048);p.add_argument('--learning-rate',type=float,default=6e-4);p.add_argument('--device',default='auto');p.add_argument('--seed',type=int,default=20260824);a=p.parse_args()
 if a.output_dir.exists() and any(a.output_dir.iterdir()):raise FileExistsError(a.output_dir)
 a.output_dir.mkdir(parents=True,exist_ok=True)
 random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);device=device_for(a.device)
 z=np.load(a.data,allow_pickle=True);x=z['x'].astype(np.float32);c=z['candidate_features'].astype(np.float32);coords=z['candidate_coordinates'].astype(np.float32);valid=z['candidate_valid'];y=z['y'].astype(np.float32);meta=[json.loads(str(v)) for v in z['meta']];n=len(x);tr=int(.70*n);va=int(.85*n)
 tx,ty,tw=_build_regression_rows(x,c,coords,valid,y,meta,0,tr,4,200.,a.seed);print(json.dumps({'train_rows':len(tx),'feature_dim':tx.shape[1],'device':str(device)}),flush=True)
 # robustly clip rare >250km supervision created by random support rows
 yn=np.linalg.norm(ty,axis=1)*SCALE_KM;keep=yn<=250.;tx=tx[keep];ty=ty[keep];tw=tw[keep]
 ds=TensorDataset(torch.from_numpy(tx),torch.from_numpy(ty),torch.from_numpy(tw));loader=DataLoader(ds,batch_size=a.batch_size,shuffle=True,num_workers=0)
 model=MDN(tx.shape[1],a.components,a.width).to(device);opt=torch.optim.AdamW(model.parameters(),lr=a.learning_rate,weight_decay=.02);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.epochs)
 # validation NLL on a deterministic sample of locally supervised rows
 vx,vy,vw=_build_regression_rows(x,c,coords,valid,y,meta,tr,va,3,200.,a.seed+1,chunk=5000);sel=np.linspace(0,len(vx)-1,min(180000,len(vx)),dtype=np.int64);vx=vx[sel];vy=vy[sel];vw=vw[sel]
 best=None;hist=[]
 for epoch in range(1,a.epochs+1):
  model.train();total=0.;den=0.
  for xb,yb,wb in loader:
   xb=xb.to(device);yb=yb.to(device);wb=wb.to(device);opt.zero_grad(set_to_none=True);loss=nll(model,xb,yb,wb);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step();total+=float(loss.detach().cpu())*len(xb);den+=len(xb)
  sched.step();model.eval();vals=[]
  with torch.no_grad():
   for s in range(0,len(vx),4096):vals.append(float(nll(model,torch.from_numpy(vx[s:s+4096]).to(device),torch.from_numpy(vy[s:s+4096]).to(device),torch.from_numpy(vw[s:s+4096]).to(device)).cpu())*len(vx[s:s+4096]))
  vn=sum(vals)/len(vx);row={'epoch':epoch,'train_nll':total/den,'validation_nll':vn};hist.append(row);print(json.dumps(row),flush=True)
  if best is None or vn<best[0]:
   state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};best=(vn,epoch,state)
   torch.save({'state_dict':state,'feature_dim':int(tx.shape[1]),'components':a.components,'width':a.width,'selected_epoch':epoch},a.output_dir/'mdn_refiner.pt')
 assert best;model.load_state_dict(best[2]);del tx,ty,tw,vx,vy,vw,ds,loader;gc.collect()
 coarse=Booster(model_file=str(a.coarse_model));cfg=json.loads(a.coarse_config.read_text());it=int(cfg.get('selected_iteration',coarse.num_trees()));temp=float(cfg.get('probability_temperature',.2))
 vf,_=_flatten_inference_features(x[tr:va],c[tr:va],valid[tr:va]);raw=_scores_to_matrix(coarse.predict(vf,num_iteration=it),valid[tr:va]);parent=_softmax(raw,valid[tr:va],temp);fpi,fmu,_=predict_mdn(model,vf,8192,device);mpi,mmu=matrix_modes(fpi,fmu,valid[tr:va],a.components);ei=_country_indices(meta,tr,va,'Ethiopia');baseg=metrics(parent,coords[tr:va],valid[tr:va],y[tr:va]);basee=metrics(parent[ei],coords[tr:va][ei],valid[tr:va][ei],y[tr:va][ei])
 # Search the policy on Ethiopia validation first. Full-global expansion creates
 # up to 192 support points per example, so evaluating every policy globally is
 # wasteful and can exhaust host memory. Only the strongest local policies are
 # checked against the global non-regression gate below.
 candidates=[]
 ep=parent[ei];empi=mpi[ei];emmu=mmu[ei];ecoords=coords[tr:va][ei];evalid=valid[tr:va][ei];ey=y[tr:va][ei]
 for keepmass in (.2,.35,.5,.65,.8):
  for alpha in (.75,1.,1.25):
   for pm in (.2,.25,.35,.5):
    for mt in (.6,1.,1.6):
     pp,xy,vv=expand(ep,empi,emmu,ecoords,evalid,keepmass,alpha,pm,mt);e=metrics(pp,xy,vv,ey);del pp,xy,vv
     if e['within_25km']<basee['within_25km']-.003:continue
     score=(e['broad_area_score']-basee['broad_area_score'])+.0005*(basee['mean_error_km']-e['mean_error_km'])+.1*(e['within_25km']-basee['within_25km']);candidates.append((score,keepmass,alpha,pm,mt,e))
 candidates.sort(key=lambda row:row[0],reverse=True);bestpol=None
 for score,keepmass,alpha,pm,mt,e in candidates[:20]:
  pp,xy,vv=expand(parent,mpi,mmu,coords[tr:va],valid[tr:va],keepmass,alpha,pm,mt);g=metrics(pp,xy,vv,y[tr:va]);del pp,xy,vv;gc.collect()
  if g['broad_area_score']<baseg['broad_area_score']-.001:continue
  bestpol=(score,keepmass,alpha,pm,mt,e,g);break
 if bestpol is None:raise RuntimeError('no globally admissible MDN support policy among top local policies')
 _,km,alpha,pm,mt,ve,vg=bestpol
 df,_=_flatten_inference_features(x[va:],c[va:],valid[va:]);draw=_scores_to_matrix(coarse.predict(df,num_iteration=it),valid[va:]);dp=_softmax(draw,valid[va:],temp);dpi,dmu,_=predict_mdn(model,df,8192,device);dpi,dmu=matrix_modes(dpi,dmu,valid[va:],a.components);dpp,dxy,dvv=expand(dp,dpi,dmu,coords[va:],valid[va:],km,alpha,pm,mt);dei=_country_indices(meta,va,n,'Ethiopia');dg=metrics(dpp,dxy,dvv,y[va:]);de=metrics(dpp[dei],dxy[dei],dvv[dei],y[va:][dei]);bdg=metrics(dp,coords[va:],valid[va:],y[va:]);bde=metrics(dp[dei],coords[va:][dei],valid[va:][dei],y[va:][dei])
 a.output_dir.mkdir(parents=True,exist_ok=True);torch.save({'state_dict':best[2],'feature_dim':model.net[1].in_features,'components':a.components,'width':a.width,'selected_epoch':best[1]},a.output_dir/'mdn_refiner.pt');policy={'parent_keep_mass':km,'mean_scale':alpha,'parent_probability_max':pm,'mixture_temperature':mt};(a.output_dir/'policy.json').write_text(json.dumps(policy,indent=2)+'\n');report={'schema':'candidate-mdn-refiner-v1','parameter_count':sum(p.numel() for p in model.parameters()),'selected_epoch':best[1],'policy':policy,'validation_global':vg,'validation_ethiopia':ve,'baseline_validation_global':baseg,'baseline_validation_ethiopia':basee,'development_global':dg,'development_ethiopia':de,'baseline_development_global':bdg,'baseline_development_ethiopia':bde,'history':hist,'protocol':'MDN fit on first 70%; validation NLL selects epoch; validation broad-area selects support policy with global/within25 non-regression; final block diagnostic only'};(a.output_dir/'metrics.json').write_text(json.dumps(report,indent=2)+'\n');write_info(a.output_dir,ModelInfo(subsystem='location/candidate_mdn_refiner',version='v1',status='research-challenger',description='Multimodal Gaussian-mixture local residual density under frozen coarse candidate probability.',metrics={'validation':ve,'development':de},lineage={'dataset':str(a.data),'coarse_model':str(a.coarse_model),'restore_tag':'production-boost-preflight-2026-08-24'},training={'components':a.components,'width':a.width,'selected_epoch':best[1],'policy':policy,'seed':a.seed},notes=['Coarse regional probabilities remain frozen; MDN only expands local support.','Development is diagnostic, not pristine prospective evaluation.']));print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
