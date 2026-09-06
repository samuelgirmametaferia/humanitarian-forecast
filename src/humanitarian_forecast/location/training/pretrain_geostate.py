#!/usr/bin/env python3
"""Global pretraining for the GeoState history backbone on 157k location examples.

The dense Ethiopia grid has too few early-regime labels to train a 10M+ model
from scratch. This stage first teaches the shared temporal encoder global conflict
motion/geometry using the existing cutoff-safe 32-candidate task, then exports a
backbone state that can be fine-tuned against dense H3 support.
"""
from __future__ import annotations

import argparse,json,random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from humanitarian_forecast.location.models.geostate import GeoStateConfig,GeoStateTransformer

PRESETS={
 'S':dict(d_model=256,heads=8,layers=6,ff_dim=768),
 'M':dict(d_model=384,heads=8,layers=8,ff_dim=1152),
 'L':dict(d_model=768,heads=12,layers=16,ff_dim=3072),
}

def device_for(name):
 if name=='auto':
  if torch.backends.mps.is_available():return torch.device('mps')
  if torch.cuda.is_available():return torch.device('cuda')
  return torch.device('cpu')
 return torch.device(name)

def soft_target(coords,valid,target,radius_km=100.):
 d=torch.linalg.vector_norm(coords-target[:,None],dim=-1)*1000.
 score=-.5*(d/radius_km).square();score=score.masked_fill(~valid,-1e9)
 return torch.softmax(score,dim=-1),d

class PretrainModel(nn.Module):
 def __init__(self,event_dim,sequence_length,candidate_dim,preset):
  super().__init__();cfg=GeoStateConfig(event_dim,sequence_length,(0,),(1,),**PRESETS[preset]);self.backbone=GeoStateTransformer(cfg);d=cfg.d_model
  self.candidate=nn.Sequential(nn.LayerNorm(candidate_dim),nn.Linear(candidate_dim,d),nn.GELU(),nn.Linear(d,d),nn.LayerNorm(d))
  self.bias=nn.Sequential(nn.Linear(candidate_dim,128),nn.GELU(),nn.Linear(128,1))
 def forward(self,x,candidate,valid):
  h=self.backbone.encode_history(x);q=self.backbone.query(h)
  # Candidate features 6/7 are absolute latitude/longitude normalized by
  # 90/180 in the causal dataset builder. Training this shared encoder here
  # gives later dense H3 cells a globally pretrained geographic embedding.
  latlon=torch.stack([candidate[...,6]*90.0,candidate[...,7]*180.0],dim=-1)
  geo=self.backbone.coordinate_encoder(self.backbone.coordinate_features(latlon))
  c=self.candidate(candidate)+geo
  logits=torch.einsum('bd,bkd->bk',q,c)/(q.shape[-1]**.5)+self.bias(candidate).squeeze(-1);return logits.masked_fill(~valid,-1e9)

def metrics(logits,coords,valid,target):
 p=torch.softmax(logits,dim=-1);d=torch.linalg.vector_norm(coords-target[:,None],dim=-1)*1000.;d=d.masked_fill(~valid,float('inf'));top=logits.argmax(-1);e=d[torch.arange(len(d)),top]
 q,_=soft_target(coords,valid,target);ce=-(q*torch.log(p.clamp_min(1e-12))).sum(-1).mean()
 return {'distance_soft_ce':float(ce),'mean_error_km':float(e.mean()),'median_error_km':float(e.median()),'within_25km':float((e<=25).float().mean()),'within_100km':float((e<=100).float().mean())}

@torch.no_grad()
def collect(model,x,cand,coords,valid,target,lo,hi,batch,device):
 model.eval();outs=[]
 for s in range(lo,hi,batch):
  t=min(hi,s+batch);outs.append(model(x[s:t].to(device),cand[s:t].to(device),valid[s:t].to(device)).cpu())
 logits=torch.cat(outs);return metrics(logits,coords[lo:hi],valid[lo:hi],target[lo:hi])

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--preset',choices=tuple(PRESETS),default='M');p.add_argument('--epochs',type=int,default=8);p.add_argument('--batch-size',type=int,default=256);p.add_argument('--learning-rate',type=float,default=2e-4);p.add_argument('--weight-decay',type=float,default=.03);p.add_argument('--seed',type=int,default=20260824);p.add_argument('--device',default='auto');a=p.parse_args()
 if a.output_dir.exists() and any(a.output_dir.iterdir()):raise FileExistsError(a.output_dir)
 random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);device=device_for(a.device)
 z=np.load(a.data);x=torch.from_numpy(z['x'].astype(np.float32));cand=torch.from_numpy(z['candidate_features'].astype(np.float32));coords=torch.from_numpy(z['candidate_coordinates'].astype(np.float32));valid=torch.from_numpy(z['candidate_valid']);target=torch.from_numpy(z['y'].astype(np.float32));labels=torch.from_numpy(z['label'].astype(np.int64));n=len(x);tr=int(.7*n);va=int(.85*n)
 model=PretrainModel(x.shape[-1],x.shape[1],cand.shape[-1],a.preset).to(device);params=sum(v.numel() for v in model.parameters());print(json.dumps({'device':str(device),'parameters':params,'preset':a.preset}),flush=True)
 loader=DataLoader(TensorDataset(torch.arange(tr)),batch_size=a.batch_size,shuffle=True);opt=torch.optim.AdamW(model.parameters(),lr=a.learning_rate,weight_decay=a.weight_decay);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.epochs)
 best=None;history=[]
 for epoch in range(1,a.epochs+1):
  model.train();total=0.;seen=0
  for (idx,) in loader:
   xb=x[idx].to(device);cb=cand[idx].to(device);vb=valid[idx].to(device);yb=target[idx].to(device);lb=labels[idx].to(device);co=coords[idx].to(device);opt.zero_grad(set_to_none=True);logits=model(xb,cb,vb);q,d=soft_target(co,vb,yb);hard=nn.functional.cross_entropy(logits,lb);soft=-(q*torch.log_softmax(logits,-1)).sum(-1).mean();expected=(torch.softmax(logits,-1)*d.masked_fill(~vb,0)).sum(-1).mean()/100.;loss=hard+.75*soft+.10*expected;loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();total+=float(loss.detach().cpu())*len(idx);seen+=len(idx)
  sched.step();val=collect(model,x,cand,coords,valid,target,tr,va,a.batch_size,device);row={'epoch':epoch,'train_loss':total/seen,'validation':val};history.append(row);print(json.dumps(row),flush=True)
  key=val['distance_soft_ce']
  if best is None or key<best[0]:best=(key,epoch,{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},val)
 assert best;_,be,state,val=best;model.load_state_dict(state);dev=collect(model,x,cand,coords,valid,target,va,n,a.batch_size,device)
 backbone={k.removeprefix('backbone.'):v for k,v in state.items() if k.startswith('backbone.') and not k.startswith('backbone.cell_embeddings.') and not k.startswith('backbone.geo_bias.') and not k.startswith('backbone.temperature.')}
 a.output_dir.mkdir(parents=True,exist_ok=True);torch.save({'backbone_state_dict':backbone,'preset':a.preset,'selected_epoch':be,'parameter_count':params},a.output_dir/'geostate_backbone.pt');report={'schema':'geostate-global-pretrain-v1','parameter_count':params,'preset':a.preset,'selected_epoch':be,'validation':val,'development':dev,'history':history,'protocol':'global 70/15/15; validation selects epoch; development historical block is diagnostic only'};(a.output_dir/'pretrain_metrics.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
