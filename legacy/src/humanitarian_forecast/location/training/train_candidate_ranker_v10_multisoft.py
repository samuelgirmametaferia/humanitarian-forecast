#!/usr/bin/env python3
"""V10 direct candidate-ranker: v9 architecture + 64 support + distance-soft Ethiopia weighting.

This is intentionally not a new model family. It uses ConflictCandidateRanker unchanged.
The only changes from v9 are (1) 64 v9-style spatial candidates, (2) distance-aware
candidate supervision, and (3) higher loss weight for Ethiopia while retaining global
replay. Recipe/epoch/calibration are selected on the chronological Ethiopia validation
block. The final 15% is read only after the winner is frozen.
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, Subset
from humanitarian_forecast.location.models.candidate_rank import ConflictCandidateRanker
from humanitarian_forecast.location.training.calibrate_candidate_center import weighted_geometric_median

SCALE=1000.0
TEMPS=(0.35,0.5,0.65,0.8,1.0,1.25)

def maps(meta,end):
    co={v:i+1 for i,v in enumerate(sorted({m['country'] for m in meta[:end]}))}
    cf={v:i+1 for i,v in enumerate(sorted({m['conflict_id'] for m in meta[:end]}))}
    return co,cf

def soft_target(coords,valid,target,radius):
    d=torch.linalg.vector_norm(coords-target[:,None],dim=-1)*SCALE
    w=torch.exp(-.5*(d/radius)**2)*valid.float(); s=w.sum(1,keepdim=True)
    bad=s[:,0]<=1e-12
    if bad.any():
        near=d.masked_fill(~valid,float('inf')).argmin(1);w[bad]=0;w[bad,near[bad]]=1;s=w.sum(1,keepdim=True)
    return w/s.clamp_min(1e-12)

def collect(model,ds,idx,device,batch=512):
    ls=[];cs=[];ys=[]
    model.eval()
    with torch.no_grad():
      for xb,fb,cb,vb,lb,yb,cob,cfb,ieb in DataLoader(Subset(ds,idx.tolist()),batch_size=batch):
        ls.append(model(xb.to(device),fb.to(device),vb.to(device),cob.to(device),cfb.to(device)).cpu());cs.append(cb);ys.append(yb)
    return torch.cat(ls),torch.cat(cs),torch.cat(ys)

def metric(logits,coords,target,temp,agg):
    p=(logits/temp).softmax(-1)
    center=(p[:,:,None]*coords).sum(1) if agg=='weighted_mean' else weighted_geometric_median(p,coords)
    e=torch.linalg.vector_norm(center-target,dim=-1)*SCALE
    d=torch.linalg.vector_norm(coords-target[:,None],dim=-1)*SCALE
    top=d[torch.arange(len(d)),logits.argmax(-1)]
    return {'samples':len(e),'mean_error_km':float(e.mean()),'median_error_km':float(e.median()),'p90_error_km':float(e.quantile(.9)),'within25':float((e<=25).float().mean()),'within50':float((e<=50).float().mean()),'within100':float((e<=100).float().mean()),'top1_mean_error_km':float(top.mean()),'oracle_mean_error_km':float(d.min(-1).values.mean())}
def calibrate(logits,coords,target):
    best=None
    for agg in ('weighted_mean','weighted_geometric_median'):
      for t in TEMPS:
        r=metric(logits,coords,target,t,agg);key=r['mean_error_km']
        if best is None or key<best[0]:best=(key,agg,t,r)
    return best

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=Path('data/location/conflict_candidates_64_spatial_v10.npz'));p.add_argument('--output-dir',type=Path,default=Path('models/location/candidate_ranker_ethiopia/v10_64_multisoft_eval'));p.add_argument('--epochs',type=int,default=12);p.add_argument('--batch-size',type=int,default=384);p.add_argument('--seed',type=int,default=20260824);p.add_argument('--device',default='mps');a=p.parse_args()
    if a.output_dir.exists() and any(a.output_dir.iterdir()):raise FileExistsError(a.output_dir)
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    z=np.load(a.data,allow_pickle=True);meta=[json.loads(str(v)) for v in z['meta']];n=len(meta);te=int(.7*n);ve=int(.85*n)
    co_map,cf_map=maps(meta,te); co=torch.tensor([co_map.get(m['country'],0) for m in meta]);cf=torch.tensor([cf_map.get(m['conflict_id'],0) for m in meta]);ie=torch.tensor([m['country']=='Ethiopia' for m in meta])
    x=torch.from_numpy(z['x']).float();f=torch.from_numpy(z['candidate_features']).float();c=torch.from_numpy(z['candidate_coordinates']).float();v=torch.from_numpy(z['candidate_valid']);lab=torch.from_numpy(z['label']).long();y=torch.from_numpy(z['y']).float();ds=TensorDataset(x,f,c,v,lab,y,co,cf,ie)
    tr=np.arange(te,dtype=np.int64); va=np.asarray([i for i in range(te,ve) if meta[i]['country']=='Ethiopia'],np.int64); de=np.asarray([i for i in range(ve,n) if meta[i]['country']=='Ethiopia'],np.int64)
    device=torch.device(a.device if a.device!='mps' or torch.backends.mps.is_available() else 'cpu')
    recipes=[
      {'name':'soft50','radius1':50.,'radius2':None,'mix':1.,'hard':.20,'center':50.,'ethiopia_weight':5.,'lr':3e-4},
      {'name':'soft100','radius1':100.,'radius2':None,'mix':1.,'hard':.15,'center':50.,'ethiopia_weight':5.,'lr':3e-4},
      {'name':'multi50_150','radius1':50.,'radius2':150.,'mix':.65,'hard':.15,'center':75.,'ethiopia_weight':8.,'lr':3e-4},
    ]
    loader=DataLoader(Subset(ds,tr.tolist()),batch_size=a.batch_size,shuffle=True)
    best=None; trials=[]
    for rid,rp in enumerate(recipes,1):
      random.seed(a.seed+rid);np.random.seed(a.seed+rid);torch.manual_seed(a.seed+rid)
      m=ConflictCandidateRanker(x.shape[-1],f.shape[-1],x.shape[1],len(co_map)+1,len(cf_map)+1).to(device);opt=torch.optim.AdamW(m.parameters(),lr=rp['lr'],weight_decay=.01);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.epochs);er=[]
      for ep in range(1,a.epochs+1):
        m.train();total=0.;den=0.
        for xb,fb,cb,vb,lb,yb,cob,cfb,ieb in loader:
          xb,fb,cb,vb,lb,yb,cob,cfb,ieb=[q.to(device) for q in (xb,fb,cb,vb,lb,yb,cob,cfb,ieb)];opt.zero_grad(set_to_none=True);logits=m(xb,fb,vb,cob,cfb);logp=torch.log_softmax(logits,-1);q1=soft_target(cb,vb,yb,rp['radius1']);q=q1
          if rp['radius2'] is not None:q=rp['mix']*q1+(1-rp['mix'])*soft_target(cb,vb,yb,rp['radius2'])
          soft=-(q*logp).sum(-1);hard=nn.functional.cross_entropy(logits,lb,reduction='none');prob=logits.softmax(-1);center=(prob[:,:,None]*cb).sum(1);cd=torch.linalg.vector_norm(center-yb,dim=-1);per=soft+rp['hard']*hard+rp['center']*cd;w=torch.where(ieb,torch.full_like(per,rp['ethiopia_weight']),torch.ones_like(per));loss=(per*w).sum()/w.sum();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step();total+=float((per*w).sum().detach().cpu());den+=float(w.sum().cpu())
        sched.step();vl,vc,vy=collect(m,ds,va,device);score,agg,temp,vr=calibrate(vl,vc,vy);row={'recipe':rid,**rp,'epoch':ep,'train_loss':total/den,'aggregation':agg,'temperature':temp,**vr};er.append(row);print(json.dumps({k:row[k] for k in ['recipe','name','epoch','train_loss','aggregation','temperature','mean_error_km','median_error_km','p90_error_km','within25','within100','top1_mean_error_km']}),flush=True)
        if best is None or score<best['mean_error_km']:
          best={**row,'state':{k:v.detach().cpu().clone() for k,v in m.state_dict().items()},'config':m.config}
      trials.append({'recipe':rid,**rp,'epochs':er})
    win=ConflictCandidateRanker(**best['config']);win.load_state_dict(best['state']);win.to(device).eval();dl,dc,dy=collect(win,ds,de,device);dev=metric(dl,dc,dy,best['temperature'],best['aggregation'])
    a.output_dir.mkdir(parents=True,exist_ok=False);torch.save({'model_config':best['config'],'model_state':best['state'],'selected':{k:v for k,v in best.items() if k not in ('state','config')},'data':str(a.data),'scope':'evaluation'},a.output_dir/'model.pt')
    report={'model':'candidate_ranker_v10_64_multisoft','architecture':'ConflictCandidateRanker (unchanged v9 family)','data':str(a.data),'rows':{'global_train':len(tr),'ethiopia_validation':len(va),'ethiopia_development':len(de)},'selected':{k:v for k,v in best.items() if k not in ('state','config')},'development':dev,'trials':trials,'protocol':'global first70 training with Ethiopia loss upweight; recipe/epoch/calibration selected on Ethiopia 70-85 validation; final15 Ethiopia read once after winner frozen'};(a.output_dir/'metrics.json').write_text(json.dumps(report,indent=2)+'\n');print('FINAL',json.dumps({'selected':report['selected'],'development':dev},indent=2),flush=True)
if __name__=='__main__':main()
