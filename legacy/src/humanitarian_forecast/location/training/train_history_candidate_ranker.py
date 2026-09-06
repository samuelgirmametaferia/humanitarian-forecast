#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from humanitarian_forecast.location.models.history_candidate import HistoryCandidateRanker, candidate_distances, ranking_loss

SCALE_KM = 1000.0


def collect(model, loader, device):
    logits=[]; features=[]; targets=[];model.eval()
    with torch.no_grad():
        for x,y in loader:
            logits.append(model(x.to(device)).cpu());features.append(x);targets.append(y)
    return torch.cat(logits),torch.cat(features),torch.cat(targets)


def metrics(logits, features, target):
    distance=candidate_distances(features,target)*SCALE_KM
    distance=distance.masked_fill(features[:,:,0] <= .5,float('inf'))
    selected=distance[torch.arange(len(distance)),logits.argmax(-1)]
    oracle=distance.min(-1).values
    return {"samples":len(target),"mean_error_km":float(selected.mean()),
            "median_error_km":float(selected.median()),"p90_error_km":float(selected.quantile(.9)),
            "within_25km":float((selected<=25).float().mean()),
            "candidate_oracle_mean_km":float(oracle.mean())}


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--epochs',type=int,default=12)
    p.add_argument('--batch-size',type=int,default=768);p.add_argument('--learning-rate',type=float,default=3e-4)
    p.add_argument('--distance-weight',type=float,default=.5);p.add_argument('--seed',type=int,default=20260816)
    a=p.parse_args();random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu');print('device',device,flush=True)
    d=np.load(a.data);x=torch.from_numpy(d['x']).float();y=torch.from_numpy(d['y']).float();n=len(x)
    te=int(.7*n);ve=int(.85*n)
    loaders=[DataLoader(TensorDataset(x[i:j],y[i:j]),batch_size=a.batch_size,shuffle=(i==0))
             for i,j in ((0,te),(te,ve),(ve,n))]
    model=HistoryCandidateRanker(x.shape[-1],x.shape[1]).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=a.learning_rate,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.epochs);a.output_dir.mkdir(parents=True,exist_ok=True)
    best=float('inf');stale=0
    for epoch in range(1,a.epochs+1):
        model.train();total=0
        for xb,yb in loaders[0]:
            xb,yb=xb.to(device),yb.to(device);opt.zero_grad(set_to_none=True);logits=model(xb)
            loss,_,_=ranking_loss(logits,xb,yb,a.distance_weight);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();total+=loss.detach().item()*len(xb)
        scheduler.step();vl,vx,vy=collect(model,loaders[1],device);vm=metrics(vl,vx,vy)
        print(f"epoch={epoch:02d} train_loss={total/te:.4f} val_mean={vm['mean_error_km']:.2f} val_median={vm['median_error_km']:.2f}",flush=True)
        if vm['mean_error_km']<best:
            best=vm['mean_error_km'];stale=0;torch.save({'model_config':model.config,'model_state':{k:v.detach().cpu() for k,v in model.state_dict().items()},'selection_metric':'validation_mean_error'},a.output_dir/'history_ranker_best.pt')
        else:
            stale+=1
            if stale>=4:break
    state=torch.load(a.output_dir/'history_ranker_best.pt',map_location='cpu',weights_only=False);model.load_state_dict(state['model_state']);model.to(device)
    vl,vx,vy=collect(model,loaders[1],device);tl,tx,ty=collect(model,loaders[2],device)
    report={'validation':metrics(vl,vx,vy),'untouched_test':metrics(tl,tx,ty),'data_policy':'UCDP+ReliefWeb only; Telegram excluded'}
    (a.output_dir/'history_ranker_metrics.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
