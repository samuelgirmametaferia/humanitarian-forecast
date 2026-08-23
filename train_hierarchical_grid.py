#!/usr/bin/env python3
from __future__ import annotations

import argparse,json,random
from pathlib import Path
import numpy as np,torch
from torch.utils.data import DataLoader,TensorDataset
from hierarchical_grid_model import HierarchicalGridClassifier,distance_aware_loss


def key(value,resolution):return round(float(value)/resolution)

def metrics(logits,country_mask,target_latlon,class_latlon):
    prediction=class_latlon[logits.masked_fill(~country_mask,-1e9).argmax(-1)]
    delta=prediction-target_latlon;mean_lat=torch.deg2rad((prediction[:,0]+target_latlon[:,0])/2)
    error=torch.sqrt((delta[:,1]*torch.cos(mean_lat)*111.32).square()+(delta[:,0]*111.32).square())
    return {'samples':len(error),'mean_error_km':float(error.mean()),'median_error_km':float(error.median()),
            'p90_error_km':float(error.quantile(.9)),'within_25km':float((error<=25).float().mean())}


def collect(model,loader,device,class_country):
    logits=[];targets=[];countries=[]
    model.eval()
    with torch.no_grad():
        for x,c,k,h,_,latlon in loader:
            logits.append(model(x.to(device),c.to(device),k.to(device),h.to(device)).cpu())
            targets.append(latlon);countries.append(c)
    l=torch.cat(logits);t=torch.cat(targets);c=torch.cat(countries)
    return l,class_country[None,:]==c[:,None],t


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--resolution',type=float,default=.5);p.add_argument('--epochs',type=int,default=15);p.add_argument('--batch-size',type=int,default=512)
    p.add_argument('--learning-rate',type=float,default=3e-4);p.add_argument('--distance-weight',type=float,default=2.0);p.add_argument('--seed',type=int,default=20260816)
    a=p.parse_args();random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    d=np.load(a.data);meta=[json.loads(str(v)) for v in d['meta']];n=len(meta);te=int(.7*n);ve=int(.85*n)
    countries={v:i+1 for i,v in enumerate(sorted({m['country'] for m in meta[:te]}))}
    conflicts={v:i+1 for i,v in enumerate(sorted({m['conflict_id'] for m in meta[:te]}))}
    cells=sorted({(m['country'],key(m['target_lat'],a.resolution),key(m['target_lon'],a.resolution)) for m in meta[:te]})
    classes={v:i for i,v in enumerate(cells)};class_latlon=torch.tensor([[v[1]*a.resolution,v[2]*a.resolution] for v in cells]).float()
    class_country=torch.tensor([countries.get(v[0],0) for v in cells])
    x=torch.from_numpy(d['x']).float();country=torch.tensor([countries.get(m['country'],0) for m in meta]);conflict=torch.tensor([conflicts.get(m['conflict_id'],0) for m in meta])
    horizon=torch.tensor([m['gap_days'] for m in meta]).float();labels=torch.tensor([classes.get((m['country'],key(m['target_lat'],a.resolution),key(m['target_lon'],a.resolution)),-1) for m in meta])
    latlon=torch.tensor([[m['target_lat'],m['target_lon']] for m in meta]).float();dataset=TensorDataset(x,country,conflict,horizon,labels,latlon)
    loaders=[DataLoader(torch.utils.data.Subset(dataset,range(i,j)),batch_size=a.batch_size,shuffle=(i==0)) for i,j in ((0,te),(te,ve),(ve,n))]
    device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu');print('device',device,'classes',len(cells),flush=True)
    model=HierarchicalGridClassifier(x.shape[-1],x.shape[1],len(countries)+1,len(conflicts)+1,len(cells)).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=a.learning_rate,weight_decay=.01);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.epochs)
    class_latlon_device=class_latlon.to(device);class_country_device=class_country.to(device);a.output_dir.mkdir(parents=True,exist_ok=True);best=float('inf');stale=0
    for epoch in range(1,a.epochs+1):
        model.train();total=0
        for xb,cb,kb,hb,yb,lb in loaders[0]:
            xb,cb,kb,hb,yb,lb=[v.to(device) for v in (xb,cb,kb,hb,yb,lb)];opt.zero_grad(set_to_none=True)
            logits=model(xb,cb,kb,hb);mask=class_country_device[None,:]==cb[:,None]
            loss,_,_=distance_aware_loss(logits,yb,lb,class_latlon_device,mask,a.distance_weight);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();total+=loss.detach().item()*len(xb)
        scheduler.step();vl,vm,vt=collect(model,loaders[1],device,class_country);value=metrics(vl,vm,vt,class_latlon)['mean_error_km']
        print(f'epoch={epoch:02d} train_loss={total/te:.4f} validation_mean_km={value:.2f}',flush=True)
        if value<best:best=value;stale=0;torch.save({'model_config':model.config,'model_state':{k:v.detach().cpu() for k,v in model.state_dict().items()},'countries':countries,'conflicts':conflicts,'cells':cells,'resolution':a.resolution},a.output_dir/'grid_best.pt')
        else:
            stale+=1
            if stale>=4:break
    state=torch.load(a.output_dir/'grid_best.pt',map_location='cpu',weights_only=False);model.load_state_dict(state['model_state']);model.to(device)
    vl,vm,vt=collect(model,loaders[1],device,class_country);tl,tm,tt=collect(model,loaders[2],device,class_country)
    report={'resolution_degrees':a.resolution,'classes':len(cells),'validation':metrics(vl,vm,vt,class_latlon),'untouched_test':metrics(tl,tm,tt,class_latlon),'unknown_target_cell_rate':{'validation':float((labels[te:ve]<0).float().mean()),'test':float((labels[ve:]<0).float().mean())},'data_policy':'training-vocabulary cells only; Telegram excluded'}
    (a.output_dir/'grid_metrics.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
