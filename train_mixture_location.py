#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from mixture_location_model import MixtureLocationTransformer, mixture_nll

SCALE = 1000.0

def objective(logits, centers, sigmas, target, ranking_weight, ranking_temperature):
    """Density fit plus a detached nearest-component ranking target."""
    nll = mixture_nll(logits, centers, sigmas, target)
    if ranking_weight <= 0:
        return nll
    error2 = ((centers - target[:, None, :]) ** 2).sum(-1).detach()
    soft_nearest = torch.softmax(-error2 / (2 * ranking_temperature ** 2), dim=-1)
    ranking = -(soft_nearest * torch.log_softmax(logits, dim=-1)).sum(-1).mean()
    return nll + ranking_weight * ranking

def collect(model, loader, device):
    ls=[]; cs=[]; ss=[]; ys=[]; model.eval()
    with torch.no_grad():
        for x,y in loader:
            l,c,s=model(x.to(device)); ls.append(l.cpu());cs.append(c.cpu());ss.append(s.cpu());ys.append(y)
    return torch.cat(ls),torch.cat(cs),torch.cat(ss),torch.cat(ys)

def report(logits, centers, sigmas, target, multiplier, floor):
    probabilities=logits.softmax(-1); errors=torch.linalg.vector_norm(centers-target[:,None,:],dim=-1)*SCALE
    top=probabilities.argmax(-1); top_error=errors[torch.arange(len(errors)),top]
    best_error=errors.min(-1).values
    radii=torch.maximum(sigmas*SCALE*multiplier,torch.tensor(floor))
    union=(errors<=radii).any(-1)
    return {"samples":len(target),"top1_mean_error_km":float(top_error.mean()),
            "top1_median_error_km":float(top_error.median()),"top1_p90_error_km":float(top_error.quantile(.9)),
            "best_of_k_median_error_km":float(best_error.median()),"best_of_k_p90_error_km":float(best_error.quantile(.9)),
            "union_circle_coverage":float(union.float().mean()),"median_component_radius_km":float(radii.median())}

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=20);p.add_argument('--batch-size',type=int,default=384);p.add_argument('--learning-rate',type=float,default=2e-4)
    p.add_argument('--components',type=int,default=5);p.add_argument('--target-coverage',type=float,default=.8);p.add_argument('--radius-floor-km',type=float,default=50);p.add_argument('--seed',type=int,default=20260810)
    p.add_argument('--ranking-weight',type=float,default=0.0);p.add_argument('--ranking-temperature-km',type=float,default=100.0)
    a=p.parse_args();random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu');print('device',device)
    d=np.load(a.data);x=torch.from_numpy(d['x']).float();y=torch.from_numpy(d['y']).float();n=len(x);te=int(n*.7);ve=int(n*.85)
    loaders=[DataLoader(TensorDataset(x[i:j],y[i:j]),batch_size=a.batch_size,shuffle=(i==0)) for i,j in ((0,te),(te,ve),(ve,n))]
    
    # Phase 1: Train on x[:te], validate on x[te:ve]
    print(f"--- Starting Phase 1: training on x[:{te}] (70%), validating on x[{te}:{ve}] (15%) ---")
    model=MixtureLocationTransformer(x.shape[-1],x.shape[1],a.components).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=a.learning_rate,weight_decay=.01)
    
    a.output_dir.mkdir(parents=True,exist_ok=True);best_val_obj=float('inf');best_epoch=1;best_state=None
    for epoch in range(1,a.epochs+1):
        model.train();total=0
        for xb,yb in loaders[0]:
            xb,yb=xb.to(device),yb.to(device);opt.zero_grad(set_to_none=True);l,c,s=model(xb);loss=objective(l,c,s,yb,a.ranking_weight,a.ranking_temperature_km/SCALE);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();total+=loss.item()*len(xb)
        vl,vc,vs,vy=collect(model,loaders[1],device);v=float(objective(vl,vc,vs,vy,a.ranking_weight,a.ranking_temperature_km/SCALE))
        print(f'epoch={epoch:02d} train_objective={total/te:.4f} val_objective={v:.4f}')
        if v<best_val_obj:
            best_val_obj=v;best_epoch=epoch;best_state={k:v.cpu() for k,v in model.state_dict().items()}
            
    print(f"Phase 1 finished. Best validation objective: {best_val_obj:.4f} at epoch {best_epoch}")
    
    # Compute calibration multiplier using Phase 1 best model on loaders[1]
    model.load_state_dict(best_state);model.to(device)
    vl,vc,vs,vy=collect(model,loaders[1],device)
    ratios=(torch.linalg.vector_norm(vc-vy[:,None,:],dim=-1)/vs).min(-1).values
    multiplier=float(ratios.quantile(a.target_coverage))
    print(f"Calibration multiplier: {multiplier:.4f}")
    
    # Phase 2: Train on x[:ve] (0 to ve) for scaled epochs
    phase2_epochs = max(5, int(best_epoch * 0.85))
    print(f"--- Starting Phase 2: training on x[:{ve}] (85%) for {phase2_epochs} epochs ---")
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    model2=MixtureLocationTransformer(x.shape[-1],x.shape[1],a.components).to(device)
    opt2=torch.optim.AdamW(model2.parameters(),lr=a.learning_rate,weight_decay=.01)
    
    phase2_loader=DataLoader(TensorDataset(x[:ve],y[:ve]),batch_size=a.batch_size,shuffle=True)
    swa_state = None
    swa_count = 0
    swa_start_epoch = max(1, phase2_epochs - 4)
    
    for epoch in range(1,phase2_epochs+1):
        model2.train();total=0
        for xb,yb in phase2_loader:
            xb,yb=xb.to(device),yb.to(device);opt2.zero_grad(set_to_none=True);l,c,s=model2(xb);loss=objective(l,c,s,yb,a.ranking_weight,a.ranking_temperature_km/SCALE);loss.backward();torch.nn.utils.clip_grad_norm_(model2.parameters(),1);opt2.step();total+=loss.item()*len(xb)
        print(f'epoch={epoch:02d} train_objective={total/ve:.4f}')
        
        if epoch >= swa_start_epoch:
            current_state = {k: v.cpu().clone() for k, v in model2.state_dict().items()}
            if swa_state is None:
                swa_state = current_state
            else:
                for k in swa_state:
                    swa_state[k] += current_state[k]
            swa_count += 1
            
    for k in swa_state:
        if swa_state[k].is_floating_point():
            swa_state[k] /= swa_count
        else:
            swa_state[k] = torch.div(swa_state[k], swa_count, rounding_mode='floor')
            
    # Save the final Phase 2 SWA model
    torch.save({'model_config':model2.config,'model_state':swa_state,'data_policy':'UCDP+ReliefWeb only; Telegram excluded'},a.output_dir/'mixture_best.pt')
    
    # Evaluate Phase 2 model
    model2.load_state_dict(swa_state);model2.to(device);model2.eval()
    vl2,vc2,vs2,vy2=collect(model2,loaders[1],device)
    tl2,tc2,ts2,ty2=collect(model2,loaders[2],device)
    
    result={'components':a.components,'ranking_weight':a.ranking_weight,'ranking_temperature_km':a.ranking_temperature_km,'calibration_multiplier':multiplier,'target_coverage':a.target_coverage,'radius_floor_km':a.radius_floor_km,
            'validation':report(vl2,vc2,vs2,vy2,multiplier,a.radius_floor_km),'untouched_test':report(tl2,tc2,ts2,ty2,multiplier,a.radius_floor_km),
            'data_policy':'UCDP georeferenced events + ReliefWeb context; no Telegram'}
    (a.output_dir/'mixture_metrics.json').write_text(json.dumps(result,indent=2));(a.output_dir/'mixture_calibration.json').write_text(json.dumps({'multiplier':multiplier,'target_coverage':a.target_coverage,'minimum_radius_km':a.radius_floor_km},indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
