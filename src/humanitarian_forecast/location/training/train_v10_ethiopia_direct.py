#!/usr/bin/env python3
"""Direct Ethiopia successor to candidate-ranker v9.

This deliberately stays inside the v9 model family.  It initializes from the
70%-trained v9 phase-1 checkpoint, keeps the exact v9 event/candidate feature
space, and fine-tunes increasingly large portions of the same Transformer
using the original v9 objective (candidate CE + weighted-center distance).
Model/epoch/temperature selection uses Ethiopia chronological validation only;
the final 15% is read only after the winning recipe is frozen.
"""
from __future__ import annotations

import argparse, copy, json, random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

from humanitarian_forecast.location.models.candidate_rank import ConflictCandidateRanker, loss_fn
from humanitarian_forecast.location.training.calibrate_candidate_center import weighted_geometric_median

SCALE=1000.0
TEMPS=(0.5,0.65,0.8,1.0,1.25,1.5)


def identity_maps(meta,end):
    countries={v:i+1 for i,v in enumerate(sorted({m['country'] for m in meta[:end]}))}
    conflicts={v:i+1 for i,v in enumerate(sorted({m['conflict_id'] for m in meta[:end]}))}
    return countries,conflicts


def collect(model,dataset,idx,batch,device):
    out=[]; cs=[]; ys=[]
    model.eval()
    with torch.no_grad():
        for xb,fb,cb,vb,lb,yb,cob,cfb in DataLoader(Subset(dataset,idx.tolist()),batch_size=batch):
            out.append(model(xb.to(device),fb.to(device),vb.to(device),cob.to(device),cfb.to(device)).cpu())
            cs.append(cb);ys.append(yb)
    return torch.cat(out),torch.cat(cs),torch.cat(ys)


def metrics_from_logits(logits,coords,target,temp):
    p=(logits/temp).softmax(-1)
    center=weighted_geometric_median(p,coords)
    e=torch.linalg.vector_norm(center-target,dim=-1)*SCALE
    d=torch.linalg.vector_norm(coords-target[:,None],dim=-1)*SCALE
    top=d[torch.arange(len(d)),logits.argmax(-1)]
    return {
        'samples':len(e),
        'geomedian_mean_error_km':float(e.mean()),
        'geomedian_median_error_km':float(e.median()),
        'geomedian_p90_error_km':float(e.quantile(.9)),
        'geomedian_within_25km':float((e<=25).float().mean()),
        'geomedian_within_50km':float((e<=50).float().mean()),
        'geomedian_within_100km':float((e<=100).float().mean()),
        'top1_mean_error_km':float(top.mean()),
        'top1_median_error_km':float(top.median()),
        'oracle_mean_error_km':float(d.min(-1).values.mean()),
    }


def best_temp(logits,coords,target):
    rows=[]
    for t in TEMPS:
        r=metrics_from_logits(logits,coords,target,t); rows.append((r['geomedian_mean_error_km'],t,r))
    return min(rows,key=lambda x:x[0])


def set_mode(model,mode):
    for p in model.parameters():p.requires_grad=False
    if mode=='late': prefixes=('context.','candidate.','bias.','country.','conflict.')
    elif mode=='last_block': prefixes=('encoder.layers.2.','context.','candidate.','bias.','country.','conflict.')
    elif mode=='all': prefixes=('',)
    else: raise ValueError(mode)
    names=[]
    for n,p in model.named_parameters():
        if n.startswith(prefixes):p.requires_grad=True;names.append(n)
    return names


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',type=Path,default=Path('data/location/conflict_candidates_32_spatial_v5.npz'))
    ap.add_argument('--parent',type=Path,default=Path('models/location/candidate_ranker/v9_phase1_ensemble_repro/v9_phase1.pt'))
    ap.add_argument('--output-dir',type=Path,default=Path('models/location/candidate_ranker_ethiopia/v10_direct_v9lineage_eval'))
    ap.add_argument('--epochs',type=int,default=20)
    ap.add_argument('--batch-size',type=int,default=128)
    ap.add_argument('--seed',type=int,default=20260824)
    ap.add_argument('--device',default='cpu',choices=('cpu','mps'))
    a=ap.parse_args()
    if a.output_dir.exists() and any(a.output_dir.iterdir()):raise FileExistsError(a.output_dir)
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    z=np.load(a.data,allow_pickle=True);meta=[json.loads(str(v)) for v in z['meta']];n=len(meta);te=int(.7*n);ve=int(.85*n)
    countries,conflicts=identity_maps(meta,te)
    country=torch.tensor([countries.get(m['country'],0) for m in meta]); conflict=torch.tensor([conflicts.get(m['conflict_id'],0) for m in meta])
    x=torch.from_numpy(z['x']).float();f=torch.from_numpy(z['candidate_features']).float();c=torch.from_numpy(z['candidate_coordinates']).float();v=torch.from_numpy(z['candidate_valid']);lab=torch.from_numpy(z['label']);y=torch.from_numpy(z['y']).float()
    ds=TensorDataset(x,f,c,v,lab,y,country,conflict)
    tr=np.asarray([i for i in range(te) if meta[i]['country']=='Ethiopia'],np.int64)
    va=np.asarray([i for i in range(te,ve) if meta[i]['country']=='Ethiopia'],np.int64)
    de=np.asarray([i for i in range(ve,n) if meta[i]['country']=='Ethiopia'],np.int64)
    device=torch.device(a.device)
    ck=torch.load(a.parent,map_location='cpu',weights_only=False);cfg=ck['model_config']
    parent=ConflictCandidateRanker(**cfg);parent.load_state_dict(ck['model_state']);parent.to(device).eval()
    vl,vc,vy=collect(parent,ds,va,512,device); base_v=best_temp(vl,vc,vy)
    dl,dc,dy=collect(parent,ds,de,512,device); base_d=metrics_from_logits(dl,dc,dy,base_v[1])
    print(json.dumps({'parent_validation':base_v[2],'parent_temperature':base_v[1],'parent_development':base_d,'rows':{'train':len(tr),'validation':len(va),'development':len(de)}},indent=2),flush=True)

    recipes=[
      ('late',1e-4,50.),('late',3e-4,50.),('late',1e-4,100.),('late',3e-4,100.),
      ('last_block',3e-5,50.),('last_block',1e-4,50.),('last_block',3e-5,100.),('last_block',1e-4,100.),
      ('all',1e-5,50.),('all',3e-5,50.),('all',1e-5,100.),('all',3e-5,100.),
    ]
    loader=DataLoader(Subset(ds,tr.tolist()),batch_size=a.batch_size,shuffle=True)
    best=None; trials=[]
    for ri,(mode,lr,cw) in enumerate(recipes,1):
        random.seed(a.seed+ri);np.random.seed(a.seed+ri);torch.manual_seed(a.seed+ri)
        m=ConflictCandidateRanker(**cfg);m.load_state_dict(ck['model_state']);m.to(device);names=set_mode(m,mode)
        opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=lr,weight_decay=.01)
        sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.epochs)
        erows=[]
        for ep in range(1,a.epochs+1):
            m.train();tot=0.
            for batch in loader:
                xb,fb,cb,vb,lb,yb,cob,cfb=[q.to(device) for q in batch];opt.zero_grad(set_to_none=True)
                logits=m(xb,fb,vb,cob,cfb);loss,ce,ed,cd=loss_fn(logits,cb,yb,lb,0.,cw);loss.backward();torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad],1.);opt.step();tot+=float(loss.detach().cpu())*len(xb)
            sched.step(); ql,qc,qy=collect(m,ds,va,512,device); score,t,r=best_temp(ql,qc,qy)
            row={'recipe':ri,'mode':mode,'lr':lr,'center_weight':cw,'epoch':ep,'train_loss':tot/len(tr),'temperature':t,**r};erows.append(row)
            print(json.dumps({k:row[k] for k in ['recipe','mode','lr','center_weight','epoch','train_loss','temperature','geomedian_mean_error_km','geomedian_median_error_km','geomedian_within_100km','top1_mean_error_km']}),flush=True)
            if best is None or score<best['geomedian_mean_error_km']:
                best={**row,'state':{k:v.detach().cpu().clone() for k,v in m.state_dict().items()},'trainable':sorted(names)}
        trials.append({'recipe':ri,'mode':mode,'lr':lr,'center_weight':cw,'epochs':erows})
    assert best
    win=ConflictCandidateRanker(**cfg);win.load_state_dict(best['state']);win.to(device).eval(); dl,dc,dy=collect(win,ds,de,512,device); dev=metrics_from_logits(dl,dc,dy,best['temperature'])
    a.output_dir.mkdir(parents=True,exist_ok=False)
    torch.save({'model_config':cfg,'model_state':best['state'],'parent':str(a.parent),'recipe':{k:best[k] for k in ['mode','lr','center_weight','epoch','temperature']},'scope':'evaluation'},a.output_dir/'model.pt')
    report={'model':'candidate_ranker_v10_direct_v9lineage_ethiopia_eval','parent':str(a.parent),'data':str(a.data),'rows':{'train':len(tr),'validation':len(va),'development':len(de)},'parent_validation':base_v[2],'parent_temperature':base_v[1],'parent_development':base_d,'selected':{k:v for k,v in best.items() if k not in ('state','trainable')},'development':dev,'delta_vs_parent':{'validation_geomedian_mean_km':best['geomedian_mean_error_km']-base_v[2]['geomedian_mean_error_km'],'development_geomedian_mean_km':dev['geomedian_mean_error_km']-base_d['geomedian_mean_error_km'],'development_within100':dev['geomedian_within_100km']-base_d['geomedian_within_100km'],'development_top1_mean_km':dev['top1_mean_error_km']-base_d['top1_mean_error_km']},'trials':trials,'note':'Direct v9-family Ethiopia fine-tune. Final 15% only read after validation winner frozen.'}
    (a.output_dir/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    print('FINAL',json.dumps({k:report[k] for k in ['selected','development','delta_vs_parent']},indent=2),flush=True)
if __name__=='__main__':main()
