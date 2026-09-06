#!/usr/bin/env python3
"""Direct v9 -> v10 Ethiopia specialization using v9's original center-distance objective.

Recipe selection uses the phase-1 v9 checkpoint and Ethiopia rows from the first
70% global chronology, with Ethiopia rows in 70-85% used only to choose the
fine-tuning mode/lr/epoch and geometric-median temperature.  The frozen recipe
is then applied to the real v9 phase-2 (85%-trained) checkpoint and evaluated
on Ethiopia rows in the final 15%.

This stays in the validated v9 model family: identical candidate support,
features, Transformer, center-distance loss, and geometric-median aggregation.
"""
from __future__ import annotations
import argparse, copy, json, random
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from humanitarian_forecast.location.models.candidate_rank import ConflictCandidateRanker, loss_fn
from humanitarian_forecast.location.training.calibrate_candidate_center import weighted_geometric_median

SCALE=1000.0
TEMPS=(0.5,0.65,0.8,1.0,1.25,1.5,2.0)

def maps(meta,end):
    c={v:i+1 for i,v in enumerate(sorted({m['country'] for m in meta[:end]}))}
    f={v:i+1 for i,v in enumerate(sorted({m['conflict_id'] for m in meta[:end]}))}
    return c,f

def dataset_for(z,meta,end):
    c,f=maps(meta,end)
    cid=torch.tensor([c.get(m['country'],0) for m in meta]); fid=torch.tensor([f.get(m['conflict_id'],0) for m in meta])
    ds=TensorDataset(torch.from_numpy(z['x']).float(),torch.from_numpy(z['candidate_features']).float(),torch.from_numpy(z['candidate_coordinates']).float(),torch.from_numpy(z['candidate_valid']),torch.from_numpy(z['label']),torch.from_numpy(z['y']).float(),cid,fid)
    return ds,c,f

def set_mode(model,mode):
    if mode=='all':
        for p in model.parameters(): p.requires_grad=True
    else:
        for p in model.parameters(): p.requires_grad=False
        prefixes=('context.','candidate.','bias.','country.','conflict.') if mode=='late' else ('context.','candidate.','bias.')
        for n,p in model.named_parameters():
            if n.startswith(prefixes): p.requires_grad=True
    return [n for n,p in model.named_parameters() if p.requires_grad]

def collect(model,ds,idx,device):
    dl=DataLoader(Subset(ds,idx.tolist()),batch_size=512)
    ls=[]; cs=[]; ys=[]
    model.eval()
    with torch.no_grad():
        for x,f,c,v,l,y,cid,fid in dl:
            ls.append(model(x.to(device),f.to(device),v.to(device),cid.to(device),fid.to(device)).cpu()); cs.append(c); ys.append(y)
    return torch.cat(ls),torch.cat(cs),torch.cat(ys)

def pm(logits,coords,y,temp):
    p=(logits/temp).softmax(-1); pred=weighted_geometric_median(p,coords)
    e=torch.linalg.vector_norm(pred-y,dim=-1)*SCALE
    return {'samples':len(e),'mean_error_km':float(e.mean()),'median_error_km':float(e.median()),'p90_error_km':float(e.quantile(.9)),'within_25km':float((e<=25).float().mean()),'within_50km':float((e<=50).float().mean()),'within_100km':float((e<=100).float().mean()),'within_200km':float((e<=200).float().mean())}

def best_temp(logits,coords,y):
    rows=[{'temperature':t,**pm(logits,coords,y,t)} for t in TEMPS]
    return min(rows,key=lambda r:r['mean_error_km']),rows

def train(parent_state,ds,idx,mode,lr,epochs,device,seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    m=ConflictCandidateRanker(**parent_state['model_config']); m.load_state_dict(parent_state['model_state']); m.to(device)
    names=set_mode(m,mode); opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=lr,weight_decay=.01)
    dl=DataLoader(Subset(ds,idx.tolist()),batch_size=128,shuffle=True)
    for ep in range(1,epochs+1):
        m.train(); total=0.0
        # MPS scaled-dot-product attention cannot train with dropout. Keep the
        # inherited Transformer deterministic during Ethiopia specialization;
        # eval() disables dropout but does not disable gradients.
        m.encoder.eval()
        for batch in dl:
            x,f,c,v,l,y,cid,fid=[q.to(device) for q in batch]; opt.zero_grad(set_to_none=True)
            logits=m(x,f,v,cid,fid); loss,_,_,_=loss_fn(logits,c,y,l,0.0,50.0); loss.backward(); torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad],1.0); opt.step(); total+=float(loss.detach().cpu())*len(x)
        yield ep,total/len(idx),m,names

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',type=Path,default=Path('data/location/conflict_candidates_32_spatial_v5.npz')); ap.add_argument('--phase1-parent',type=Path,default=Path('models/location/candidate_ranker/v9_phase1_ensemble_repro/v9_phase1.pt')); ap.add_argument('--phase2-parent',type=Path,default=Path('models/location/candidate_ranker/v9/candidate_ranker_calibrated.pt')); ap.add_argument('--output-dir',type=Path,default=Path('models/location/candidate_ranker_ethiopia/v10_center_direct_eval')); ap.add_argument('--seed',type=int,default=20260824); a=ap.parse_args()
    if a.output_dir.exists() and any(a.output_dir.iterdir()): raise FileExistsError(a.output_dir)
    z=np.load(a.data,allow_pickle=True); meta=[json.loads(str(r)) for r in z['meta']]; n=len(meta); e70=int(.70*n); e85=int(.85*n); device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    i70=np.asarray([i for i in range(e70) if meta[i]['country']=='Ethiopia']); iv=np.asarray([i for i in range(e70,e85) if meta[i]['country']=='Ethiopia']); i85=np.asarray([i for i in range(e85) if meta[i]['country']=='Ethiopia']); it=np.asarray([i for i in range(e85,n) if meta[i]['country']=='Ethiopia'])
    ds70,c70,f70=dataset_for(z,meta,e70); ds85,c85,f85=dataset_for(z,meta,e85)
    p1=torch.load(a.phase1_parent,map_location='cpu',weights_only=False); p2=torch.load(a.phase2_parent,map_location='cpu',weights_only=False)
    assert p1['model_config']['countries']==len(c70)+1 and p1['model_config']['conflicts']==len(f70)+1
    assert p2['model_config']['countries']==len(c85)+1 and p2['model_config']['conflicts']==len(f85)+1
    base=ConflictCandidateRanker(**p1['model_config']); base.load_state_dict(p1['model_state']); base.to(device)
    bl,bc,by=collect(base,ds70,iv,device); base_sel,base_trials=best_temp(bl,bc,by)
    sweep=[('scorer',5e-5),('scorer',1e-4),('scorer',2e-4),('late',5e-5),('late',1e-4),('late',2e-4),('all',1e-5),('all',3e-5)]
    rows=[]; best=None
    for trial,(mode,lr) in enumerate(sweep,1):
        for ep,loss,m,names in train(p1,ds70,i70,mode,lr,10,device,a.seed+trial):
            l,c,y=collect(m,ds70,iv,device); sel,_=best_temp(l,c,y); row={'trial':trial,'mode':mode,'lr':lr,'epoch':ep,'train_loss':loss,**sel}; rows.append(row); print(json.dumps(row),flush=True)
            if best is None or row['mean_error_km']<best['mean_error_km']:
                best={**row,'trainable':names}
    # Apply only the validation-selected recipe to the actual 85%-trained v9 parent.
    winner=None
    for ep,loss,m,names in train(p2,ds85,i85,best['mode'],best['lr'],best['epoch'],device,a.seed+100): winner=m
    assert winner is not None
    parent2=ConflictCandidateRanker(**p2['model_config']); parent2.load_state_dict(p2['model_state']); parent2.to(device)
    pl,pc,py=collect(parent2,ds85,it,device); wl,wc,wy=collect(winner,ds85,it,device)
    parent_test=pm(pl,pc,py,base_sel['temperature']); challenger_test=pm(wl,wc,wy,best['temperature'])
    report={'model':'v10_direct_v9_ethiopia_center','selection_protocol':'phase1 v9 + Ethiopia first70; select mode/lr/epoch/temp on Ethiopia 70-85; apply frozen recipe to phase2 v9 trained85; final15 Ethiopia reporting','rows':{'train70':len(i70),'validation':len(iv),'train85':len(i85),'test':len(it)},'v9_validation_calibration':base_sel,'selected':best,'v9_final15':parent_test,'v10_final15':challenger_test,'delta_final15':{k:challenger_test[k]-parent_test[k] for k in challenger_test if isinstance(challenger_test[k],(int,float)) and k!='samples'}}
    a.output_dir.mkdir(parents=True,exist_ok=False); torch.save({'model_config':p2['model_config'],'model_state':{k:v.detach().cpu() for k,v in winner.state_dict().items()},'parent':str(a.phase2_parent),'recipe':best,'calibration':{'aggregation':'weighted_geometric_median','temperature':best['temperature']}},a.output_dir/'model.pt'); (a.output_dir/'metrics.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__': main()
