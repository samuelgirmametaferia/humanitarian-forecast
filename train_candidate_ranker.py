#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,random
from pathlib import Path
import numpy as np,torch
from torch.utils.data import DataLoader,TensorDataset
from candidate_rank_model import ConflictCandidateRanker,loss_fn
SCALE=1000.
def collect(model,loader,device):
    ls=[];cs=[];ys=[];model.eval()
    with torch.no_grad():
        for x,f,c,v,_,y,country,conflict in loader:ls.append(model(x.to(device),f.to(device),v.to(device),country.to(device),conflict.to(device)).cpu());cs.append(c);ys.append(y)
    return torch.cat(ls),torch.cat(cs),torch.cat(ys)
def metrics(logits,coordinates,target,weighted_center=False):
    d=torch.linalg.vector_norm(coordinates-target[:,None],dim=-1)*SCALE
    if weighted_center:
        prediction=(logits.softmax(-1)[:,:,None]*coordinates).sum(1);e=torch.linalg.vector_norm(prediction-target,dim=-1)*SCALE
    else:e=d[torch.arange(len(d)),logits.argmax(-1)]
    o=d.min(-1).values
    return {'samples':len(e),'mean_error_km':float(e.mean()),'median_error_km':float(e.median()),'p90_error_km':float(e.quantile(.9)),'within_25km':float((e<=25).float().mean()),'candidate_oracle_mean_km':float(o.mean())}
def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--epochs',type=int,default=15);p.add_argument('--batch-size',type=int,default=512);p.add_argument('--learning-rate',type=float,default=3e-4);p.add_argument('--distance-weight',type=float,default=2.0);p.add_argument('--center-weight',type=float,default=0.0);p.add_argument('--seed',type=int,default=20260816);a=p.parse_args()
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);d=np.load(a.data);x=torch.from_numpy(d['x']).float();f=torch.from_numpy(d['candidate_features']).float();c=torch.from_numpy(d['candidate_coordinates']).float();v=torch.from_numpy(d['candidate_valid']);label=torch.from_numpy(d['label']);y=torch.from_numpy(d['y']).float();meta=[json.loads(str(z)) for z in d['meta']];n=len(x);te=int(.7*n);ve=int(.85*n);device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu');print('device',device,flush=True)
    def identity_maps(end):
        countries={z:i+1 for i,z in enumerate(sorted({m['country'] for m in meta[:end]}))};conflicts={z:i+1 for i,z in enumerate(sorted({m['conflict_id'] for m in meta[:end]}))};return countries,conflicts
    def make_dataset(country_map,conflict_map):
        countries=torch.tensor([country_map.get(m['country'],0) for m in meta]);conflicts=torch.tensor([conflict_map.get(m['conflict_id'],0) for m in meta]);return TensorDataset(x,f,c,v,label,y,countries,conflicts)
    country1,conflict1=identity_maps(te);dataset1=make_dataset(country1,conflict1);loaders=[DataLoader(torch.utils.data.Subset(dataset1,range(i,j)),batch_size=a.batch_size,shuffle=(i==0)) for i,j in ((0,te),(te,ve),(ve,n))]
    def fresh(country_map,conflict_map):return ConflictCandidateRanker(x.shape[-1],f.shape[-1],x.shape[1],len(country_map)+1,len(conflict_map)+1).to(device)
    model=fresh(country1,conflict1);opt=torch.optim.AdamW(model.parameters(),lr=a.learning_rate,weight_decay=.01);schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.epochs);a.output_dir.mkdir(parents=True,exist_ok=True);best=float('inf');best_epoch=1;best_state=None;stale=0
    for epoch in range(1,a.epochs+1):
        model.train();total=0
        for xb,fb,cb,vb,lb,yb,countryb,conflictb in loaders[0]:
            xb,fb,cb,vb,lb,yb,countryb,conflictb=[z.to(device) for z in (xb,fb,cb,vb,lb,yb,countryb,conflictb)];opt.zero_grad(set_to_none=True);logits=model(xb,fb,vb,countryb,conflictb);loss,_,_,_=loss_fn(logits,cb,yb,lb,a.distance_weight,a.center_weight);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();total+=loss.detach().item()*len(xb)
        schedule.step();vl,vc,vy=collect(model,loaders[1],device);value=metrics(vl,vc,vy,a.center_weight>0)['mean_error_km'];print(f'phase1 epoch={epoch:02d} train_loss={total/te:.4f} validation_mean_km={value:.2f}',flush=True)
        if value<best:best=value;best_epoch=epoch;best_state={k:z.detach().cpu().clone() for k,z in model.state_dict().items()};stale=0
        else:
            stale+=1
            if stale>=4:break
    # Retrain from scratch on 85% for the validation-selected epoch count.
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);country2,conflict2=identity_maps(ve);dataset2=make_dataset(country2,conflict2);model=fresh(country2,conflict2);opt=torch.optim.AdamW(model.parameters(),lr=a.learning_rate,weight_decay=.01);phase2=DataLoader(torch.utils.data.Subset(dataset2,range(0,ve)),batch_size=a.batch_size,shuffle=True)
    for epoch in range(1,best_epoch+1):
        model.train()
        for xb,fb,cb,vb,lb,yb,countryb,conflictb in phase2:
            xb,fb,cb,vb,lb,yb,countryb,conflictb=[z.to(device) for z in (xb,fb,cb,vb,lb,yb,countryb,conflictb)];opt.zero_grad(set_to_none=True);logits=model(xb,fb,vb,countryb,conflictb);loss,_,_,_=loss_fn(logits,cb,yb,lb,a.distance_weight,a.center_weight);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
        print(f'phase2 epoch={epoch:02d}/{best_epoch}',flush=True)
    torch.save({'model_config':model.config,'model_state':{k:z.detach().cpu() for k,z in model.state_dict().items()},'selected_epochs':best_epoch},a.output_dir/'candidate_ranker_best.pt')
    # Phase-1 validation is the honest selection estimate; phase-2 is test only.
    phase1=fresh(country1,conflict1);phase1.load_state_dict(best_state);vl,vc,vy=collect(phase1,loaders[1],device);test_loader=DataLoader(torch.utils.data.Subset(dataset2,range(ve,n)),batch_size=a.batch_size);tl,tc,ty=collect(model,test_loader,device)
    report={'selected_epochs':best_epoch,'distance_weight':a.distance_weight,'center_weight':a.center_weight,'prediction':'probability_weighted_center' if a.center_weight>0 else 'top_candidate','validation':metrics(vl,vc,vy,a.center_weight>0),'untouched_test':metrics(tl,tc,ty,a.center_weight>0),'protocol':'phase1 train70/validate15; phase2 fresh train85/test15','data_policy':'cutoff-safe candidates; Telegram excluded'}
    (a.output_dir/'candidate_ranker_metrics.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
