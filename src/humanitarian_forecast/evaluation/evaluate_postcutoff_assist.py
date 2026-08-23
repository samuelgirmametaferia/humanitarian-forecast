#!/usr/bin/env python3
"""Retrospective-only reweighting from events observed after each target date."""
from __future__ import annotations
import argparse,bisect,json
from collections import defaultdict
from pathlib import Path
import sys
import numpy as np,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from humanitarian_forecast.location.models.candidate_rank import ConflictCandidateRanker

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();d=np.load(a.data);meta=[json.loads(str(z)) for z in d['meta']];n=len(meta);te=int(.7*n);ve=int(.85*n)
    countries={z:i+1 for i,z in enumerate(sorted({m['country'] for m in meta[:ve]}))};conflicts={z:i+1 for i,z in enumerate(sorted({m['conflict_id'] for m in meta[:ve]}))};state=torch.load(a.checkpoint,map_location='cpu',weights_only=False);model=ConflictCandidateRanker(**state['model_config']);model.load_state_dict(state['model_state']);model.eval()
    occurrence=defaultdict(list)
    for m in meta:occurrence[(str(m['conflict_id']),round(m['target_lat'],5),round(m['target_lon'],5))].append(int(np.datetime64(m['target_date'],'D').astype(int)))
    configs=[(h,b) for h in (30,90,365,100000) for b in (.5,1.,2.,4.,8.)]
    def evaluate(start,end,choices):
        X=torch.from_numpy(d['x'][start:end]).float();F=torch.from_numpy(d['candidate_features'][start:end]).float();C=torch.from_numpy(d['candidate_coordinates'][start:end]).float();V=torch.from_numpy(d['candidate_valid'][start:end]);Y=torch.from_numpy(d['y'][start:end]).float();ci=torch.tensor([countries.get(m['country'],0) for m in meta[start:end]]);ki=torch.tensor([conflicts.get(m['conflict_id'],0) for m in meta[start:end]])
        logits=[]
        with torch.no_grad():
            for i in range(0,len(X),512):logits.append(model(X[i:i+512],F[i:i+512],V[i:i+512],ci[i:i+512],ki[i:i+512]))
        logits=torch.cat(logits);future=torch.zeros((len(X),len(choices),F.shape[1]))
        for i,m in enumerate(meta[start:end]):
            day=int(np.datetime64(m['target_date'],'D').astype(int));conflict=str(m['conflict_id'])
            for j in range(F.shape[1]):
                if not V[i,j]:continue
                lat=round(float(F[i,j,6]*90),5);lon=round(float(F[i,j,7]*180),5);days=occurrence[(conflict,lat,lon)];left=bisect.bisect_right(days,day)
                for q,(h,_) in enumerate(choices):future[i,q,j]=bisect.bisect_right(days,day+h)-left
        results=[]
        for q,(h,boost) in enumerate(choices):
            probability=(logits+boost*torch.log1p(future[:,q])).softmax(-1);prediction=(probability[:,:,None]*C).sum(1);error=torch.linalg.vector_norm(prediction-Y,dim=-1)*1000
            results.append({'future_window_days':h,'boost':boost,'mean_error_km':float(error.mean()),'median_error_km':float(error.median()),'p90_error_km':float(error.quantile(.9)),'within_25km':float((error<=25).float().mean())})
        return results
    validation=evaluate(te,ve,configs);selected=min(validation,key=lambda r:r['mean_error_km']);choice=(selected['future_window_days'],selected['boost']);test=evaluate(ve,n,[choice])[0]
    report={'mode':'retrospective post-cutoff assisted; not deployable for real future forecasts','selected_on_validation':selected,'untouched_test':test,'future_information':'same-conflict location occurrences strictly after target date; target-day record excluded'};a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
