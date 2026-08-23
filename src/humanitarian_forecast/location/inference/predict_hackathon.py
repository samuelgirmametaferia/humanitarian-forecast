#!/usr/bin/env python3
"""Hackathon predictor: center forecast plus ranked high-recall location set."""
from __future__ import annotations
import argparse,bisect,json,math
from collections import defaultdict
from pathlib import Path
import numpy as np,torch
from humanitarian_forecast.location.models.candidate_rank import ConflictCandidateRanker

def aggregate_center(probability, coordinates, method):
    center=(probability[:,None]*coordinates).sum(0)
    if method!='weighted_geometric_median':return center
    for _ in range(20):
        distance=torch.linalg.vector_norm(coordinates-center[None],dim=-1).clamp_min(1e-5)
        weight=probability/distance
        center=(weight[:,None]*coordinates).sum(0)/weight.sum()
    return center

def history_reference(events, method):
    valid=events[:,0]>.5;positions=events[:,1:3]
    if method=='anchor':return torch.zeros(2,dtype=events.dtype)
    if method.startswith('recent_'):
        count=int(method.split('_')[1]);valid=valid[-count:];positions=positions[-count:]
    if method.startswith('decay_'):
        scale=float(method.split('_')[1][:-1]);days=torch.expm1(events[:,3]*6).clamp_min(0);weight=torch.exp(-days/scale)*valid
    else:weight=valid.float()
    return (positions*weight[:,None]).sum(0)/weight.sum().clamp_min(1e-6)

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=Path('data/location/conflict_candidates_32_spatial_v5.npz'));p.add_argument('--checkpoint',type=Path,default=Path('models/location/candidate_ranker/v9/candidate_ranker_calibrated.pt'));p.add_argument('--index',type=int,default=-1);p.add_argument('--post-cutoff-assist',action='store_true');p.add_argument('--retrospective-reconstruction',action='store_true',help='Use the target-day record after cutoff to select the nearest generated candidate');p.add_argument('--top',type=int,default=32);a=p.parse_args()
    if a.post_cutoff_assist and a.retrospective_reconstruction:raise SystemExit('choose only one assisted mode')
    d=np.load(a.data);meta=[json.loads(str(z)) for z in d['meta']];n=len(meta);train_end=int(.85*n);i=a.index if a.index>=0 else n+a.index
    if not 0<=i<n:raise SystemExit(f'index must be in [0,{n-1}]')
    countries={z:j+1 for j,z in enumerate(sorted({m['country'] for m in meta[:train_end]}))};conflicts={z:j+1 for j,z in enumerate(sorted({m['conflict_id'] for m in meta[:train_end]}))};state=torch.load(a.checkpoint,map_location='cpu',weights_only=False);model=ConflictCandidateRanker(**state['model_config']);model.load_state_dict(state['model_state']);model.eval();m=meta[i]
    x=torch.from_numpy(d['x'][i:i+1]).float();f=torch.from_numpy(d['candidate_features'][i:i+1]).float();c=torch.from_numpy(d['candidate_coordinates'][i:i+1]).float();v=torch.from_numpy(d['candidate_valid'][i:i+1]);country=torch.tensor([countries.get(m['country'],0)]);conflict=torch.tensor([conflicts.get(m['conflict_id'],0)])
    with torch.no_grad():logits=model(x,f,v,country,conflict)[0]
    future_note='not used'
    if a.post_cutoff_assist:
        occurrence=defaultdict(list)
        for row in meta:
            occurrence[(str(row['conflict_id']),round(row['target_lat'],5),round(row['target_lon'],5))].append(int(np.datetime64(row['target_date'],'D').astype(int)))
        day=int(np.datetime64(m['target_date'],'D').astype(int));boost=torch.zeros_like(logits)
        for j in range(len(logits)):
            if not v[0,j]:continue
            key=(str(m['conflict_id']),round(float(f[0,j,6]*90),5),round(float(f[0,j,7]*180),5));days=occurrence[key];left=bisect.bisect_right(days,day);count=bisect.bisect_right(days,day+30)-left;boost[j]=math.log1p(count)
        logits+=boost;future_note='30-day post-target same-conflict occurrences reweighted candidates (retrospective only)'
    if a.retrospective_reconstruction:
        target=torch.from_numpy(d['y'][i]).float();distance=torch.linalg.vector_norm(c[0]-target[None],dim=-1).masked_fill(~v[0],float('inf'));selected=int(distance.argmin());logits.fill_(-1e9);logits[selected]=0
        future_note='target-day UCDP record selected the nearest generated candidate (retrospective reconstruction; not a future forecast)'
    calibration=state.get('center_calibration',{'aggregation':'weighted_mean','temperature':1.0});temperature=float(calibration['temperature']);aggregation=str(calibration['aggregation']);reference_name=str(calibration.get('history_reference','none'));candidate_weight=float(calibration.get('candidate_weight',1.0))
    if a.retrospective_reconstruction:reference_name='none';candidate_weight=1.0
    probability=(logits/temperature).softmax(-1);center=aggregate_center(probability,c[0],aggregation)
    if reference_name!='none':center=candidate_weight*center+(1-candidate_weight)*history_reference(x[0],reference_name)
    anchor_lat=float(m['anchor_lat']);anchor_lon=float(m['anchor_lon']);lat=anchor_lat+float(center[1])*1000/111.32;lon=anchor_lon+float(center[0])*1000/(111.32*max(.1,math.cos(math.radians(anchor_lat))))
    order=torch.argsort(probability,descending=True);candidates=[]
    for rank,j in enumerate(order[:min(a.top,int(v[0].sum()))],1):
        j=int(j);candidates.append({'rank':rank,'probability':round(float(probability[j]),6),'latitude':round(float(f[0,j,6]*90),5),'longitude':round(float(f[0,j,7]*180),5)})
    cutoff=str(np.datetime64(m['target_date'])-np.timedelta64(int(m['gap_days']),'D'))
    warning=('Retrospective reconstruction, not a deployable forecast.' if a.retrospective_reconstruction else 'Research/hackathon output. Candidate-set error is best-of-set, not top-1 center error.')
    print(json.dumps({'country':m['country'],'conflict':m['conflict'],'observation_cutoff':cutoff,'forecast_horizon_days':int(m['gap_days']),'center_forecast':{'latitude':round(lat,5),'longitude':round(lon,5)},'center_calibration':{'aggregation':aggregation,'temperature':temperature,'history_reference':reference_name,'candidate_weight':candidate_weight},'ranked_high_recall_candidates':candidates,'post_cutoff_assistance':future_note,'warning':warning},indent=2))
if __name__=='__main__':main()
