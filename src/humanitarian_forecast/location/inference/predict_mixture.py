#!/usr/bin/env python3
"""Produce multiple probability-weighted circles with optional situation overrides."""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np,torch
from humanitarian_forecast.location.models.mixture import MixtureLocationTransformer

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--calibration',type=Path,required=True);p.add_argument('--index',type=int,default=-1);p.add_argument('--scenario',type=Path)
    a=p.parse_args();d=np.load(a.data);i=a.index if a.index>=0 else len(d['x'])+a.index;x=d['x'][i].copy();meta=json.loads(str(d['meta'][i]));scenario=json.loads(a.scenario.read_text()) if a.scenario else {}
    latest=x[-1];
    if 'forecast_horizon_days' in scenario:
        requested_gap=max(1,int(scenario['forecast_horizon_days']));original_gap=max(1,int(meta['gap_days']));delta=requested_gap-original_gap
        valid=x[:,0]>.5;days_ago=np.expm1(x[valid,3]*6.0);x[valid,3]=np.log1p(np.maximum(0,days_ago+delta))/6.0
        target_date=np.datetime64(meta['target_date'])+np.timedelta64(delta,'D');year_start=target_date.astype('datetime64[Y]');day_of_year=int((target_date-year_start)/np.timedelta64(1,'D'))+1;angle=2*math.pi*day_of_year/365.25
        if x.shape[1]>=19:x[valid,17]=math.sin(angle);x[valid,18]=math.cos(angle)
    if 'fatalities' in scenario: latest[4]=math.log1p(max(0,scenario['fatalities']))/6
    if 'civilian_fatalities' in scenario: latest[5]=math.log1p(max(0,scenario['civilian_fatalities']))/6
    if 'violence_type' in scenario: latest[6:9]=0; latest[5+int(scenario['violence_type'])]=1
    if 'source_count' in scenario: latest[10]=math.log1p(max(0,scenario['source_count']))/5
    for key,index in [('reliefweb_activity',11),('reliefweb_attack',12),('reliefweb_harm',13),('reliefweb_displacement',14)]:
        if key in scenario: latest[index]=math.log1p(max(0,scenario[key]))/8
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=False);model=MixtureLocationTransformer(**state['model_config']);model.load_state_dict(state['model_state']);model.eval()
    with torch.no_grad(): logits,centers,sigmas=model(torch.from_numpy(x[None]).float())
    probs=logits.softmax(-1)[0].numpy();centers=centers[0].numpy()*1000;sigmas=sigmas[0].numpy()*1000;cal=json.loads(a.calibration.read_text());alat=float(meta['anchor_lat']);alon=float(meta['anchor_lon']);pred=[]
    for rank,k in enumerate(np.argsort(-probs),1):
        east,north=centers[k];lat=alat+north/111.32;lon=alon+east/(111.32*max(.1,math.cos(math.radians(alat))));radius=max(cal['minimum_radius_km'],sigmas[k]*cal['multiplier'])
        pred.append({'rank':rank,'probability':round(float(probs[k]),4),'center_latitude_coarse':round(lat*4)/4,'center_longitude_coarse':round(lon*4)/4,'uncertainty_radius_km':round(radius)})
    print(json.dumps({'country':meta['country'],'conflict':meta['conflict'],'forecast_horizon_days':int(scenario.get('forecast_horizon_days',meta['gap_days'])),'scenario_overrides':scenario,'predictions':pred,'warning':'Research-only coarse predictions; not verified fronts or evacuation orders'},indent=2))
if __name__=='__main__':main()
