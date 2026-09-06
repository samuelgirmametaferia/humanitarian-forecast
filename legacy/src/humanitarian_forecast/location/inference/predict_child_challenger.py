#!/usr/bin/env python3
"""Shadow inference for the parent-preserving coarse-to-fine location challenger."""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np
from humanitarian_forecast.location.inference.candidate_child_support import CandidateChildSupport

def to_latlon(anchor_lat:float,anchor_lon:float,xy:np.ndarray)->tuple[float,float]:
 east=float(xy[0])*1000.;north=float(xy[1])*1000.;lat=anchor_lat+north/111.32;lon=anchor_lon+east/(111.32*max(.1,math.cos(math.radians(anchor_lat))));return lat,lon

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,default=Path('data/location/conflict_candidates_32_motion_actor_transfer_v8.npz'));p.add_argument('--coarse-model',type=Path,default=Path('models/location/geo_lambdarank_actor_transfer_ethiopia/v1/geo_lambdarank.txt'));p.add_argument('--coarse-config',type=Path,default=Path('models/location/geo_lambdarank_actor_transfer_ethiopia/v1/ensemble_config.json'));p.add_argument('--refiner-dir',type=Path,default=Path('models/location/candidate_refiner/actor_transfer_v1'));p.add_argument('--policy',type=Path,required=True);p.add_argument('--gate-model',type=Path);p.add_argument('--index',type=int,default=-1);p.add_argument('--top',type=int,default=20);a=p.parse_args()
 z=np.load(a.data,allow_pickle=True);meta=[json.loads(str(v)) for v in z['meta']];n=len(meta);i=a.index if a.index>=0 else n+a.index
 if not 0<=i<n:raise SystemExit(f'index must be in [0,{n-1}]')
 model=CandidateChildSupport(a.coarse_model,a.coarse_config,a.refiner_dir,a.policy,a.gate_model);r=model.predict(z['x'][i:i+1],z['candidate_features'][i:i+1],z['candidate_coordinates'][i:i+1],z['candidate_valid'][i:i+1]);m=meta[i];prob=r.probability[0];valid=r.valid[0];coords=r.coordinates[0];order=np.argsort(-prob);rows=[]
 for j in order:
  if not valid[j] or len(rows)>=a.top:continue
  lat,lon=to_latlon(float(m['anchor_lat']),float(m['anchor_lon']),coords[j]);rows.append({'rank':len(rows)+1,'probability':round(float(prob[j]),7),'latitude':round(lat,5),'longitude':round(lon,5),'parent_candidate_index':int(r.parent_index[0,j]),'refined_child':bool(r.is_child[0,j])})
 center=(prob[:,None]*coords).sum(0);clat,clon=to_latlon(float(m['anchor_lat']),float(m['anchor_lon']),center);cutoff=str(np.datetime64(m['target_date'])-np.timedelta64(int(m['gap_days']),'D'))
 print(json.dumps({'schema':'location-child-shadow-forecast/v1','country':m['country'],'conflict':m.get('conflict'),'observation_cutoff':cutoff,'forecast_horizon_days':int(m['gap_days']),'center_forecast':{'latitude':round(clat,5),'longitude':round(clon,5)},'ranked_support':rows,'support_size':int(valid.sum()),'warning':'Shadow research challenger only; does not replace the promoted production location model.'},indent=2))
if __name__=='__main__':main()
