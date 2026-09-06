#!/usr/bin/env python3
"""Measure local-H3 support oracle around ranked historical candidates.

This is an oracle/coverage diagnostic only: target coordinates are used solely to
measure whether local H3 expansion can represent the eventual location. The
coarse ranker determines which parent candidates are expanded, while H3 grid
rings supply previously unseen nearby support.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import h3
import numpy as np
from lightgbm import Booster
from humanitarian_forecast.location.training.train_geo_lambdarank_v2 import _flatten_inference_features,_scores_to_matrix

EARTH_RADIUS_KM=6371.0088

def haversine(lat,lon,tlat,tlon):
 p1=np.deg2rad(lat);p2=np.deg2rad(tlat);dp=p2-p1;dl=np.deg2rad(tlon-lon);a=np.sin(dp/2)**2+np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
 return EARTH_RADIUS_KM*2*np.arctan2(np.sqrt(a),np.sqrt(np.maximum(1-a,0)))

def evaluate(x,candidates,valid,meta,indices,model,iteration,resolution,topk,ring):
 flat,_=_flatten_inference_features(x[indices],candidates[indices],valid[indices]);raw=_scores_to_matrix(model.predict(flat,num_iteration=iteration),valid[indices]);order=np.argsort(-raw,axis=1);errors=[];sizes=[]
 for q,i in enumerate(indices):
  points={}
  for j in order[q,:topk]:
   if not valid[i,j]:continue
   lat=float(candidates[i,j,6]*90.);lon=float(candidates[i,j,7]*180.);points[(round(lat,6),round(lon,6))]=(lat,lon)
   base=h3.latlng_to_cell(lat,lon,resolution)
   for cell in h3.grid_disk(base,ring):
    clat,clon=h3.cell_to_latlng(cell);points[(round(clat,6),round(clon,6))]=(clat,clon)
  arr=np.asarray(list(points.values()),np.float64);tlat=float(meta[i]['target_lat']);tlon=float(meta[i]['target_lon']);d=haversine(arr[:,0],arr[:,1],tlat,tlon);errors.append(float(d.min()));sizes.append(len(arr))
 e=np.asarray(errors)
 return {'samples':len(e),'mean_oracle_km':float(e.mean()),'median_oracle_km':float(np.median(e)),'p90_oracle_km':float(np.quantile(e,.9)),'within_10km':float((e<=10).mean()),'within_25km':float((e<=25).mean()),'within_50km':float((e<=50).mean()),'within_100km':float((e<=100).mean()),'average_support_points':float(np.mean(sizes))}

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--coarse-model',type=Path,required=True);p.add_argument('--coarse-config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--resolution',type=int,default=5);p.add_argument('--country',default='Ethiopia');a=p.parse_args()
 z=np.load(a.data,allow_pickle=True);x=z['x'].astype(np.float32);c=z['candidate_features'].astype(np.float32);valid=z['candidate_valid'];meta=[json.loads(str(v)) for v in z['meta']];n=len(x);tr=int(.7*n);va=int(.85*n);model=Booster(model_file=str(a.coarse_model));cfg=json.loads(a.coarse_config.read_text());iteration=int(cfg.get('selected_iteration',model.num_trees()));country=a.country.casefold()
 splits={'validation':np.asarray([i for i in range(tr,va) if str(meta[i].get('country','')).casefold()==country],np.int64),'development':np.asarray([i for i in range(va,n) if str(meta[i].get('country','')).casefold()==country],np.int64)}
 report={'schema':'local-h3-candidate-oracle-v1','resolution':a.resolution,'country':a.country,'coarse_model':str(a.coarse_model),'selected_iteration':iteration,'oracle_warning':'Target coordinates are used only for coverage/oracle measurement; these metrics are not deployable forecast performance.','splits':{}}
 for name,idx in splits.items():
  report['splits'][name]={}
  for topk in (1,3,5,10,32):
   for ring in (1,2,3,4):
    key=f'top{topk}_ring{ring}';m=evaluate(x,c,valid,meta,idx,model,iteration,a.resolution,topk,ring);report['splits'][name][key]=m;print(json.dumps({'split':name,'setting':key,**m}),flush=True)
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'output':str(a.output),'development_top32_ring4':report['splits']['development']['top32_ring4']},indent=2))
if __name__=='__main__':main()
