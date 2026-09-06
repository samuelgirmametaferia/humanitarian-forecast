#!/usr/bin/env python3
"""Export raw actor-transfer and propagation-kernel scores in a LightGBM-only process."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import lightgbm as lgb
import numpy as np
from humanitarian_forecast.location.training.train_geo_lambdarank_v2 import _flatten_features as actor_features, _scores_to_matrix as actor_matrix
from humanitarian_forecast.location.training.train_geo_lambdarank_kernel import _flatten_features as kernel_features, _scores_to_matrix as kernel_matrix

def export_one(bundle,lo,hi,booster,iteration,builder,to_matrix):
    f,_,_,_=builder(bundle['x'][lo:hi].astype(np.float32),bundle['candidate_features'][lo:hi].astype(np.float32),bundle['candidate_valid'][lo:hi],bundle['candidate_coordinates'][lo:hi].astype(np.float32),bundle['y'][lo:hi].astype(np.float32))
    return to_matrix(booster.predict(f,num_iteration=iteration),bundle['candidate_valid'][lo:hi]).astype(np.float32)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--actor-dir',type=Path,required=True);p.add_argument('--kernel-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    z=np.load(a.data); n=len(z['x']);tr=int(.70*n);va=int(.85*n)
    ac=json.loads((a.actor_dir/'ensemble_config.json').read_text());kc=json.loads((a.kernel_dir/'ensemble_config.json').read_text())
    ab=lgb.Booster(model_file=str(a.actor_dir/'geo_lambdarank.txt')); kb=lgb.Booster(model_file=str(a.kernel_dir/'geo_lambdarank.txt'))
    print('actor validation',flush=True); av=export_one(z,tr,va,ab,int(ac['selected_iteration']),actor_features,actor_matrix)
    print('actor development',flush=True); ad=export_one(z,va,n,ab,int(ac['selected_iteration']),actor_features,actor_matrix)
    print('kernel validation',flush=True); kv=export_one(z,tr,va,kb,int(kc['selected_iteration']),kernel_features,kernel_matrix)
    print('kernel development',flush=True); kd=export_one(z,va,n,kb,int(kc['selected_iteration']),kernel_features,kernel_matrix)
    a.output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.output,actor_validation=av,actor_development=ad,kernel_validation=kv,kernel_development=kd)
    print(json.dumps({'actor_validation':list(av.shape),'kernel_validation':list(kv.shape),'output':str(a.output)},indent=2))
if __name__=='__main__':main()
