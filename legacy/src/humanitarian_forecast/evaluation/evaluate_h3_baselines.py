#!/usr/bin/env python3
"""Evaluate simple dense-H3 humanitarian location baselines.

These baselines establish whether a new dense neural model beats trivial support
choices: historical target frequency, last observed cell, and a causal
recency-weighted history kernel. They use the same global chronological split
boundaries embedded in the H3 dataset.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def cell_error(centroids: np.ndarray, chosen: np.ndarray, truth: np.ndarray) -> np.ndarray:
    c=centroids[chosen]
    north=(c[:,0]-truth[:,0])*111.32
    east=(c[:,1]-truth[:,1])*111.32*np.cos(np.deg2rad(truth[:,0]))
    return np.sqrt(east*east+north*north)


def report(chosen: np.ndarray, centroids: np.ndarray, truth: np.ndarray) -> dict[str,float]:
    e=cell_error(centroids,chosen,truth)
    return {
        'samples':float(len(e)),
        'mean_error_km':float(e.mean()),
        'median_error_km':float(np.median(e)),
        'p90_error_km':float(np.quantile(e,.9)),
        'within_25km':float(np.mean(e<=25)),
        'within_50km':float(np.mean(e<=50)),
        'within_100km':float(np.mean(e<=100)),
        'within_200km':float(np.mean(e<=200)),
    }


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    z=np.load(a.data,allow_pickle=True);rows=[json.loads(str(v)) for v in z['meta']];truth=np.asarray([[float(r['target_lat']),float(r['target_lon'])] for r in rows],dtype=np.float32);src=z['source_indices'];tr=int(z['source_train_end']);va=int(z['source_validation_end']);train=src<tr;val=(src>=tr)&(src<va);dev=src>=va
    result={'schema':'dense-h3-baselines-v1','source':str(a.data),'resolutions':{}}
    for r in map(int,z['resolutions']):
        cent=z[f'centroids_r{r}'];target=z[f'target_index_r{r}'];hist=z[f'history_index_r{r}'];cells=len(cent)
        valid_train=train&(target>=0);freq=np.bincount(target[valid_train],minlength=cells).astype(np.float64);freq=(freq+1e-3)/(freq.sum()+1e-3*cells);frequency_choice=np.full(len(target),int(freq.argmax()),dtype=np.int64)
        last=np.full(len(target),int(freq.argmax()),dtype=np.int64);kernel=np.zeros((len(target),),dtype=np.int64)
        for i,h in enumerate(hist):
            ok=h[h>=0]
            if len(ok):last[i]=int(ok[-1])
            score=freq.copy()*0.05
            # Later sequence positions receive exponentially more mass. Kernel
            # spreads one graph ring around observed cells to avoid exact-cell persistence.
            for pos,cell in enumerate(h):
                if cell<0:continue
                score[int(cell)]+=float(np.exp((pos-len(h)+1)/4.0))
            kernel[i]=int(score.argmax())
        rr={}
        for name,choice in [('historical_frequency',frequency_choice),('last_observed_cell',last),('recency_history_kernel',kernel)]:
            rr[name]={'validation':report(choice[val],cent,truth[val]),'development':report(choice[dev],cent,truth[dev])}
        result['resolutions'][str(r)]=rr
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
