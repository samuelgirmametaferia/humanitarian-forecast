#!/usr/bin/env python3
"""Proper-score ensemble of frozen v9, actor-transfer, and propagation experts.

All model/temperature/weight choices use chronological validation only. The
objective is cross-entropy to a fixed 100 km distance-soft candidate target,
which rewards probability on the correct broad area rather than exact point
selection. The later partition is reporting-only.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

SCALE_KM = 1000.0
SOFT_RADIUS_KM = 100.0


def softmax(scores: np.ndarray, valid: np.ndarray, temperature: float) -> np.ndarray:
    scaled = np.asarray(scores, dtype=np.float64) / max(float(temperature), 1e-5)
    scaled = np.where(valid, scaled, -np.inf)
    mx = np.max(scaled, axis=1, keepdims=True)
    ex = np.exp(np.where(valid, scaled - mx, -np.inf))
    ex = np.where(valid, ex, 0.0)
    return ex / np.maximum(ex.sum(axis=1, keepdims=True), 1e-12)


def distance_soft_target(coordinates: np.ndarray, valid: np.ndarray, target: np.ndarray, radius_km: float = SOFT_RADIUS_KM) -> np.ndarray:
    distance = np.linalg.norm(coordinates - target[:, None], axis=-1) * SCALE_KM
    weight = np.exp(-0.5 * (distance / radius_km) ** 2)
    weight = np.where(valid, weight, 0.0)
    # If all mass numerically vanishes, recover with the nearest valid candidate.
    total = weight.sum(axis=1, keepdims=True)
    bad = total[:, 0] <= 1e-30
    if np.any(bad):
        d = np.where(valid[bad], distance[bad], np.inf)
        nearest = np.argmin(d, axis=1)
        weight[bad] = 0.0
        weight[np.where(bad)[0], nearest] = 1.0
        total = weight.sum(axis=1, keepdims=True)
    return weight / total


def soft_ce(target_distribution: np.ndarray, probability: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is None:
        q = target_distribution; p = probability
    else:
        q = target_distribution[mask]; p = probability[mask]
    if len(q) == 0:
        return math.inf
    return float(-np.mean(np.sum(q * np.log(np.maximum(p, 1e-12)), axis=1)))


def calibrate_temperature(scores: np.ndarray, valid: np.ndarray, q: np.ndarray, mask: np.ndarray | None = None) -> tuple[float, np.ndarray, float]:
    best = (math.inf, 1.0, None)
    for temperature in np.geomspace(0.10, 4.0, 45):
        probability = softmax(scores, valid, float(temperature))
        ce = soft_ce(q, probability, mask)
        if ce < best[0] - 1e-12:
            best = (ce, float(temperature), probability)
    assert best[2] is not None
    return best[1], best[2], best[0]


def ensemble_grid(probabilities: list[np.ndarray], q: np.ndarray, mask: np.ndarray | None, step: float = 0.025, global_guard: tuple[np.ndarray,float] | None = None):
    units = int(round(1.0 / step))
    best = (math.inf, None, None)
    for a in range(units + 1):
        for b in range(units + 1 - a):
            c = units - a - b
            weights = np.asarray([a,b,c], dtype=np.float64) / units
            blend = weights[0]*probabilities[0] + weights[1]*probabilities[1] + weights[2]*probabilities[2]
            ce = soft_ce(q, blend, mask)
            if global_guard is not None:
                global_q, ceiling = global_guard
                if soft_ce(global_q, blend) > ceiling:
                    continue
            if ce < best[0] - 1e-12:
                best = (ce, weights, blend)
    if best[1] is None:
        raise RuntimeError('no ensemble weights satisfy constraints')
    return best


def metrics(probability: np.ndarray, coordinates: np.ndarray, valid: np.ndarray, target: np.ndarray, q: np.ndarray, mask: np.ndarray | None = None) -> dict[str,float]:
    if mask is not None:
        probability=probability[mask];coordinates=coordinates[mask];valid=valid[mask];target=target[mask];q=q[mask]
    distance=np.linalg.norm(coordinates-target[:,None],axis=-1)*SCALE_KM
    distance=np.where(valid,distance,np.inf)
    top=np.argmax(probability,axis=1); error=distance[np.arange(len(distance)),top]
    out={
        'samples':float(len(error)),
        'distance_soft_cross_entropy':soft_ce(q,probability),
        'mean_error_km':float(np.mean(error)),
        'median_error_km':float(np.median(error)),
        'p90_error_km':float(np.quantile(error,.90)),
        'candidate_oracle_mean_km':float(np.mean(np.min(distance,axis=1))),
        'mean_entropy':float(np.mean(-np.sum(probability*np.log(np.maximum(probability,1e-12)),axis=1))),
    }
    for radius in (25,50,100,200):
        inside=distance<=radius
        out[f'within_{radius}km']=float(np.mean(error<=radius))
        out[f'probability_mass_within_{radius}km']=float(np.mean(np.sum(probability*inside,axis=1)))
    order=np.argsort(-probability,axis=1)
    for k in (3,5):
        chosen=np.take_along_axis(distance,order[:,:k],axis=1)
        out[f'top{k}_within_100km']=float(np.mean(np.any(chosen<=100,axis=1)))
        out[f'top{k}_within_200km']=float(np.mean(np.any(chosen<=200,axis=1)))
    out['broad_area_score']=(.55*out['probability_mass_within_100km']+.20*out['probability_mass_within_200km']+.15*out['within_100km']+.10*out['top3_within_100km'])
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--v9-data',type=Path,required=True)
    p.add_argument('--actor-data',type=Path,required=True)
    p.add_argument('--v9-export',type=Path,required=True)
    p.add_argument('--tree-export',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--weight-step',type=float,default=.025)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    v9=np.load(a.v9_data,allow_pickle=True); actor=np.load(a.actor_data,allow_pickle=True)
    for key in ('y','candidate_coordinates','candidate_valid','meta','label'):
        if not np.array_equal(v9[key],actor[key]):raise ValueError(f'v9/actor alignment failure: {key}')
    n=len(v9['y']);tr=int(.70*n);va=int(.85*n)
    coords=v9['candidate_coordinates'].astype(np.float32);valid=v9['candidate_valid'];target=v9['y'].astype(np.float32)
    meta=[json.loads(str(z)) for z in v9['meta']]
    country=np.asarray([m.get('country','') for m in meta])
    val_eth=country[tr:va]=='Ethiopia';dev_eth=country[va:]=='Ethiopia'
    ve=np.load(a.v9_export);te=np.load(a.tree_export)
    val_scores=[ve['validation_logits'],te['actor_validation'],te['kernel_validation']]
    dev_scores=[ve['development_logits'],te['actor_development'],te['kernel_development']]
    qv=distance_soft_target(coords[tr:va],valid[tr:va],target[tr:va]);qd=distance_soft_target(coords[va:],valid[va:],target[va:])
    names=['v9_phase1','actor_transfer','propagation_kernel']
    val_p=[];dev_p=[];calibration={}
    for name,vs,ds in zip(names,val_scores,dev_scores):
        temp,pv,ce=calibrate_temperature(vs,valid[tr:va],qv)
        pd=softmax(ds,valid[va:],temp)
        val_p.append(pv);dev_p.append(pd);calibration[name]={'temperature':temp,'validation_distance_soft_ce':ce}
    global_ce,global_w,global_blend=ensemble_grid(val_p,qv,None,a.weight_step)
    global_dev=sum(float(w)*p for w,p in zip(global_w,dev_p))
    # Ethiopia-specific weights are allowed only if their global validation CE is
    # within 0.5% of the globally selected proper-score ensemble.
    ceiling=global_ce*1.005
    eth_ce,eth_w,eth_blend=ensemble_grid(val_p,qv,val_eth,a.weight_step,global_guard=(qv,ceiling))
    eth_dev=sum(float(w)*p for w,p in zip(eth_w,dev_p))
    report={
      'schema':'broad-area-distance-soft-ensemble-v1',
      'objective':{'target':'Gaussian distance-soft candidate distribution','radius_km':SOFT_RADIUS_KM,'selection':'temperatures and convex weights use chronological validation only','weight_step':a.weight_step},
      'alignment':{'samples':n,'candidate_coordinates_equal':True,'candidate_valid_equal':True,'targets_equal':True,'metadata_equal':True},
      'expert_calibration':calibration,
      'global_selection':{'validation_distance_soft_ce':global_ce,'weights':dict(zip(names,map(float,global_w))),'validation':metrics(global_blend,coords[tr:va],valid[tr:va],target[tr:va],qv),'development':metrics(global_dev,coords[va:],valid[va:],target[va:],qd),'ethiopia_validation':metrics(global_blend,coords[tr:va],valid[tr:va],target[tr:va],qv,val_eth),'ethiopia_development':metrics(global_dev,coords[va:],valid[va:],target[va:],qd,dev_eth)},
      'ethiopia_guarded_selection':{'validation_distance_soft_ce':eth_ce,'global_validation_ce_ceiling':ceiling,'weights':dict(zip(names,map(float,eth_w))),'ethiopia_validation':metrics(eth_blend,coords[tr:va],valid[tr:va],target[tr:va],qv,val_eth),'ethiopia_development':metrics(eth_dev,coords[va:],valid[va:],target[va:],qd,dev_eth),'global_validation':metrics(eth_blend,coords[tr:va],valid[tr:va],target[tr:va],qv),'global_development':metrics(eth_dev,coords[va:],valid[va:],target[va:],qd)},
      'experts':{name:{'validation':metrics(pv,coords[tr:va],valid[tr:va],target[tr:va],qv),'development':metrics(pd,coords[va:],valid[va:],target[va:],qd),'ethiopia_validation':metrics(pv,coords[tr:va],valid[tr:va],target[tr:va],qv,val_eth),'ethiopia_development':metrics(pd,coords[va:],valid[va:],target[va:],qd,dev_eth)} for name,pv,pd in zip(names,val_p,dev_p)},
      'evaluation_caveat':'The final historical block is a development benchmark after iterative research, not a pristine prospective test. The v9 phase-1 expert is reproduced on the first 70% specifically to avoid validation leakage from its 85%-trained production checkpoint.',
      'output_scope':'broad-area probability research; exact point metrics are diagnostics only',
    }
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'expert_calibration':calibration,'global_selection':report['global_selection'],'ethiopia_guarded_selection':report['ethiopia_guarded_selection']},indent=2))
if __name__=='__main__':main()
