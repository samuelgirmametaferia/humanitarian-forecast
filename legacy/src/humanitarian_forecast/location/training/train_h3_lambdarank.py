#!/usr/bin/env python3
"""Dense H3 LambdaRank expert for Ethiopia broad-area forecasting.

Unlike the historical 32-candidate rankers, this model can score every H3 cell in
an Ethiopia boundary grid. Training uses cutoff-safe event history and static
population/accessibility context; labelled target location is used only to form
relevance grades and stratified negative sampling.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from lightgbm import Booster, LGBMRanker, log_evaluation

from humanitarian_forecast.core.model_store import ModelInfo, write_info

EARTH_KM = 111.32


def _relative_cells(anchor: np.ndarray, cells: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    north = (cells[:, 0] - anchor[0]) * EARTH_KM
    east = (cells[:, 1] - anchor[1]) * EARTH_KM * np.cos(np.deg2rad(anchor[0]))
    dist = np.sqrt(east * east + north * north)
    return east, north, dist


def _features_one(events: np.ndarray, anchor: np.ndarray, cells: np.ndarray, static: np.ndarray) -> np.ndarray:
    east, north, distance = _relative_cells(anchor, cells)
    cell_rel = np.stack([east / 1000.0, north / 1000.0], axis=-1)
    bearing = np.arctan2(east, north)
    base = [
        cells[:, 0:1] / 90.0,
        cells[:, 1:2] / 180.0,
        cell_rel[:, 0:1],
        cell_rel[:, 1:2],
        (distance / 1000.0)[:, None],
        np.sin(bearing)[:, None],
        np.cos(bearing)[:, None],
        np.log1p(distance / 1000.0)[:, None],
    ]

    valid = events[:, 0] > 0.5
    hist_xy = events[:, 1:3]
    days = np.maximum(0.0, np.expm1(events[:, 3] * 6.0))
    fatal = np.maximum(0.0, np.expm1(events[:, 4] * 6.0))
    civ = np.maximum(0.0, np.expm1(events[:, 5] * 6.0))
    diff = cell_rel[:, None, :] - hist_xy[None, :, :]
    dkm = np.linalg.norm(diff, axis=-1) * 1000.0
    mask = valid[None, :].astype(np.float32)

    feats = list(base)
    for radius in (25.0, 50.0, 100.0, 200.0, 400.0):
        spatial = np.exp(-0.5 * (dkm / radius) ** 2) * mask
        for half_life in (3.0, 14.0, 60.0, 180.0):
            temporal = np.exp(-np.log(2.0) * days[None, :] / half_life)
            feats.append((spatial * temporal).sum(axis=1, keepdims=True))
        w = spatial * np.exp(-np.log(2.0) * days[None, :] / 30.0)
        feats.append((w * np.log1p(fatal)[None, :]).sum(axis=1, keepdims=True))
        feats.append((w * np.log1p(civ)[None, :]).sum(axis=1, keepdims=True))

    masked = np.where(valid[None, :], dkm, np.inf)
    ordered = np.sort(masked, axis=1)
    for rank in (0, 1, 2, 3, 7):
        val = ordered[:, min(rank, ordered.shape[1] - 1)]
        feats.append((np.where(np.isfinite(val), val, 2000.0) / 1000.0)[:, None])

    # Explicit directional continuation relative to the recent motion vectors.
    for step in (events[-1, 19:21], events[-4:, 19:21].mean(axis=0)):
        sn = float(np.linalg.norm(step))
        cn = np.linalg.norm(cell_rel, axis=1)
        dot = cell_rel @ step
        feats.extend([
            np.full((len(cells), 1), step[0], np.float32),
            np.full((len(cells), 1), step[1], np.float32),
            (dot / np.maximum(cn * max(sn, 1e-6), 1e-6))[:, None],
            (dot / max(sn, 1e-6))[:, None],
        ])

    # Last-event and recent-history summaries let the tree condition the kernels
    # on violence type, severity and report/source state.
    last = events[-1]
    recent = events[-4:].mean(axis=0)
    keep = np.asarray([4,5,6,7,8,9,10,11,12,13,14,21,22,23,24], dtype=np.int64)
    history_summary = np.concatenate([last[keep], recent[keep]]).astype(np.float32)
    feats.append(np.repeat(history_summary[None, :], len(cells), axis=0))
    feats.append(static.astype(np.float32, copy=False))
    return np.concatenate(feats, axis=1).astype(np.float32, copy=False)


def _distance(cells: np.ndarray, truth: np.ndarray) -> np.ndarray:
    north = (cells[:, 0] - truth[0]) * EARTH_KM
    east = (cells[:, 1] - truth[1]) * EARTH_KM * np.cos(np.deg2rad(truth[0]))
    return np.sqrt(east * east + north * north).astype(np.float32)


def _relevance(d: np.ndarray) -> np.ndarray:
    y = np.zeros(len(d), np.int32)
    y[d <= 200] = 1
    y[d <= 100] = 2
    y[d <= 50] = 3
    y[d <= 25] = 4
    return y


def _sample_cells(rng: np.random.Generator, cells: np.ndarray, anchor: np.ndarray, truth: np.ndarray, count: int) -> np.ndarray:
    n = len(cells)
    dt = _distance(cells, truth)
    _, _, da = _relative_cells(anchor, cells)
    chosen: set[int] = {int(np.argmin(dt))}
    # Label-neighbour sampling gives each training query multiple graded positives.
    for radius, cap in ((25, 48), (50, 48), (100, 64), (200, 64)):
        pool = np.flatnonzero(dt <= radius)
        if len(pool):
            take = min(cap, len(pool))
            chosen.update(map(int, rng.choice(pool, size=take, replace=False)))
    # Hard spatial negatives around current conflict position.
    pool = np.flatnonzero(da <= 350.0)
    if len(pool):
        take = min(max(32, count // 3), len(pool))
        chosen.update(map(int, rng.choice(pool, size=take, replace=False)))
    remaining = count - len(chosen)
    if remaining > 0:
        pool = np.setdiff1d(np.arange(n), np.fromiter(chosen, dtype=np.int64), assume_unique=False)
        if len(pool): chosen.update(map(int, rng.choice(pool, size=min(remaining, len(pool)), replace=False)))
    arr = np.fromiter(chosen, dtype=np.int64)
    if len(arr) > count:
        # Always retain the nearest truth cell and otherwise randomly trim.
        positive = int(np.argmin(dt))
        rest = arr[arr != positive]
        arr = np.concatenate([[positive], rng.choice(rest, size=count-1, replace=False)])
    return arr


def _build_train(x, anchors, truth, cells, static, indices, sample_count, seed):
    rng = np.random.default_rng(seed)
    xs=[]; ys=[]; groups=[]
    for pos,i in enumerate(indices):
        chosen=_sample_cells(rng,cells,anchors[i],truth[i],sample_count)
        feats=_features_one(x[i],anchors[i],cells[chosen],static[chosen])
        d=_distance(cells[chosen],truth[i])
        xs.append(feats);ys.append(_relevance(d));groups.append(len(chosen))
        if (pos+1)%200==0: print(f"training queries {pos+1}/{len(indices)}",flush=True)
    return np.concatenate(xs),np.concatenate(ys),np.asarray(groups,np.int32)


def _softmax(scores: np.ndarray, t: float) -> np.ndarray:
    z=scores/max(float(t),1e-4);z-=z.max();p=np.exp(z);return p/np.maximum(p.sum(),1e-12)


def _score_matrix(model, x, anchors, truth, cells, static, indices, iteration):
    """Score each query/cell once; calibration can reuse this matrix cheaply."""
    scores=np.empty((len(indices),len(cells)),np.float32)
    distances=np.empty_like(scores)
    for pos,i in enumerate(indices):
        feat=_features_one(x[i],anchors[i],cells,static)
        scores[pos]=model.predict(feat,num_iteration=iteration).astype(np.float32)
        distances[pos]=_distance(cells,truth[i])
    return scores,distances


def _metrics_from_scores(scores: np.ndarray, distances: np.ndarray, temperature: float, export=False):
    z=scores/max(float(temperature),1e-4);z=z-z.max(axis=1,keepdims=True)
    probability=np.exp(z,dtype=np.float64);probability/=np.maximum(probability.sum(axis=1,keepdims=True),1e-12)
    top=np.argmax(scores,axis=1);e=distances[np.arange(len(distances)),top]
    top3_idx=np.argpartition(-scores,kth=min(2,scores.shape[1]-1),axis=1)[:,:3]
    t3=np.take_along_axis(distances,top3_idx,axis=1).min(axis=1)
    out={"samples":float(len(e)),"mean_error_km":float(e.mean()),"median_error_km":float(np.median(e)),"p90_error_km":float(np.quantile(e,.9)),"mean_entropy":float(np.mean(-(probability*np.log(np.maximum(probability,1e-12))).sum(axis=1)))}
    for r in (25,50,100,200):
        out[f"within_{r}km"]=float(np.mean(e<=r));out[f"probability_mass_within_{r}km"]=float(np.mean((probability*(distances<=r)).sum(axis=1)))
    out["top3_within_100km"]=float(np.mean(t3<=100));out["broad_area_score"]=.55*out["probability_mass_within_100km"]+.20*out["probability_mass_within_200km"]+.15*out["within_100km"]+.10*out["top3_within_100km"]
    return out,(probability.astype(np.float32) if export else None)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--static-context',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--resolution',type=int,default=5);p.add_argument('--samples-per-query',type=int,default=384);p.add_argument('--estimators',type=int,default=600);p.add_argument('--seed',type=int,default=20260824);p.add_argument('--init-model',type=Path,help='Optional globally pretrained LightGBM model using the same 103-feature contract.')
    a=p.parse_args()
    if a.output_dir.exists() and any(a.output_dir.iterdir()):raise FileExistsError(a.output_dir)
    z=np.load(a.data,allow_pickle=True);s=np.load(a.static_context,allow_pickle=True)
    x=z['x'].astype(np.float32);rows=[json.loads(str(v)) for v in z['meta']]
    anchors=np.asarray([[r['anchor_lat'],r['anchor_lon']] for r in rows],np.float32);truth=np.asarray([[r['target_lat'],r['target_lon']] for r in rows],np.float32)
    cells=z[f'centroids_r{a.resolution}'].astype(np.float32);static=s[f'features_r{a.resolution}'].astype(np.float32)
    source=z['source_indices'].astype(np.int64);tr=int(z['source_train_end']);va=int(z['source_validation_end'])
    train_idx=np.flatnonzero(source<tr);val_idx=np.flatnonzero((source>=tr)&(source<va));dev_idx=np.flatnonzero(source>=va)
    print(json.dumps({'train_queries':len(train_idx),'validation_queries':len(val_idx),'development_queries':len(dev_idx),'cells':len(cells)}),flush=True)
    tx,ty,group=_build_train(x,anchors,truth,cells,static,train_idx,a.samples_per_query,a.seed)
    print(json.dumps({'train_rows':len(tx),'feature_dim':tx.shape[1]}),flush=True)
    initial_trees=0
    if a.init_model:
        initial_trees=Booster(model_file=str(a.init_model)).num_trees()
        print(json.dumps({'init_model':str(a.init_model),'initial_trees':initial_trees}),flush=True)
    ranker=LGBMRanker(objective='lambdarank',metric='ndcg',label_gain=[0,1,3,7,15],n_estimators=a.estimators,learning_rate=.025,num_leaves=63,min_child_samples=60,colsample_bytree=.8,reg_lambda=6.,reg_alpha=.2,n_jobs=-1,verbosity=-1,random_state=a.seed)
    ranker.fit(tx,ty,group=group,callbacks=[log_evaluation(50)],init_model=str(a.init_model) if a.init_model else None)
    del tx,ty
    final_trees=ranker.booster_.num_trees()
    best=None
    validation_cache={}
    steps=sorted(set(list(range(initial_trees+50,final_trees+1,50))+[final_trees])) if initial_trees else sorted(set(list(range(100,final_trees+1,100))+[final_trees]))
    for iteration in steps:
        val_scores,val_distances=_score_matrix(ranker,x,anchors,truth,cells,static,val_idx,iteration)
        validation_cache[iteration]=(val_scores,val_distances)
        for temp in np.geomspace(.12,4.,16):
            report,_=_metrics_from_scores(val_scores,val_distances,float(temp))
            key=report['broad_area_score']
            if best is None or key>best[0]:best=(key,iteration,float(temp),report)
        print(json.dumps({'iteration':iteration,'best_so_far':{'iteration':best[1],'temperature':best[2],'validation':best[3]}}),flush=True)
    assert best is not None
    _,iteration,temp,validation=best
    val_scores,val_distances=validation_cache[iteration]
    _,val_prob=_metrics_from_scores(val_scores,val_distances,temp,export=True)
    dev_scores,dev_distances=_score_matrix(ranker,x,anchors,truth,cells,static,dev_idx,iteration)
    development,dev_prob=_metrics_from_scores(dev_scores,dev_distances,temp,export=True)
    a.output_dir.mkdir(parents=True,exist_ok=True);ranker.booster_.save_model(str(a.output_dir/'h3_lambdarank.txt'),num_iteration=iteration)
    np.savez_compressed(a.output_dir/'h3_lambdarank_probabilities.npz',validation=val_prob,development=dev_prob,validation_indices=val_idx,development_indices=dev_idx,cells=cells)
    report={'schema':'dense-h3-lambdarank-v1','resolution':a.resolution,'cells':len(cells),'selected_iteration':iteration,'initial_trees':initial_trees,'final_trees':final_trees,'init_model':str(a.init_model) if a.init_model else None,'temperature':temp,'feature_dim':int(ranker.n_features_in_),'validation':validation,'development':development,'protocol':'global 70/15/15 source cutoffs projected onto Ethiopia rows; training uses sampled dense H3 cells; validation selects iteration and temperature; development is diagnostic only'}
    (a.output_dir/'h3_lambdarank_metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    write_info(a.output_dir,ModelInfo(subsystem='location/h3_lambdarank',version='v1',status='research-challenger',description='Dense H3 propagation/motion LambdaRank expert with population/accessibility context.',metrics={'validation':validation,'development':development},lineage={'data':str(a.data),'static_context':str(a.static_context),'init_model':str(a.init_model) if a.init_model else None,'restore_tag':'production-boost-preflight-2026-08-24'},training={'resolution':a.resolution,'cells':len(cells),'samples_per_query':a.samples_per_query,'selected_iteration':iteration,'initial_trees':initial_trees,'final_trees':final_trees,'temperature':temp,'seed':a.seed},notes=['Scores every H3 cell rather than historical candidate coordinates.','Development block is not a pristine prospective test.']))
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
