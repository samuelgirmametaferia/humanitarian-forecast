#!/usr/bin/env python3
"""Dense H3 LambdaRank with full-conflict recurrence and transition memory.

This restores long-memory signals that historical candidate generation provided
implicitly, without restricting the output support to previously observed points.
Every memory lookup is cutoff-safe.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import io
import json
import math
import zipfile
from collections import defaultdict
from pathlib import Path

import h3
import numpy as np
from lightgbm import LGBMRanker, log_evaluation

from humanitarian_forecast.core.model_store import ModelInfo, write_info
from humanitarian_forecast.location.training import train_h3_lambdarank as base


class ConflictMemory:
    def __init__(self, ucdp: Path, resolution: int, cells: np.ndarray, cell_ids: np.ndarray):
        self.resolution=resolution
        self.cell_ids=[str(v) for v in cell_ids]
        self.events=defaultdict(list)
        self.transitions=defaultdict(list)
        grouped=defaultdict(list)
        with zipfile.ZipFile(ucdp) as archive:
            name=next(v for v in archive.namelist() if v.lower().endswith('.csv'))
            with archive.open(name) as raw, io.TextIOWrapper(raw,encoding='utf-8-sig') as text:
                for row in csv.DictReader(text):
                    try:
                        if int(row['where_prec'])>4 or int(row['date_prec'])>3:continue
                        conflict=str(row['conflict_new_id']).strip();day=int(np.datetime64(row['date_start'][:10],'D').astype(int));lat=float(row['latitude']);lon=float(row['longitude']);eid=int(row['id'])
                    except (ValueError,KeyError,TypeError):continue
                    if not conflict:continue
                    cell=h3.latlng_to_cell(lat,lon,resolution);grouped[conflict].append((day,eid,cell))
        for conflict,rows in grouped.items():
            rows.sort()
            for day,_,cell in rows:self.events[(conflict,cell)].append(day)
            for prev,cur in zip(rows,rows[1:]):
                self.transitions[(conflict,prev[2],cur[2])].append(cur[0])
        for index in (self.events,self.transitions):
            for days in index.values():days.sort()
        # Precompute grid neighbors only for output cells.
        self.neighbors1=[tuple(h3.grid_disk(cell,1)) for cell in self.cell_ids]
        self.neighbors2=[tuple(h3.grid_disk(cell,2)) for cell in self.cell_ids]

    @staticmethod
    def _count(days,cutoff,window=None):
        right=bisect.bisect_right(days,cutoff)
        if window is None:return right
        return right-bisect.bisect_left(days,cutoff-window,0,right)

    def features(self, conflict: str, anchor_lat: float, anchor_lon: float, cutoff: int) -> np.ndarray:
        source=h3.latlng_to_cell(anchor_lat,anchor_lon,self.resolution);out=np.zeros((len(self.cell_ids),16),np.float32)
        for j,cell in enumerate(self.cell_ids):
            ed=self.events.get((conflict,cell),());td=self.transitions.get((conflict,source,cell),())
            ec=[self._count(ed,cutoff,w) for w in (None,30,90,365)]
            tc=[self._count(td,cutoff,w) for w in (None,30,90,365)]
            right=bisect.bisect_right(ed,cutoff);age=(cutoff-ed[right-1]) if right else 9999
            n1_30=n1_90=n2_90=0
            for neigh in self.neighbors1[j]:
                nd=self.events.get((conflict,neigh),());n1_30+=self._count(nd,cutoff,30);n1_90+=self._count(nd,cutoff,90)
            for neigh in self.neighbors2[j]:
                nd=self.events.get((conflict,neigh),());n2_90+=self._count(nd,cutoff,90)
            out[j]=[
                math.log1p(ec[0])/6,math.log1p(ec[1])/4,math.log1p(ec[2])/4,math.log1p(ec[3])/4,
                math.log1p(tc[0])/5,math.log1p(tc[1])/4,math.log1p(tc[2])/4,math.log1p(tc[3])/4,
                math.log1p(max(0,age))/6,float(right==0),
                math.log1p(n1_30)/4,math.log1p(n1_90)/4,math.log1p(n2_90)/4,
                float(cell==source),float(source in self.neighbors1[j]),
                float(ec[0]>0),
            ]
        return out


def feature_one(x,anchors,cells,static,rows,i,memory,selection=None):
    use_cells=cells if selection is None else cells[selection];use_static=static if selection is None else static[selection]
    f=base._features_one(x[i],anchors[i],use_cells,use_static);row=rows[i];target_day=int(np.datetime64(row['target_date'],'D').astype(int));cutoff=target_day-int(row['gap_days']);mf=memory.features(str(row['conflict_id']),float(row['anchor_lat']),float(row['anchor_lon']),cutoff);mf=mf if selection is None else mf[selection]
    return np.concatenate([f,mf],axis=1)


def build_train(x,anchors,truth,cells,static,rows,indices,memory,count,seed):
    rng=np.random.default_rng(seed);xs=[];ys=[];groups=[]
    for pos,i in enumerate(indices):
        chosen=base._sample_cells(rng,cells,anchors[i],truth[i],count);xs.append(feature_one(x,anchors,cells,static,rows,i,memory,chosen));ys.append(base._relevance(base._distance(cells[chosen],truth[i])));groups.append(len(chosen))
        if (pos+1)%200==0:print(f'training queries {pos+1}/{len(indices)}',flush=True)
    return np.concatenate(xs),np.concatenate(ys),np.asarray(groups,np.int32)


def score_matrix(model,x,anchors,truth,cells,static,rows,indices,memory,iteration):
    scores=np.empty((len(indices),len(cells)),np.float32);dist=np.empty_like(scores)
    for pos,i in enumerate(indices):
        scores[pos]=model.predict(feature_one(x,anchors,cells,static,rows,i,memory),num_iteration=iteration).astype(np.float32);dist[pos]=base._distance(cells,truth[i])
    return scores,dist


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--static-context',type=Path,required=True);p.add_argument('--ucdp-events',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--resolution',type=int,default=4);p.add_argument('--samples-per-query',type=int,default=256);p.add_argument('--estimators',type=int,default=700);p.add_argument('--seed',type=int,default=20260824);a=p.parse_args()
    if a.output_dir.exists() and any(a.output_dir.iterdir()):raise FileExistsError(a.output_dir)
    z=np.load(a.data,allow_pickle=True);s=np.load(a.static_context,allow_pickle=True);x=z['x'].astype(np.float32);rows=[json.loads(str(v)) for v in z['meta']];anchors=np.asarray([[r['anchor_lat'],r['anchor_lon']] for r in rows],np.float32);truth=np.asarray([[r['target_lat'],r['target_lon']] for r in rows],np.float32);cells=z[f'centroids_r{a.resolution}'].astype(np.float32);cell_ids=z[f'cells_r{a.resolution}'];static=s[f'features_r{a.resolution}'].astype(np.float32);source=z['source_indices'].astype(np.int64);tr=int(z['source_train_end']);va=int(z['source_validation_end']);ti=np.flatnonzero(source<tr);vi=np.flatnonzero((source>=tr)&(source<va));di=np.flatnonzero(source>=va);memory=ConflictMemory(a.ucdp_events,a.resolution,cells,cell_ids)
    print(json.dumps({'train_queries':len(ti),'validation_queries':len(vi),'development_queries':len(di),'cells':len(cells),'memory_event_keys':len(memory.events),'memory_transition_keys':len(memory.transitions)}),flush=True)
    tx,ty,group=build_train(x,anchors,truth,cells,static,rows,ti,memory,a.samples_per_query,a.seed);print(json.dumps({'train_rows':len(tx),'feature_dim':tx.shape[1]}),flush=True)
    ranker=LGBMRanker(objective='lambdarank',metric='ndcg',label_gain=[0,1,3,7,15],n_estimators=a.estimators,learning_rate=.025,num_leaves=63,min_child_samples=60,colsample_bytree=.82,reg_lambda=6.,reg_alpha=.2,n_jobs=-1,verbosity=-1,random_state=a.seed);ranker.fit(tx,ty,group=group,callbacks=[log_evaluation(50)]);del tx,ty
    best=None;cache={}
    for iteration in sorted(set(list(range(100,a.estimators+1,100))+[a.estimators])):
        sc,dd=score_matrix(ranker,x,anchors,truth,cells,static,rows,vi,memory,iteration);cache[iteration]=(sc,dd)
        for temp in np.geomspace(.10,4.,18):
            rep,_=base._metrics_from_scores(sc,dd,float(temp));key=rep['broad_area_score']
            if best is None or key>best[0]:best=(key,iteration,float(temp),rep)
        print(json.dumps({'iteration':iteration,'best_so_far':{'iteration':best[1],'temperature':best[2],'validation':best[3]}}),flush=True)
    _,iteration,temp,validation=best;vsc,vdd=cache[iteration];_,vp=base._metrics_from_scores(vsc,vdd,temp,export=True);dsc,ddd=score_matrix(ranker,x,anchors,truth,cells,static,rows,di,memory,iteration);development,dp=base._metrics_from_scores(dsc,ddd,temp,export=True)
    a.output_dir.mkdir(parents=True,exist_ok=True);ranker.booster_.save_model(str(a.output_dir/'h3_memory_lambdarank.txt'),num_iteration=iteration);np.savez_compressed(a.output_dir/'h3_memory_lambdarank_probabilities.npz',validation=vp,development=dp,validation_indices=vi,development_indices=di,cells=cells)
    report={'schema':'dense-h3-memory-lambdarank-v1','resolution':a.resolution,'cells':len(cells),'selected_iteration':iteration,'temperature':temp,'feature_dim':int(ranker.n_features_in_),'validation':validation,'development':development,'protocol':'full-conflict event/destination/transition memory is cutoff-safe; validation selects iteration/temperature; development diagnostic only'};(a.output_dir/'h3_memory_lambdarank_metrics.json').write_text(json.dumps(report,indent=2)+'\n');write_info(a.output_dir,ModelInfo(subsystem='location/h3_lambdarank_memory',version='v1',status='research-challenger',description='Dense H3 propagation ranker with full-conflict recurrence and transition memory.',metrics={'validation':validation,'development':development},lineage={'data':str(a.data),'static_context':str(a.static_context),'ucdp':str(a.ucdp_events),'baseline':'models/location/h3_lambdarank/ethiopia_r4_v1','restore_tag':'production-boost-preflight-2026-08-24'},training={'selected_iteration':iteration,'temperature':temp,'samples_per_query':a.samples_per_query,'seed':a.seed},notes=['Restores full-history recurrence without restricting output to historical coordinates.','All UCDP memory features are truncated at each forecast cutoff.']));print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
