#!/usr/bin/env python3
"""Dense H3 LambdaRank with cutoff-safe cross-conflict actor geography."""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np
from lightgbm import LGBMRanker,log_evaluation
from humanitarian_forecast.core.model_store import ModelInfo,write_info
from humanitarian_forecast.data.augment_candidate_actor_transfer import load_indices,other_counts,encode
from humanitarian_forecast.location.training import train_h3_lambdarank as base

class ActorContext:
    def __init__(self,ucdp:Path,cells:np.ndarray):
        self.idx=load_indices(ucdp)
        self.conflict_actors,self.ap,self.ah,self.ao,self.cp,self.ch,self.co=self.idx
        self.half=[(math.floor(float(lat)*2),math.floor(float(lon)*2)) for lat,lon in cells]
        self.one=[(math.floor(float(lat)),math.floor(float(lon))) for lat,lon in cells]
        self.half_unique,self.half_inv=self._unique(self.half)
        self.one_unique,self.one_inv=self._unique(self.one)
    @staticmethod
    def _unique(keys):
        mapping={};unique=[];inv=np.empty(len(keys),np.int32)
        for i,k in enumerate(keys):
            j=mapping.get(k)
            if j is None:j=len(unique);mapping[k]=j;unique.append(k)
            inv[i]=j
        return unique,inv
    def features(self,conflict:str,cutoff:int)->np.ndarray:
        a,b=self.conflict_actors.get(conflict,('',''))
        side=[]
        for actor in (a,b):
            blocks=[]
            for actor_index,conflict_index,keys,inv in ((self.ah,self.ch,self.half_unique,self.half_inv),(self.ao,self.co,self.one_unique,self.one_inv)):
                values=np.asarray([encode(other_counts(actor_index,conflict_index,actor,conflict,key,cutoff)) for key in keys],np.float32)
                blocks.append(values[inv])
            side.append(np.concatenate(blocks,axis=1))
        ea,eb=side
        # 16 raw (two actors x two scales x four windows) plus eight symmetric summaries.
        combo=np.stack([
            np.maximum(ea[:,0],eb[:,0]),ea[:,0]+eb[:,0],
            np.maximum(ea[:,2],eb[:,2]),ea[:,2]+eb[:,2],
            np.maximum(ea[:,3],eb[:,3]),ea[:,3]+eb[:,3],
            np.maximum(ea[:,6],eb[:,6]),ea[:,6]+eb[:,6],
        ],axis=1)
        return np.concatenate([ea,eb,combo],axis=1).astype(np.float32)

def feature_one(x,anchors,cells,static,rows,i,actors,selection=None):
    chosen=cells if selection is None else cells[selection];st=static if selection is None else static[selection]
    f=base._features_one(x[i],anchors[i],chosen,st)
    row=rows[i];target_day=int(np.datetime64(row['target_date'],'D').astype(int));cutoff=target_day-int(row['gap_days'])
    af=actors.features(str(row['conflict_id']),cutoff);af=af if selection is None else af[selection]
    return np.concatenate([f,af],axis=1)

def build_train(x,anchors,truth,cells,static,rows,indices,actors,count,seed):
    rng=np.random.default_rng(seed);xs=[];ys=[];groups=[]
    for pos,i in enumerate(indices):
        chosen=base._sample_cells(rng,cells,anchors[i],truth[i],count)
        xs.append(feature_one(x,anchors,cells,static,rows,i,actors,chosen));ys.append(base._relevance(base._distance(cells[chosen],truth[i])));groups.append(len(chosen))
        if (pos+1)%200==0:print(f'training queries {pos+1}/{len(indices)}',flush=True)
    return np.concatenate(xs),np.concatenate(ys),np.asarray(groups,np.int32)

def score_matrix(model,x,anchors,truth,cells,static,rows,indices,actors,iteration):
    scores=np.empty((len(indices),len(cells)),np.float32);dist=np.empty_like(scores)
    for pos,i in enumerate(indices):
        scores[pos]=model.predict(feature_one(x,anchors,cells,static,rows,i,actors),num_iteration=iteration).astype(np.float32);dist[pos]=base._distance(cells,truth[i])
    return scores,dist

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--static-context',type=Path,required=True);p.add_argument('--ucdp-events',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--resolution',type=int,default=5);p.add_argument('--samples-per-query',type=int,default=384);p.add_argument('--estimators',type=int,default=700);p.add_argument('--seed',type=int,default=20260824);a=p.parse_args()
    if a.output_dir.exists() and any(a.output_dir.iterdir()):raise FileExistsError(a.output_dir)
    z=np.load(a.data,allow_pickle=True);s=np.load(a.static_context,allow_pickle=True);x=z['x'].astype(np.float32);rows=[json.loads(str(v)) for v in z['meta']];anchors=np.asarray([[r['anchor_lat'],r['anchor_lon']] for r in rows],np.float32);truth=np.asarray([[r['target_lat'],r['target_lon']] for r in rows],np.float32);cells=z[f'centroids_r{a.resolution}'].astype(np.float32);static=s[f'features_r{a.resolution}'].astype(np.float32);source=z['source_indices'].astype(np.int64);tr=int(z['source_train_end']);va=int(z['source_validation_end']);ti=np.flatnonzero(source<tr);vi=np.flatnonzero((source>=tr)&(source<va));di=np.flatnonzero(source>=va);actors=ActorContext(a.ucdp_events,cells)
    print(json.dumps({'train_queries':len(ti),'validation_queries':len(vi),'development_queries':len(di),'cells':len(cells),'half_actor_cells':len(actors.half_unique),'one_actor_cells':len(actors.one_unique)}),flush=True)
    tx,ty,group=build_train(x,anchors,truth,cells,static,rows,ti,actors,a.samples_per_query,a.seed);print(json.dumps({'train_rows':len(tx),'feature_dim':tx.shape[1]}),flush=True)
    ranker=LGBMRanker(objective='lambdarank',metric='ndcg',label_gain=[0,1,3,7,15],n_estimators=a.estimators,learning_rate=.025,num_leaves=63,min_child_samples=60,colsample_bytree=.82,reg_lambda=6.,reg_alpha=.2,n_jobs=-1,verbosity=-1,random_state=a.seed);ranker.fit(tx,ty,group=group,callbacks=[log_evaluation(50)]);del tx,ty
    best=None;cache={}
    for iteration in sorted(set(list(range(100,a.estimators+1,100))+[a.estimators])):
        sc,dd=score_matrix(ranker,x,anchors,truth,cells,static,rows,vi,actors,iteration);cache[iteration]=(sc,dd)
        for temp in np.geomspace(.10,4.,18):
            rep,_=base._metrics_from_scores(sc,dd,float(temp));key=rep['broad_area_score']
            if best is None or key>best[0]:best=(key,iteration,float(temp),rep)
        print(json.dumps({'iteration':iteration,'best_so_far':{'iteration':best[1],'temperature':best[2],'validation':best[3]}}),flush=True)
    _,iteration,temp,validation=best;vsc,vdd=cache[iteration];_,vp=base._metrics_from_scores(vsc,vdd,temp,export=True);dsc,ddd=score_matrix(ranker,x,anchors,truth,cells,static,rows,di,actors,iteration);development,dp=base._metrics_from_scores(dsc,ddd,temp,export=True)
    a.output_dir.mkdir(parents=True,exist_ok=True);ranker.booster_.save_model(str(a.output_dir/'h3_actor_lambdarank.txt'),num_iteration=iteration);np.savez_compressed(a.output_dir/'h3_actor_lambdarank_probabilities.npz',validation=vp,development=dp,validation_indices=vi,development_indices=di,cells=cells)
    report={'schema':'dense-h3-actor-lambdarank-v1','resolution':a.resolution,'cells':len(cells),'selected_iteration':iteration,'temperature':temp,'feature_dim':int(ranker.n_features_in_),'validation':validation,'development':development,'protocol':'same chronological Ethiopia gate as dense H3 v1; actor geography uses only events at/before each cutoff and subtracts current-conflict activity'};(a.output_dir/'h3_actor_lambdarank_metrics.json').write_text(json.dumps(report,indent=2)+'\n');write_info(a.output_dir,ModelInfo(subsystem='location/h3_lambdarank_actor',version='v1',status='research-challenger',description='Dense H3 propagation LambdaRank with cross-conflict actor geography.',metrics={'validation':validation,'development':development},lineage={'data':str(a.data),'static_context':str(a.static_context),'ucdp':str(a.ucdp_events),'baseline':'models/location/h3_lambdarank/ethiopia_v1','restore_tag':'production-boost-preflight-2026-08-24'},training={'selected_iteration':iteration,'temperature':temp,'samples_per_query':a.samples_per_query,'seed':a.seed},notes=['Actor histories are cutoff-safe and current-conflict activity is subtracted.','Development is diagnostic, not pristine prospective evaluation.']));print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
