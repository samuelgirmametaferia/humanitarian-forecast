#!/usr/bin/env python3
"""Train a hierarchical H3 child ranker beneath the frozen coarse candidate model.

The parent LambdaRank owns regional probability.  For training only, the two
historical candidates nearest the target are used as local parent groups; every
H3-r5 cell in a small ring is ranked by distance relevance.  At inference the
same scorer can be applied to every valid coarse parent without target access.
"""
from __future__ import annotations
import argparse,json,math,random
from pathlib import Path
import h3
import numpy as np
from lightgbm import LGBMRanker,log_evaluation
from humanitarian_forecast.core.model_store import ModelInfo,write_info
from humanitarian_forecast.location.training.train_geo_lambdarank_v2 import _flatten_inference_features
EARTH_KM=111.32

def parent_matrix(x,c,valid):
 flat,_=_flatten_inference_features(x,c,valid);out=np.zeros((*valid.shape,flat.shape[-1]),np.float32);out[valid]=flat;return out

def child_support(parent_abs,parent_xy,resolution,ring):
 plat,plon=float(parent_abs[0]),float(parent_abs[1]);base=h3.latlng_to_cell(plat,plon,resolution);rows=[]
 for cell in h3.grid_disk(base,ring):
  lat,lon=h3.cell_to_latlng(cell);north=(lat-plat)*EARTH_KM;east=(lon-plon)*EARTH_KM*math.cos(math.radians(plat));off=np.asarray([east/1000.,north/1000.],np.float32);dist=math.hypot(east,north);bearing=math.atan2(east,north);xy=parent_xy+off
  rows.append((cell,np.asarray([lat,lon],np.float32),xy,np.asarray([off[0],off[1],dist/1000.,math.sin(bearing),math.cos(bearing),float(h3.grid_distance(base,cell))/max(ring,1),lat/90.,lon/180.,xy[0],xy[1]],np.float32)))
 return rows

def relevance(d):
 y=np.zeros(len(d),np.int32);y[d<=100]=1;y[d<=50]=2;y[d<=25]=3;y[d<=10]=4;return y

def child_dynamic_features(events:np.ndarray,child_xy:np.ndarray)->np.ndarray:
 """Query-specific local geometry for each child, without target access."""
 hist=events[-8:];hvalid=hist[:,0]>.5;hxy=hist[:,1:3]
 delta=child_xy[:,None,:]-hxy[None,:,:];dist=np.linalg.norm(delta,axis=-1)
 dist=np.where(hvalid[None,:],dist,np.nan)
 def one(pos,default=2.0):
  if len(hist)>=abs(pos) and hvalid[pos]:return dist[:,pos]
  return np.full(len(child_xy),default,np.float32)
 def stats(last_n):
  d=dist[:,-last_n:]
  with np.errstate(invalid='ignore'):
   mn=np.nanmin(d,axis=1);mean=np.nanmean(d,axis=1)
  return np.nan_to_num(mn,nan=2.0),np.nan_to_num(mean,nan=2.0)
 dlast=one(-1);dprev=one(-2);min2,_=stats(2);min4,mean4=stats(4);min8,mean8=stats(8)
 if hvalid.any():
  recent=hxy[hvalid][-4:];centroid=recent.mean(axis=0);centroid_delta=child_xy-centroid[None,:]
 else:centroid_delta=np.zeros_like(child_xy)
 centroid_dist=np.linalg.norm(centroid_delta,axis=1)
 if hvalid[-1]:last_delta=child_xy-hxy[-1][None,:]
 else:last_delta=np.zeros_like(child_xy)
 def relation(step):
  step=np.asarray(step,np.float32);sn=max(float(np.linalg.norm(step)),1e-5);vn=np.maximum(np.linalg.norm(last_delta,axis=1),1e-5)
  dot=np.sum(last_delta*step[None,:],axis=1);cross=last_delta[:,0]*step[1]-last_delta[:,1]*step[0]
  return np.stack([dot/(vn*sn),cross/(vn*sn),dot/sn],axis=1)
 last_step=events[-1,19:21] if events.shape[1]>=21 else np.zeros(2,np.float32)
 recent_step=events[-4:,19:21].mean(axis=0) if events.shape[1]>=21 else np.zeros(2,np.float32)
 kernels=np.stack([np.exp(-dlast/.025),np.exp(-dlast/.05),np.exp(-dlast/.10)],axis=1)
 return np.concatenate([
  last_delta,
  np.stack([dlast,dprev,min2,min4,mean4,min8,mean8],axis=1),
  kernels,
  centroid_delta,
  centroid_dist[:,None],
  relation(last_step),
  relation(recent_step),
 ],axis=1).astype(np.float32)

def choose_queries(meta,tr,max_queries,seed):
 all_idx=np.arange(tr,dtype=np.int64);eth=np.asarray([i for i in all_idx if str(meta[i].get('country','')).casefold()=='ethiopia'],np.int64)
 if max_queries>=tr:return all_idx
 rng=np.random.default_rng(seed);base=np.linspace(0,tr-1,max_queries,dtype=np.int64);jitter=rng.integers(-2,3,size=len(base));base=np.clip(base+jitter,0,tr-1);return np.unique(np.concatenate([base,eth]))

def build_train(x,c,coords,valid,y,meta,indices,resolution,ring,parents_per_query):
 xs=[];ys=[];groups=[];weights=[];cache={};pm=parent_matrix(x[indices],c[indices],valid[indices])
 for qi,i in enumerate(indices):
  d=np.linalg.norm(coords[i]-y[i,None],axis=-1)*1000.;d=np.where(valid[i],d,np.inf);parents=np.argsort(d)[:parents_per_query]
  for j in parents:
   if not valid[i,j]:continue
   plat=float(c[i,j,6]*90.);plon=float(c[i,j,7]*180.);key=(round(plat,5),round(plon,5),resolution,ring);support=cache.get(key)
   if support is None:
    generated=child_support((plat,plon),coords[i,j],resolution,ring)
    # Store and consume parent-relative offsets on both cache misses and hits.
    # child_support returns anchor-relative xy; adding that directly to the
    # parent again would double-count the parent displacement on cache misses.
    support=[(cell,ll,child_xy-coords[i,j],features) for cell,ll,child_xy,features in generated]
    cache[key]=support
   # cached xy is stored as offset from the parent because anchor-relative parent xy changes by query
   rows=[];child_xy=[]
   for cell,ll,offxy,lf in support:
    current_child_xy=coords[i,j]+offxy
    current_lf=lf.copy()
    # The final two local channels are anchor-relative child x/y, so they must
    # be rebuilt for the current query even when absolute-parent support is cached.
    current_lf[-2:]=current_child_xy
    rows.append(np.concatenate([pm[qi,j],current_lf]));child_xy.append(current_child_xy)
   child_xy=np.asarray(child_xy,np.float32);rows=np.asarray(rows,np.float32)
   rows=np.concatenate([rows,child_dynamic_features(x[i],child_xy)],axis=1)
   dist=np.linalg.norm(child_xy-y[i,None],axis=-1)*1000.;lab=relevance(dist)
   if lab.max()==0:continue
   xs.append(rows);ys.append(lab);groups.append(len(rows));weights.append(np.full(len(rows),5. if str(meta[i].get('country','')).casefold()=='ethiopia' else 1.,np.float32))
  if (qi+1)%2500==0:print(f'child groups queries={qi+1:,}/{len(indices):,} rows={sum(len(v) for v in xs):,}',flush=True)
 return np.concatenate(xs),np.concatenate(ys),np.asarray(groups,np.int32),np.concatenate(weights)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--resolution',type=int,default=5);p.add_argument('--ring',type=int,default=2);p.add_argument('--parents-per-query',type=int,default=2);p.add_argument('--max-train-queries',type=int,default=20000);p.add_argument('--estimators',type=int,default=600);p.add_argument('--seed',type=int,default=20260824);a=p.parse_args()
 if a.output_dir.exists() and any(a.output_dir.iterdir()):raise FileExistsError(a.output_dir)
 random.seed(a.seed);np.random.seed(a.seed);z=np.load(a.data,allow_pickle=True);x=z['x'].astype(np.float32);c=z['candidate_features'].astype(np.float32);coords=z['candidate_coordinates'].astype(np.float32);valid=z['candidate_valid'];y=z['y'].astype(np.float32);meta=[json.loads(str(v)) for v in z['meta']];n=len(x);tr=int(.70*n);idx=choose_queries(meta,tr,a.max_train_queries,a.seed);tx,ty,tg,tw=build_train(x,c,coords,valid,y,meta,idx,a.resolution,a.ring,a.parents_per_query);print(json.dumps({'train_queries_sampled':len(idx),'ranking_groups':len(tg),'train_rows':len(tx),'feature_dim':tx.shape[1],'positive_rows':int((ty>0).sum())}),flush=True)
 ranker=LGBMRanker(objective='lambdarank',metric='ndcg',label_gain=[0,1,3,7,15],n_estimators=a.estimators,learning_rate=.03,num_leaves=63,min_child_samples=80,colsample_bytree=.82,reg_lambda=7.,reg_alpha=.25,n_jobs=-1,verbosity=-1,random_state=a.seed);ranker.fit(tx,ty,group=tg,sample_weight=tw,callbacks=[log_evaluation(50)]);a.output_dir.mkdir(parents=True,exist_ok=True);ranker.booster_.save_model(str(a.output_dir/'h3_child_ranker.txt'));report={'schema':'candidate-h3-child-ranker-v1','resolution':a.resolution,'ring':a.ring,'parents_per_query':a.parents_per_query,'train_queries_sampled':len(idx),'ranking_groups':len(tg),'train_rows':len(tx),'feature_dim':int(tx.shape[1]),'estimators':a.estimators,'protocol':'ranker fit only on global first70%; target used only to choose local training parents and distance-relevance labels; no validation/development labels used'};(a.output_dir/'training.json').write_text(json.dumps(report,indent=2)+'\n');write_info(a.output_dir,ModelInfo(subsystem='location/candidate_h3_child_ranker',version='v1',status='research-pretrain',description='Hierarchical local H3 child ranker under frozen coarse candidate support.',metrics={},lineage={'dataset':str(a.data),'restore_tag':'production-boost-preflight-2026-08-24'},training=report,notes=['Parent regional probabilities are not trained here.','Evaluation must tune conditional temperature/mass allocation on validation only.']));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
