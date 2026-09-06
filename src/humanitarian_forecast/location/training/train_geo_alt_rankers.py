#!/usr/bin/env python3
"""Benchmark XGBoost LambdaMART and CatBoost YetiRank as candidate experts.

Training uses a fixed global replay subset plus every Horn/Ethiopia query in the
first 70% chronological block.  Full 15% validation and 15% development blocks
are scored.  A frozen LightGBM actor-transfer expert is optionally included for
validation-only convex probability ensembling.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from xgboost import XGBRanker
from catboost import CatBoostRanker,Pool
from lightgbm import Booster
from humanitarian_forecast.core.model_store import ModelInfo,write_info
from humanitarian_forecast.location.training.train_geo_lambdarank_v2 import _flatten_features,_scores_to_matrix,_softmax,metrics

HORN={"ethiopia","eritrea","somalia","sudan","south sudan","djibouti","kenya"}

def select_queries(meta,tr,count,seed):
    horn=np.asarray([i for i in range(tr) if str(meta[i].get('country','')).casefold() in HORN],np.int64)
    rng=np.random.default_rng(seed);allidx=np.arange(tr,dtype=np.int64);rest=np.setdiff1d(allidx,horn,assume_unique=False);need=max(0,count-len(horn));sample=rng.choice(rest,size=min(need,len(rest)),replace=False) if need else np.empty(0,np.int64)
    return np.sort(np.concatenate([horn,sample]))

def eth_mask(meta,lo,hi):return np.asarray([str(meta[i].get('country','')).casefold()=='ethiopia' for i in range(lo,hi)])

def calibrate(raw,coords,valid,target,eth):
    best=None
    for t in np.geomspace(.08,6.,28):
        p=_softmax(raw,valid,float(t));g=metrics(raw,p,coords,valid,target);e=metrics(raw[eth],p[eth],coords[eth],valid[eth],target[eth]);row=(e['broad_area_score'],float(t),g,e,p)
        if best is None or row[0]>best[0]:best=row
    return best

def score_flat(model,features,kind):
    if kind=='xgb':return model.predict(features).astype(np.float64)
    return np.asarray(model.predict(features),np.float64)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--train-queries',type=int,default=30000);p.add_argument('--seed',type=int,default=20260824);p.add_argument('--baseline-model',type=Path);p.add_argument('--baseline-config',type=Path);a=p.parse_args()
    if a.output_dir.exists() and any(a.output_dir.iterdir()):raise FileExistsError(a.output_dir)
    z=np.load(a.data,allow_pickle=True);x=z['x'].astype(np.float32);c=z['candidate_features'].astype(np.float32);coords=z['candidate_coordinates'].astype(np.float32);valid=z['candidate_valid'];target=z['y'].astype(np.float32);meta=[json.loads(str(v)) for v in z['meta']];n=len(x);tr=int(.7*n);va=int(.85*n)
    qi=select_queries(meta,tr,a.train_queries,a.seed);print(json.dumps({'train_queries':len(qi),'ethiopia':sum(str(meta[i].get('country','')).casefold()=='ethiopia' for i in qi),'horn':sum(str(meta[i].get('country','')).casefold() in HORN for i in qi)}),flush=True)
    tx,ty,tg,_=_flatten_features(x[qi],c[qi],valid[qi],coords[qi],target[qi]);train_rows_count=len(tx);qid=np.repeat(np.arange(len(qi),dtype=np.int32),tg);print(json.dumps({'train_rows':train_rows_count,'feature_dim':tx.shape[1]}),flush=True)
    vx,_,vg,_=_flatten_features(x[tr:va],c[tr:va],valid[tr:va],coords[tr:va],target[tr:va]);dx,_,dg,_=_flatten_features(x[va:],c[va:],valid[va:],coords[va:],target[va:]);ethv=eth_mask(meta,tr,va);ethd=eth_mask(meta,va,n)
    models={};rawv={};rawd={};reports={}
    xgb=XGBRanker(objective='rank:ndcg',n_estimators=550,learning_rate=.035,max_depth=8,min_child_weight=20,subsample=.85,colsample_bytree=.8,reg_lambda=8.,reg_alpha=.25,tree_method='hist',lambdarank_pair_method='topk',lambdarank_num_pair_per_sample=12,n_jobs=-1,random_state=a.seed,verbosity=0)
    xgb.fit(tx,ty,qid=qid,verbose=False);models['xgb']=xgb
    cat=CatBoostRanker(iterations=500,learning_rate=.045,depth=8,loss_function='YetiRank',eval_metric='NDCG:top=5',l2_leaf_reg=8.,random_seed=a.seed,verbose=100,thread_count=-1,allow_writing_files=False)
    cat.fit(Pool(tx,ty,group_id=qid));models['catboost']=cat
    del tx,ty,qid
    for name,m in models.items():
        rv=_scores_to_matrix(score_flat(m,vx,'xgb' if name=='xgb' else 'cat'),valid[tr:va]);rd=_scores_to_matrix(score_flat(m,dx,'xgb' if name=='xgb' else 'cat'),valid[va:]);cal=calibrate(rv,coords[tr:va],valid[tr:va],target[tr:va],ethv);_,temp,vglo,veth,pv=cal;pd=_softmax(rd,valid[va:],temp);dglo=metrics(rd,pd,coords[va:],valid[va:],target[va:]);deth=metrics(rd[ethd],pd[ethd],coords[va:][ethd],valid[va:][ethd],target[va:][ethd]);rawv[name]=rv;rawd[name]=rd;reports[name]={'temperature':temp,'validation':vglo,'validation_ethiopia':veth,'development':dglo,'development_ethiopia':deth};print(json.dumps({name:reports[name]},indent=2),flush=True)
    # Optional baseline and convex probability ensemble.  Keep candidate support fixed.
    ensemble=None
    if a.baseline_model and a.baseline_config:
        b=Booster(model_file=str(a.baseline_model));cfg=json.loads(a.baseline_config.read_text());bit=int(cfg['selected_iteration']);bt=float(cfg['probability_temperature']);brv=_scores_to_matrix(b.predict(vx,num_iteration=bit),valid[tr:va]);brd=_scores_to_matrix(b.predict(dx,num_iteration=bit),valid[va:]);bpv=_softmax(brv,valid[tr:va],bt);bpd=_softmax(brd,valid[va:],bt)
        probs_v={'baseline':bpv};probs_d={'baseline':bpd}
        for name in models:
            probs_v[name]=_softmax(rawv[name],valid[tr:va],reports[name]['temperature']);probs_d[name]=_softmax(rawd[name],valid[va:],reports[name]['temperature'])
        best=None
        grid=np.linspace(0,1,9)
        for wx in grid:
            for wc in grid:
                if wx+wc>1:continue
                wb=1-wx-wc;pv=wb*probs_v['baseline']+wx*probs_v['xgb']+wc*probs_v['catboost'];score=np.log(np.maximum(pv,1e-12));g=metrics(score,pv,coords[tr:va],valid[tr:va],target[tr:va]);e=metrics(score[ethv],pv[ethv],coords[tr:va][ethv],valid[tr:va][ethv],target[tr:va][ethv]);row=(e['broad_area_score'],wb,wx,wc,g,e)
                if best is None or row[0]>best[0]:best=row
        assert best;_,wb,wx,wc,vglo,veth=best;pd=wb*probs_d['baseline']+wx*probs_d['xgb']+wc*probs_d['catboost'];sd=np.log(np.maximum(pd,1e-12));dglo=metrics(sd,pd,coords[va:],valid[va:],target[va:]);deth=metrics(sd[ethd],pd[ethd],coords[va:][ethd],valid[va:][ethd],target[va:][ethd]);ensemble={'weights':{'baseline':wb,'xgb':wx,'catboost':wc},'validation':vglo,'validation_ethiopia':veth,'development':dglo,'development_ethiopia':deth};print(json.dumps({'ensemble':ensemble},indent=2),flush=True)
    a.output_dir.mkdir(parents=True,exist_ok=True);models['xgb'].save_model(str(a.output_dir/'xgb_ranker.json'));models['catboost'].save_model(str(a.output_dir/'catboost_ranker.cbm'));np.savez_compressed(a.output_dir/'alt_ranker_logits.npz',xgb_validation=rawv['xgb'],xgb_development=rawd['xgb'],catboost_validation=rawv['catboost'],catboost_development=rawd['catboost']);report={'schema':'geo-alt-rankers-v1','train_queries':len(qi),'train_rows':int(train_rows_count),'models':reports,'ensemble':ensemble,'protocol':'fixed global/Horn replay subset from first 70%; full 15% validation calibration/ensemble selection; final 15% development diagnostic only'};(a.output_dir/'alt_ranker_metrics.json').write_text(json.dumps(report,indent=2)+'\n');write_info(a.output_dir,ModelInfo(subsystem='location/geo_alt_rankers',version='v1',status='research-challenger',description='XGBoost LambdaMART and CatBoost YetiRank candidate experts with optional LightGBM convex ensemble.',metrics={'models':reports,'ensemble':ensemble},lineage={'dataset':str(a.data),'restore_tag':'production-boost-preflight-2026-08-24'},training={'train_queries':len(qi),'seed':a.seed},notes=['Alternative ranking libraries installed as research dependencies.','Development remains diagnostic.']));print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
