#!/usr/bin/env python3
"""Export saved MDN mixture modes without importing LightGBM/OpenMP."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import torch
from torch import nn

class MDN(nn.Module):
 def __init__(self,din:int,k:int,width:int):
  super().__init__();self.k=k;self.net=nn.Sequential(nn.LayerNorm(din),nn.Linear(din,width),nn.GELU(),nn.Linear(width,width),nn.GELU(),nn.Linear(width,256),nn.GELU());self.out=nn.Linear(256,k*5)
 def forward(self,x):
  z=self.out(self.net(x)).view(len(x),self.k,5);return z[...,0],z[...,1:3],z[...,3:5].clamp(-5.,-1.)

def history_context(events):
 valid=events[:,:,0]>.5;w=valid.astype(np.float32)[:,:,None];count=np.maximum(w.sum(1),1.);mean=(events*w).sum(1)/count;last=events[:,-1];recent4=events[:,-4:].mean(1);previous4=events[:,-8:-4].mean(1);delta4=recent4-previous4;selected=np.asarray([1,2,3,4,5,9,10,11,12,13,14,19,20,21,22,23,24],np.int64);return np.concatenate([last[:,selected],mean[:,selected],recent4[:,selected],delta4[:,selected]],1).astype(np.float32)
def recent_distance(events,candidates):
 cxy=candidates[...,1:3];hxy=events[:,-8:,1:3];hv=events[:,-8:,0]>.5;d=np.linalg.norm(cxy[:,:,None,:]-hxy[:,None,:,:],axis=-1);d=np.where(hv[:,None,:],d,np.nan)
 with np.errstate(invalid='ignore'):out=np.concatenate([np.nanmin(d[:,:,-4:],-1,keepdims=True),np.nanmean(d[:,:,-4:],-1,keepdims=True),np.nanmin(d,-1,keepdims=True),np.nanmean(d,-1,keepdims=True)],-1)
 return np.nan_to_num(out,nan=2.,posinf=2.,neginf=2.).astype(np.float32)
def motion(events,candidates):
 cxy=candidates[...,1:3];last=np.repeat(events[:,-1,19:21][:,None,:],candidates.shape[1],1);recent=np.repeat(events[:,-4:,19:21].mean(1)[:,None,:],candidates.shape[1],1)
 def rel(step):
  cn=np.linalg.norm(cxy,axis=-1,keepdims=True);sn=np.linalg.norm(step,axis=-1,keepdims=True);dot=(cxy*step).sum(-1,keepdims=True);cos=dot/np.maximum(cn*sn,1e-5);cross=cxy[...,0:1]*step[...,1:2]-cxy[...,1:2]*step[...,0:1];proj=dot/np.maximum(sn,1e-5);return [step,cos,cross,proj]
 return np.concatenate([cxy,*rel(last),*rel(recent)],-1).astype(np.float32)
def flat_features(events,candidates,valid):
 full=np.concatenate([candidates,motion(events,candidates),recent_distance(events,candidates)],-1).astype(np.float32);group=valid.sum(1).astype(np.int32);return np.concatenate([full[valid],np.repeat(history_context(events),group,axis=0)],1).astype(np.float32)
def export(model,events,candidates,valid,batch,device):
 f=flat_features(events,candidates,valid);pis=[];mus=[];sigs=[];model.eval()
 with torch.no_grad():
  for s in range(0,len(f),batch):
   l,m,ls=model(torch.from_numpy(f[s:s+batch]).to(device));pis.append(torch.softmax(l,-1).cpu().numpy());mus.append(m.cpu().numpy());sigs.append(torch.exp(ls).cpu().numpy())
 pi=np.zeros((*valid.shape,model.k),np.float32);mu=np.zeros((*valid.shape,model.k,2),np.float32);sig=np.zeros_like(mu);pi[valid]=np.concatenate(pis);mu[valid]=np.concatenate(mus);sig[valid]=np.concatenate(sigs);return pi,mu,sig

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--batch-size',type=int,default=8192);p.add_argument('--device',default='auto');a=p.parse_args();device=torch.device('mps' if a.device=='auto' and torch.backends.mps.is_available() else ('cuda' if a.device=='auto' and torch.cuda.is_available() else ('cpu' if a.device=='auto' else a.device)));ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False);model=MDN(int(ck['feature_dim']),int(ck['components']),int(ck['width']));model.load_state_dict(ck['state_dict']);model.to(device);z=np.load(a.data,allow_pickle=True);x=z['x'].astype(np.float32);c=z['candidate_features'].astype(np.float32);v=z['candidate_valid'];n=len(x);tr=int(.70*n);va=int(.85*n);vpi,vmu,vsig=export(model,x[tr:va],c[tr:va],v[tr:va],a.batch_size,device);dpi,dmu,dsig=export(model,x[va:],c[va:],v[va:],a.batch_size,device);a.output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.output,validation_pi=vpi,validation_mu=vmu,validation_sigma=vsig,development_pi=dpi,development_mu=dmu,development_sigma=dsig,selected_epoch=np.asarray(ck['selected_epoch']),components=np.asarray(ck['components']));print(json.dumps({'output':str(a.output),'device':str(device),'validation_shape':list(vpi.shape),'development_shape':list(dpi.shape),'selected_epoch':int(ck['selected_epoch'])},indent=2))
if __name__=='__main__':main()
