#!/usr/bin/env python3
"""Reproduce v9's 70%-trained phase-1 model and export broad-area logits.

This exists so ensemble weights can be selected on chronological validation
without leaking through v9's later 85%-trained production checkpoint.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from humanitarian_forecast.location.models.candidate_rank import ConflictCandidateRanker, loss_fn


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--learning-rate', type=float, default=3e-4)
    p.add_argument('--seed', type=int, default=20260816)
    p.add_argument('--device', default='auto')
    args = p.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f'refusing to overwrite non-empty output dir: {args.output_dir}')

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    d = np.load(args.data, allow_pickle=True)
    x = torch.from_numpy(d['x']).float()
    f = torch.from_numpy(d['candidate_features']).float()
    c = torch.from_numpy(d['candidate_coordinates']).float()
    v = torch.from_numpy(d['candidate_valid'])
    label = torch.from_numpy(d['label'])
    y = torch.from_numpy(d['y']).float()
    meta = [json.loads(str(z)) for z in d['meta']]
    n = len(x); train_end = int(.70*n); valid_end = int(.85*n)
    if args.device == 'auto':
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    countries = {z:i+1 for i,z in enumerate(sorted({m['country'] for m in meta[:train_end]}))}
    conflicts = {z:i+1 for i,z in enumerate(sorted({m['conflict_id'] for m in meta[:train_end]}))}
    country = torch.tensor([countries.get(m['country'],0) for m in meta])
    conflict = torch.tensor([conflicts.get(m['conflict_id'],0) for m in meta])
    dataset = TensorDataset(x,f,c,v,label,y,country,conflict)
    loader = DataLoader(torch.utils.data.Subset(dataset, range(train_end)), batch_size=args.batch_size, shuffle=True)
    model = ConflictCandidateRanker(x.shape[-1], f.shape[-1], x.shape[1], len(countries)+1, len(conflicts)+1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=.01)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    for epoch in range(1,args.epochs+1):
        model.train(); total=0.0
        for xb,fb,cb,vb,lb,yb,countryb,conflictb in loader:
            xb,fb,cb,vb,lb,yb,countryb,conflictb=[z.to(device) for z in (xb,fb,cb,vb,lb,yb,countryb,conflictb)]
            optimizer.zero_grad(set_to_none=True)
            logits=model(xb,fb,vb,countryb,conflictb)
            loss,_,_,_=loss_fn(logits,cb,yb,lb,0.0,50.0)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
            total += float(loss.detach().cpu())*len(xb)
        schedule.step(); print(f'v9 phase1 epoch={epoch:02d}/{args.epochs} loss={total/train_end:.6f}',flush=True)

    def collect(lo:int,hi:int) -> np.ndarray:
        out=[]; model.eval()
        subset=DataLoader(torch.utils.data.Subset(dataset,range(lo,hi)),batch_size=1024,shuffle=False)
        with torch.no_grad():
            for xb,fb,cb,vb,lb,yb,countryb,conflictb in subset:
                out.append(model(xb.to(device),fb.to(device),vb.to(device),countryb.to(device),conflictb.to(device)).cpu().numpy())
        return np.concatenate(out)
    val_logits=collect(train_end,valid_end); dev_logits=collect(valid_end,n)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    torch.save({'model_config':model.config,'model_state':{k:z.detach().cpu() for k,z in model.state_dict().items()},'epochs':args.epochs,'seed':args.seed,'training_fraction':0.70,'purpose':'honest validation export for broad-area ensemble'},args.output_dir/'v9_phase1.pt')
    np.savez_compressed(args.output_dir/'probability_export.npz',validation_logits=val_logits.astype(np.float32),development_logits=dev_logits.astype(np.float32))
    (args.output_dir/'info.json').write_text(json.dumps({'data':str(args.data),'train_end':train_end,'validation_end':valid_end,'samples':n,'epochs':args.epochs,'seed':args.seed,'loss':'candidate CE + 50*probability-weighted center distance; v9 frozen recipe reproduction','note':'This phase-1 reproduction is for validation-only broad-area ensembling. It does not replace v9.'},indent=2)+'\n')
    print(json.dumps({'validation_logits':list(val_logits.shape),'development_logits':list(dev_logits.shape),'output':str(args.output_dir)},indent=2))

if __name__=='__main__': main()
