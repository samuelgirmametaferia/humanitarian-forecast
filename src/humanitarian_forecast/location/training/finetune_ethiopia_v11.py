#!/usr/bin/env python3
"""Ethiopia-adapted fine-tune of the v11 spillover candidate ranker.

Trains the standard phase-1 model on the global 0-70% split (so Ethiopia
validation rows stay untouched), then fine-tunes on Ethiopia train rows only
and selects the fine-tune epoch on Ethiopia validation. Development rows are
never trained on or selected on.
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader, TensorDataset
from humanitarian_forecast.location.models.candidate_rank import ConflictCandidateRanker, loss_fn

SCALE = 1000.


def collect(model, loader, device):
    ls = []
    model.eval()
    with torch.no_grad():
        for x, f, c, v, _, y, country, conflict in loader:
            ls.append(model(x.to(device), f.to(device), v.to(device), country.to(device), conflict.to(device)).cpu())
    return torch.cat(ls)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path('data/location/conflict_candidates_64_spillover_v6.npz'))
    p.add_argument('--output-dir', type=Path, default=Path('models/location/candidate_ranker/v11_ethiopia_finetune'))
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--ft-epochs', type=int, default=6)
    p.add_argument('--ft-lr', type=float, default=5e-5)
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--distance-weight', type=float, default=2.0)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    d = np.load(a.data)
    x = torch.from_numpy(d['x']).float(); f = torch.from_numpy(d['candidate_features']).float()
    c = torch.from_numpy(d['candidate_coordinates']).float(); v = torch.from_numpy(d['candidate_valid'])
    label = torch.from_numpy(d['label']); y = torch.from_numpy(d['y']).float()
    meta = [json.loads(str(z)) for z in d['meta']]
    n = len(x); te = int(.7 * n); ve = int(.85 * n)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print('device', device, flush=True)

    countries = {z: i + 1 for i, z in enumerate(sorted({m['country'] for m in meta[:te]}))}
    conflicts = {z: i + 1 for i, z in enumerate(sorted({m['conflict_id'] for m in meta[:te]}))}
    ctry = torch.tensor([countries.get(m['country'], 0) for m in meta])
    cflt = torch.tensor([conflicts.get(m['conflict_id'], 0) for m in meta])
    dataset = TensorDataset(x, f, c, v, label, y, ctry, cflt)

    eth = np.asarray([i for i, m in enumerate(meta) if m['country'] == 'Ethiopia'])
    eth_train = eth[eth < te]
    eth_val = eth[(eth >= te) & (eth < ve)]
    eth_dev = eth[eth >= ve]
    print(f'ethiopia rows: train={len(eth_train)} validation={len(eth_val)} development={len(eth_dev)}', flush=True)

    def fresh():
        return ConflictCandidateRanker(x.shape[-1], f.shape[-1], x.shape[1], len(countries) + 1, len(conflicts) + 1).to(device)

    def run_epoch(model, rows, lr, shuffle):
        loader = DataLoader(torch.utils.data.Subset(dataset, rows), batch_size=a.batch_size, shuffle=shuffle)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.01)
        model.train(); total = 0
        for xb, fb, cb, vb, lb, yb, countryb, conflictb in loader:
            xb, fb, cb, vb, lb, yb, countryb, conflictb = [z.to(device) for z in (xb, fb, cb, vb, lb, yb, countryb, conflictb)]
            opt.zero_grad(set_to_none=True)
            logits = model(xb, fb, vb, countryb, conflictb)
            loss, _, _, _ = loss_fn(logits, cb, yb, lb, a.distance_weight)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1); opt.step()
            total += loss.detach().item() * len(xb)
        return total / len(rows)

    # Phase-1: global train on 0-70%, select on global validation (te-ve).
    model = fresh()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    train_loader = DataLoader(torch.utils.data.Subset(dataset, range(0, te)), batch_size=a.batch_size, shuffle=True)
    val_loader = DataLoader(torch.utils.data.Subset(dataset, range(te, ve)), batch_size=a.batch_size)
    best = float('inf'); best_state = None; best_epoch = 1; stale = 0
    for epoch in range(1, a.epochs + 1):
        model.train(); total = 0
        for xb, fb, cb, vb, lb, yb, countryb, conflictb in train_loader:
            xb, fb, cb, vb, lb, yb, countryb, conflictb = [z.to(device) for z in (xb, fb, cb, vb, lb, yb, countryb, conflictb)]
            opt.zero_grad(set_to_none=True)
            logits = model(xb, fb, vb, countryb, conflictb)
            loss, _, _, _ = loss_fn(logits, cb, yb, lb, a.distance_weight)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1); opt.step()
            total += loss.detach().item() * len(xb)
        schedule.step()
        vl = collect(model, val_loader, device)
        vd = torch.linalg.vector_norm(c[te:ve] - y[te:ve, None], dim=-1) * SCALE
        value = float(vd[torch.arange(vl.shape[0]), vl.argmax(-1)].mean())
        print(f'phase1 epoch={epoch:02d} train_loss={total/te:.4f} validation_mean_km={value:.2f}', flush=True)
        if value < best:
            best = value; best_epoch = epoch
            best_state = {k: z.detach().cpu().clone() for k, z in model.state_dict().items()}; stale = 0
        else:
            stale += 1
            if stale >= 4: break
    model.load_state_dict(best_state)
    base_state = {k: z.clone() for k, z in model.state_dict().items()}

    # Ethiopia fine-tune, selected on Ethiopia validation.
    def eth_metrics(model, rows):
        loader = DataLoader(torch.utils.data.Subset(dataset, rows), batch_size=256)
        logits = collect(model, loader, device)
        dd = torch.linalg.vector_norm(c[rows] - y[rows, None], dim=-1) * SCALE
        e = dd[torch.arange(len(rows)), logits.argmax(-1)]
        return {'median_km': float(e.median()), 'w20': float((e <= 20).float().mean()),
                'w50': float((e <= 50).float().mean())}

    print('base ethiopia validation:', json.dumps(eth_metrics(model, eth_val)), flush=True)
    ft_best = eth_metrics(model, eth_val)['median_km']; ft_best_state = {k: z.clone() for k, z in model.state_dict().items()}; ft_best_epoch = 0
    for epoch in range(1, a.ft_epochs + 1):
        run_epoch(model, eth_train, a.ft_lr, True)
        m = eth_metrics(model, eth_val)
        print(f'finetune epoch={epoch:02d} eth_val={json.dumps(m)}', flush=True)
        if m['median_km'] < ft_best:
            ft_best = m['median_km']; ft_best_epoch = epoch
            ft_best_state = {k: z.detach().cpu().clone() for k, z in model.state_dict().items()}
    model.load_state_dict(ft_best_state)

    a.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({'model_config': model.config, 'model_state': {k: z.detach().cpu() for k, z in model.state_dict().items()},
                'base_state': base_state, 'base_epoch': best_epoch, 'finetune_epoch': ft_best_epoch,
                'country_map': countries, 'conflict_map': conflicts},
               a.output_dir / 'candidate_ranker_best.pt')
    report = {'base_epoch': best_epoch, 'finetune_epoch': ft_best_epoch,
              'ethiopia_validation': eth_metrics(model, eth_val),
              'ethiopia_development': eth_metrics(model, eth_dev),
              'protocol': 'global phase1 train70/validate15; ethiopia-only finetune selected on ethiopia validation; development untouched'}
    (a.output_dir / 'candidate_ranker_metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
