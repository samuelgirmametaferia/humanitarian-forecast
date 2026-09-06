#!/usr/bin/env python3
"""Candidate-ranker trainer with distance-softened labels.

Identical protocol to ``train_candidate_ranker`` (phase-1 70/15 selection,
phase-2 fresh 85% retrain) except that the hard argmin label is replaced by a
proximity kernel over candidates: ``t_j ∝ exp(-d_j / tau)`` with d_j the
candidate's distance to the truth in kilometres. When dozens of candidates
cluster within a few kilometres of the truth, a one-hot label is mostly noise
at 20 km scale; the kernel makes every near candidate count.
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader, TensorDataset
from humanitarian_forecast.location.models.candidate_rank import ConflictCandidateRanker

SCALE = 1000.


def soft_targets(coordinates, target, valid, tau_km):
    d = torch.linalg.vector_norm(coordinates - target[:, None], dim=-1) * SCALE
    logk = -d / tau_km
    logk = logk.masked_fill(~valid, -1e9)
    return logk.softmax(-1)


def collect(model, loader, device):
    ls = []; model.eval()
    with torch.no_grad():
        for x, f, c, v, _, y, country, conflict in loader:
            ls.append(model(x.to(device), f.to(device), v.to(device), country.to(device), conflict.to(device)).cpu())
    return torch.cat(ls)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--learning-rate', type=float, default=3e-4)
    p.add_argument('--distance-weight', type=float, default=2.0)
    p.add_argument('--tau-km', type=float, default=15.0, help='Proximity kernel scale for soft labels.')
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
    print('device', device, 'tau_km', a.tau_km, flush=True)

    def identity_maps(end):
        countries = {z: i + 1 for i, z in enumerate(sorted({m['country'] for m in meta[:end]}))}
        conflicts = {z: i + 1 for i, z in enumerate(sorted({m['conflict_id'] for m in meta[:end]}))}
        return countries, conflicts

    def make_dataset(country_map, conflict_map):
        countries = torch.tensor([country_map.get(m['country'], 0) for m in meta])
        conflicts = torch.tensor([conflict_map.get(m['conflict_id'], 0) for m in meta])
        return TensorDataset(x, f, c, v, label, y, countries, conflicts)

    country1, conflict1 = identity_maps(te); dataset1 = make_dataset(country1, conflict1)
    loaders = [DataLoader(torch.utils.data.Subset(dataset1, range(i, j)), batch_size=a.batch_size, shuffle=(i == 0))
               for i, j in ((0, te), (te, ve), (ve, n))]

    def fresh(country_map, conflict_map):
        return ConflictCandidateRanker(x.shape[-1], f.shape[-1], x.shape[1], len(country_map) + 1, len(conflict_map) + 1).to(device)

    def phase_loss(logits, cb, yb, vb):
        t = soft_targets(cb, yb, vb, a.tau_km)
        ce = -(t * logits.log_softmax(-1).clamp(-30, 0)).sum(-1).mean()
        prob = logits.softmax(-1)
        dist = torch.linalg.vector_norm(cb - yb[:, None], dim=-1) * SCALE
        expected = (prob * dist).sum(-1).mean()
        return ce + a.distance_weight * expected, ce

    model = fresh(country1, conflict1)
    opt = torch.optim.AdamW(model.parameters(), lr=a.learning_rate, weight_decay=.01)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    best = float('inf'); best_epoch = 1; best_state = None; stale = 0
    for epoch in range(1, a.epochs + 1):
        model.train(); total = 0
        for xb, fb, cb, vb, lb, yb, countryb, conflictb in loaders[0]:
            xb, fb, cb, vb, yb, countryb, conflictb = [z.to(device) for z in (xb, fb, cb, vb, yb, countryb, conflictb)]
            opt.zero_grad(set_to_none=True)
            logits = model(xb, fb, vb, countryb, conflictb)
            loss, ce = phase_loss(logits, cb, yb, vb)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1); opt.step()
            total += loss.detach().item() * len(xb)
        schedule.step()
        vl = collect(model, loaders[1], device)
        vd = torch.linalg.vector_norm(c[te:ve] - y[te:ve, None], dim=-1) * SCALE
        vdist = vd[torch.arange(vl.shape[0]), vl.argmax(-1)]
        value = float(vdist.mean())
        print(f'phase1 epoch={epoch:02d} train_loss={total/te:.4f} validation_mean_km={value:.2f}', flush=True)
        if value < best:
            best = value; best_epoch = epoch
            best_state = {k: z.detach().cpu().clone() for k, z in model.state_dict().items()}; stale = 0
        else:
            stale += 1
            if stale >= 4: break

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    country2, conflict2 = identity_maps(ve); dataset2 = make_dataset(country2, conflict2)
    model = fresh(country2, conflict2)
    opt = torch.optim.AdamW(model.parameters(), lr=a.learning_rate, weight_decay=.01)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, best_epoch)
    phase2 = DataLoader(torch.utils.data.Subset(dataset2, range(0, ve)), batch_size=a.batch_size, shuffle=True)
    for epoch in range(1, best_epoch + 1):
        model.train()
        for xb, fb, cb, vb, lb, yb, countryb, conflictb in phase2:
            xb, fb, cb, vb, yb, countryb, conflictb = [z.to(device) for z in (xb, fb, cb, vb, yb, countryb, conflictb)]
            opt.zero_grad(set_to_none=True)
            logits = model(xb, fb, vb, countryb, conflictb)
            loss, ce = phase_loss(logits, cb, yb, vb)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1); opt.step()
        schedule.step()
        print(f'phase2 epoch={epoch:02d}/{best_epoch}', flush=True)

    torch.save({'model_config': model.config,
                'model_state': {k: z.detach().cpu() for k, z in model.state_dict().items()},
                'selected_epochs': best_epoch, 'tau_km': a.tau_km},
               a.output_dir / 'candidate_ranker_best.pt')
    test_loader = DataLoader(torch.utils.data.Subset(dataset2, range(ve, n)), batch_size=a.batch_size)
    vl = collect(model, loaders[1], device)
    tl = collect(model, test_loader, device)

    def report(logits, lo, hi):
        dd = torch.linalg.vector_norm(c[lo:hi] - y[lo:hi, None], dim=-1) * SCALE
        e = dd[torch.arange(logits.shape[0]), logits.argmax(-1)]
        o = torch.where(v[lo:hi], dd, torch.full((), 1e9)).min(1).values
        return {'samples': int(logits.shape[0]), 'mean_error_km': float(e.mean()),
                'median_error_km': float(e.median()), 'within_25km': float((e <= 25).float().mean()),
                'candidate_oracle_mean_km': float(o.mean())}

    out = {'selected_epochs': best_epoch, 'tau_km': a.tau_km, 'distance_weight': a.distance_weight,
           'validation': report(vl, te, ve), 'untouched_test': report(tl, ve, n),
           'protocol': 'phase1 train70/validate15 soft-label selection; phase2 fresh train85/test15'}
    (a.output_dir / 'candidate_ranker_metrics.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
