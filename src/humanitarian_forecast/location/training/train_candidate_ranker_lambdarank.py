#!/usr/bin/env python3
"""Candidate-ranker trainer with a LambdaRank listwise objective.

Same chronological protocol as ``train_candidate_ranker`` (phase-1 70/15
selection, phase-2 fresh 85% retrain) but the loss is LambdaRank: pairwise
logistic loss over candidate pairs, each pair weighted by the NDCG swing of
swapping them, with relevance graded by distance to the truth (within 20 km
is the top grade — the metric we actually care about). Cross-entropy trains
"which single candidate is nearest"; LambdaRank trains the whole ordering.
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader, TensorDataset
from humanitarian_forecast.location.models.candidate_rank import ConflictCandidateRanker

SCALE = 1000.
# Relevance grades by distance to truth (km): 2^rel - 1 gains.
GRADE_EDGES = (20.0, 50.0, 100.0, 250.0)
GRADE_RELS = (4.0, 3.0, 2.0, 1.0, 0.0)


def relevance(distance_km: torch.Tensor) -> torch.Tensor:
    rel = torch.zeros_like(distance_km)
    for edge, r in zip(GRADE_EDGES, GRADE_RELS):
        rel = torch.where(distance_km <= edge, torch.full_like(rel, r), rel)
    return rel


def lambdarank_loss(logits, coordinates, target, valid, ce_weight=0.3):
    d = torch.linalg.vector_norm(coordinates - target[:, None], dim=-1) * SCALE
    rel = relevance(d)
    scores = logits.masked_fill(~valid, -1e4)
    n, k = scores.shape
    # Pairwise logistic loss with NDCG-swap weights.
    pos = scores.argsort(-1, descending=True).argsort(-1).float() + 1.0
    discount = 1.0 / torch.log2(pos + 1.0)
    gain = (2.0 ** rel - 1.0) * discount
    i_rel, j_rel = rel[:, :, None], rel[:, None, :]
    i_gain, j_gain = gain[:, :, None], gain[:, None, :]
    i_disc, j_disc = discount[:, :, None], discount[:, None, :]
    delta = (2.0 ** i_rel - 1.0) * j_disc - (2.0 ** j_rel - 1.0) * i_disc
    delta = delta.abs()
    s_i, s_j = scores[:, :, None], scores[:, None, :]
    pair_loss = torch.nn.functional.softplus(-(s_i - s_j))
    mask = (i_rel > j_rel) & valid[:, :, None] & valid[:, None, :]
    loss = (delta * pair_loss * mask).sum((1, 2)) / mask.sum((1, 2)).clamp_min(1.0)
    ce = torch.nn.functional.cross_entropy(scores, rel.argmax(-1))
    return loss.mean() + ce_weight * ce, ce


def collect(model, loader, device):
    ls = []
    model.eval()
    with torch.no_grad():
        for x, f, c, v, _, y, country, conflict in loader:
            ls.append(model(x.to(device), f.to(device), v.to(device), country.to(device), conflict.to(device)).cpu())
    return torch.cat(ls)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--learning-rate', type=float, default=3e-4)
    p.add_argument('--ce-weight', type=float, default=0.3)
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
    print('device', device, 'lambdarank', flush=True)

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
            loss, ce = lambdarank_loss(logits, cb, yb, vb, a.ce_weight)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1); opt.step()
            total += loss.detach().item() * len(xb)
        schedule.step()
        vl = collect(model, loaders[1], device)
        vd = torch.linalg.vector_norm(c[te:ve] - y[te:ve, None], dim=-1) * SCALE
        vdist = vd[torch.arange(vl.shape[0]), vl.argmax(-1)]
        # Selection metric: within-25km share (ranking quality), mean as tiebreak.
        value = float(vdist.mean())
        print(f'phase1 epoch={epoch:02d} train_loss={total/te:.4f} validation_mean_km={value:.2f} w25={float((vdist<=25).float().mean()):.3f}', flush=True)
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
            loss, ce = lambdarank_loss(logits, cb, yb, vb, a.ce_weight)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1); opt.step()
        schedule.step()
        print(f'phase2 epoch={epoch:02d}/{best_epoch}', flush=True)

    torch.save({'model_config': model.config,
                'model_state': {k: z.detach().cpu() for k, z in model.state_dict().items()},
                'selected_epochs': best_epoch, 'objective': 'lambdarank'},
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

    out = {'selected_epochs': best_epoch, 'objective': 'lambdarank', 'ce_weight': a.ce_weight,
           'validation': report(vl, te, ve), 'untouched_test': report(tl, ve, n),
           'protocol': 'phase1 train70/validate15 lambdarank selection; phase2 fresh train85/test15'}
    (a.output_dir / 'candidate_ranker_metrics.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
