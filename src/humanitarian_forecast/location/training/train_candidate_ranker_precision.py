#!/usr/bin/env python3
"""Candidate-ranker trainer with UCDP geolocation-precision weighting.

Same chronological protocol as ``train_candidate_ranker``, but each row's
loss is weighted by its target's UCDP ``where_prec``: exactly-geolocated
targets (where_prec==1) carry full weight, named-place-radius targets less.
Roughly half of Ethiopia rows are geolocated to a named place with a
25-100 km radius — training the fine-scale discrimination equally on those
labels injects coordinate noise exactly at the scale we are trying to learn.

``--precision-weights 1,0.4,0.1`` maps where_prec 1/2/3+ to weights;
pass ``1,0,0`` to train only on exactly-geolocated targets.
"""
from __future__ import annotations
import argparse, csv, io, json, random, zipfile
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader, TensorDataset
from humanitarian_forecast.location.models.candidate_rank import ConflictCandidateRanker

SCALE = 1000.


def target_precision(meta, ucdp_zip: Path) -> np.ndarray:
    """where_prec of each row's target event, from the raw GED export."""
    prec = {}
    with zipfile.ZipFile(ucdp_zip) as archive:
        name = next(v for v in archive.namelist() if v.endswith('.csv'))
        with archive.open(name) as raw:
            for row in csv.DictReader(io.TextIOWrapper(raw, encoding='utf-8-sig')):
                try:
                    key = (row['date_start'][:10], round(float(row['latitude']), 5), round(float(row['longitude']), 5))
                except (ValueError, KeyError):
                    continue
                prec.setdefault(key, int(row['where_prec']))
    out = np.zeros(len(meta), dtype=np.float32)
    for i, m in enumerate(meta):
        key = (m['target_date'], round(float(m['target_lat']), 5), round(float(m['target_lon']), 5))
        out[i] = prec.get(key, 0)
    return out


def collect(model, loader, device):
    ls = []
    model.eval()
    with torch.no_grad():
        for x, f, c, v, _, y, country, conflict, _w in loader:
            ls.append(model(x.to(device), f.to(device), v.to(device), country.to(device), conflict.to(device)).cpu())
    return torch.cat(ls)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--ucdp-events', type=Path, default=Path('data/raw/ged261-csv.zip'))
    p.add_argument('--precision-weights', default='1,0.4,0.1',
                   help='comma weights for where_prec 1/2/3+ (e.g. 1,0,0 = exact only)')
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--learning-rate', type=float, default=3e-4)
    p.add_argument('--distance-weight', type=float, default=2.0)
    p.add_argument('--d-model', type=int, default=128)
    p.add_argument('--layers', type=int, default=3)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    w1, w2, w3 = (float(v) for v in a.precision_weights.split(','))
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    d = np.load(a.data)
    x = torch.from_numpy(d['x']).float(); f = torch.from_numpy(d['candidate_features']).float()
    c = torch.from_numpy(d['candidate_coordinates']).float(); v = torch.from_numpy(d['candidate_valid'])
    label = torch.from_numpy(d['label']); y = torch.from_numpy(d['y']).float()
    meta = [json.loads(str(z)) for z in d['meta']]
    n = len(x); te = int(.7 * n); ve = int(.85 * n)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    prec = target_precision(meta, a.ucdp_events)
    weight = np.where(prec <= 1, w1, np.where(prec == 2, w2, w3)).astype(np.float32)
    weight = torch.from_numpy(weight)
    kept = int((weight[:te] > 0).sum())
    print(f'device {device} precision_weights=({w1},{w2},{w3}) train rows kept={kept}/{te}', flush=True)

    def identity_maps(end):
        countries = {z: i + 1 for i, z in enumerate(sorted({m['country'] for m in meta[:end]}))}
        conflicts = {z: i + 1 for i, z in enumerate(sorted({m['conflict_id'] for m in meta[:end]}))}
        return countries, conflicts

    def make_dataset(country_map, conflict_map):
        countries = torch.tensor([country_map.get(m['country'], 0) for m in meta])
        conflicts = torch.tensor([conflict_map.get(m['conflict_id'], 0) for m in meta])
        return TensorDataset(x, f, c, v, label, y, countries, conflicts, weight)

    country1, conflict1 = identity_maps(te); dataset1 = make_dataset(country1, conflict1)
    loaders = [DataLoader(torch.utils.data.Subset(dataset1, range(i, j)), batch_size=a.batch_size, shuffle=(i == 0))
               for i, j in ((0, te), (te, ve), (ve, n))]

    def fresh(country_map, conflict_map):
        return ConflictCandidateRanker(x.shape[-1], f.shape[-1], x.shape[1], len(country_map) + 1,
                                       len(conflict_map) + 1, d_model=a.d_model, layers=a.layers).to(device)

    def run_epoch(model, loader, lr, opt=None):
        if opt is None:
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.01)
        model.train(); total = 0
        for xb, fb, cb, vb, lb, yb, countryb, conflictb, wb in loader:
            xb, fb, cb, vb, lb, yb, countryb, conflictb, wb = [z.to(device) for z in (xb, fb, cb, vb, lb, yb, countryb, conflictb, wb)]
            opt.zero_grad(set_to_none=True)
            logits = model(xb, fb, vb, countryb, conflictb)
            ce = torch.nn.functional.cross_entropy(logits, lb, reduction='none')
            prob = logits.softmax(-1)
            dist = torch.linalg.vector_norm(cb - yb[:, None], dim=-1)
            expected = (prob * dist).sum(-1)
            loss = (wb * (ce + a.distance_weight * expected)).sum() / wb.sum().clamp_min(1e-6)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1); opt.step()
            total += loss.detach().item() * len(xb)
        return total

    model = fresh(country1, conflict1)
    opt = torch.optim.AdamW(model.parameters(), lr=a.learning_rate, weight_decay=.01)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    best = float('inf'); best_epoch = 1; best_state = None; stale = 0
    for epoch in range(1, a.epochs + 1):
        total = run_epoch(model, loaders[0], a.learning_rate, opt)
        schedule.step()
        vl = collect(model, loaders[1], device)
        vd = torch.linalg.vector_norm(c[te:ve] - y[te:ve, None], dim=-1) * SCALE
        vdist = vd[torch.arange(vl.shape[0]), vl.argmax(-1)]
        print(f'phase1 epoch={epoch:02d} train_loss={total/te:.4f} validation_mean_km={float(vdist.mean()):.2f} w25={float((vdist<=25).float().mean()):.3f}', flush=True)
        if float(vdist.mean()) < best:
            best = float(vdist.mean()); best_epoch = epoch
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
        run_epoch(model, phase2, a.learning_rate, opt)
        schedule.step()
        print(f'phase2 epoch={epoch:02d}/{best_epoch}', flush=True)

    torch.save({'model_config': model.config,
                'model_state': {k: z.detach().cpu() for k, z in model.state_dict().items()},
                'selected_epochs': best_epoch, 'precision_weights': [w1, w2, w3]},
               a.output_dir / 'candidate_ranker_best.pt')
    test_loader = DataLoader(torch.utils.data.Subset(dataset2, range(ve, n)), batch_size=a.batch_size)
    vl = collect(model, loaders[1], device)
    tl = collect(model, test_loader, device)

    def report(logits, lo, hi):
        dd = torch.linalg.vector_norm(c[lo:hi] - y[lo:hi, None], dim=-1) * SCALE
        e = dd[torch.arange(logits.shape[0]), logits.argmax(-1)]
        return {'samples': int(logits.shape[0]), 'mean_error_km': float(e.mean()),
                'median_error_km': float(e.median()), 'within_25km': float((e <= 25).float().mean())}

    out = {'selected_epochs': best_epoch, 'precision_weights': [w1, w2, w3],
           'validation': report(vl, te, ve), 'untouched_test': report(tl, ve, n),
           'protocol': 'phase1 train70/validate15 precision-weighted; phase2 fresh train85/test15'}
    (a.output_dir / 'candidate_ranker_metrics.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
