#!/usr/bin/env python3
"""Two-stage candidate cascade for fine-resolution ranking.

Stage 1 is the standard v12 recipe trained on subsets; stage 2 is a fresh
ranker that only sees stage-1's top-``--shortlist`` candidates, so it learns
the fine distinction among plausible sites instead of spreading capacity over
the whole 96-candidate pool.

Leak-free by construction:

* stage-2 training rows carry **out-of-fold** stage-1 scores (two fold models
  trained on disjoint halves of the 0-70% block each score the other half);
* the stage-1 model used at evaluation time is trained on the full 0-70%
  block and never sees validation rows;
* stage-2 is selected on the untouched validation block, scored once on
  development.

Stage-2 candidate features are the v7 features plus two cascade features
(stage-1 probability, stage-1 rank).
"""
from __future__ import annotations

import argparse, json, random
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader, TensorDataset
from humanitarian_forecast.location.models.candidate_rank import ConflictCandidateRanker, loss_fn

SCALE = 1000.


def identity_maps(meta, end):
    countries = {z: i + 1 for i, z in enumerate(sorted({m['country'] for m in meta[:end]}))}
    conflicts = {z: i + 1 for i, z in enumerate(sorted({m['conflict_id'] for m in meta[:end]}))}
    return countries, conflicts


def train_model(x, f, c, v, label, y, ctry, cflt, rows, epochs, lr, batch_size, device, seed,
                distance_weight=2.0, log_prefix='', select_rows=None):
    """Train a ranker on ``rows``; if ``select_rows`` given, select the epoch
    by mean error on them, else run all epochs (fixed budget)."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model = ConflictCandidateRanker(x.shape[-1], f.shape[-1], x.shape[1],
                                    int(ctry.max()) + 1, int(cflt.max()) + 1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.01)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    loader = DataLoader(TensorDataset(x[rows], f[rows], c[rows], v[rows], label[rows], y[rows],
                                      ctry[rows], cflt[rows]), batch_size=batch_size, shuffle=True)
    best = float('inf'); best_state = None
    for epoch in range(1, epochs + 1):
        model.train(); total = 0
        for xb, fb, cb, vb, lb, yb, cb1, cb2 in loader:
            xb, fb, cb, vb, lb, yb, cb1, cb2 = [z.to(device) for z in (xb, fb, cb, vb, lb, yb, cb1, cb2)]
            opt.zero_grad(set_to_none=True)
            logits = model(xb, fb, vb, cb1, cb2)
            loss, _, _, _ = loss_fn(logits, cb, yb, lb, distance_weight)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1); opt.step()
            total += loss.detach().item() * len(xb)
        schedule.step()
        line = f'{log_prefix} epoch={epoch:02d} train_loss={total/len(rows):.4f}'
        if select_rows is not None:
            probs = score(model, x, f, v, ctry, cflt, select_rows, device)
            dd = torch.linalg.vector_norm(c[select_rows] - y[select_rows, None], dim=-1) * SCALE
            e = dd[torch.arange(len(select_rows)), probs.argmax(-1)]
            value = float(e.mean())
            line += f' select_mean_km={value:.2f}'
            if value < best:
                best = value
                best_state = {k: z.detach().cpu().clone() for k, z in model.state_dict().items()}
        print(line, flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_state if best_state is not None else {k: z.detach().cpu() for k, z in model.state_dict().items()}


def score(model, x, f, v, ctry, cflt, rows, device, batch=256):
    model.eval()
    out = []
    with torch.no_grad():
        for lo in range(0, len(rows), batch):
            sl = rows[lo:lo + batch]
            out.append(model(x[sl].to(device), f[sl].to(device), v[sl].to(device),
                             ctry[sl].to(device), cflt[sl].to(device)).cpu())
    return torch.cat(out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path('data/location/conflict_candidates_96_spillover_v7.npz'))
    p.add_argument('--output-dir', type=Path, default=Path('models/location/candidate_ranker/v12_cascade'))
    p.add_argument('--shortlist', type=int, default=12)
    p.add_argument('--fold-epochs', type=int, default=8)
    p.add_argument('--stage2-epochs', type=int, default=12)
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()

    d = np.load(a.data, allow_pickle=True)
    x = torch.from_numpy(d['x']).float(); f = torch.from_numpy(d['candidate_features']).float()
    c = torch.from_numpy(d['candidate_coordinates']).float(); v = torch.from_numpy(d['candidate_valid'])
    label = torch.from_numpy(d['label']); y = torch.from_numpy(d['y']).float()
    meta = [json.loads(str(z)) for z in d['meta']]
    n = len(x); te = int(.7 * n); ve = int(.85 * n)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print('device', device, flush=True)
    countries, conflicts = identity_maps(meta, te)
    ctry = torch.tensor([countries.get(m['country'], 0) for m in meta])
    cflt = torch.tensor([conflicts.get(m['conflict_id'], 0) for m in meta])

    # --- Stage 1: cross-fit folds on the 0-70% block -----------------------
    # Fold/stage-1 states are cached in the output dir so a rerun after a crash
    # does not retrain them.
    a.output_dir.mkdir(parents=True, exist_ok=True)

    def cached(key, build):
        path = a.output_dir / f'{key}.pt'
        if path.exists():
            state = torch.load(path, map_location='cpu', weights_only=False)
            model = ConflictCandidateRanker(state['model_config']['event_dim'], state['model_config']['candidate_dim'],
                                            state['model_config']['sequence_length'], state['model_config']['countries'],
                                            state['model_config']['conflicts']).to(device)
            model.load_state_dict(state['model_state'])
            print(f'{key}: loaded cached', flush=True)
            return model, state['model_state']
        model, model_state = build()
        torch.save({'model_config': model.config, 'model_state': model_state}, path)
        return model, model_state

    half = te // 2
    fold_a, _ = cached('fold_a', lambda: train_model(
        x, f, c, v, label, y, ctry, cflt, np.arange(0, half),
        a.fold_epochs, 3e-4, a.batch_size, device, a.seed, log_prefix='foldA'))
    fold_b, _ = cached('fold_b', lambda: train_model(
        x, f, c, v, label, y, ctry, cflt, np.arange(half, te),
        a.fold_epochs, 3e-4, a.batch_size, device, a.seed + 1, log_prefix='foldB'))
    full_s1, full_state = cached('stage1_full', lambda: train_model(
        x, f, c, v, label, y, ctry, cflt, np.arange(0, te),
        a.fold_epochs + 4, 3e-4, a.batch_size, device, a.seed + 2,
        log_prefix='stage1', select_rows=np.arange(te, ve)))

    # --- Stage-2 dataset: shortlist + cascade features ----------------------
    K = c.shape[1]
    S = a.shortlist
    f2 = torch.zeros(n, S, f.shape[-1] + 2)
    c2 = torch.zeros(n, S, 2)
    v2 = torch.zeros(n, S, dtype=torch.bool)
    label2 = torch.zeros(n, dtype=torch.long)
    y2 = y.clone()

    def fill_stage2(rows, s1_logits):
        s1_probs = torch.where(v[rows], s1_logits.softmax(-1), torch.zeros(()))
        order = torch.argsort(torch.where(v[rows], s1_logits, torch.full((), -1e9)), -1, descending=True)
        topk = order[:, :S]
        rows_t = torch.as_tensor(rows)
        f2[rows_t] = torch.cat([f[rows_t][torch.arange(len(rows_t))[:, None], topk],
                                s1_probs[torch.arange(len(rows_t))[:, None], topk][..., None],
                                (torch.arange(S, dtype=torch.float32) / S)[None, :, None].expand(len(rows_t), S, 1)], -1)
        c2[rows_t] = c[rows_t][torch.arange(len(rows_t))[:, None], topk]
        v2[rows_t] = v[rows_t][torch.arange(len(rows_t))[:, None], topk]
        dd = torch.linalg.vector_norm(c2[rows_t] - y[rows_t, None], dim=-1) * SCALE
        label2[rows_t] = dd.argmin(-1)

    # Out-of-fold stage-1 scores for the training block.
    la = score(fold_a, x, f, v, ctry, cflt, np.arange(half, te), device)
    fill_stage2(np.arange(half, te), la)
    lb = score(fold_b, x, f, v, ctry, cflt, np.arange(0, half), device)
    fill_stage2(np.arange(0, half), lb)
    # Full stage-1 scores for evaluation blocks (never trained on them).
    lv = score(full_s1, x, f, v, ctry, cflt, np.arange(te, ve), device)
    fill_stage2(np.arange(te, ve), lv)
    ld = score(full_s1, x, f, v, ctry, cflt, np.arange(ve, n), device)
    fill_stage2(np.arange(ve, n), ld)

    # --- Stage 2: fresh ranker over the shortlist ---------------------------
    s2_model, s2_state = train_model(x, f2, c2, v2, label2, y2, ctry, cflt, np.arange(0, te),
                                     a.stage2_epochs, 3e-4, a.batch_size, device, a.seed + 3,
                                     log_prefix='stage2',
                                     select_rows=np.arange(te, ve))

    # --- Evaluation ----------------------------------------------------------
    eth = np.asarray([i for i, m in enumerate(meta) if m['country'] == 'Ethiopia'])
    results = {}
    with torch.no_grad():
        for name, rows in (('validation', eth[(eth >= te) & (eth < ve)]), ('development', eth[eth >= ve])):
            s2_logits = score(s2_model, x, f2, v2, ctry, cflt, rows, device)
            s1_logits = score(full_s1, x, f, v, ctry, cflt, rows, device)
            dd1 = torch.linalg.vector_norm(c[rows] - y[rows, None], dim=-1) * SCALE
            dd2 = torch.linalg.vector_norm(c2[rows] - y[rows, None], dim=-1) * SCALE
            e_s1 = dd1[torch.arange(len(rows)), s1_logits.argmax(-1)]
            e_s2 = dd2[torch.arange(len(rows)), s2_logits.argmax(-1)]
            # top-k via stage-2 ordering of the shortlist
            order2 = torch.argsort(torch.where(v2[rows], s2_logits, torch.full((), -1e9)), -1, descending=True)
            res = {'samples': int(len(rows)),
                   'stage1_top1_w20': round(float((e_s1 <= 20).float().mean()), 3),
                   'cascade_top1_w20': round(float((e_s2 <= 20).float().mean()), 3),
                   'stage1_median_km': round(float(e_s1.median()), 1),
                   'cascade_median_km': round(float(e_s2.median()), 1)}
            for topk in (3, 5, 10):
                if topk <= S:
                    ek = dd2[torch.arange(len(rows))[:, None], order2[:, :topk]].min(1).values
                    res[f'cascade_w20_at_top{topk}'] = round(float((ek <= 20).float().mean()), 3)
            shortlist_oracle = torch.where(v2[rows], dd2, torch.full((), 1e9)).min(1).values
            res['shortlist_oracle_w20'] = round(float((shortlist_oracle <= 20).float().mean()), 3)
            results[name] = res
    print(json.dumps(results, indent=2))

    a.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({'model_config': s2_model.config, 'model_state': s2_state,
                'stage1_state': full_state, 'stage1_config': full_s1.config,
                'country_map': countries, 'conflict_map': conflicts,
                'shortlist': S, 'stage2_feature_offset': int(f.shape[-1])},
               a.output_dir / 'candidate_ranker_best.pt')
    (a.output_dir / 'candidate_ranker_metrics.json').write_text(json.dumps(
        {'results': results, 'shortlist': S, 'fold_epochs': a.fold_epochs,
         'stage2_epochs': a.stage2_epochs,
         'protocol': 'stage1 0-70% (validation-selected); stage2 trained on out-of-fold stage-1 shortlists; development untouched'}, indent=2) + '\n')


if __name__ == '__main__':
    main()
