#!/usr/bin/env python3
"""V10: conservative Ethiopia specialization of the validated v9 candidate ranker.

The global v9 phase-1 representation is the parent.  Only the late scoring
modules are updated on Ethiopia rows that lie strictly before chronological
validation.  The loss uses a 100 km Gaussian distance-soft target plus a small
hard-label term and KL distillation back to frozen v9.  Epoch, learning rate,
distillation strength, and output temperature are selected on Ethiopia
validation only.  The final historical block is reporting-only development.

This is a humanitarian broad-area research challenger.  Exact coordinate
metrics are diagnostics and are not a tactical output interface.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset

from humanitarian_forecast.core.model_store import ModelInfo, write_info
from humanitarian_forecast.location.models.candidate_rank import ConflictCandidateRanker
from humanitarian_forecast.location.training.calibrate_candidate_center import weighted_geometric_median
from humanitarian_forecast.location.training.ensemble_broad_area_probabilities import (
    distance_soft_target,
    metrics as broad_metrics,
    soft_ce,
    softmax,
)

SCALE_KM = 1000.0


def _torch_soft_target(coordinates: torch.Tensor, valid: torch.Tensor, target: torch.Tensor, radius_km: float) -> torch.Tensor:
    distance = torch.linalg.vector_norm(coordinates - target[:, None], dim=-1) * SCALE_KM
    weight = torch.exp(-0.5 * (distance / radius_km) ** 2) * valid.float()
    total = weight.sum(dim=1, keepdim=True)
    bad = total[:, 0] <= 1e-30
    if bad.any():
        masked = distance.masked_fill(~valid, float('inf'))
        nearest = masked.argmin(dim=1)
        weight[bad] = 0.0
        weight[bad, nearest[bad]] = 1.0
        total = weight.sum(dim=1, keepdim=True)
    return weight / total.clamp_min(1e-12)


def _country_conflict_ids(meta: list[dict], train_end: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, int], dict[object, int]]:
    countries = {z: i + 1 for i, z in enumerate(sorted({m['country'] for m in meta[:train_end]}))}
    conflicts = {z: i + 1 for i, z in enumerate(sorted({m['conflict_id'] for m in meta[:train_end]}))}
    country = torch.tensor([countries.get(m['country'], 0) for m in meta], dtype=torch.long)
    conflict = torch.tensor([conflicts.get(m['conflict_id'], 0) for m in meta], dtype=torch.long)
    return country, conflict, countries, conflicts


def _collect(model: nn.Module, dataset: TensorDataset, indices: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    loader = DataLoader(Subset(dataset, indices.tolist()), batch_size=batch_size, shuffle=False)
    out: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for xb, fb, cb, vb, lb, yb, countryb, conflictb in loader:
            logits = model(xb.to(device), fb.to(device), vb.to(device), countryb.to(device), conflictb.to(device))
            out.append(logits.cpu().numpy())
    return np.concatenate(out, axis=0)


def _point_metrics(probability: np.ndarray, coordinates: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pt = torch.from_numpy(probability.astype(np.float32, copy=False))
    co = torch.from_numpy(coordinates.astype(np.float32, copy=False))
    y = torch.from_numpy(target.astype(np.float32, copy=False))
    with torch.no_grad():
        center = weighted_geometric_median(pt, co)
        err = torch.linalg.vector_norm(center - y, dim=-1) * SCALE_KM
    return {
        'samples': int(len(err)),
        'geomedian_mean_error_km': float(err.mean()),
        'geomedian_median_error_km': float(err.median()),
        'geomedian_p90_error_km': float(err.quantile(0.90)),
        'geomedian_within_25km': float((err <= 25).float().mean()),
        'geomedian_within_50km': float((err <= 50).float().mean()),
        'geomedian_within_100km': float((err <= 100).float().mean()),
    }


def _calibrate(scores: np.ndarray, valid: np.ndarray, q: np.ndarray) -> tuple[float, np.ndarray, float]:
    best: tuple[float, float, np.ndarray] | None = None
    for temperature in np.geomspace(0.35, 5.0, 45):
        p = softmax(scores, valid, float(temperature))
        ce = soft_ce(q, p)
        if best is None or ce < best[0] - 1e-12:
            best = (ce, float(temperature), p)
    assert best is not None
    return best[1], best[2], best[0]


def _set_trainable(model: ConflictCandidateRanker) -> list[str]:
    # Preserve v9's temporal Transformer completely.  Adapt only the final
    # context transform, candidate scorer/bias and Ethiopia-relevant embeddings.
    for p in model.parameters():
        p.requires_grad = False
    trainable_prefixes = ('context.', 'candidate.', 'bias.', 'country.', 'conflict.')
    names: list[str] = []
    for name, p in model.named_parameters():
        if name.startswith(trainable_prefixes):
            p.requires_grad = True
            names.append(name)
    return names


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=Path, default=Path('data/location/conflict_candidates_32_spatial_v5.npz'))
    ap.add_argument('--parent', type=Path, default=Path('models/location/candidate_ranker/v9_phase1_ensemble_repro/v9_phase1.pt'))
    ap.add_argument('--output-dir', type=Path, default=Path('models/location/candidate_ranker_ethiopia/v10_v9_adapter'))
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--seed', type=int, default=20260824)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--soft-radius-km', type=float, default=100.0)
    args = ap.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f'refusing to overwrite non-empty output dir: {args.output_dir}')

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    bundle = np.load(args.data, allow_pickle=True)
    x = torch.from_numpy(bundle['x']).float()
    f = torch.from_numpy(bundle['candidate_features']).float()
    coords = torch.from_numpy(bundle['candidate_coordinates']).float()
    valid = torch.from_numpy(bundle['candidate_valid']).bool()
    label = torch.from_numpy(bundle['label']).long()
    target = torch.from_numpy(bundle['y']).float()
    meta = [json.loads(str(z)) for z in bundle['meta']]
    n = len(x); train_end = int(.70 * n); valid_end = int(.85 * n)
    country, conflict, countries, conflicts = _country_conflict_ids(meta, train_end)
    dataset = TensorDataset(x, f, coords, valid, label, target, country, conflict)

    train_idx = np.asarray([i for i in range(train_end) if meta[i].get('country') == 'Ethiopia'], dtype=np.int64)
    val_idx = np.asarray([i for i in range(train_end, valid_end) if meta[i].get('country') == 'Ethiopia'], dtype=np.int64)
    dev_idx = np.asarray([i for i in range(valid_end, n) if meta[i].get('country') == 'Ethiopia'], dtype=np.int64)
    if min(len(train_idx), len(val_idx), len(dev_idx)) == 0:
        raise RuntimeError('missing Ethiopia rows in one chronological partition')

    if args.device == 'auto':
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    checkpoint = torch.load(args.parent, map_location='cpu', weights_only=False)
    config = checkpoint['model_config']
    parent = ConflictCandidateRanker(**config)
    parent.load_state_dict(checkpoint['model_state'])
    parent.to(device).eval()

    val_valid = bundle['candidate_valid'][val_idx]
    val_coords = bundle['candidate_coordinates'][val_idx].astype(np.float32, copy=False)
    val_target = bundle['y'][val_idx].astype(np.float32, copy=False)
    dev_valid = bundle['candidate_valid'][dev_idx]
    dev_coords = bundle['candidate_coordinates'][dev_idx].astype(np.float32, copy=False)
    dev_target = bundle['y'][dev_idx].astype(np.float32, copy=False)
    qv = distance_soft_target(val_coords, val_valid, val_target, args.soft_radius_km)
    qd = distance_soft_target(dev_coords, dev_valid, dev_target, args.soft_radius_km)

    base_val_logits = _collect(parent, dataset, val_idx, 1024, device)
    base_dev_logits = _collect(parent, dataset, dev_idx, 1024, device)
    base_t, base_val_p, base_val_ce = _calibrate(base_val_logits, val_valid, qv)
    base_dev_p = softmax(base_dev_logits, dev_valid, base_t)
    baseline = {
        'temperature': base_t,
        'validation': broad_metrics(base_val_p, val_coords, val_valid, val_target, qv),
        'validation_point': _point_metrics(base_val_p, val_coords, val_target),
        'development': broad_metrics(base_dev_p, dev_coords, dev_valid, dev_target, qd),
        'development_point': _point_metrics(base_dev_p, dev_coords, dev_target),
    }
    print(json.dumps({'baseline_v9_ethiopia_calibrated': baseline}, indent=2), flush=True)

    # Small pre-declared sweep.  All choices use validation only; development is
    # not touched until the single winning configuration is frozen.
    sweep = [
        {'lr': 1.0e-4, 'distill': 0.5},
        {'lr': 2.0e-4, 'distill': 0.5},
        {'lr': 1.0e-4, 'distill': 2.0},
        {'lr': 2.0e-4, 'distill': 2.0},
    ]
    hard_weight = 0.15
    train_loader = DataLoader(Subset(dataset, train_idx.tolist()), batch_size=args.batch_size, shuffle=True)
    best: dict | None = None
    trials: list[dict] = []

    for trial_id, hp in enumerate(sweep, start=1):
        student = ConflictCandidateRanker(**config)
        student.load_state_dict(checkpoint['model_state'])
        student.to(device)
        trainable = _set_trainable(student)
        optimizer = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=hp['lr'], weight_decay=.02)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
        epoch_rows: list[dict] = []
        for epoch in range(1, args.epochs + 1):
            student.train()
            # The inherited v9 temporal representation is frozen. Keep its dropout
            # disabled as well, so late-layer specialization sees the exact parent
            # representation rather than a stochastic surrogate. This also avoids
            # unsupported MPS attention-dropout training.
            student.event_norm.eval()
            student.event_projection.eval()
            student.encoder.eval()
            running = 0.0
            for xb, fb, cb, vb, lb, yb, countryb, conflictb in train_loader:
                xb, fb, cb, vb, lb, yb, countryb, conflictb = [z.to(device) for z in (xb, fb, cb, vb, lb, yb, countryb, conflictb)]
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    base_logits = parent(xb, fb, vb, countryb, conflictb)
                    base_probability = base_logits.softmax(dim=-1)
                logits = student(xb, fb, vb, countryb, conflictb)
                q = _torch_soft_target(cb, vb, yb, args.soft_radius_km)
                logp = torch.log_softmax(logits, dim=-1)
                soft_loss = -(q * logp).sum(dim=-1).mean()
                hard_loss = nn.functional.cross_entropy(logits, lb)
                distill = nn.functional.kl_div(logp, base_probability, reduction='batchmean')
                loss = soft_loss + hard_weight * hard_loss + hp['distill'] * distill
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], 1.0)
                optimizer.step(); running += float(loss.detach().cpu()) * len(xb)
            schedule.step()

            val_logits = _collect(student, dataset, val_idx, 1024, device)
            temp, val_p, val_ce = _calibrate(val_logits, val_valid, qv)
            val_report = broad_metrics(val_p, val_coords, val_valid, val_target, qv)
            row = {
                'trial': trial_id, 'epoch': epoch, 'learning_rate': hp['lr'], 'distill_weight': hp['distill'],
                'train_loss': running / len(train_idx), 'temperature': temp,
                'validation_distance_soft_ce': val_ce,
                'validation_broad_area_score': val_report['broad_area_score'],
                'validation_mean_error_km': val_report['mean_error_km'],
                'validation_median_error_km': val_report['median_error_km'],
                'validation_within_100km': val_report['within_100km'],
            }
            epoch_rows.append(row); print(json.dumps(row), flush=True)
            key = val_ce
            if best is None or key < best['validation_distance_soft_ce'] - 1e-12:
                best = {
                    **row,
                    'state': {k: v.detach().cpu().clone() for k, v in student.state_dict().items()},
                    'validation_logits': val_logits.astype(np.float32),
                    'validation_probability': val_p.astype(np.float32),
                    'validation_report': val_report,
                    'trainable_parameters': trainable,
                }
        trials.append({'trial': trial_id, **hp, 'epochs': epoch_rows})

    assert best is not None
    winner = ConflictCandidateRanker(**config)
    winner.load_state_dict(best['state']); winner.to(device).eval()
    dev_logits = _collect(winner, dataset, dev_idx, 1024, device)
    dev_p = softmax(dev_logits, dev_valid, best['temperature'])
    development = broad_metrics(dev_p, dev_coords, dev_valid, dev_target, qd)
    development_point = _point_metrics(dev_p, dev_coords, dev_target)
    validation_point = _point_metrics(best['validation_probability'], val_coords, val_target)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.save({
        'model_config': config,
        'model_state': best['state'],
        'parent': str(args.parent),
        'parent_family': 'candidate_ranker/v9',
        'adaptation': 'frozen v9 temporal encoder; Ethiopia-only late scorer fine-tune',
        'selected_trial': best['trial'],
        'selected_epoch': best['epoch'],
        'temperature': best['temperature'],
        'soft_radius_km': args.soft_radius_km,
    }, args.output_dir / 'v10_v9_ethiopia_adapter.pt')
    np.savez_compressed(
        args.output_dir / 'probability_export.npz',
        validation_logits=best['validation_logits'],
        validation_probability=best['validation_probability'],
        development_logits=dev_logits.astype(np.float32),
        development_probability=dev_p.astype(np.float32),
        validation_indices=val_idx,
        development_indices=dev_idx,
    )

    report = {
        'model': 'candidate_ranker_v10_v9_ethiopia_adapter',
        'status': 'research-challenger',
        'parent': 'models/location/candidate_ranker/v9_phase1_ensemble_repro/v9_phase1.pt',
        'protocol': {
            'train': 'Ethiopia rows contained in first 70% chronological global partition only',
            'selection': 'Ethiopia validation (70-85%) distance-soft cross-entropy only',
            'development': '85-100% reporting only; repeatedly inspected historical block, not pristine test',
            'soft_target_radius_km': args.soft_radius_km,
            'hard_label_weight': hard_weight,
            'sweep': sweep,
            'epochs_per_trial': args.epochs,
            'train_rows': int(len(train_idx)), 'validation_rows': int(len(val_idx)), 'development_rows': int(len(dev_idx)),
        },
        'selected': {k: v for k, v in best.items() if k not in {'state', 'validation_logits', 'validation_probability', 'validation_report', 'trainable_parameters'}},
        'trainable_parameters': best['trainable_parameters'],
        'v9_ethiopia_calibrated_baseline': baseline,
        'validation': best['validation_report'],
        'validation_point_geometric_median': validation_point,
        'development': development,
        'development_point_geometric_median': development_point,
        'delta_vs_v9': {
            'validation_distance_soft_ce': best['validation_report']['distance_soft_cross_entropy'] - baseline['validation']['distance_soft_cross_entropy'],
            'validation_broad_area_score': best['validation_report']['broad_area_score'] - baseline['validation']['broad_area_score'],
            'validation_mean_error_km': best['validation_report']['mean_error_km'] - baseline['validation']['mean_error_km'],
            'development_distance_soft_ce': development['distance_soft_cross_entropy'] - baseline['development']['distance_soft_cross_entropy'],
            'development_broad_area_score': development['broad_area_score'] - baseline['development']['broad_area_score'],
            'development_mean_error_km': development['mean_error_km'] - baseline['development']['mean_error_km'],
            'development_geomedian_mean_error_km': development_point['geomedian_mean_error_km'] - baseline['development_point']['geomedian_mean_error_km'],
        },
        'trials': trials,
        'evaluation_caveat': 'No promotion claim is made from the repeatedly inspected development block. Rolling-origin and prospective frozen forecasts remain required.',
        'output_scope': 'coarse humanitarian broad-area forecasting; point metrics diagnostic only',
    }
    (args.output_dir / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    write_info(args.output_dir, ModelInfo(
        subsystem='location/candidate_ranker_ethiopia',
        version='v10_v9_adapter',
        status='research-challenger',
        description='Conservative Ethiopia late-layer specialization initialized from validated v9 with 100 km distance-soft supervision and v9 distillation.',
        metrics={
            'validation': best['validation_report'], 'development': development,
            'validation_point_geometric_median': validation_point, 'development_point_geometric_median': development_point,
            'delta_vs_v9': report['delta_vs_v9'],
        },
        lineage={'parent': str(args.parent), 'dataset': str(args.data), 'validated_reference': 'models/location/candidate_ranker/v9'},
        training={'seed': args.seed, 'selected_trial': best['trial'], 'selected_epoch': best['epoch'], 'learning_rate': best['learning_rate'], 'distill_weight': best['distill_weight'], 'train_rows': len(train_idx)},
        calibration={'temperature': best['temperature'], 'selection': 'Ethiopia chronological validation distance-soft cross-entropy'},
        notes=['v9 is untouched.', 'Final historical block is development, not pristine test.', 'Requires rolling-origin/prospective confirmation before promotion.'],
    ))
    print(json.dumps({
        'selected': report['selected'],
        'delta_vs_v9': report['delta_vs_v9'],
        'validation': report['validation'],
        'development': report['development'],
        'validation_point_geometric_median': validation_point,
        'development_point_geometric_median': development_point,
        'output_dir': str(args.output_dir),
    }, indent=2), flush=True)


if __name__ == '__main__':
    main()
