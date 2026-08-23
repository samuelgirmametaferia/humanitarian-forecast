#!/usr/bin/env python3
"""Predict the next war zone using the hierarchical location model.

Takes parameters like forecast horizon, current activity levels, and geographic
context to produce probability-weighted circle predictions for the next conflict
location. Research-only output; not verified front lines or evacuation orders.
"""

from __future__ import annotations

import argparse, json, math
from pathlib import Path
import numpy as np, torch
from hierarchical_location_model import HierarchicalLocationTransformer


def main():
    p = argparse.ArgumentParser(description="Predict next war zone location")
    p.add_argument('--model', type=Path, required=True, help="Path to model checkpoint .pt file")
    p.add_argument('--data', type=Path, required=True, help="Path to prepared .npz data file")
    p.add_argument('--index', type=int, default=-1, help="Index into data (default: last example)")
    p.add_argument('--horizon-days', type=int, default=7, help="Forecast horizon in days (1-90)")
    p.add_argument('--fatalities', type=float, default=0, help="Logged fatalities scenario override")
    p.add_argument('--civilian-fatalities', type=float, default=0, help="Logged civilian fatalities override")
    p.add_argument('--violence-type', type=int, default=None, choices=range(3),
                   help="Violence type override (0=best, 1=intermediate, 2=worst)")
    p.add_argument('--source-count', type=float, default=None, help="Logged source count override")
    p.add_argument('--reliefweb-activity', type=float, default=None, help="Logged reliefweb activity override")
    p.add_argument('--reliefweb-attack', type=float, default=None, help="Logged reliefweb attack override")
    p.add_argument('--reliefweb-harm', type=float, default=None, help="Logged reliefweb harm override")
    p.add_argument('--reliefweb-displacement', type=float, default=None, help="Logged reliefweb displacement override")
    p.add_argument('--output', type=Path, default=None, help="Output JSON file (default: print to stdout)")
    a = p.parse_args()

    # Load model checkpoint
    state = torch.load(a.model, map_location='cpu', weights_only=False)
    model = HierarchicalLocationTransformer(**state['model_config'])
    model.load_state_dict(state['model_state'])
    model.eval()

    # Load data
    d = np.load(a.data)
    i = a.index if a.index >= 0 else len(d['x']) + a.index
    x = d['x'][i].copy()
    meta = json.loads(str(d['meta'][i]))

    # Apply scenario overrides (parameter-based war zone prediction)
    latest = x[-1]

    # Forecast horizon parameter - encode as angle and log-horizon feature
    if 'forecast_horizon_days' in vars(a):
        requested_gap = max(1, int(a.horizon_days))
    else:
        requested_gap = max(1, int(a.horizon_days))
    original_gap = max(1, int(meta['gap_days']))
    delta = requested_gap - original_gap
    days_ago = np.expm1(x[max(0, i-1):i, 3] * 6.0) if x.shape[1] > 3 else np.array([0.0])
    x[:, 3] = np.log1p(np.maximum(0, days_ago + delta)) / 6.0

    # Apply fatalities parameter (log1p transformed, divided by 6 as in training)
    if a.fatalities > 0:
        latest[4] = math.log1p(max(0, a.fatalities)) / 6
    if a.civilian_fatalities > 0:
        latest[5] = math.log1p(max(0, a.civilian_fatalities)) / 6

    # Apply violence type parameter
    if a.violence_type is not None:
        latest[6:9] = 0
        latest[5 + a.violence_type] = 1

    # Apply source count parameter
    if a.source_count is not None:
        latest[10] = math.log1p(max(0, a.source_count)) / 5

    # Apply ReliefWeb scenario parameters
    reliefweb_keys = [
        ('reliefweb_activity', 11),
        ('reliefweb_attack', 12),
        ('reliefweb_harm', 13),
        ('reliefweb_displacement', 14),
    ]
    for key, idx in reliefweb_keys:
        if locals().get(key, None) is not None:
            latest[idx] = math.log1p(max(0, locals()[key])) / 8

    # Add temporal features (day-of-year sin/cos) based on forecast horizon
    target_date = np.datetime64(meta['target_date']) + np.timedelta64(delta, 'D')
    year_start = target_date.astype('datetime64[Y]')
    day_of_year = int((target_date - year_start) / np.timedelta64(1, 'D')) + 1
    angle = 2 * math.pi * day_of_year / 365.25
    if x.shape[1] >= 19:
        x[:, 17] = math.sin(angle)
        x[:, 18] = math.cos(angle)

    # Run inference
    with torch.no_grad():
        inp = torch.from_numpy(x[None]).float()
        logits, centers, sigmas = model(inp, horizon_days=torch.tensor([a.horizon_days]))

    # Post-process predictions
    probs = logits.softmax(-1)[0].numpy()
    centers = centers[0].numpy() * 1000  # convert from normalized to km
    sigmas = sigmas[0].numpy() * 1000

    # Compute predicted coordinates from anchor + offsets
    alat = float(meta['anchor_lat'])
    alon = float(meta['anchor_lon'])

    predictions = []
    for rank, k in enumerate(np.argsort(-probs), 1):
        east, north = centers[k]
        lat = alat + north / 111.32
        lon = alon + east / (111.32 * max(0.1, math.cos(math.radians(alat))))
        radius = max(50, sigmas[k])  # minimum 50km radius
        predictions.append({
            'rank': rank,
            'probability': round(float(probs[k]), 4),
            'center_latitude_coarse': round(lat * 4) / 4,
            'center_longitude_coarse': round(lon * 4) / 4,
            'uncertainty_radius_km': round(radius),
        })

    result = {
        'country': meta['country'],
        'conflict': meta['conflict'],
        'forecast_horizon_days': int(a.horizon_days),
        'scenario_overrides': {
            'fatalities': a.fatalities,
            'civilian_fatalities': a.civilian_fatalities,
            'violence_type': a.violence_type,
            'source_count': a.source_count,
        },
        'predictions': predictions,
        'warning': 'Research-only coarse predictions; not verified fronts or evacuation orders. '
                   'Do not use for operational decision-making.',
    }

    # Human-readable output
    output_text = json.dumps(result, indent=2)

    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(output_text)
        print(f"Predictions written to {a.output}")
    else:
        print(output_text)


if __name__ == '__main__':
    main()