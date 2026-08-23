from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from flask import Flask, render_template, request, jsonify

from hierarchical_location_model import HierarchicalLocationTransformer


app = Flask(__name__, template_folder='templates', static_folder='static')

# Global model variable - loaded once at startup
model = None
model_config = None

# Data meta for feature preparation
data_meta = None
data_x = None


def load_model(checkpoint_path: Path):
    global model, model_config
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model_config = state['model_config']
    model = HierarchicalLocationTransformer(**state['model_config'])
    model.load_state_dict(state['model_state'])
    model.eval()
    print(f"Model loaded from {checkpoint_path}")


def load_data(data_path: Path):
    global data_x, data_meta
    import numpy as np
    d = np.load(data_path)
    data_x = d['x']
    # Store meta for lookup; in production would use proper indexing
    data_meta = [json.loads(str(m)) for m in d['meta']]
    print(f"Data loaded from {data_path}, {len(data_x)} examples")


def prepare_features(meta, horizon_days=7, fatalities=0, civilian_fatalities=0,
                    violence_type=None, source_count=None,
                    reliefweb_activity=None, reliefweb_attack=None,
                    reliefweb_harm=None, reliefweb_displacement=None):
    """Prepare feature vector from parameters and metadata.

    Uses only approved data sources - UCDP + ReliefWeb.
    """
    # Start with a default feature sequence (16 events, 15 features)
    n_events = 16
    n_features = 19
    features = np.zeros((n_events, n_features), dtype=np.float32)

    # Encode forecast horizon as day-of-year sin/cos
    from datetime import date, timedelta
    target_date = date.fromisoformat(meta['target_date']) + timedelta(days=horizon_days)
    year_start = target_date.replace(month=1, day=1)
    day_of_year = (target_date - year_start).days + 1
    angle = 2 * math.pi * day_of_year / 365.25
    features[-1, 17] = math.sin(angle)  # sin feature
    features[-1, 18] = math.cos(angle)  # cos feature

    # Encode current event (latest) with scenario parameters
    latest = features[-1]  # last event row

    # Fatalities parameter
    if fatalities > 0:
        latest[4] = math.log1p(fatalities) / 6
    if civilian_fatalities > 0:
        latest[5] = math.log1p(civilian_fatalities) / 6

    # Violence type one-hot
    if violence_type is not None:
        latest[6:9] = 0
        latest[5 + violence_type] = 1

    # Source count
    if source_count is not None:
        latest[10] = math.log1p(source_count) / 5

    # ReliefWeb features (approved source)
    if reliefweb_activity is not None:
        latest[11] = math.log1p(reliefweb_activity) / 8
    if reliefweb_attack is not None:
        latest[12] = math.log1p(reliefweb_attack) / 8
    if reliefweb_harm is not None:
        latest[13] = math.log1p(reliefweb_harm) / 8
    if reliefweb_displacement is not None:
        latest[14] = math.log1p(reliefweb_displacement) / 8

    # Anchor coordinates for geolocation
    alat = float(meta.get('anchor_lat', 0))
    alon = float(meta.get('anchor_lon', 0))

    return features, alat, alon


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for war zone prediction with approved data only."""
    try:
        data = request.get_json()

        # Extract parameters
        horizon_days = data.get('horizon', 7)
        fatalities = data.get('fatalities', 0)
        civilian_fatalities = data.get('civilian_fatalities', 0)
        violence_type = data.get('violence_type')
        source_count = data.get('source_count')
        reliefweb_activity = data.get('reliefweb_activity')
        reliefweb_attack = data.get('reliefweb_attack')
        reliefweb_harm = data.get('reliefweb_harm')
        reliefweb_displacement = data.get('reliefweb_displacement')

        # Get example index (or use last from dataset)
        example_idx = data.get('index', -1)

        if not model or not data_meta:
            return jsonify({'error': 'Model not loaded'}), 500

        # Bounds check
        if example_idx < 0 or example_idx >= len(data_meta):
            example_idx = len(data_meta) - 1

        # Prepare features using only approved parameters
        features, alat, alon = prepare_features(
            data_meta[example_idx],
            horizon_days=horizon_days,
            fatalities=fatalities,
            civilian_fatalities=civilian_fatalities,
            violence_type=violence_type,
            source_count=source_count,
            reliefweb_activity=reliefweb_activity,
            reliefweb_attack=reliefweb_attack,
            reliefweb_harm=reliefweb_harm,
            reliefweb_displacement=reliefweb_displacement,
        )

        # Run inference
        with torch.no_grad():
            inp = torch.from_numpy(features[None]).float()
            logits, probs, head_logits, head_centers, head_sigmas, uncert = model(inp)

        # Post-process
        probs_val = float(probs.numpy().flatten()[0])
        centers_km = head_centers.numpy() * 1000
        sigmas_km = head_sigmas.numpy() * 1000

        alat = float(alat)
        alon = float(alon)

        predictions = []
        for rank, k in enumerate(np.argsort(-probs_val) if hasattr(probs_val, 'flatten') else [0], 1):
            # Use the top prediction
            east, north = centers_km[0]
            lat = alat + north / 111.32
            lon = alon + east / (111.32 * max(0.1, math.cos(math.radians(alat))))
            radius = max(50, float(sigmas_km[0]))
            predictions.append({
                'rank': rank,
                'probability': round(probs_val, 4),
                'center_latitude_coarse': round(lat * 4) / 4,
                'center_longitude_coarse': round(lon * 4) / 4,
                'uncertainty_radius_km': round(radius),
            })

        result = {
            'country': data_meta[example_idx]['country'] if data_meta else 'unknown',
            'conflict': data_meta[example_idx]['conflict'] if data_meta else 'unknown',
            'forecast_horizon_days': horizon_days,
            'predictions': predictions,
            'warning': 'Research-only coarse predictions; not verified fronts or evacuation orders. '
                       'Do not use for operational decision-making. '
                       'Data: UCDP georeferenced events + ReliefWeb context only. '
                       'Telegram excluded per data policy.',
        }

        return jsonify(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'data_examples': len(data_x) if data_x else 0,
        'data_policy': 'UCDP + ReliefWeb only; Telegram excluded'
    })


if __name__ == '__main__':
    # Load model and data at startup
    checkpoint_path = Path('checkpoints_mixture_geo_v2/mixture_best.pt')
    data_path = Path('data/location/next_location_geo_v2.npz')

    if checkpoint_path.exists():
        load_model(checkpoint_path)
    else:
        print(f"Warning: checkpoint not found at {checkpoint_path}")

    if data_path.exists():
        load_data(data_path)
    else:
        print(f"Warning: data not found at {data_path}")

    # Run the web server
    print("Starting web server at http://0.0.0.0:5000")
    print("Data policy: UCDP georeferenced events + ReliefWeb context; Telegram excluded")
    app.run(host='0.0.0.0', port=5000, debug=False)