# Humanitarian area-risk forecaster

This project trains a **coarse, probabilistic humanitarian-risk model**. It is
not a tactical tracker and must not be used to publish exact positions, routes,
individual identities, or real-time force movements.

The training design has two stages:

1. International pretraining on UCDP GED georeferenced conflict events, augmented
   with daily ReliefWeb conflict-report activity. This teaches generic temporal
   escalation and persistence patterns across geographic cells and countries.
2. Ethiopia fine-tuning on extracted Telegram events grouped into coarse named
   administrative areas. The base and fine-tuned checkpoints are saved separately.

The model is a compact temporal Transformer suitable for Apple MPS and a 16 GB
Mac. Calling a model “large” does not make it accurate; this architecture is sized
to the available data and hardware and is evaluated with a chronological holdout.

## Safety and interpretation

- Predictions are relative risk signals, not verified facts or evacuation orders.
- Outputs are delayed and aggregated to named administrative areas.
- A forecast must display its observation cutoff and uncertainty.
- Human review and independent sources are required before public communication.
- ReliefWeb report activity is a proxy for documented humanitarian concern, not
  ground-truth battlefield control.

## Run

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

.venv/bin/python download_reliefweb.py --start-year 2000 --end-year 2026
.venv/bin/python build_datasets.py \
  --reliefweb data/reliefweb_conflict.jsonl.gz \
  --ucdp data/raw/ged261-csv.zip \
  --ethiopia-db /Users/sam/Documents/wfp/database.sqlite

.venv/bin/python train.py --stage base --data-dir data/processed_v2 \
  --checkpoint-dir checkpoints_v2
.venv/bin/python train.py --stage ethiopia \
  --data-dir data/processed_v3 --checkpoint-dir checkpoints_v4 \
  --resume checkpoints_v2/base_best.pt --ranking-weight 0.2
```

Use `--max-records` during development. The downloader is resumable and writes
one gzip JSON object per report.

See `MODEL_CARD.md` before using either checkpoint.

## Current recommended model: Telegram-free location forecast

The current location pipeline uses only UCDP georeferenced events and ReliefWeb
context. Telegram-derived data is excluded.

```bash
.venv/bin/python build_next_location_dataset.py \
  --ucdp data/raw/ged261-csv.zip \
  --reliefweb data/reliefweb_balanced.jsonl.gz \
  --output data/location/next_location.npz

PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python train_location.py \
  --data data/location/next_location.npz \
  --output-dir checkpoints_location
```

Render a coarse historical example:

```bash
.venv/bin/python predict_location.py \
  --data data/location/next_location.npz \
  --checkpoint checkpoints_location/location_best.pt \
  --calibration checkpoints_location/location_calibration.json \
  --index -1
```

## Multimodal scenario model

The higher-capability location model returns five probability-weighted circles and
accepts current-situation overrides:

```bash
.venv/bin/python predict_mixture.py \
  --data data/location/next_location_geo_v2.npz \
  --checkpoint checkpoints_mixture_geo_v2/mixture_best.pt \
  --calibration checkpoints_mixture_geo_v2/mixture_calibration.json \
  --scenario example_scenario.json \
  --index -1
```

Scenario fields are `forecast_horizon_days` (1–90), `fatalities`,
`civilian_fatalities`, `violence_type` (1–3),
`source_count`, `reliefweb_activity`, `reliefweb_attack`, `reliefweb_harm`, and
`reliefweb_displacement`.

The current geography-aware model reaches 245.3 km top-1 mean error on the
untouched future test interval. The project stretch goal is 25 km; that target is
not yet achieved. See `plan.md` for the measured path toward it.
