# Humanitarian Location Forecast — Validated v9 Reference

## Recommended historical/demo inference

The validated reference checkpoint now lives at:

```text
models/location/candidate_ranker/v9/candidate_ranker_calibrated.pt
```

Run it through the unified controller:

```bash
.venv/bin/python main.py run location.predict -- --index -1 --top 32
```

For the explicitly retrospective reconstruction mode:

```bash
.venv/bin/python main.py run location.predict -- --index -1 --retrospective-reconstruction
```

That mode uses the target-day UCDP record and is a reconstruction, not a forecast of an unknown future event.

## Verified fixed-test results

All 23,690 examples in the chronological June 2023–December 2025 test interval remain included.

| Output | Result | Meaning |
|---|---:|---|
| Prospective calibrated center, median error | **61.01 km** | One strict-cutoff center |
| Prospective calibrated center, mean error | **194.36 km** | One strict-cutoff center |
| Prospective calibrated center, p90 | **435.09 km** | One strict-cutoff center |
| Prospective within 25 km | **34.39%** | One strict-cutoff center |
| 32-location candidate oracle mean | **35.99 km** | Distance to closest issued candidate; diagnostic only |

The 35.99 km result is a high-recall candidate-set diagnostic and must not be presented as top-1 accuracy.

## Model

The v9 model combines a 3-layer Transformer event-history encoder, country and conflict embeddings, 32 cutoff-safe conflict-location candidates, candidate frequency/recency/transition features, 7/30/90/365-day activity, candidate-centered activity/fatality rings at 25/50/100/250 km scales, and a probability-weighted center objective. Its validation-selected output aggregation is a weighted geometric median at temperature 1.0.

The production workflow retrains this validated recipe on all available labels and then fine-tunes an embedding-compatible copy on Ethiopia. Those all-data artifacts are intentionally marked as having no fresh holdout evaluation; use v9's `info.blt` for the measured reference performance.
