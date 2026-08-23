# Model Card: Humanitarian Forecast

## Status

**Research system. Not approved for autonomous operational decisions.**

The repository now distinguishes validated reference models from all-data production retrains. A production retrain consumes every available label and therefore does not receive a fabricated new holdout score; its `info.blt` points back to the validated ancestor used to select the recipe.

## Validated next-location reference

Model directory:

```text
models/location/candidate_ranker/v9/
```

Promoted checkpoint:

```text
models/location/candidate_ranker/v9/candidate_ranker_calibrated.pt
```

Recipe: `conflict_candidate_transformer_spatial_rings_geometric_median_v9`.

### Data

- UCDP GED 26.1 georeferenced events
- ReliefWeb country/day context
- Telegram excluded from this location pipeline
- 157,932 chronologically ordered next-event examples
- 1,517 loaded UCDP conflicts; 120 countries represented in usable sequences
- 16 historical events per example
- 19 event-history features (`geo-v2`)
- up to 32 cutoff-safe conflict-history candidate locations
- 28 features per candidate, including spatial activity/fatality rings

### Fixed chronological protocol

- first 70%: phase-1 training
- next 15%: model/calibration selection
- first 85%: fresh phase-2 retraining for the validation-selected epoch count
- final 15%: untouched test evaluation
- fixed test size: 23,690 examples
- test interval: June 2023 through December 2025

### Architecture and objective

- 3-layer Transformer history encoder
- width 128, 4 attention heads, FFN width 384, dropout 0.1
- learned country and conflict embeddings
- candidate encoder + learned candidate bias
- 677,599 parameters
- training loss: candidate cross-entropy + `50 ×` probability-weighted center distance
- expected-candidate-distance weight: `0`
- selected epoch: 12
- output calibration: temperature `1.0`, weighted geometric median

### Untouched-test performance

| Metric | Result |
|---|---:|
| Median center error | **61.01 km** |
| Mean center error | **194.36 km** |
| P90 center error | **435.09 km** |
| Within 25 km | **34.39%** |
| Candidate oracle mean | **35.99 km** |

The candidate-oracle metric asks how close the best of 32 generated candidates was after seeing the target. It is a diagnostic of candidate coverage, not top-1 forecasting accuracy.

## Temporal risk references

Historical global and Ethiopia risk artifacts have been migrated to:

```text
models/risk/base/vN/
models/risk/ethiopia/vN/
```

Their legacy metrics are preserved in each version's `info.blt`. The validated reference versions used by the new production workflow are `models/risk/base/v2/` and `models/risk/ethiopia/v4/`.

### Ethiopia v6 spatial-risk challenger

Model directory:

```text
models/risk/ethiopia/v6/
```

v6 is a **coarse humanitarian early-warning** challenger, not a precise next-event location model. It preserves the v5 local temporal-risk predictor as a stable expert and adds cutoff-safe PRIO-GRID neighborhood histories. The spatial tensor contains the eight local channels plus first- and second-ring historical neighbor aggregates and historical active-neighbor counts. A separate spatial model is optimized for future intensity so the rare-event ranking and magnitude objectives do not fight over one checkpoint.

The escalation ensemble weight, checkpoint epochs, affine probability calibration, classification threshold, and split-conformal intensity radii are selected only on the chronological validation partition. The final 15% has been inspected during iterative research and is therefore reported as a **development holdout**, not as a pristine prospective test.

| Metric | v5 development | v6 development | Relative change |
|---|---:|---:|---:|
| Escalation average precision | 0.13436 | **0.14688** | **+9.32%** |
| Intensity MAE (log1p) | 0.16383 | **0.13684** | **-16.47%** |
| Brier score | 0.06023 | **0.06009** | lower is better |
| Log loss | 0.23368 | **0.23258** | lower is better |

The paired bootstrap 95% interval for the v6-v5 average-precision difference on the development block is **+0.0040 to +0.0253**. This is a development diagnostic, not a substitute for future prospective scoring. Promotion still requires rolling-origin and genuinely prospective confirmation.

v6 also stores validation split-conformal absolute residual radii for intensity at 80%, 90%, and 95% target coverage. These intervals describe historical model uncertainty; they are not guarantees about field conditions.

## Production all-data policy

`main.py workflow full` trains global models on all currently available labels, then fine-tunes Ethiopia-specialized copies. The default new version destinations are:

```text
models/risk/base/v3/
models/location/candidate_ranker/v10/
models/risk/ethiopia/v5/
models/location/candidate_ranker_ethiopia/v1/
```

These directories are created when the workflow is actually run. Production training includes data that formerly belonged to the evaluation period, but each historical training example remains causal: its features and candidate bank use only observations available by that example's own cutoff.

Because no data is withheld from an all-data production artifact, its `info.blt` contains `held_out_evaluation: false`. Generalization claims continue to come from the corresponding validated reference until a later-period evaluation becomes available.

## Required interpretation

- Forecasts are probabilistic research signals, not verified incidents.
- Candidate-set oracle metrics are not top-1 accuracy.
- Training loss is not a substitute for held-out geographic error.
- Exact model performance claims must identify the model version and evaluation protocol.
- Public/operational humanitarian decisions require independent corroboration and human review.
